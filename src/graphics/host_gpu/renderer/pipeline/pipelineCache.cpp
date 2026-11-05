#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/aftermath.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "common/threads.h"
#include "common/switches.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/computePipelineQueue.h"
#include "graphics/host_gpu/renderer/pipeline/drawSpeculation.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shaderCallLink.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// Set once by the PipelineCache constructor, before any shader is translated: the device has
// descriptor indexing and --bindless asked for it.
bool g_bindless_images = false;

// Set while a look-ahead predicts a draw's or dispatch's programs: permutations it compiles are
// journaled only once a real draw or dispatch uses them (see Permutation::unrecorded).
thread_local bool t_speculative_lookup = false;

struct SpeculativeLookup {
	SpeculativeLookup() { t_speculative_lookup = true; }
	~SpeculativeLookup() { t_speculative_lookup = false; }
	KYTY_CLASS_NO_COPY(SpeculativeLookup);
};

// A compute shader as PipelineCache::m_compute_sources keys it: its address, and bit 63 for wave32.
uint64_t ComputeSourceKey(const HW::ComputeShaderInfo& cs) {
	return cs.cs_regs.data_addr | (cs.cs_regs.wave_size == 32u ? (uint64_t {1} << 63u) : 0u);
}

void DestroyPipelineObjects(const GraphicContext& graphics, const PipelineCache::Pipeline& pipeline) {
	if (pipeline.pipeline != nullptr) {
		graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	}
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	if (pipeline.pixel_set_layout != nullptr) {
		graphics.device.destroyDescriptorSetLayout(pipeline.pixel_set_layout, nullptr);
	}
}

void RecordPipelineCreated(std::chrono::steady_clock::time_point begin) {
	Common::DebugCounters::Add(Common::DebugCounters::Counter::PipelinesCreated);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::PipelineCreateNs,
	                           static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                                     std::chrono::steady_clock::now() - begin)
	                                                     .count()));
}

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC3:{}:{:08x}:{:08x}:{:08x}:{}:opt={}\n", KYTY_SHADER_CACHE_KEY,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid,
	                   static_cast<uint32_t>(Config::GetShaderOptimizationType()));
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// KYTY_PRO_SWAPPC_LINK with KYTY_PRO_SWAPPC_LINK_OFF_FILE: while the file exists, dispatches use
// the unlinked program (A/B in one run).
bool LinkShaderCalls(const ShaderParams& params, ShaderParams& linked) {
	static FileSwitch off(KYTY_GETENV("KYTY_PRO_SWAPPC_LINK_OFF_FILE"), "swappc-link");
	return !off.Exists() && ShaderCallLink::Link(params, linked);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// GPU-written bytes fail the read instead of waiting for a readback (upstream reads them back
	// through TryReadBufferBacking); the shader then keeps such reads on the GPU.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, values.data(), values.size_bytes());
}

// Guest memory never lives in the lowest megabyte. A walk through a null base there is on a
// path the shader guards with its own null test (seen in Ghost of Yotei), so it reads zero.
constexpr uint64_t NullGuestRegionEnd = 0x100000;

// Per-draw resource tables can share a tracker page with GPU-written data, which protects the
// whole page. Reading their clean bytes from the backing avoids a fault and a GPU wait; bytes
// the GPU did write still fault and read back.
// Whether [address, address + size) lies in mapped guest memory. A look-ahead (pipeline prefetch)
// can evaluate a table pointer the game has not written yet: the title crashed the host reading
// 0x4000bd6d6000 (2026-10-19). Such reads give zero, as the null region does.
static bool GuestRangeMapped(uint64_t address, uint64_t size) {
	return size == 0 || Libs::LibKernel::Memory::MappedRangeSize(address, size) == size;
}

bool ReadShaderGuestMemoryOnGpuThread(void*, uint64_t address, std::span<uint32_t> values) {
	if (address < NullGuestRegionEnd || !GuestRangeMapped(address, values.size_bytes())) {
		std::fill(values.begin(), values.end(), 0u);
		return true;
	}
	Libs::LibKernel::Memory::ReadGuestOnGpuThread(address, values.data(), values.size_bytes());
	return true;
}

// A whole 64-byte block for the reads above, when each of them would be a plain copy.
bool ReadShaderGuestBlockOnGpuThread(void*, uint64_t address, std::span<uint32_t> values) {
	if (address < NullGuestRegionEnd || !GuestRangeMapped(address, values.size_bytes())) {
		return false;
	}
	return Libs::LibKernel::Memory::TryReadGuestPlainOnGpuThread(address, values.data(),
	                                                             values.size_bytes());
}

// The draw speculation thread's reader for all three kinds of read (userdata: the BufferCache).
// It reads the backing only, since a guest address can fault, and fails on pages the lock-free
// hint calls GPU-dirty. Its results only matter when adoption on the GPU thread repeats the read
// through the readers above and gets the same result.
bool ReadShaderGuestMemoryAhead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	const auto* buffers = static_cast<const BufferCache*>(userdata);
	if (values.empty() || buffers->IsPageGpuDirtyHint(address) ||
	    buffers->IsPageGpuDirtyHint(address + values.size_bytes() - 1u)) {
		return false;
	}
	return Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes());
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

// Shaders that failed to translate or compile are listed in _PipelineCache/failed_shaders.txt
// (one hex hash per line, kept across runs). A failure prints a red console line; a listed shader
// that now translates and compiles prints a magenta line and leaves the list, so a translator fix
// is visible in the console the first time the game uses that shader again.
class ShaderOutcomes {
public:
	void Failed(uint64_t hash, const char* stage) {
		std::scoped_lock lock(m_mutex);
		Load();
		if (!m_failed.insert(hash).second) {
			return;
		}
		Save();
		std::printf("\x1b[91mshader %016llx failed (%s): skipped, listed in %s\x1b[0m\n",
		            static_cast<unsigned long long>(hash), stage, Path().string().c_str());
		std::fflush(stdout);
	}
	void Works(uint64_t hash) {
		std::scoped_lock lock(m_mutex);
		Load();
		if (m_failed.erase(hash) == 0) {
			return;
		}
		Save();
		std::printf("\x1b[95mshader %016llx works now: translated and compiled (it failed before)"
		            "\x1b[0m\n",
		            static_cast<unsigned long long>(hash));
		std::fflush(stdout);
	}

private:
	static std::filesystem::path Path() { return std::filesystem::path("_PipelineCache") / "failed_shaders.txt"; }
	void Load() {
		if (m_loaded) {
			return;
		}
		m_loaded = true;
		if (auto* file = std::fopen(Path().string().c_str(), "rb"); file != nullptr) {
			unsigned long long hash = 0;
			while (std::fscanf(file, "%llx", &hash) == 1) {
				m_failed.insert(hash);
			}
			std::fclose(file);
		}
	}
	void Save() const {
		std::error_code ec;
		std::filesystem::create_directories(Path().parent_path(), ec);
		if (auto* file = std::fopen(Path().string().c_str(), "wb"); file != nullptr) {
			for (const auto hash: m_failed) {
				std::fprintf(file, "%016llx\n", static_cast<unsigned long long>(hash));
			}
			std::fclose(file);
		}
	}

	std::mutex                   m_mutex;
	bool                         m_loaded = false;
	std::unordered_set<uint64_t> m_failed;
};

ShaderOutcomes& Outcomes() {
	static ShaderOutcomes outcomes;
	return outcomes;
}

// A skipped draw or dispatch is visible in the frame; name the shader once per hash.
void ReportMaterializeFailure(uint64_t shader_hash) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> reported;
	std::scoped_lock                    lock(mutex);
	if (reported.insert(shader_hash).second) {
		std::printf("Shader 0x%016" PRIx64 ": resource materialization failed; skipping its "
		            "draws/dispatches until it resolves\n",
		            shader_hash);
	}
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

// A journal record's compile options; `params` and the options point into the record.
ShaderRecompiler::CompileOptions ReplayOptions(ShaderPrecompile::PermutationRecord& record,
                                               ShaderParams&                       params) {
	params.code            = record.code;
	params.back_code       = record.back_code;
	params.hash            = record.hash;
	params.user_data_count = record.user_data_count;
	ShaderRecompiler::CompileOptions options;
	options.stage          = record.stage;
	options.shader_hash    = record.hash;
	options.user_data      = std::span(params.user_data).first(params.user_data_count);
	options.back_code      = params.back_code;
	options.user_data_base = record.user_data_base;
	options.wave_size      = record.wave_size;
	options.dump_ir        = false;
	options.early_dump     = false;
	options.dump_label     = "ShaderPrecompile";
	options.bindless_images = g_bindless_images;
	std::visit(
	    [&](auto& info) {
		    using Info = std::decay_t<decltype(info)>;
		    if constexpr (std::is_same_v<Info, ShaderVertexInputInfo>)
			    options.input_info.vertex = &info;
		    else if constexpr (std::is_same_v<Info, ShaderPixelInputInfo>)
			    options.input_info.pixel = &info;
		    else
			    options.input_info.compute = &info;
	    },
	    record.info);
	return options;
}

// KYTY_DEBUG_REPLAY_VERIFY=1 (diagnostic): the first live use of each replayed permutation also
// compiles it from the live inputs and reports a SPIR-V difference (see VerifyReplayed).
bool ReplayVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = KYTY_GETENV("KYTY_DEBUG_REPLAY_VERIFY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

} // namespace

// KYTY_PERMUTATION_LOG=1 (diagnostic): describe why each runtime-compiled permutation is new.
[[nodiscard]] static bool PermutationLogEnabled() {
	static const bool enabled = [] {
		const char* value = KYTY_GETENV("KYTY_PERMUTATION_LOG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// The first few fields in which two specializations differ, e.g. "img[2].mip_count 1->12".
[[nodiscard]] static std::string
SpecializationDiff(const ShaderRecompiler::IR::ResourceSpecialization& a,
                   const ShaderRecompiler::IR::ResourceSpecialization& b) {
	std::string text;
	size_t      shown = 0;
	const auto  add   = [&](const char* kind, size_t index, const char* field, uint64_t from,
                         uint64_t to) {
        if (from != to && shown++ < 6) {
            text += fmt::format(" {}[{}].{} {}->{}", kind, index, field, from, to);
        }
	};
	if (a.buffers.size() != b.buffers.size() || a.images.size() != b.images.size()) {
		return fmt::format(" counts buffers {}->{} images {}->{}", a.buffers.size(),
		                   b.buffers.size(), a.images.size(), b.images.size());
	}
	for (size_t i = 0; i < a.buffers.size(); i++) {
		const auto& x = a.buffers[i];
		const auto& y = b.buffers[i];
		add("buf", i, "stride", x.packed_stride, y.packed_stride);
		add("buf", i, "format", static_cast<uint64_t>(x.descriptor_format),
		    static_cast<uint64_t>(y.descriptor_format));
		add("buf", i, "swizzle", x.descriptor_swizzle, y.descriptor_swizzle);
		add("buf", i, "oob", static_cast<uint64_t>(x.zero_stride_oob),
		    static_cast<uint64_t>(y.zero_stride_oob));
	}
	for (size_t i = 0; i < a.images.size(); i++) {
		const auto& x = a.images[i];
		const auto& y = b.images[i];
		add("img", i, "class", static_cast<uint64_t>(x.numeric_class),
		    static_cast<uint64_t>(y.numeric_class));
		add("img", i, "dim", static_cast<uint64_t>(x.dimension), static_cast<uint64_t>(y.dimension));
		add("img", i, "mip_count", x.mip_count, y.mip_count);
		add("img", i, "conversion", static_cast<uint64_t>(x.conversion_format),
		    static_cast<uint64_t>(y.conversion_format));
		add("img", i, "swizzle", x.shader_swizzle, y.shader_swizzle);
		add("img", i, "indirect_root", x.indirect_root, y.indirect_root);
		add("img", i, "indirect_offset", x.indirect_mapping_offset, y.indirect_mapping_offset);
		add("img", i, "indirect_iterations", x.indirect_search_iterations,
		    y.indirect_search_iterations);
		add("img", i, "cube", x.cube ? 1u : 0u, y.cube ? 1u : 0u);
		add("img", i, "fmask", x.fmask ? 1u : 0u, y.fmask ? 1u : 0u);
		add("img", i, "clamp_half", x.clamp_half_float ? 1u : 0u, y.clamp_half_float ? 1u : 0u);
	}
	return text.empty() ? " (same specialization)" : text;
}

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		uint64_t              code_variant    = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
		// Compiled by a speculative lookup (a look-ahead): its journal record, appended when a
		// draw or dispatch first uses the permutation, so mispredictions stay out of the journal.
		std::optional<ShaderPrecompile::PermutationRecord> unrecorded;
		// KYTY_DEBUG_REPLAY_VERIFY=1: the journal record a replayed permutation was compiled from,
		// until its first live use checks it (VerifyReplayed).
		std::shared_ptr<const ShaderPrecompile::PermutationRecord> replay_source;
		size_t                                                     replay_record = 0;
	};

	// A permutation's shader module and info, compiled on any thread.
	struct CompiledModule {
		ShaderRecompiler::IR::CompiledShaderInfo program;
		vk::ShaderModule                         module      = nullptr;
		size_t                                   spirv_words = 0;
		// The emitter rejected the planned program: the shader is skipped like an untranslatable one.
		bool                                     failed      = false;
	};

	// A permutation compiling on a worker thread.
	struct PendingPermutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint32_t                                     push_data_cursor = 0;
		std::future<CompiledModule>                  compiled;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::MaterializationMemo    memo;
		std::vector<Permutation>                    permutations;
		// Permutations only grow and never repeat a (push data start, specialization) pair, so
		// the last hit stays valid while the specialization and push data cursor are unchanged.
		uint32_t                                    last_permutation = UINT32_MAX;
		uint32_t                                    last_push_cursor = 0;
		bool                                        skip_dispatch = false;
		std::vector<PendingPermutation>             pending;
		// Set once a GPU-thread refresh succeeded (its plan is compiled): only then may the draw
		// speculation thread refresh this plan (see Speculate).
		std::atomic<bool>                           speculable {false};
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	static const char* StageName(ShaderType stage) {
		switch (stage) {
			case ShaderType::Vertex: return "vs";
			case ShaderType::Mesh: return "ms";
			case ShaderType::Local: return "ls";
			case ShaderType::TessellationControl: return "hs";
			case ShaderType::TessellationEvaluation: return "ds";
			case ShaderType::Pixel: return "ps";
			case ShaderType::Compute: return "cs";
			default: EXIT("invalid pipeline shader stage\n");
		}
		return nullptr;
	}

	// Compiles a translated permutation to its SPIR-V module. Reads only its arguments, so it may
	// run on a worker thread.
	static CompiledModule CompileModule(vk::Device device, std::span<const uint32_t> code,
	                                    const ShaderRecompiler::CompileOptions&             options,
	                                    ShaderRecompiler::TranslateResult                   translated,
	                                    const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                    uint32_t push_data_start_dword) {
		const char*                     stage_name = StageName(options.stage);
		ShaderRecompiler::CompileResult result;
		try {
			result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization,
			                                          push_data_start_dword);
		} catch (const ShaderRecompiler::Spirv::ValidationError& error) {
			Log::WriteToConsoleAndLog(fmt::format("Warning: skipping shader 0x{:016x}: {}\n",
			                                      options.shader_hash, error.what()));
			return {.failed = true};
		}
		DumpShaderOriginal(stage_name, options.shader_hash, code);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
		Aftermath::NoteGuestShader(result.spirv, options.shader_hash, stage_name);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		return {
		    .program     = std::move(result.program).TakeCompiledInfo(),
		    .module      = module,
		    .spirv_words = result.spirv.size(),
		};
	}

	// Numbers a compiled permutation.
	Permutation MakePermutation(const ShaderRecompiler::CompileOptions&      options,
	                            ShaderRecompiler::IR::ResourceSpecialization specialization,
	                            CompiledModule                               compiled) {
		if (PermutationLogEnabled()) [[unlikely]] {
			std::printf("spirv: id=%llu %s hash=%016llx words=%zu\n",
			            static_cast<unsigned long long>(next_shader_id + 1), StageName(options.stage),
			            static_cast<unsigned long long>(options.shader_hash), compiled.spirv_words);
		}
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(compiled.spirv_words), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(compiled.program),
		    .handle         = {.id = ++next_shader_id, .module = compiled.module},
		};
	}

	// KYTY_DEBUG_REPLAY_VERIFY=1, on the first live use of a replayed permutation: compiles it again
	// from its journal record and from the live inputs, as Get would compile it now, and reports
	// different SPIR-V with every journaled input that differs. The replayed module stays in use.
	template <typename InputInfo>
	void VerifyReplayed(const ShaderParams& params, const InputInfo& input_info, ShaderType stage,
	                    uint32_t push_data_cursor, const Permutation& permutation) {
		const auto spirv_hash = [](ShaderRecompiler::TranslateResult translated,
		                           const ShaderRecompiler::CompileOptions&             options,
		                           const ShaderRecompiler::IR::ResourceSpecialization& specialization,
		                           uint32_t push_data_start) -> uint64_t {
			if (translated.skip_dispatch) {
				return 0;
			}
			const auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
			                                                     specialization, push_data_start);
			return XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t));
		};
		auto         record = *permutation.replay_source;
		ShaderParams replay_params;
		const auto   replay_options = ReplayOptions(record, replay_params);
		const auto   replay_hash =
		    spirv_hash(ShaderRecompiler::TranslateProgram(replay_params.code, replay_options),
		               replay_options, record.specialization, record.push_data_start_dword);

		auto live_input  = input_info;
		live_input.stage = {};
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		auto live_options = MakeOptions(stage, params.hash, user_data, params.back_code, live_input);
		live_options.dump_ir    = false;
		live_options.early_dump = false;
		const auto live_hash =
		    spirv_hash(ShaderRecompiler::TranslateProgram(params.code, live_options), live_options,
		               permutation.specialization, push_data_cursor);
		if (live_hash == replay_hash) {
			return;
		}

		// What the journal would hold for the live compile, against the replayed record.
		const auto live = ShaderPrecompile::Capture(params, live_options, permutation.specialization,
		                                            push_data_cursor, live_input);
		std::string differences;
		const auto  differ = [&](const char* name, auto a, auto b) {
			if (a != b) {
				differences += fmt::format(" {}:{}->{}", name, a, b);
			}
		};
		differ("push_start", record.push_data_start_dword, live.push_data_start_dword);
		differ("user_data_base", record.user_data_base, live.user_data_base);
		differ("user_data_count", record.user_data_count, live.user_data_count);
		differ("wave", record.wave_size, live.wave_size);
		differ("code_words", record.code.size(), live.code.size());
		differ("back_code_words", record.back_code.size(), live.back_code.size());
		const auto& replayed_spec = record.specialization;
		const auto& live_spec     = live.specialization;
		differ("buffers", replayed_spec.buffers.size(), live_spec.buffers.size());
		for (size_t i = 0; i < std::min(replayed_spec.buffers.size(), live_spec.buffers.size()); i++) {
			if (!(replayed_spec.buffers[i] == live_spec.buffers[i])) {
				differences += fmt::format(" buffer[{}]", i);
			}
		}
		differ("images", replayed_spec.images.size(), live_spec.images.size());
		for (size_t i = 0; i < std::min(replayed_spec.images.size(), live_spec.images.size()); i++) {
			if (!(replayed_spec.images[i] == live_spec.images[i])) {
				differences += fmt::format(" image[{}]", i);
			}
		}
		if (const auto* r = std::get_if<ShaderComputeInputInfo>(&record.info)) {
			const auto& l = std::get<ShaderComputeInputInfo>(live.info);
			differ("float_mode", uint32_t {r->float_mode}, uint32_t {l.float_mode});
			differ("lds_storage", r->lds_storage, l.lds_storage);
			differ("lds_dwords", r->lds_size_dwords, l.lds_size_dwords);
			differ("scratch_dwords", r->scratch_size_dwords, l.scratch_size_dwords);
			differ("host_subgroup", r->host_subgroup_size, l.host_subgroup_size);
			differ("info_wave", r->wave_size, l.wave_size);
			differ("thread_ids", r->thread_ids_num, l.thread_ids_num);
			differ("workgroup_register", r->workgroup_register, l.workgroup_register);
			differ("tg_size_en", r->tg_size_en, l.tg_size_en);
			differ("thread_dimensions", r->dispatch_thread_dimensions, l.dispatch_thread_dimensions);
			for (int i = 0; i < 3; i++) {
				differ("threads", r->threads_num[i], l.threads_num[i]);
				differ("group_id", r->group_id[i], l.group_id[i]);
			}
		}
		// Equal encodings mean the difference comes from an input the journal does not hold.
		std::vector<uint8_t> record_bytes;
		std::vector<uint8_t> live_bytes;
		const bool           encoded = ShaderPrecompile::Encode(record, record_bytes) &&
		                     ShaderPrecompile::Encode(live, live_bytes);
		PipelineCacheLog("Shader replay verify: hash={:016x} id={} record={} stage={} spirv "
		                 "replay={:016x} live={:016x} journal_inputs={}{}",
		                 params.hash, permutation.handle.id, permutation.replay_record,
		                 static_cast<uint32_t>(stage), replay_hash, live_hash,
		                 !encoded                      ? "unencodable"
		                 : record_bytes == live_bytes ? "equal"
		                                              : "differ",
		                 differences);
	}

	template <typename InputInfo>
	static ShaderRecompiler::CompileOptions MakeOptions(ShaderType stage, uint64_t hash,
	                                                    std::span<const uint32_t> user_data,
	                                                    std::span<const uint32_t> back_code,
	                                                    InputInfo&                input_info) {
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = hash;
		options.user_data   = user_data;
		options.back_code   = back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.bindless_images = g_bindless_images;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size      = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return options;
	}

	// A background translation's own copies of its inputs; `options` points into them, so the
	// job stays where it was allocated.
	template <typename InputInfo>
	struct TranslationInput {
		std::vector<uint32_t>            code;
		std::vector<uint32_t>            back_code;
		std::array<uint32_t, 40>         user_data {};
		InputInfo                        input_info;
		ShaderRecompiler::CompileOptions options;
	};

	template <typename InputInfo>
	static std::unique_ptr<TranslationInput<InputInfo>>
	CopyTranslationInput(ShaderType stage, const ShaderParams& params, const InputInfo& input_info) {
		static_assert(std::tuple_size_v<decltype(params.user_data)> == 40);
		auto input = std::make_unique<TranslationInput<InputInfo>>();
		input->code.assign(params.code.begin(), params.code.end());
		input->back_code.assign(params.back_code.begin(), params.back_code.end());
		input->user_data  = params.user_data;
		input->input_info = input_info;
		input->input_info.stage = {};
		input->options = MakeOptions(stage, params.hash,
		                             std::span(input->user_data).first(params.user_data_count),
		                             input->back_code, input->input_info);
		// Worker threads never write the shader log.
		input->options.dump_ir    = false;
		input->options.early_dump = false;
		return input;
	}

	// Posts a translation or module compile to the worker threads, counted in `jobs` until its
	// result is set.
	template <typename Job>
	void PostJob(Job&& job, bool urgent) {
		jobs->in_flight.fetch_add(1, std::memory_order_relaxed);
		const bool posted = workers->Post(
		    [counts = jobs, job = std::forward<Job>(job)]() mutable {
			    job();
			    counts->finished.fetch_add(1, std::memory_order_relaxed);
			    counts->in_flight.fetch_sub(1, std::memory_order_release);
		    },
		    urgent);
		if (!posted) {
			jobs->in_flight.fetch_sub(1, std::memory_order_relaxed);
		}
	}

	template <typename T>
	[[nodiscard]] static bool IsReady(const std::future<T>& future) {
		return future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
	}

	// Null when the job was dropped (the worker threads stopped) or failed.
	template <typename T>
	[[nodiscard]] static std::optional<T> TakeResult(std::future<T>& future) {
		try {
			return future.get();
		} catch (const std::exception&) {
			return std::nullopt;
		}
	}

	// Whether the shader stores to buffers, writes images or uses GDS, found by decoding only.
	// Buffer atomics do not count (see StoresData in renderDraw.cpp).
	bool StoresData(ShaderType stage, const ShaderParams& params) {
		if (const auto known = stores_data.find(params.hash); known != stores_data.end()) {
			return known->second;
		}
		namespace Decoder = ShaderRecompiler::Decoder;
		const auto stores = [](const Decoder::Program& program) {
			return std::ranges::any_of(program.instructions, [](const Decoder::Instruction& inst) {
				if (inst.family == Decoder::Family::DS && inst.gds) {
					return true;
				}
				const auto name = magic_enum::enum_name(inst.opcode);
				return name.starts_with("BUFFER_STORE") || name.starts_with("TBUFFER_STORE") ||
				       name.starts_with("FLAT_STORE") || name.starts_with("GLOBAL_STORE") ||
				       name.starts_with("IMAGE_STORE") || name.starts_with("IMAGE_ATOMIC");
			});
		};
		// Decoded as TranslateProgram decodes it.
		Decoder::Program front;
		if (!params.back_code.empty() || stage == ShaderType::Local) {
			front = Decoder::DecodeFrontProgram(params.code);
		} else {
			Decoder::DecodeProgram(params.code, front);
		}
		bool result = stores(front);
		if (!result && !params.back_code.empty()) {
			Decoder::Program back;
			Decoder::DecodeProgram(params.back_code, back);
			result = stores(back);
		}
		stores_data.emplace(params.hash, result);
		return result;
	}

	// Starts translating a source nothing has translated or queued yet (ProgramWait::Prefetch for
	// a stage whose push data position is unknown until an earlier stage compiles).
	template <typename InputInfo>
	void QueueSourceTranslation(const ShaderParams& params, const InputInfo& input_info,
	                            ShaderType stage) {
		if (workers == nullptr ||
		    Config::GetShaderLogDirection() != Config::LogDirection::Silent) {
			return;
		}
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		lookup_key.code_variant    = params.code_variant;
		BuildStageStaticKey(input_info, lookup_key.static_state);
		if (programs.contains(lookup_key) || pending_sources.contains(lookup_key)) {
			return;
		}
		std::promise<ShaderRecompiler::TranslateResult> promise;
		pending_sources.emplace(lookup_key, promise.get_future());
		PostJob(
		    [input = CopyTranslationInput(stage, params, input_info),
		     promise = std::move(promise)]() mutable {
			    promise.set_value(ShaderRecompiler::TranslateProgram(input->code, input->options));
		    },
		    false);
	}

	// With ProgramWait::Defer (a draw, with asynchronous pipelines) or ProgramWait::Prefetch (a
	// look-ahead prediction), a permutation not compiled yet is translated and compiled on the
	// worker threads, and the result is empty with `*pending` set; the same lookup later picks
	// it up. Defer jobs go ahead of queued prefetches.
	template <typename InputInfo>
	// known_source: the entry the draw speculation found for these very stage inputs (see
	// PreparedGraphicsStages), which spares building the key and looking it up.
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info, uint32_t& push_data_cursor,
	                  ProgramWait wait = ProgramWait::Wait, bool* pending = nullptr,
	                  const void* known_source = nullptr) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
			if (ShaderParams linked; LinkShaderCalls(params, linked)) {
				return Get(linked, input_info, push_data_cursor, wait, pending);
			}
		}
		const bool background = wait != ProgramWait::Wait && workers != nullptr &&
		                        Config::GetShaderLogDirection() == Config::LogDirection::Silent;
		const bool urgent     = wait == ProgramWait::Defer;

		// KYTY_PRO_SKIP_SHADERS=<hash>[,<hash>...] (PS5 Pro experiment 2026-10-29): draws and dispatches
		// of these shaders are left out, also when the program comes from the precompiled store.
		static const Common::Switches::HashList pro_skip_early {KYTY_GETENV("KYTY_PRO_SKIP_SHADERS")};
		if (!pro_skip_early.hashes.empty() && pro_skip_early.Contains(params.hash)) [[unlikely]] {
			static std::atomic_bool logged {false};
			if (!logged.exchange(true)) {
				std::printf("KYTY_PRO_SKIP_SHADERS: skipping shader 0x%016llx\n",
				            static_cast<unsigned long long>(params.hash));
				std::fflush(stdout);
			}
			return {};
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryOnGpuThread,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    // TryReadGpuCleanBacking fails for a range when any byte is GPU-dirty or unbacked.
		    .specialization_block_reads = true,
		    .read_memory_block          = ReadShaderGuestBlockOnGpuThread,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
		}
		// Refreshes a known source and returns its permutation for the refreshed specialization,
		// if it has one yet.
		const auto existing = [&](SourceEntry& source) -> std::optional<ShaderProgram> {
			if (!AdoptSpeculation(source, stage, user_data, params.Base(), runtime, wait)) {
				DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::ResourceRefresh);
				LibKernel::Memory::t_debug_reading_shader = params.hash;
				LibKernel::Memory::t_debug_reading_stage  = static_cast<uint32_t>(stage);
				const bool materialized = ShaderRecompiler::IR::MaterializeResources(
				    source.resource_plan, runtime, source.resources, source.specialization,
				    &source.memo);
				LibKernel::Memory::t_debug_reading_shader = 0;
				if (!materialized) {
					// Skip this draw or dispatch, not the session; the next use materializes again.
					ReportMaterializeFailure(params.hash);
					return ShaderProgram {};
				}
				// Sources the precompile replay inserted get here for their first refresh.
				if (!source.speculable.load(std::memory_order_relaxed)) {
					source.speculable.store(true, std::memory_order_release);
				}
			}
			if (stats_enabled) {
				CountRefresh(source);
			}
			DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::ProgramMatch);
			const auto matches = [&](const Permutation& candidate) {
				const auto& layout = candidate.program.bindings;
				return layout.push_data_start_dword == layout.PushStartFor(push_data_cursor) &&
				       candidate.specialization == source.specialization;
			};
			auto index = source.last_permutation;
			if (index >= source.permutations.size() ||
			    ((!source.memo.reused || source.last_push_cursor != push_data_cursor) &&
			     !matches(source.permutations[index]))) {
				index = static_cast<uint32_t>(
				    std::ranges::find_if(source.permutations, matches) -
				    source.permutations.begin());
			}
			if (index < source.permutations.size()) {
				auto& permutation       = source.permutations[index];
				if (permutation.replay_source && !t_speculative_lookup) [[unlikely]] {
					VerifyReplayed(params, input_info, stage, push_data_cursor, permutation);
					permutation.replay_source.reset();
				}
				if (permutation.unrecorded && !t_speculative_lookup) [[unlikely]] {
					(void)recording.Append(*permutation.unrecorded);
					permutation.unrecorded.reset();
				}
				source.last_permutation = index;
				source.last_push_cursor = push_data_cursor;
				input_info.stage = {.program = &permutation.program, .resources = &source.resources};
				permutation.program.bindings.AdvancePushData(push_data_cursor);
				return permutation.handle;
			}
			return std::nullopt;
		};
		// Refreshed below without a permutation yet: the lookup finds it again, to compile one.
		const SourceEntry* refreshed = nullptr;
		if (known_source != nullptr) {
			auto& source = *static_cast<SourceEntry*>(const_cast<void*>(known_source));
			if (verify_speculation) {
				// KYTY_VERIFY_SPEC=1: the known source must be the entry the key finds.
				lookup_key.stage           = stage;
				lookup_key.hash            = params.hash;
				lookup_key.user_data_count = params.user_data_count;
				lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
				lookup_key.code_variant    = params.code_variant;
				BuildStageStaticKey(input_info, lookup_key.static_state);
				const auto found = programs.find(lookup_key);
				if (found == programs.end() || &found->second != &source) {
					EXIT("draw speculation: known source 0x%016llx is not the entry its key finds\n",
					     static_cast<unsigned long long>(params.hash));
				}
			}
			if (source.skip_dispatch) {
				return {};
			}
			if (auto program = existing(source)) {
				return *program;
			}
			refreshed = &source;
		}
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		lookup_key.code_variant    = params.code_variant;
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto entry = programs.find(lookup_key);
		if (entry != programs.end() && entry->second.skip_dispatch) {
			return {};
		}
		if (entry != programs.end() && &entry->second != refreshed) {
			if (auto program = existing(entry->second)) {
				return *program;
			}
		}

		const auto defer = [&] {
			if (pending != nullptr) {
				*pending = true;
			}
			// The refresh above may have changed the specialization without a permutation to match
			// it, so the next lookup must search instead of trusting the last hit.
			if (entry != programs.end()) {
				entry->second.last_permutation = UINT32_MAX;
			}
			return ShaderProgram {};
		};
		auto options = MakeOptions(stage, params.hash, user_data, params.back_code, input_info);
		std::optional<ShaderRecompiler::TranslateResult> translated;
		if (entry == programs.end()) {
			// A new source: translate it first.
			if (auto queued = pending_sources.find(lookup_key); queued != pending_sources.end()) {
				if (background && !IsReady(queued->second)) {
					return defer();
				}
				translated = TakeResult(queued->second);
				pending_sources.erase(queued);
			} else {
				if (PermutationLogEnabled()) [[unlikely]] {
					LogNewPermutation(stage, params.hash, entry, push_data_cursor);
				}
				if (background) {
					std::promise<ShaderRecompiler::TranslateResult> promise;
					pending_sources.emplace(lookup_key, promise.get_future());
					PostJob(
					    [input   = CopyTranslationInput(stage, params, input_info),
					     promise = std::move(promise)]() mutable {
						    promise.set_value(
						        ShaderRecompiler::TranslateProgram(input->code, input->options));
					    },
					    urgent);
					return defer();
				}
			}
			if (!translated) {
				translated = ShaderRecompiler::TranslateProgram(params.code, options);
			}
			if (translated->skip_dispatch) {
				// KYTY_SKIP_UNTRANSLATABLE=1 (PS5 Pro experiment 2026-10-29): the shader is left out
				// (its draws and dispatches skipped) and its code saved as _pro_shaders\<hash>.bin,
				// so that one run collects every shader the emulator cannot translate yet.
				static const bool skip_untranslatable =
				    KYTY_GETENV("KYTY_SKIP_UNTRANSLATABLE") != nullptr;
				if (skip_untranslatable) {
					std::filesystem::create_directories("_pro_shaders");
					const auto path = fmt::format("_pro_shaders/{:016x}.bin", params.hash);
					if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
						std::fwrite(params.code.data(), sizeof(uint32_t), params.code.size(), file);
						std::fclose(file);
					}
					std::printf("Skipped untranslatable shader 0x%016llx (stage %d, %zu dwords) -> %s\n",
					            static_cast<unsigned long long>(params.hash), static_cast<int>(stage),
					            params.code.size(), path.c_str());
					std::fflush(stdout);
					std::unique_lock lock(programs_mutex);
					auto skipped = programs.try_emplace(lookup_key, ShaderRecompiler::IR::ExtractResourcePlan(
					                                                    translated->program)).first;
					skipped->second.skip_dispatch = true;
					return {};
				}
				// No shader is left out: a shader we cannot translate ends the
				// session, so that it gets fixed instead of silently missing from the picture.
				Outcomes().Failed(params.hash, "translation");
				EXIT("shader 0x%016llx could not be translated (see the warning above)\n",
				     static_cast<unsigned long long>(params.hash));
			}
			{
				std::unique_lock lock(programs_mutex);
				entry = programs.try_emplace(lookup_key,
				    ShaderRecompiler::IR::ExtractResourcePlan(translated->program)).first;
			}
			if (!ShaderRecompiler::IR::MaterializeResources(
			        entry->second.resource_plan, runtime, entry->second.resources,
			        entry->second.specialization, &entry->second.memo)) {
				// Skip this draw or dispatch, not the session; the next use materializes again.
				ReportMaterializeFailure(params.hash);
				return {};
			}
			entry->second.speculable.store(true, std::memory_order_release);
		}
		auto& source = entry->second;

		// Then compile the permutation for this specialization and push data position.
		std::optional<CompiledModule> compiled;
		const auto queued = std::ranges::find_if(source.pending, [&](const PendingPermutation& p) {
			return p.push_data_cursor == push_data_cursor && p.specialization == source.specialization;
		});
		if (queued != source.pending.end()) {
			if (background && !IsReady(queued->compiled)) {
				return defer();
			}
			compiled = TakeResult(queued->compiled);
			source.pending.erase(queued);
		} else {
			if (PermutationLogEnabled() && !translated) [[unlikely]] {
				LogNewPermutation(stage, params.hash, entry, push_data_cursor);
			}
			if (background) {
				std::promise<CompiledModule> promise;
				source.pending.push_back({.specialization   = source.specialization,
				                          .push_data_cursor = push_data_cursor,
				                          .compiled         = promise.get_future()});
				PostJob(
				    [device = device, input = CopyTranslationInput(stage, params, input_info),
				     translated = std::move(translated), specialization = source.specialization,
				     push_data_cursor, promise = std::move(promise)]() mutable {
					    if (!translated) {
						    translated = ShaderRecompiler::TranslateProgram(input->code, input->options);
					    }
					    promise.set_value(CompileModule(device, input->code, input->options,
					                                    std::move(*translated), specialization,
					                                    push_data_cursor));
				    },
				    urgent);
				return defer();
			}
		}
		if (!compiled) {
			if (!translated) {
				translated = ShaderRecompiler::TranslateProgram(params.code, options);
			}
			compiled = CompileModule(device, params.code, options, std::move(*translated),
			                         source.specialization, push_data_cursor);
		}
		if (compiled->failed) {
			// As above: the emitter rejected the program; end the session instead of skipping it.
			Outcomes().Failed(params.hash, "SPIR-V emit");
			EXIT("shader 0x%016llx failed SPIR-V validation (see the warning above)\n",
			     static_cast<unsigned long long>(params.hash));
		}
		source.permutations.push_back(
		    MakePermutation(options, source.specialization, std::move(*compiled)));
		Outcomes().Works(params.hash);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::ShadersCompiled);
		source.last_permutation = static_cast<uint32_t>(source.permutations.size() - 1u);
		source.last_push_cursor = push_data_cursor;
		auto& permutation = source.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &source.resources};
		// A linked program names this run's call targets: it is not journaled.
		if (recording.IsOpen() && params.base_address == 0) {
			auto record = ShaderPrecompile::Capture(params, options, permutation.specialization,
			                                        push_data_cursor, input_info);
			if (t_speculative_lookup) {
				permutation.unrecorded = std::move(record);
			} else {
				(void)recording.Append(record);
			}
		}
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, program_source]: programs) {
			counts[static_cast<size_t>(key.stage)] += program_source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// On the draw speculation thread: refreshes a known source's resources for a future draw
	// into `out`, logging its reads (see SpeculatedStage). Insertions into programs (the GPU
	// thread's and the precompile replay's) hold programs_mutex. A source is refreshed only once
	// a GPU-thread refresh succeeded (its plan is compiled), through the caller's plan scratch slot.
	template <typename InputInfo>
	bool Speculate(const ShaderParams& params, const InputInfo& input_info, ShaderType stage,
	               const BufferCache& buffers, SpeculatedStage& out) {
		thread_local ProgramKey key;
		key.stage           = stage;
		key.hash            = params.hash;
		key.user_data_count = params.user_data_count;
		key.code_size       = static_cast<uint32_t>(params.code.size());
		key.code_variant    = params.code_variant;
		BuildStageStaticKey(input_info, key.static_state);
		SourceEntry* source = nullptr;
		{
			std::shared_lock lock(programs_mutex);
			if (auto entry = programs.find(key); entry != programs.end()) {
				source = &entry->second;
			}
		}
		if (source == nullptr || !source->speculable.load(std::memory_order_acquire)) {
			speculation_failures[source == nullptr ? 1 : 2].fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadShaderGuestMemoryAhead,
		    .userdata                   = const_cast<BufferCache*>(&buffers),
		    .read_specialization_memory = ReadShaderGuestMemoryAhead,
		    .specialization_block_reads = true,
		    .read_memory_block          = ReadShaderGuestMemoryAhead,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			std::copy_n(input_info.workgroup_counts, 3, out.workgroup_counts.begin());
		}
		if (!ShaderRecompiler::IR::MaterializeResourcesLogged(source->resource_plan, runtime,
		                                                      out.snapshot, out.specialization,
		                                                      out.reads)) {
			speculation_failures[3].fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		out.user_data.assign(user_data.begin(), user_data.end());
		out.shader_base = params.Base();
		out.source      = source;
		return true;
	}

	// The GPU thread's current draw was speculated: takes the stage's resources instead of
	// refreshing when they are what a refresh would produce now. That holds when the source, user
	// data and shader base are the same and every read the speculation made gives the same result
	// again. KYTY_DEBUG_AB=specprep refreshes instead in every other window.
	bool AdoptSpeculation(SourceEntry& source, ShaderType stage, std::span<const uint32_t> user_data,
	                      uint64_t shader_base, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                      ProgramWait wait) {
		if (wait == ProgramWait::Prefetch ||
		    (stage == ShaderType::Compute && !SpeculateDispatchesEnabled())) {
			return false;
		}
		auto* draw         = t_speculated_draw;
		auto* stage_result = draw == nullptr ? nullptr
		                     : &draw->stages[stage == ShaderType::Pixel ? SpeculatedDraw::Pixel
		                                                                : SpeculatedDraw::Vertex];
		static const bool ab      = AbSelected("specprep");
		bool              adopted = false;
		const auto        reads_unchanged = [&] {
			DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::SpeculationReads);
			if (draw->reads_current) {
				// The reads stand (see DrawSpeculator::ReadsCurrent). KYTY_VERIFY_SPEC_READS=1
				// makes them again anyway and reports differences.
				speculation.reads_kept++;
				if (verify_speculation_reads &&
				    !ShaderRecompiler::IR::ReadsUnchanged(stage_result->reads, runtime)) {
					const auto missed = ++speculation.reads_kept_changed;
					if (missed <= 16 || missed % 1000 == 0) {
						std::printf("speculation reads verify: kept reads changed (%llu of %llu)\n",
						            static_cast<unsigned long long>(missed),
						            static_cast<unsigned long long>(speculation.reads_kept));
					}
				}
				return true;
			}
			return ShaderRecompiler::IR::ReadsUnchanged(stage_result->reads, runtime);
		};
		if (stage_result == nullptr || stage_result->source == nullptr) {
			speculation.unspeculated++;
		} else {
			const bool same_inputs =
			    stage_result->source == &source && stage_result->shader_base == shader_base &&
			    std::ranges::equal(stage_result->user_data, user_data) &&
			    (stage != ShaderType::Compute ||
			     std::ranges::equal(stage_result->workgroup_counts, runtime.workgroup_counts));
			// One adoption per stage: a second lookup of this stage in the draw refreshes.
			stage_result->source = nullptr;
			if (!same_inputs) {
				speculation.mismatched++;
			} else if (ab && AbFeatureOff()) {
				speculation.skipped++;
			} else if (!reads_unchanged()) {
				speculation.changed++;
			} else {
				std::swap(source.resources, stage_result->snapshot);
				std::swap(source.specialization, stage_result->specialization);
				source.memo.valid  = false;
				source.memo.reused = false;
				speculation.adopted++;
				adopted = true;
				// KYTY_VERIFY_SPEC_COMPUTE=<s>: only dispatches' adoptions, from s seconds after
				// the first one on (a route's loading stays at full speed and keeps its timing).
				static const char* verify_compute = KYTY_GETENV("KYTY_VERIFY_SPEC_COMPUTE");
				bool               verify_now     = false;
				if (stage == ShaderType::Compute) {
					compute_adopted++;
					if (verify_compute != nullptr) [[unlikely]] {
						static const auto from =
						    std::chrono::steady_clock::now() +
						    std::chrono::seconds(std::strtoul(verify_compute, nullptr, 10));
						verify_now = std::chrono::steady_clock::now() >= from;
					}
				}
				if (verify_speculation || verify_now) {
					VerifyAdoption(source, runtime);
				}
			}
		}
		if (speculation_stats && (++speculation.stages % 100000u) == 0) {
			std::printf("draw-speculation: stages=%llu adopted=%llu (compute %llu) changed=%llu "
			            "mismatched=%llu unspeculated=%llu skipped=%llu inputs taken=%llu changed=%llu "
			            "reads kept=%llu kept-changed=%llu\n",
			            static_cast<unsigned long long>(speculation.stages),
			            static_cast<unsigned long long>(speculation.adopted),
			            static_cast<unsigned long long>(compute_adopted),
			            static_cast<unsigned long long>(speculation.changed),
			            static_cast<unsigned long long>(speculation.mismatched),
			            static_cast<unsigned long long>(speculation.unspeculated),
			            static_cast<unsigned long long>(speculation.skipped),
			            static_cast<unsigned long long>(speculation.inputs_taken),
			            static_cast<unsigned long long>(speculation.inputs_changed),
			            static_cast<unsigned long long>(speculation.reads_kept),
			            static_cast<unsigned long long>(speculation.reads_kept_changed));
		}
		return adopted;
	}

	// KYTY_VERIFY_SPEC=1: an adopted refresh must equal a refresh made now.
	static void VerifyAdoption(const SourceEntry&                      source,
	                           const ShaderRecompiler::IR::SrtRuntime& runtime) {
		ShaderRecompiler::IR::ResourceSnapshot       snapshot;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		const bool ok = ShaderRecompiler::IR::MaterializeResources(source.resource_plan, runtime,
		                                                           snapshot, specialization);
		const auto& adopted = source.resources;
		const char* field   = nullptr;
		if (!ok) {
			field = "result";
		} else if (snapshot.buffers != adopted.buffers) {
			field = "buffers";
		} else if (snapshot.images != adopted.images) {
			field = "images";
		} else if (snapshot.samplers != adopted.samplers) {
			field = "samplers";
		} else if (snapshot.flattened_srt != adopted.flattened_srt) {
			field = "flattened SRT";
		} else if (snapshot.user_data != adopted.user_data) {
			field = "user data";
		} else if (snapshot.uniform_fill != adopted.uniform_fill) {
			field = "uniform fill";
		} else if (specialization != source.specialization) {
			field = "specialization";
		}
		if (field != nullptr) {
			EXIT("draw speculation adopted a stale refresh: hash=0x%016llx field=%s\n",
			     static_cast<unsigned long long>(source.resource_plan.shader_hash), field);
		}
	}

	// KYTY_PERMUTATION_LOG: why this lookup compiles: a new shader, a new static state of a
	// known one (and which static words differ), or a new specialization of a known source.
	template <typename Iterator>
	void LogNewPermutation(ShaderType stage, uint64_t hash, Iterator entry,
	                       uint32_t push_data_cursor) {
		std::string reason;
		if (entry != programs.end() && !entry->second.permutations.empty()) {
			const auto& source = entry->second;
			const auto& last   = source.permutations.back();
			reason = fmt::format(" new specialization ({} existing):", source.permutations.size()) +
			         SpecializationDiff(last.specialization, source.specialization);
			const auto start = last.program.bindings.PushStartFor(push_data_cursor);
			if (last.program.bindings.push_data_start_dword != start) {
				reason += fmt::format(" push_start {}->{}", last.program.bindings.push_data_start_dword,
				                      start);
			}
		} else {
			const ProgramKey* other = nullptr;
			size_t            same  = 0;
			for (const auto& [key, source]: programs) {
				if (key.hash == hash && key.stage == stage && !(key == lookup_key)) {
					other = &key;
					same++;
				}
			}
			if (other == nullptr) {
				reason = " new shader";
			} else {
				reason = fmt::format(" new static state ({} other keys):", same);
				const auto& a = other->static_state;
				const auto& b = lookup_key.static_state;
				if (a.size() != b.size()) {
					reason += fmt::format(" words {}->{}", a.size(), b.size());
				}
				size_t shown = 0;
				for (size_t i = 0; i < std::min(a.size(), b.size()) && shown < 6; i++) {
					if (a[i] != b[i]) {
						reason += fmt::format(" w[{}] {:#x}->{:#x}", i, a[i], b[i]);
						shown++;
					}
				}
				if (other->user_data_count != lookup_key.user_data_count) {
					reason += fmt::format(" user_data {}->{}", other->user_data_count,
					                      lookup_key.user_data_count);
				}
			}
		}
		const auto now = std::chrono::steady_clock::now().time_since_epoch();
		std::printf("permutation: stage=%u hash=%016llx t=%.3f%s\n", static_cast<uint32_t>(stage),
		            static_cast<unsigned long long>(hash),
		            std::chrono::duration<double>(now).count(), reason.c_str());
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		// Debugging aids: KYTY_VERIFY_SRT=1 checks every cached resource refresh against the
		// reference SRT walker and aborts on a difference; KYTY_SRT_STATS=1 logs how often
		// refreshes reuse descriptors; KYTY_DEBUG_SRT_RUNS=0 reads flat SRT slots one by one (A/B).
		const auto enabled = [](const char* value) {
			return value != nullptr && std::strcmp(value, "1") == 0;
		};
		if (enabled(KYTY_GETENV("KYTY_VERIFY_SRT"))) {
			ShaderRecompiler::IR::SetResourceMaterializationVerification(true);
		}
		stats_enabled      = enabled(KYTY_GETENV("KYTY_SRT_STATS"));
		speculation_stats  = enabled(KYTY_GETENV("KYTY_DEBUG_SPEC_STATS"));
		verify_speculation       = enabled(KYTY_GETENV("KYTY_VERIFY_SPEC"));
		verify_speculation_reads = enabled(KYTY_GETENV("KYTY_VERIFY_SPEC_READS"));
		if (const char* runs = KYTY_GETENV("KYTY_DEBUG_SRT_RUNS");
		    runs != nullptr && std::strcmp(runs, "0") == 0) {
			ShaderRecompiler::IR::SetFlatRunReads(false);
		}
	}

	void CountRefresh(const SourceEntry& source) {
		constexpr uint64_t Interval = 100000;
		stats.refreshes++;
		stats.memoizable += source.resource_plan.compiled != nullptr &&
		                    source.resource_plan.compiled->memoizable;
		stats.reused += source.memo.reused;
		if (stats.refreshes % Interval == 0) {
			PipelineCacheLog("Shader resources: {} refreshes, {} memoizable, {} reused descriptors",
			                 stats.refreshes, stats.memoizable, stats.reused);
		}
	}
	~ProgramCache() {
		for (auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
			// The worker threads have stopped: each job either finished or was dropped.
			for (auto& queued: entry.pending) {
				if (queued.compiled.valid() && IsReady(queued.compiled)) {
					if (auto compiled = TakeResult(queued.compiled)) {
						device.destroyShaderModule(compiled->module, nullptr);
					}
				}
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	// The GPU thread inserts into programs under this lock, and the draw speculation thread looks
	// up under it. Entries are never removed while both run.
	std::shared_mutex                                             programs_mutex;
	// Adoption counts (GPU thread), printed every 100000 stages with KYTY_DEBUG_SPEC_STATS=1.
	struct {
		uint64_t stages       = 0;
		uint64_t adopted      = 0;
		uint64_t changed      = 0;
		uint64_t mismatched   = 0;
		uint64_t unspeculated = 0;
		uint64_t skipped      = 0;
		// Prepared stage inputs (see PreparedGraphicsStages) taken, and found changed.
		uint64_t inputs_taken   = 0;
		uint64_t inputs_changed = 0;
		// Adoptions whose reads were kept without making them again (see
		// DrawSpeculator::ReadsCurrent), and, with KYTY_VERIFY_SPEC_READS=1, how many would have
		// read differently.
		uint64_t reads_kept         = 0;
		uint64_t reads_kept_changed = 0;
	} speculation;
	// Adoptions by dispatches (KYTY_SPECULATE_DISPATCHES), part of speculation.adopted.
	uint64_t compute_adopted      = 0;
	bool speculation_stats        = false;
	bool verify_speculation       = false;
	bool verify_speculation_reads = false;
	// See PipelineCache::SpeculationFailures (index 0 is counted there).
	std::array<std::atomic<uint64_t>, 4> speculation_failures {};
	// New sources translating on worker threads.
	std::unordered_map<ProgramKey, std::future<ShaderRecompiler::TranslateResult>, ProgramKeyHash>
	    pending_sources;
	// Whether a shader (by hash) stores data; see StoresData.
	std::unordered_map<uint64_t, bool> stores_data;
	// Runs background translations; null without pipeline libraries.
	PipelineLibraryCache*                                       workers = nullptr;
	// Background translations and module compiles: running or queued, and finished so far.
	struct JobCounts {
		std::atomic<uint32_t> in_flight {0};
		std::atomic<uint64_t> finished {0};
	};
	std::shared_ptr<JobCounts> jobs = std::make_shared<JobCounts>();
	ProgramKey                                                  lookup_key;
	vk::Device                                                  device;
	uint64_t                                                    next_shader_id = 0;
	ShaderPrecompile::Journal                                   recording;
	bool                                                        stats_enabled = false;
	struct {
		uint64_t refreshes  = 0;
		uint64_t memoizable = 0;
		uint64_t reused     = 0;
	} stats;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	g_bindless_images = graphics.bindless_enabled && Config::BindlessImagesEnabled();
	ShaderRecompiler::Spirv::SetMaximalReconvergence(graphics.maximal_reconvergence_enabled);
	InitializeDriverCache();
	// Before the precompile replay, which queues the journal's compute pipelines on it.
	m_compute_queue = std::make_unique<ComputePipelineQueue>(Config::PipelineCompileThreads(),
	                                                         Config::PipelineCompileReplayThreads());
	if (m_driver_cache) {
		m_last_save   = std::chrono::steady_clock::now();
		m_save_thread = std::jthread([this](const std::stop_token& stop) { SaveThread(stop); });
	}
	InitializeShaderPrecompile();
	// Linking libraries without fast linking costs about as much as a full compile.
	if (m_graphics.pipeline_library_enabled && m_graphics.pipeline_library_fast_linking) {
		m_libraries = std::make_unique<PipelineLibraryCache>(m_graphics, m_driver_cache);
		m_program_cache->workers = m_libraries.get();
	}
}

PipelineCache::~PipelineCache() {
	// Save also stops the pipeline-library link thread. Pipelines linked from the libraries are
	// destroyed below, and the libraries after them.
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
			if (pipeline->pixel_set_layout != nullptr) {
				m_graphics.device.destroyDescriptorSetLayout(pipeline->pixel_set_layout, nullptr);
			}
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	// Compute pipelines no dispatch took. Save stopped the queue: every result is set.
	for (auto& [id, build]: m_compute_builds) {
		(void)id;
		build.pipeline->pipeline = build.result.get();
		DestroyPipelineObjects(m_graphics, *build.pipeline);
	}
	m_compute_queue.reset();
	m_libraries.reset();
	for (uint32_t index = 0; index < m_driver_cache.count; index++) {
		m_graphics.device.destroyPipelineCache(m_driver_cache.shards[index], nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	// The key hashes the recompiler and pipeline sources, including uncommitted edits, so records
	// written by different shader code never match.
	if (std::string_view(KYTY_SHADER_CACHE_KEY) == "unknown") {
		PipelineCacheLog("Vulkan pipeline cache: disabled (unknown shader source key)");
		return;
	}

	m_driver_cache_path = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	// Shard i > 0 is "<title>.<i>.bin". Without pipelineCreationCacheControl only shard 0 is used.
	const auto shard_path = [&](uint32_t index) {
		return index == 0 ? m_driver_cache_path
		                  : std::filesystem::path("_PipelineCache") /
		                        fmt::format("{}.{}.bin", title_id, index);
	};
	const uint32_t max_shards =
	    m_graphics.pipeline_creation_cache_control ? DriverPipelineCache::kMaxShards : 1;
	// The driver adds nothing to a cache this large (2831710399 bytes stayed the same across runs
	// that compiled new pipelines), so a new shard takes the new pipelines.
	constexpr size_t kShardFull = 2500000000;
	size_t           last_loaded = 0;
	for (uint32_t index = 0; index < max_shards; index++) {
		const auto path = shard_path(index);
		if (index > 0 && !Common::File::IsFileExisting(path)) {
			break;
		}
		auto cache = LoadDriverCacheShard(path, &last_loaded);
		if (cache == nullptr) {
			break;
		}
		m_driver_cache.shards[m_driver_cache.count++] = cache;
		m_driver_cache_write_path                     = path;
		// An empty (new or invalidated) shard takes the new pipelines; later shards are not used.
		if (last_loaded == 0) {
			break;
		}
	}
	if (m_driver_cache.count != 0 && m_driver_cache.count < max_shards && last_loaded >= kShardFull) {
		const auto path  = shard_path(m_driver_cache.count);
		size_t     empty = 0;
		auto cache = LoadDriverCacheShard(path, &empty);
		if (cache != nullptr) {
			PipelineCacheLog("Vulkan pipeline cache: shard {} is full ({} bytes); new pipelines go to {}",
			                 m_driver_cache.count - 1, last_loaded, Common::PathToString(path));
			m_driver_cache.shards[m_driver_cache.count++] = cache;
			m_driver_cache_write_path                     = path;
		}
	}
	if (m_driver_cache.count == 0) {
		PipelineCacheLog("Vulkan pipeline cache: disabled");
	} else if (m_driver_cache.count > 1) {
		PipelineCacheLog("Vulkan pipeline cache: {} shards, writing {}", m_driver_cache.count,
		                 Common::PathToString(m_driver_cache_write_path));
	}
}

vk::PipelineCache PipelineCache::LoadDriverCacheShard(const std::filesystem::path& cache_path,
                                                      size_t*                      loaded) {
	*loaded                 = 0;
	const auto path         = Common::PathToString(cache_path);
	const bool cache_exists = Common::File::IsFileExisting(cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	vk::PipelineCache cache  = nullptr;
	auto              result = m_graphics.device.createPipelineCache(&create, nullptr, &cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: {} failed ({})", path, vk::to_string(result));
		return nullptr;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty {}", path);
	}
	*loaded = initial_data.size();
	return cache;
}

void PipelineCache::InitializeShaderPrecompile() {
	// Reuse the Release/source-key/driver gate: records from different recompiler or pipeline
	// sources carry a different key and are discarded.
	if (!Config::ShaderPrecompileEnabled() || !m_driver_cache ||
	    m_driver_cache_path.empty())
		return;
	auto path = m_driver_cache_path;
	path.replace_extension(".shaders");
	auto        key = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	std::string app_version;
	(void)Loader::SystemContentParamSfoGetString("APP_VER", &app_version);
	key += app_version;
	auto records = m_program_cache->recording.Open(path, key);
	if (records.empty()) return;
	PipelineCacheLog("Shader precompile: replaying {} recorded permutations", records.size());
	m_precompile_done.store(false, std::memory_order_release);
	m_precompile_thread = std::jthread(
	    [this, records = std::move(records)]() mutable { ReplayPrecompiled(std::move(records)); });
}

void PipelineCache::WaitForPrecompile() {
	if (m_precompile_done.load(std::memory_order_acquire)) return;
	std::lock_guard lock(m_precompile_join_mutex);
	if (m_precompile_thread.joinable()) m_precompile_thread.join();
	m_precompile_done.store(true, std::memory_order_release);
}

void PipelineCache::ReplayPrecompiled(std::vector<ShaderPrecompile::PermutationRecord> records) {
	KYTY_PROFILER_THREAD("ShaderPrecompile");
	// The first draw waits for the replay, so it translates and compiles the records on several
	// threads without the cache lock, then adds them in order under it.
	struct Replayed {
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		ProgramCache::CompiledModule                     compiled;
		bool                                             valid = false;
	};
	const auto options_for = ReplayOptions;

	const auto            started = std::chrono::steady_clock::now();
	std::vector<Replayed> replayed(records.size());
	// Compute records first, each group in journal order: their pipelines are queued as soon as
	// their modules exist, and their driver compiles are the longest.
	std::vector<size_t> order(records.size());
	std::iota(order.begin(), order.end(), size_t {0});
	const auto compute_end = std::stable_partition(order.begin(), order.end(), [&](size_t index) {
		return records[index].stage == ShaderType::Compute;
	});
	std::span<const size_t> indices;
	std::atomic_size_t      next {0};
	const auto              work = [&] {
		for (size_t position = next.fetch_add(1); position < indices.size();
		     position        = next.fetch_add(1)) {
			const auto   index  = indices[position];
			auto&        record = records[index];
			ShaderParams params;
			const auto   options    = options_for(record, params);
			auto         translated = ShaderRecompiler::TranslateProgram(params.code, options);
			// Reject inconsistent metadata before ApplyResourceSpecialization's hard assertions.
			// Live guest lookups still compile normally if a record cannot be replayed.
			if (translated.skip_dispatch || !translated.program.resource_tracking_complete ||
			    translated.program.info.buffers.size() != record.specialization.buffers.size() ||
			    translated.program.info.images.size() > record.specialization.images.size()) {
				continue;
			}
			auto& result = replayed[index];
			result.plan  = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			result.compiled =
			    ProgramCache::CompileModule(m_graphics.device, params.code, options, std::move(translated),
			                                record.specialization, record.push_data_start_dword);
			result.valid = !result.compiled.failed;
		}
	};
	// As before the compute pipeline queue (whose thread count is configured): translating with
	// more threads gained little and widens any race in the translator.
	const auto threads   = std::clamp(std::thread::hardware_concurrency(), 2u, 8u) - 1u;
	const auto translate = [&](std::span<const size_t> group) {
		indices = group;
		next.store(0);
		std::vector<std::jthread> helpers;
		for (uint32_t i = 1; i < threads; i++) {
			helpers.emplace_back(work);
		}
		work();
	};

	size_t compiled = 0;
	size_t skipped  = 0;
	size_t queued   = 0;
	// The program id each record's permutation got (0: skipped or known already).
	std::vector<uint64_t> record_ids(records.size(), 0);
	// Adds a replayed record's permutation; null when the record failed or is known already.
	const auto insert = [&](size_t index) -> const ProgramCache::Permutation* {
		auto& record = records[index];
		auto& result = replayed[index];
		if (!result.valid) {
			++skipped;
			return nullptr;
		}
		ShaderParams                    params;
		const auto                      options = options_for(record, params);
		ProgramCache::ProgramKey        key;
		key.stage           = record.stage;
		key.hash            = record.hash;
		key.user_data_count = record.user_data_count;
		key.code_size       = static_cast<uint32_t>(record.code.size());
		std::visit([&](auto& info) { BuildStageStaticKey(info, key.static_state); }, record.info);
		auto entry = m_program_cache->programs.find(key);
		if (entry != m_program_cache->programs.end() &&
		    std::ranges::any_of(entry->second.permutations, [&](const auto& candidate) {
			    const auto& layout = candidate.program.bindings;
			    return candidate.specialization == record.specialization &&
				       layout.push_data_start_dword == layout.PushStartFor(record.push_data_start_dword);
		    })) {
			m_graphics.device.destroyShaderModule(result.compiled.module, nullptr);
			return nullptr;
		}
		if (entry == m_program_cache->programs.end()) {
			std::unique_lock insert_lock(m_program_cache->programs_mutex);
			entry = m_program_cache->programs.try_emplace(std::move(key), std::move(*result.plan))
			            .first;
		}
		entry->second.permutations.push_back(m_program_cache->MakePermutation(
		    options, record.specialization, std::move(result.compiled)));
		++compiled;
		if (ReplayVerifyEnabled()) [[unlikely]] {
			auto& added         = entry->second.permutations.back();
			added.replay_source = std::make_shared<const ShaderPrecompile::PermutationRecord>(record);
			added.replay_record = index;
		}
		record_ids[index] = entry->second.permutations.back().handle.id;
		return &entry->second.permutations.back();
	};

	translate(std::span(order).first(static_cast<size_t>(compute_end - order.begin())));
	{
		// Pipeline layouts of bindless shaders need the bindless table, made after this cache.
		m_context_ready_future.wait();
		// KYTY_DEBUG_REPLAY_PIPELINES=0: the replay still adds the journal's modules, but builds
		// no pipelines from them; each dispatch compiles its own, as before the compute queue.
		static const bool build_pipelines = [] {
			const char* value = KYTY_GETENV("KYTY_DEBUG_REPLAY_PIPELINES");
			return value == nullptr || std::strcmp(value, "0") != 0;
		}();
		Common::LockGuard lock(m_mutex);
		for (auto index = order.begin(); index != compute_end; ++index) {
			const auto* permutation = insert(*index);
			if (permutation == nullptr || !build_pipelines) {
				continue;
			}
			// The layouts read only the program; the dispatch that needs the pipeline takes it.
			ShaderComputeInputInfo input_info {};
			input_info.stage = {.program = &permutation->program};
			queued += StartComputeBuild(input_info, permutation->handle, ComputeOrigin::Replay,
			                            false)
			              ? 1u
			              : 0u;
		}
	}
	const auto compute_done = std::chrono::steady_clock::now();
	translate(std::span(order).subspan(static_cast<size_t>(compute_end - order.begin())));
	{
		Common::LockGuard lock(m_mutex);
		for (auto index = compute_end; index != order.end(); ++index) {
			(void)insert(*index);
		}
	}
	// KYTY_DEBUG_REPLAY_DETERMINISM=1: translates every record twice more, the two copies on
	// different threads at the same time, and names each shader whose SPIR-V differs between
	// them: the translator must give one result for one input on any thread.
	static const bool determinism = [] {
		const char* value = KYTY_GETENV("KYTY_DEBUG_REPLAY_DETERMINISM");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (determinism) {
		struct Copy {
			uint64_t        hash   = 0;
			std::thread::id thread = {};
			bool            valid  = false;
		};
		std::vector<std::array<Copy, 2>> copies(records.size());
		std::atomic_size_t               item {0};
		// Items 2i and 2i+1 are the two copies of record i: taken one after the other by
		// different threads, so they mostly run at the same time. Each copy translates its own
		// record copy, as translation may write to the record's input info.
		const auto check = [&] {
			for (size_t at = item.fetch_add(1); at < records.size() * 2u; at = item.fetch_add(1)) {
				auto         record = records[at / 2u];
				ShaderParams params;
				const auto   options    = options_for(record, params);
				auto         translated = ShaderRecompiler::TranslateProgram(params.code, options);
				auto&        copy       = copies[at / 2u][at % 2u];
				copy.thread             = std::this_thread::get_id();
				if (translated.skip_dispatch || !translated.program.resource_tracking_complete ||
				    translated.program.info.buffers.size() != record.specialization.buffers.size() ||
				    translated.program.info.images.size() > record.specialization.images.size()) {
					continue;
				}
				const auto result = ShaderRecompiler::CompileProgram(
				    std::move(translated), options, record.specialization, record.push_data_start_dword);
				copy.hash  = XXH3_64bits(result.spirv.data(), result.spirv.size() * sizeof(uint32_t));
				copy.valid = true;
			}
		};
		{
			std::vector<std::jthread> helpers;
			for (uint32_t i = 1; i < std::max(threads, 2u); i++) {
				helpers.emplace_back(check);
			}
			check();
		}
		size_t differing = 0;
		size_t same_thread = 0;
		for (size_t index = 0; index < records.size(); index++) {
			const auto& [first, second] = copies[index];
			same_thread += first.thread == second.thread;
			if (first.valid != second.valid || first.hash != second.hash) {
				differing++;
				PipelineCacheLog("Shader precompile determinism: hash={:016x} id={} stage={} record={} "
				                 "spirv {:016x} vs {:016x}{}",
				                 records[index].hash, record_ids[index],
				                 static_cast<uint32_t>(records[index].stage), index, first.hash,
				                 second.hash,
				                 first.thread == second.thread ? " (same thread)" : "");
			}
		}
		PipelineCacheLog("Shader precompile determinism: {} of {} records differ ({} pairs ran on "
		                 "one thread)",
		                 differing, records.size(), same_thread);
	}
	const auto ms = [](auto duration) {
		return std::chrono::duration<double, std::milli>(duration).count();
	};
	PipelineCacheLog("Shader precompile: replayed {} permutations on {} threads; skipped {}; "
	                 "compute modules {:.0f} ms, all {:.0f} ms; queued {} compute pipelines on {} "
	                 "threads",
	                 compiled, threads, skipped, ms(compute_done - started),
	                 ms(std::chrono::steady_clock::now() - started), queued,
	                 m_compute_queue->ThreadCount());
}

bool PipelineCache::StartComputeBuild(const ShaderComputeInputInfo& input_info,
                                      const ShaderProgram& program, ComputeOrigin origin,
                                      bool urgent) {
	if (m_compute_pipelines.contains(program.id)) {
		return false;
	}
	if (m_compute_builds.contains(program.id)) {
		if (urgent) {
			m_compute_queue->Promote(program.id);
		}
		return false;
	}
	ComputeBuild build;
	build.pipeline    = std::make_unique<Pipeline>();
	build.compile_us  = std::make_shared<std::atomic<int64_t>>(-1);
	build.shader_hash = input_info.stage.program->shader_hash;
	build.origin      = origin;
	auto create = PrepareComputePipeline(m_graphics, *build.pipeline, input_info, program.module,
	                                     m_driver_cache, origin == ComputeOrigin::Prefetch);
	build.result = m_compute_queue->Post(
	    program.id,
	    [this, create = std::move(create), compile_us = build.compile_us, origin]() mutable {
		    const auto started  = std::chrono::steady_clock::now();
		    const auto pipeline = create();
		    const auto elapsed  = std::chrono::duration_cast<std::chrono::microseconds>(
		                             std::chrono::steady_clock::now() - started)
		                             .count();
		    compile_us->store(elapsed, std::memory_order_release);
		    if (pipeline != nullptr) {
			    auto& stats = m_compute_stats[static_cast<size_t>(origin)];
			    stats.built.fetch_add(1, std::memory_order_relaxed);
			    if (elapsed < ComputeCacheHitMicroseconds) {
				    stats.hits.fetch_add(1, std::memory_order_relaxed);
			    }
		    }
		    // Saved now, not when a dispatch takes it: a run ended by force keeps a slow compile even
		    // when no dispatch needed the pipeline yet (or none ever does).
		    if (elapsed >= 2'000'000) {
			    MarkDriverCacheDirty();
		    } else if (pipeline != nullptr && elapsed >= ComputeCacheHitMicroseconds) {
			    NotePipelineCompiled();
		    }
		    return pipeline;
	    },
	    urgent, origin == ComputeOrigin::Replay);
	m_compute_builds.emplace(program.id, std::move(build));
	return true;
}

void PipelineCache::OnContextReady() {
	std::call_once(m_context_ready_once, [this] { m_context_ready.set_value(); });
}

void PipelineCache::Save() {
	// A replay waiting for the context could not finish otherwise.
	OnContextReady();
	// Join before acquiring the cache mutex or destroying the driver cache. This also covers
	// the WindowRun shutdown path, which calls Save before the PipelineCache destructor.
	WaitForPrecompile();
	if (m_save_thread.joinable()) {
		// Its write finishes first; the write below covers what it had pending.
		m_save_thread.request_stop();
		m_save_thread.join();
	}
	if (m_libraries != nullptr) {
		// The link thread writes to the driver cache, which is saved and destroyed below.
		m_libraries->Stop();
	}
	if (m_compute_queue != nullptr) {
		// As the library threads: their compiles write to the driver cache.
		m_compute_queue->Stop();
		if (!m_compute_stats_logged.exchange(true)) {
			LogComputeStats();
		}
	}
	Common::LockGuard lock(m_mutex);
	m_program_cache->recording.Close();
	if (!m_driver_cache) {
		return;
	}
	(void)WriteDriverCache();
	for (uint32_t index = 0; index < m_driver_cache.count; index++) {
		m_graphics.device.destroyPipelineCache(m_driver_cache.shards[index], nullptr);
	}
	m_driver_cache = {};
}

// Writes the driver pipeline cache to disk; called by Save under m_mutex and by the save thread
// (which Save stops first) after slow compiles, so a run ended by force (or a lost device) keeps
// what took long to compile.
bool PipelineCache::WriteDriverCache() {
	std::lock_guard      write_lock(m_driver_cache_write_mutex);
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache.Growing(), &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache.Growing(), &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return false;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_write_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return false;
	}
	auto temp_path = m_driver_cache_write_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_write_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_write_path));
		return false;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_write_path));
	return true;
}

// The color slots whose render target has a 16-bit float format (ShaderPixelInputInfo::
// target_half_float_mask), from the context's targets alone.
static uint8_t RenderTargetHalfFloatMask(const HW::Context& context) {
	uint8_t mask = 0;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX && slot < 8u; slot++) {
		const auto& rt = context.GetRenderTarget(slot);
		if (rt.base.addr != 0 && rt.info.channel_type == Prospero::ChannelType::kFloat &&
		    (rt.info.format == Prospero::ChannelLayout::k16 ||
		     rt.info.format == Prospero::ChannelLayout::k16_16 ||
		     rt.info.format == Prospero::ChannelLayout::k16_16_16_16)) {
			mask = static_cast<uint8_t>(mask | (1u << slot));
		}
	}
	return mask;
}

// Copies what preparing a non-tessellated draw's stage inputs reads (see GraphicsStageRegisters).
static void CaptureGraphicsStageRegisters(
    GraphicsStageRegisters& registers, const HW::VertexShaderInfo& vertex_regs,
    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
    const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    bool pixel_active) {
	std::memset(&registers, 0, sizeof(registers));
	std::memcpy(&registers.vertex, &vertex_regs, sizeof(registers.vertex));
	std::memcpy(&registers.pixel, &pixel_regs, sizeof(registers.pixel));
	std::memcpy(&registers.shader, &sh, sizeof(registers.shader));
	std::memcpy(&registers.blend0, &context.GetBlendControl(0), sizeof(registers.blend0));
	std::memcpy(&registers.clip, &context.GetClipControl(), sizeof(registers.clip));
	std::memcpy(&registers.mode, &context.GetModeControl(), sizeof(registers.mode));
	std::memcpy(&registers.ge, &user_config.GetGeControl(), sizeof(registers.ge));
	const auto& viewport = context.GetScreenViewport().viewports[0];
	registers.viewport0  = {viewport.xscale, viewport.yscale, viewport.xoffset, viewport.yoffset};
	std::memcpy(registers.export_mapping.data(), target_export_mapping.data(),
	            sizeof(registers.export_mapping));
	registers.shader_stages    = context.GetShaderStages();
	registers.prim_type        = static_cast<uint32_t>(user_config.GetPrimType());
	registers.rt0_blend_bypass = context.GetRenderTarget(0).info.blend_bypass;
	registers.pixel_active     = pixel_active;
	registers.rt_half_float_mask = RenderTargetHalfFloatMask(context);
}

// Whether the live registers would capture to the same bytes as `captured`
// (CaptureGraphicsStageRegisters), compared in place instead of through a copy.
static bool SameGraphicsStageRegisters(
    const GraphicsStageRegisters& captured, const HW::VertexShaderInfo& vertex_regs,
    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
    const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    bool pixel_active) {
	const auto same = [](const auto& a, const auto& b) {
		static_assert(sizeof(a) == sizeof(b));
		return std::memcmp(&a, &b, sizeof(a)) == 0;
	};
	const auto& viewport = context.GetScreenViewport().viewports[0];
	const std::array<float, 4> viewport0 {viewport.xscale, viewport.yscale, viewport.xoffset,
	                                      viewport.yoffset};
	return captured.shader_stages == context.GetShaderStages() &&
	       captured.prim_type == static_cast<uint32_t>(user_config.GetPrimType()) &&
	       captured.pixel_active == pixel_active &&
	       captured.rt_half_float_mask == RenderTargetHalfFloatMask(context) &&
	       captured.rt0_blend_bypass == context.GetRenderTarget(0).info.blend_bypass &&
	       same(captured.viewport0, viewport0) && same(captured.vertex, vertex_regs) &&
	       same(captured.pixel, pixel_regs) && same(captured.shader, sh) &&
	       same(captured.blend0, context.GetBlendControl(0)) &&
	       same(captured.clip, context.GetClipControl()) &&
	       same(captured.mode, context.GetModeControl()) &&
	       same(captured.ge, user_config.GetGeControl()) &&
	       std::memcmp(captured.export_mapping.data(), target_export_mapping.data(),
	                   sizeof(captured.export_mapping)) == 0;
}

// GetGraphicsPrograms' stage inputs and lookup parameters for these registers.
static std::array<ShaderParams, 3> PrepareGraphicsStages(
    const GraphicContext& graphics, const HW::VertexShaderInfo& vertex_regs,
    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
    const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    ShaderParams& pixel_params) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = graphics.subgroup_size;
		const auto& limits      = graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	g_draw_phases.Mark(DrawPhaseTimer::VertexParams);
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		// Only slots the shader exports to (the key then stays the same for the others).
		uint8_t exported = 0;
		for (uint32_t slot = 0; slot < 8u; slot++) {
			if (pixel_info.target_output_mode[slot] != 0) {
				exported = static_cast<uint8_t>(exported | (1u << slot));
			}
		}
		pixel_info.target_half_float_mask =
		    static_cast<uint8_t>(RenderTargetHalfFloatMask(context) & exported);
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	// Hardware ID buffer: an enabled 32-bit uint color slot that the shader does not export to
	// (SPI_SHADER_COL_FORMAT zero, CB_SHADER_MASK clear) receives VGT_OBJECT_ID | primitive id per
	// pixel (Ghost of Yotei's visibility pre-pass). KYTY_ID_BUFFER=0 turns it off.
	static const bool id_buffer_enabled = [] {
		const char* text = KYTY_GETENV("KYTY_ID_BUFFER");
		return text == nullptr || text[0] != '0';
	}();
	if (pixel_active && id_buffer_enabled && !pixel_info.dual_source_blending &&
	    vertex_info[0].logical_stage != ShaderType::Mesh) {
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if (rt.base.addr != 0 && render_target_mask_slot(context.GetRenderTargetMask(), slot) != 0 &&
			    render_target_mask_slot(sh.m_cbShaderMask, slot) == 0 &&
			    pixel_info.target_output_mode[slot] == 0 &&
			    rt.info.format == Prospero::ChannelLayout::k32 &&
			    rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.id_buffer_slot = slot;
				break;
			}
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	g_draw_phases.Mark(DrawPhaseTimer::PixelParams);
	return vertex_params;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    ProgramWait wait) {
	WaitForPrecompile();
	const bool   tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	ShaderParams pixel_params;
	std::array<ShaderParams, 3> vertex_params;
	// The draw speculation's prepared stage inputs, when the registers they were prepared from, the
	// shader map and the vertex tables are unchanged (see PreparedGraphicsStages).
	// KYTY_DEBUG_AB=stageinputs prepares them here in every other window.
	const auto take_prepared = [&]() -> SpeculatedDraw* {
		auto* draw = t_speculated_draw;
		if (tess_active || wait == ProgramWait::Prefetch || draw == nullptr ||
		    !draw->prepared.valid) {
			return nullptr;
		}
		auto& prepared = draw->prepared;
		prepared.valid = false;
		static const bool ab = AbSelected("stageinputs"); // AB-TEMP
		if (ab && AbFeatureOff()) {
			return nullptr;
		}
		const bool unchanged = [&] {
			DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::InputsCheck);
			return SameGraphicsStageRegisters(prepared.registers, vertex_regs, pixel_regs, sh,
			                                  context, user_config, target_export_mapping,
			                                  pixel_active) &&
			       ShaderMapVersion() == prepared.shader_map_version &&
			       VertexTableReadsUnchanged(prepared.vertex_tables);
		}();
		if (!unchanged) {
			m_program_cache->speculation.inputs_changed++;
			return nullptr;
		}
		CopyVertexInputInfo(vertex_info[0], prepared.vertex_info);
		vertex_params[0] = prepared.vertex_params;
		if (pixel_active) {
			pixel_info   = prepared.pixel_info;
			pixel_params = prepared.pixel_params;
		}
		if (m_program_cache->verify_speculation) {
			// KYTY_VERIFY_SPEC=1: the taken inputs must equal inputs prepared now.
			thread_local std::array<ShaderVertexInputInfo, 3> check_vertex;
			thread_local ShaderPixelInputInfo                 check_pixel;
			check_pixel = {};
			ShaderParams check_pixel_params;
			const auto   check_vertex_params =
			    PrepareGraphicsStages(m_graphics, vertex_regs, pixel_regs, sh, context, user_config,
			                          target_export_mapping, pixel_active, check_vertex, check_pixel,
			                          check_pixel_params);
			const auto same_params = [](const ShaderParams& a, const ShaderParams& b) {
				return a.code.data() == b.code.data() && a.code.size() == b.code.size() &&
				       a.back_code.data() == b.back_code.data() &&
				       a.back_code.size() == b.back_code.size() && a.hash == b.hash &&
				       a.user_data_count == b.user_data_count && a.user_data == b.user_data;
			};
			if (!SameVertexInputInfo(vertex_info[0], check_vertex[0]) ||
			    !same_params(vertex_params[0], check_vertex_params[0]) ||
			    (pixel_active && (!SamePixelInputInfo(pixel_info, check_pixel) ||
			                      !same_params(pixel_params, check_pixel_params)))) {
				EXIT("draw speculation: prepared stage inputs differ from inputs prepared now\n");
			}
		}
		g_draw_phases.Mark(DrawPhaseTimer::VertexParams);
		g_draw_phases.Mark(DrawPhaseTimer::PixelParams);
		m_program_cache->speculation.inputs_taken++;
		return draw;
	};
	const auto* prepared = take_prepared();
	if (prepared == nullptr) {
		vertex_params = PrepareGraphicsStages(m_graphics, vertex_regs, pixel_regs, sh, context,
		                                      user_config, target_export_mapping, pixel_active,
		                                      vertex_info, pixel_info, pixel_params);
	}
	const void* known_vertex =
	    prepared != nullptr ? prepared->stages[SpeculatedDraw::Vertex].source : nullptr;
	const void* known_pixel =
	    prepared != nullptr ? prepared->stages[SpeculatedDraw::Pixel].source : nullptr;
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	if (wait == ProgramWait::Defer) {
		// A draw that stores data is never skipped (see StoresData in renderDraw.cpp), so its
		// shaders compile now.
		bool stores = pixel_active && m_program_cache->StoresData(ShaderType::Pixel, pixel_params);
		for (uint32_t i = 0; i < (tess_active ? 3u : 1u) && !stores; i++) {
			stores = m_program_cache->StoresData(vertex_info[i].logical_stage, vertex_params[i]);
		}
		if (stores) {
			wait = ProgramWait::Wait;
		}
	}
	// A pending stage stops the lookups: the push data position of the next stage depends on it.
	// A look-ahead still starts translating the later stages, which that position does not affect.
	const auto translate_rest = [&](uint32_t first_vertex) {
		if (wait == ProgramWait::Prefetch) {
			for (uint32_t i = first_vertex; i < (tess_active ? 3u : 1u); i++) {
				m_program_cache->QueueSourceTranslation(vertex_params[i], vertex_info[i],
				                                        vertex_info[i].logical_stage);
			}
		}
		return result;
	};
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor, wait,
		                                    &result.pending, known_pixel);
		if (result.pending) {
			return translate_rest(0);
		}
	}
	g_draw_phases.Mark(DrawPhaseTimer::PixelProgram);
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor,
		                                        wait, &result.pending,
		                                        i == 0 ? known_vertex : nullptr);
		if (result.pending) {
			return translate_rest(i + 1);
		}
	}
	g_draw_phases.Mark(DrawPhaseTimer::VertexProgram);
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               ProgramWait wait, bool* pending) {
	WaitForPrecompile();
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor = 0;
	const auto program = m_program_cache->Get(params, input_info, push_data_cursor, wait, pending);
	if (program) {
		m_compute_sources.insert(ComputeSourceKey(regs));
	}
	return program;
}

bool DynamicDepthBoundsEnabled() {
#if defined(__APPLE__)
	return false;
#else
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DYNAMIC_DEPTH_BOUNDS");
		return text != nullptr && std::strcmp(text, "1") == 0;
	}();
	return enabled;
#endif
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline* PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs,
    vk::ImageAspectFlags feedback_aspects, bool may_defer) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	EXIT_IF(feedback_aspects && !m_graphics.attachment_feedback_loop_enabled);
	const auto color_count = static_cast<uint32_t>(colors.size());

	Common::LockGuard lock(m_mutex);
	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	static_params.attachment_feedback_loop_flags = AttachmentFeedbackPipelineFlags(
	    feedback_aspects, m_graphics.attachment_feedback_loop_dynamic_enabled);
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	if (!DynamicDepthBoundsEnabled()) {
		static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
		static_params.depth_min_bounds         = depth.depth_min_bounds;
		static_params.depth_max_bounds         = depth.depth_max_bounds;
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	// KYTY_DEBUG_AB=pipelinememo looks every key up in the map in every other window.
	static const bool ab    = AbSelected("pipelinememo");
	Pipeline*         found = nullptr;
	if (m_last_graphics_pipeline != nullptr && !(ab && AbFeatureOff()) &&
	    key == m_last_graphics_key) {
		found = m_last_graphics_pipeline;
	} else if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		found                    = iter->second.get();
		m_last_graphics_key      = key;
		m_last_graphics_pipeline = found;
	}
	if (found != nullptr) {
		if (found->optimize_pending) [[unlikely]] {
			InstallOptimizedPipeline(*found, command);
		}
		return found;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	if (may_defer && m_libraries != nullptr && Config::PipelineLibrariesEnabled()) {
		// Queue the shader parts no earlier pipeline built, and skip the draw until they are
		// compiled; the other two parts and the link take about a millisecond.
		bool ready = false;
		PrefetchLibraryParts(m_graphics, rendering, key.vertex_input, vertex_info, ps_input_info,
		                     programs, static_params, *m_libraries, m_driver_cache, &ready);
		if (!ready) {
			m_deferred_draws[key]++;
			return nullptr;
		}
	}
	uint32_t deferred_draws = 0;
	if (auto deferred = m_deferred_draws.find(key); deferred != m_deferred_draws.end()) {
		deferred_draws = deferred->second;
		m_deferred_draws.erase(deferred);
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	const auto create_start = std::chrono::steady_clock::now();
	const int library_parts =
	    CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                           ps_input_info, programs, static_params, m_libraries.get(),
	                           m_driver_cache);
	m_graphics_pipelines_created++;
	if (PermutationLogEnabled()) [[unlikely]] {
		// "libs=" lists the library parts compiled: vertex input, pre-rasterization, fragment
		// shader, fragment output ("-" when all were cached); "mono" is a monolithic pipeline.
		// "pre=" lists the shader parts a prefetch had compiled.
		const auto letters = [](uint32_t bits, const char* names) {
			std::string text;
			for (uint32_t bit = 0; names[bit] != '\0'; bit++) {
				if ((bits & (1u << bit)) != 0) {
					text += names[bit];
				}
			}
			return text.empty() ? std::string("-") : text;
		};
		const auto bits = static_cast<uint32_t>(library_parts);
		const auto now = std::chrono::steady_clock::now();
		// "deferred=" counts the draws skipped while its parts compiled; "sync" marks a pipeline
		// created synchronously although asynchronous pipelines are on.
		std::string deferral;
		if (deferred_draws != 0) {
			deferral = fmt::format(" deferred={}", deferred_draws);
		} else if (!may_defer && Config::AsyncPipelinesEnabled()) {
			deferral = " sync";
		}
		std::printf("pipeline: vs=%llu ps=%llu ms=%.1f libs=%s%s%s%s t=%.3f\n",
		            static_cast<unsigned long long>(vs_id), static_cast<unsigned long long>(ps_id),
		            std::chrono::duration<double, std::milli>(now - create_start).count(),
		            library_parts < 0 ? "mono" : letters(bits & 0xfu, "VPFO").c_str(),
		            library_parts > 0 && (bits >> 5u) != 0 ? " pre=" : "",
		            library_parts > 0 && (bits >> 5u) != 0 ? letters(bits >> 5u, "PF").c_str() : "",
		            deferral.c_str(), std::chrono::duration<double>(now.time_since_epoch()).count());
	}
	RecordPipelineCreated(create_start);
	// Driver cache hits take about a millisecond; a compile takes longer.
	if (std::chrono::steady_clock::now() - create_start >= std::chrono::milliseconds(10)) {
		NotePipelineCompiled();
	}
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	m_last_graphics_key      = iter->first;
	m_last_graphics_pipeline = iter->second.get();

	return iter->second.get();
}

// Whether a draw with these registers looks its programs up (GetGraphicsPrograms), as the draw
// path decides, and with which pixel stage activity and render target export mappings.
static bool PredictGraphicsDraw(
    const HW::Context& ctx, const HW::Shader& sh, const HW::UserConfig& user_config,
    bool&                                                                       ps_active,
    std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>& export_mapping) {
	const auto& vs = sh.GetVs();
	const auto& ps = sh.GetPs();
	if (vs.es_regs.data_addr == 0) {
		return false;
	}
	// Only primitive types the draw path accepts; RectList pipelines stay monolithic.
	switch (user_config.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kQuadListLegacy: break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) {
				return false;
			}
			break;
		default: return false;
	}
	// The draw path turns metadata color modes, resolves and depth/stencil copies into other
	// operations without translating their shaders, and a translation must not run that the real
	// draw never would.
	const auto color_mode = ctx.GetColorControl().mode;
	if (color_mode > 1) {
		return false;
	}
	const auto& override    = ctx.GetDepthRenderOverride();
	const auto& depth_regs  = ctx.GetDepthRenderTarget();
	const bool  depth_copy  = override.force_z_dirty && override.force_z_valid &&
	                        depth_regs.z_info.format != Prospero::DepthFormat::kInvalid &&
	                        depth_regs.z_read_base_addr != 0 && depth_regs.z_write_base_addr != 0 &&
	                        depth_regs.z_read_base_addr != depth_regs.z_write_base_addr;
	const bool stencil_copy =
	    override.force_stencil_dirty && override.force_stencil_valid &&
	    depth_regs.stencil_info.format != Prospero::StencilFormat::kInvalid &&
	    depth_regs.stencil_read_base_addr != 0 && depth_regs.stencil_write_base_addr != 0 &&
	    depth_regs.stencil_read_base_addr != depth_regs.stencil_write_base_addr;
	if (color_mode == 0 && (depth_copy || stencil_copy)) {
		return false;
	}

	// The draw path's decisions, made from the same registers.
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto& db          = sh_regs.db_shader_control;
	const bool  side_effect = db.shader_kill_enable || db.shader_z_export_enable ||
	                         db.shader_mask_export_enable || db.shader_dual_export_enable ||
	                         db.shader_execute_on_noop;
	const auto target_mask = ctx.GetRenderTargetMask();
	ps_active              = ps.ps_regs.data_addr != 0 &&
	            ((target_mask & sh_regs.m_cbShaderMask) != 0 || side_effect);
	export_mapping = {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if (rt.base.addr != 0 && render_target_mask_slot(target_mask, slot) != 0) {
			export_mapping[slot] = TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                                    rt.info.channel_order)
			                           .export_mapping;
		}
	}
	return true;
}

bool PipelineCache::SpeculateGraphicsPrograms(const HW::Context& ctx, const HW::Shader& sh,
                                              const HW::UserConfig& user_config,
                                              const BufferCache& buffers, SpeculatedDraw& draw) {
	for (auto& stage: draw.stages) {
		stage.source = nullptr;
	}
	draw.prepared.valid = false;
	bool ps_active = false;
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> export_mapping {};
	if (user_config.GetPrimType() == Prospero::PrimitiveType::kPatch ||
	    !PredictGraphicsDraw(ctx, sh, user_config, ps_active, export_mapping)) {
		m_program_cache->speculation_failures[0].fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	// The draw path's stage inputs, prepared as GetGraphicsPrograms prepares them, and kept with
	// what they were prepared from for the GPU thread to take (see PreparedGraphicsStages).
	thread_local std::array<ShaderVertexInputInfo, 3> vertex_info;
	thread_local ShaderPixelInputInfo                 pixel_info;
	pixel_info = {};
	auto& prepared                  = draw.prepared;
	prepared.valid                  = false;
	prepared.shader_map_version     = ShaderMapVersion();
	prepared.vertex_tables.count    = 0;
	prepared.vertex_tables.complete = true;
	t_vertex_table_reads            = &prepared.vertex_tables;
	ShaderParams pixel_params;
	const auto   vertex_params =
	    PrepareGraphicsStages(m_graphics, sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(), ctx,
	                          user_config, export_mapping, ps_active, vertex_info, pixel_info,
	                          pixel_params);
	t_vertex_table_reads = nullptr;
	CaptureGraphicsStageRegisters(prepared.registers, sh.GetVs(), sh.GetPs(),
	                              ctx.GetShaderRegisters(), ctx, user_config, export_mapping,
	                              ps_active);
	CopyVertexInputInfo(prepared.vertex_info, vertex_info[0]);
	prepared.vertex_params = vertex_params[0];
	prepared.pixel_info    = pixel_info;
	prepared.pixel_params  = pixel_params;
	prepared.valid         = true;
	bool speculated = m_program_cache->Speculate(vertex_params[0], vertex_info[0],
	                                             vertex_info[0].logical_stage, buffers,
	                                             draw.stages[SpeculatedDraw::Vertex]);
	if (ps_active) {
		speculated |= m_program_cache->Speculate(pixel_params, pixel_info, ShaderType::Pixel,
		                                          buffers, draw.stages[SpeculatedDraw::Pixel]);
	}
	return speculated;
}

bool SpeculateDispatchesEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = KYTY_GETENV("KYTY_SPECULATE_DISPATCHES");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

bool SpeculateDispatchesNow() noexcept {
	static const bool ab = SpeculateDispatchesEnabled() && AbSelected("specdispatch");
	return SpeculateDispatchesEnabled() && !(ab && AbFeatureOff());
}

bool PipelineCache::SpeculateComputeProgram(const HW::Context& ctx, const HW::Shader& sh,
                                            const uint32_t* groups, uint32_t mode,
                                            const BufferCache& buffers, SpeculatedDraw& draw) {
	for (auto& stage: draw.stages) {
		stage.source = nullptr;
	}
	draw.prepared.valid = false;
	auto       cs              = sh.GetCs();
	const bool thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	// An indirect dispatch's groups are GPU data: only the form whose program does not see them
	// (RenderExecutor::DispatchIndirect looks the program up with no group counts).
	if (cs.cs_regs.data_addr == 0 || (groups == nullptr && thread_dimensions) ||
	    (groups != nullptr && (groups[0] == 0 || groups[1] == 0 || groups[2] == 0))) {
		m_program_cache->speculation_failures[0].fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	// As CommandProcessor::DispatchDirect/DispatchIndirect, RenderExecutor::DispatchDirect/
	// DispatchIndirect and GetComputeProgram prepare the dispatch's stage inputs.
	cs.cs_regs.wave_size = Pm4::ComputeWaveSize(mode);
	ShaderComputeInputInfo input_info {};
	if (groups != nullptr) {
		input_info.dispatch_thread_dimensions = thread_dimensions;
		const std::array<uint32_t, 3> group_sizes {cs.cs_regs.num_thread_x,
		                                           cs.cs_regs.num_thread_y,
		                                           cs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size  = std::max(group_sizes[axis], 1u);
			const auto count = groups[axis];
			input_info.workgroup_counts[axis] =
			    thread_dimensions ? count / size + (count % size != 0u) : count;
		}
	}
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	auto params = PrepareProgram(cs, ctx.GetShaderRegisters(), input_info);
	if (ShaderParams linked; LinkShaderCalls(params, linked)) {
		params = linked;
	}
	return m_program_cache->Speculate(params, input_info, ShaderType::Compute, buffers,
	                                  draw.stages[SpeculatedDraw::Vertex]);
}

std::array<uint64_t, 4> PipelineCache::SpeculationFailures() const noexcept {
	std::array<uint64_t, 4> counts {};
	for (size_t i = 0; i < counts.size(); i++) {
		counts[i] = m_program_cache->speculation_failures[i].load(std::memory_order_relaxed);
	}
	return counts;
}

uint32_t PipelineCache::PrefetchGraphicsPipeline(const HW::Context& ctx, const HW::Shader& sh,
                                                 const HW::UserConfig& user_config,
                                                 ProgramWait wait, bool* pending) {
	KYTY_PROFILER_FUNCTION();
	const SpeculativeLookup speculative;
	if (m_libraries == nullptr || !Config::PipelineLibrariesEnabled()) {
		return 0;
	}
	const auto& vs = sh.GetVs();
	const auto& ps = sh.GetPs();
	bool        ps_active = false;
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX> export_mapping {};
	if (!PredictGraphicsDraw(ctx, sh, user_config, ps_active, export_mapping)) {
		return 0;
	}
	const auto  prim_type   = user_config.GetPrimType();
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  target_mask = ctx.GetRenderTargetMask();
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 pixel_info;
	const auto programs = GetGraphicsPrograms(vs, ps, sh_regs, ctx, user_config, export_mapping,
	                                          ps_active, vertex_info, pixel_info, wait);
	if (programs.pending) {
		*pending = true;
		return 0;
	}
	if (!programs.vertex[0] || (ps_active && !programs.pixel)) {
		return 0;
	}

	uint32_t samples = 0;
	if (ps_active) {
		for (const auto& output: pixel_info.stage.program->info.outputs) {
			if (output.kind != ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.index >= RENDER_COLOR_ATTACHMENTS_MAX || samples != 0) {
				continue;
			}
			const auto& rt = ctx.GetRenderTarget(output.index);
			if (rt.base.addr != 0 && render_target_mask_slot(target_mask, output.index) != 0) {
				samples = render_sample_count(rt.attrib.num_fragments);
			}
		}
	}
	// As ResolveRenderDepthTarget decides whether the draw has a depth attachment.
	const auto& z           = ctx.GetDepthRenderTarget();
	const auto& rc          = ctx.GetRenderControl();
	const auto& dc          = ctx.GetDepthControl();
	const bool  has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	const bool  depth_active =
	    dc.z_enable || dc.depth_bounds_enable || rc.depth_clear_enable || rc.copy_depth_to_color;
	const bool stencil_active =
	    has_stencil && (dc.stencil_enable || rc.stencil_clear_enable || rc.copy_stencil_to_color);
	const bool with_depth = (depth_active || stencil_active) &&
	                        (z.z_info.format != Prospero::DepthFormat::kInvalid || has_stencil);
	if (with_depth && samples == 0) {
		samples = render_sample_count(z.z_info.num_samples);
	}
	if (samples == 0 && !with_depth) {
		samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
	}
	if (samples == 0) {
		return 0;
	}

	PipelineStaticParameters static_params {};
	const auto&              clip_control = ctx.GetClipControl();
	const auto&              mc           = ctx.GetModeControl();
	static_params.negative_one_to_one     = !clip_control.dx_clip_space;
	static_params.depth_clip_enable       = clip_control.IsZClipEnabled();
	static_params.topology                = prim_type == Prospero::PrimitiveType::kPatch
	                                            ? vk::PrimitiveTopology::ePatchList
	                                            : vk::PrimitiveTopology::eTriangleList;
	static_params.samples                 = samples;
	static_params.sample_shading_enable   = ps_active && samples > 1 && pixel_info.ps_sample_shading;
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	if (!DynamicDepthBoundsEnabled()) {
		static_params.depth_bounds_test_enable = with_depth && dc.depth_bounds_enable;
		static_params.depth_min_bounds         = ctx.GetDepthBoundsMin();
		static_params.depth_max_bounds         = ctx.GetDepthBoundsMax();
	}
	static_params.cull_back                = mc.cull_back;
	static_params.cull_front               = mc.cull_front;
	static_params.face                     = mc.face;
	static_params.provoking_vtx_last       = mc.provoking_vtx_last;
	static_params.polygon_mode = ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		return 0;
	}

	// The shader parts do not depend on attachment formats or vertex input, which the prediction
	// leaves empty; a depth attachment only matters as present or not.
	PipelineRenderingState rendering {};
	if (with_depth) {
		rendering.depth_format = vk::Format::eD32Sfloat;
	}
	const PipelineVertexInputState vertex_input {};
	Common::LockGuard              lock(m_mutex);
	return PrefetchLibraryParts(m_graphics, rendering, vertex_input,
	                            std::span(vertex_info).first(programs.VertexStageCount()),
	                            ps_active ? &pixel_info : nullptr, programs, static_params,
	                            *m_libraries, m_driver_cache);
}

uint32_t PipelineCache::PrefetchComputePipeline(const HW::Context& ctx, const HW::Shader& sh,
                                                uint32_t dispatch_initiator, ProgramWait wait,
                                                bool* pending, const std::array<uint32_t, 3>* groups) {
	KYTY_PROFILER_FUNCTION();
	const SpeculativeLookup speculative;
	// Compute pipelines need no pipeline libraries: they compile on m_compute_queue.
	if (sh.GetCs().cs_regs.data_addr == 0) {
		return 0;
	}
	// The wave size comes from this dispatch's initiator, which the command processor applies
	// only when it executes the dispatch (CommandProcessor::DispatchDirect); the prefetched
	// state still holds the previous dispatch's. A wave32 shader translated as wave64 reads VALU
	// mask writes into the SGPR after their destination (Ghost of Yotei f5a93fd534e15940 lost
	// its T# that way and was skipped).
	auto cs = sh.GetCs();
	cs.cs_regs.wave_size = Pm4::ComputeWaveSize(dispatch_initiator);
	{
		Common::LockGuard lock(m_mutex);
		if (m_compute_sources.contains(ComputeSourceKey(cs))) {
			return 0;
		}
	}
	// As the dispatch path prepares it (see RenderExecutor::DispatchDirect/DispatchIndirect).
	ShaderComputeInputInfo input_info {};
	input_info.dispatch_thread_dimensions =
	    (dispatch_initiator & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	if (groups != nullptr) {
		const std::array<uint32_t, 3> group_sizes {cs.cs_regs.num_thread_x, cs.cs_regs.num_thread_y,
		                                           cs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			// Thread counts become group counts, as RenderExecutor::DispatchDirect converts them.
			const auto size  = std::max(group_sizes[axis], 1u);
			const auto count = (*groups)[axis];
			input_info.workgroup_counts[axis] =
			    input_info.dispatch_thread_dimensions ? count / size + (count % size != 0u) : count;
		}
	}
	const auto program = GetComputeProgram(cs, ctx.GetShaderRegisters(), input_info, wait, pending);
	if (!program) {
		return 0;
	}
	Common::LockGuard lock(m_mutex);
	// A specialization made from guest memory (an enumerated descriptor table, a buffer-loaded
	// value) is read here ahead of the dispatch, often before the game wrote what the dispatch
	// will see: the pipeline would be a different permutation (Ghost of Yotei b21e373811f28e31
	// compiled 61 s for one the dispatch never used). The module stays translated; the pipeline is
	// left to the dispatch.
	if (const auto* resources = input_info.stage.resources;
	    resources != nullptr && !resources->specialization_reads.empty()) {
		if (m_prefetch_deferred_logged.insert(input_info.stage.program->shader_hash).second) {
			PipelineCacheLog("prefetch: pipeline deferred to dispatch, specialization reads guest "
			                 "memory: hash={:016x} id={}",
			                 input_info.stage.program->shader_hash, program.id);
		}
		return 0;
	}
	return StartComputeBuild(input_info, program, ComputeOrigin::Prefetch, false) ? 1u : 0u;
}

void PipelineCache::LogLookahead(uint32_t draws, uint32_t parts) const {
	if (PermutationLogEnabled()) [[unlikely]] {
		std::printf("lookahead: draws=%u prefetched parts=%u t=%.3f\n", draws, parts,
		            std::chrono::duration<double>(
		                std::chrono::steady_clock::now().time_since_epoch())
		                .count());
	}
}

uint32_t PipelineCache::BackgroundShaderJobs() const noexcept {
	return m_program_cache->jobs->in_flight.load(std::memory_order_acquire);
}

uint64_t PipelineCache::BackgroundShaderJobsFinished() const noexcept {
	return m_program_cache->jobs->finished.load(std::memory_order_relaxed);
}

void PipelineCache::InstallOptimizedPipeline(Pipeline& pipeline, CommandBuffer& command) {
	const auto optimized = m_libraries->TakeOptimized(&pipeline);
	if (!optimized) {
		return;
	}
	pipeline.optimize_pending = false;
	if (*optimized == nullptr) {
		return;
	}
	// Commands recorded before this draw may still use the fast-linked pipeline.
	const auto fast_linked = pipeline.pipeline;
	pipeline.pipeline      = *optimized;
	command.GetContext().GetCommandScheduler().DeferOperation(
	    [device = m_graphics.device, fast_linked] { device.destroyPipeline(fast_linked, nullptr); });
}

// One line per compute pipeline a dispatch takes, so a log sums the GPU thread's stalls: where it
// came from, how long the GPU thread waited for it and how long the driver compiled it.
// The program id tells two permutations of one shader apart.
void PipelineCache::LogComputePipeline(uint64_t shader_hash, uint64_t program_id,
                                       ComputeOrigin                       origin,
                                       std::chrono::steady_clock::duration waited,
                                       int64_t                             compile_us) {
	const char* from = origin == ComputeOrigin::Replay     ? "replay"
	                   : origin == ComputeOrigin::Prefetch ? "prefetch"
	                                                       : "sync";
	PipelineCacheLog(
	    "ComputePipeline: hash={:016x} id={} from={} wait_ms={:.1f} compile_ms={:.1f} cache={}",
	    shader_hash, program_id, from, std::chrono::duration<double, std::milli>(waited).count(),
	    compile_us < 0 ? -1.0 : static_cast<double>(compile_us) / 1000.0,
	    compile_us < 0 ? "?" : compile_us < ComputeCacheHitMicroseconds ? "hit" : "miss");
}

// At Save: per origin, the compute pipelines compiled, how many of those compiled fast enough to
// count as driver cache hits, and how many a dispatch took (built but never taken: a look-ahead
// misprediction, or a journal permutation this run did not dispatch).
void PipelineCache::LogComputeStats() const {
	constexpr std::array<const char*, 3> Names {"replay", "prefetch", "sync"};
	std::string                          line = "Compute pipelines:";
	for (size_t i = 0; i < Names.size(); i++) {
		const auto& stats = m_compute_stats[i];
		line += fmt::format(" {} built={} hits={} taken={};", Names[i],
		                    stats.built.load(std::memory_order_relaxed),
		                    stats.hits.load(std::memory_order_relaxed),
		                    stats.taken.load(std::memory_order_relaxed));
	}
	line += fmt::format(" (hit: compile under {} ms)", ComputeCacheHitMicroseconds / 1000);
	PipelineCacheLog("{}", line);
}

void PipelineCache::MarkDriverCacheDirty() {
	{
		std::lock_guard lock(m_save_mutex);
		m_driver_cache_dirty = true;
		m_last_save_request  = std::chrono::steady_clock::now();
	}
	m_save_wake.notify_one();
}

void PipelineCache::NotePipelineCompiled() {
	static const int64_t minutes = [] {
		const char* text = KYTY_GETENV("KYTY_PIPELINE_SAVE_MINUTES");
		return text != nullptr ? std::strtoll(text, nullptr, 10) : int64_t {5};
	}();
	if (minutes <= 0 || m_compiled_since_save.fetch_add(1, std::memory_order_relaxed) + 1 < 100) {
		return;
	}
	const auto now  = std::chrono::steady_clock::now().time_since_epoch().count();
	auto       last = m_last_bulk_save.load(std::memory_order_relaxed);
	const auto interval =
	    std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::minutes(minutes))
	        .count();
	if (now - last < interval ||
	    !m_last_bulk_save.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: {} pipelines compiled since the last save, saving during play",
	                 m_compiled_since_save.exchange(0, std::memory_order_relaxed));
	MarkDriverCacheDirty();
}

// Writes the driver cache file when a slow compile made it dirty, on its own thread (not behind the
// queued compiles): at once unless it was written less than a minute ago, then a minute after that
// write, or 30 s after the last request at the latest. A compile finishing just after a write so
// reaches the file within 30 s without writing the gigabyte after every compile of a burst.
void PipelineCache::SaveThread(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("PipelineCacheSave");
	constexpr auto MinInterval = std::chrono::seconds(60);
	constexpr auto MaxDelay    = std::chrono::seconds(30);
	std::unique_lock lock(m_save_mutex);
	for (;;) {
		if (!m_save_wake.wait(lock, stop, [this] { return m_driver_cache_dirty; })) {
			return;
		}
		const auto due = std::min(m_last_save + MinInterval, m_last_save_request + MaxDelay);
		if (std::chrono::steady_clock::now() < due) {
			// Woken early by another request or a stop; the due time only moves later.
			(void)m_save_wake.wait_until(lock, stop, due, [] { return false; });
			if (stop.stop_requested()) {
				return;
			}
			continue;
		}
		m_driver_cache_dirty = false;
		lock.unlock();
		// Save stops this thread before it destroys the driver cache.
		(void)WriteDriverCache();
		lock.lock();
		m_last_save = std::chrono::steady_clock::now();
	}
}

bool PipelineCache::ComputePipelineReady(const ShaderComputeInputInfo& input_info,
                                         const ShaderProgram&          compute_program) {
	EXIT_IF(!compute_program);
	Common::LockGuard lock(m_mutex);
	if (m_compute_pipelines.contains(compute_program.id)) {
		return true;
	}
	(void)StartComputeBuild(input_info, compute_program, ComputeOrigin::Sync, true);
	auto& build = m_compute_builds.at(compute_program.id);
	if (build.result.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
		return true;
	}
	if (!build.deferred) {
		// The line its dispatch prints later shows no wait: the queue waited instead.
		build.deferred = true;
		PipelineCacheLog("ComputePipeline: hash={:016x} id={} not ready, dispatch deferred",
		                 build.shader_hash, compute_program.id);
	}
	return false;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	const auto                            wait_start = std::chrono::steady_clock::now();
	std::shared_future<vk::Pipeline>      result;
	std::shared_ptr<std::atomic<int64_t>> compile_us;
	ComputeOrigin                         origin_waited = ComputeOrigin::Sync;
	{
		Common::LockGuard lock(m_mutex);
		if (auto iter = m_compute_pipelines.find(compute_program.id);
		    iter != m_compute_pipelines.end()) {
			return *iter->second;
		}
		if (graphics_debug_dump_enabled()) {
			ShaderDbgDumpInputInfo(input_info);
		}
		// Built by the precompile replay or a look-ahead, or compiling: this dispatch waits for it
		// first. Otherwise it compiles on a worker thread too, ahead of the queued builds.
		(void)StartComputeBuild(input_info, compute_program, ComputeOrigin::Sync, true);
		const auto& build = m_compute_builds.at(compute_program.id);
		result            = build.result;
		compile_us        = build.compile_us;
		origin_waited     = build.origin;
	}
	// Without m_mutex: other lookups (draws, the look-ahead, the replay) go on meanwhile. The build
	// stays in m_compute_builds, so nothing queues this program again.
	vk::Pipeline pipeline = nullptr;
	{
		KYTY_PROFILER_BLOCK("PipelineCache::WaitForCompute");
		// Not started yet (the workers busy with other compiles, such as the replay's long ones):
		// this thread would only wait, so it compiles the pipeline itself.
		if (!m_compute_queue->RunHere(compute_program.id) &&
		    result.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
			// Already compiling on a worker: this dispatch waits for that compile.
			PipelineCacheLog("ComputePipeline: id={} waits for a running {} build",
			                 compute_program.id,
			                 origin_waited == ComputeOrigin::Replay     ? "replay"
			                 : origin_waited == ComputeOrigin::Prefetch ? "prefetch"
			                                                            : "sync");
		}
		pipeline = result.get();
	}
	const auto waited = std::chrono::steady_clock::now() - wait_start;

	Common::LockGuard lock(m_mutex);
	auto              build  = m_compute_builds.find(compute_program.id);
	EXIT_IF(build == m_compute_builds.end());
	auto              cached = std::move(build->second.pipeline);
	auto              origin = build->second.origin;
	m_compute_builds.erase(build);
	auto compiled_us = compile_us->load(std::memory_order_acquire);
	if (pipeline == nullptr) {
		// The worker threads stopped or the compile failed: compile here, as before them.
		DestroyPipelineObjects(m_graphics, *cached);
		cached = std::make_unique<Pipeline>();
		origin = ComputeOrigin::Sync;
		const auto compile_start = std::chrono::steady_clock::now();
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                       m_driver_cache);
		compiled_us = std::chrono::duration_cast<std::chrono::microseconds>(
		                  std::chrono::steady_clock::now() - compile_start)
		                  .count();
		auto& stats = m_compute_stats[static_cast<size_t>(origin)];
		stats.built.fetch_add(1, std::memory_order_relaxed);
		if (compiled_us < ComputeCacheHitMicroseconds) {
			stats.hits.fetch_add(1, std::memory_order_relaxed);
		}
		if (compiled_us >= 2'000'000) {
			MarkDriverCacheDirty();
		}
	} else {
		cached->pipeline = pipeline;
	}
	m_compute_stats[static_cast<size_t>(origin)].taken.fetch_add(1, std::memory_order_relaxed);
	LogComputePipeline(input_info.stage.program->shader_hash, compute_program.id, origin,
	                   std::chrono::steady_clock::now() - wait_start, compiled_us);
	// KYTY_DEBUG_NAME_PIPELINES=1 (capture runs set it): RenderDoc shows each compute pipeline as
	// "cs <guest hash>", so a capture's dispatches map to shaders (rd_query.py 'shader').
	static const bool name_pipelines =
	    KYTY_GETENV("KYTY_DEBUG_NAME_PIPELINES") != nullptr || Config::RenderDocEnabled();
	if (name_pipelines && cached->pipeline != nullptr &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT != nullptr) [[unlikely]] {
		char name[64] = {};
		std::snprintf(name, sizeof(name), "cs %016llx id %llu",
		              static_cast<unsigned long long>(input_info.stage.program->shader_hash),
		              static_cast<unsigned long long>(compute_program.id));
		vk::DebugUtilsObjectNameInfoEXT info {};
		info.objectType   = vk::ObjectType::ePipeline;
		info.objectHandle = reinterpret_cast<uint64_t>(static_cast<VkPipeline>(cached->pipeline));
		info.pObjectName  = name;
		(void)m_graphics.device.setDebugUtilsObjectNameEXT(&info);
	}
	m_compute_pipelines_created++;
	if (PermutationLogEnabled()) [[unlikely]] {
		const auto now = std::chrono::steady_clock::now();
		std::printf("pipeline: cs=%llu ms=%.1f%s t=%.3f\n",
		            static_cast<unsigned long long>(compute_program.id),
		            std::chrono::duration<double, std::milli>(waited).count(),
		            origin != ComputeOrigin::Sync ? " pre=C" : "",
		            std::chrono::duration<double>(now.time_since_epoch()).count());
	}
	RecordPipelineCreated(std::chrono::steady_clock::now() - std::chrono::microseconds(compiled_us));

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
