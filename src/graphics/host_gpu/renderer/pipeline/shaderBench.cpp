#include "graphics/host_gpu/renderer/pipeline/shaderBench.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/shaderPrecompile.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <spirv-tools/libspirv.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <xxhash.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Offline shader bench: replays the records of a precompile journal (<cache>.shaders) the way
// PipelineCache::ReplayPrecompiled does, writes their SPIR-V, and can time each compute module's
// vkCreateComputePipelines without a pipeline cache (the method of _yotei_slowshaders/harness/
// spvtime.cpp: a random OpModuleProcessed defeats the driver's own disk cache, the subgroup size
// is required as the renderer requires it, and the descriptor layout comes from the module).

namespace Libs::Graphics::ShaderBench {
namespace {

using ShaderPrecompile::PermutationRecord;

struct Options {
	std::filesystem::path    journal;
	std::filesystem::path    out_dir;
	std::vector<std::string> hashes;      // hex prefixes
	std::optional<ShaderType> stage;
	bool                     time          = false;
	bool                     validate      = false;
	bool                     bindless      = true;
	int                      hw_bounds     = -1; // -1: from the device (or on without Vulkan)
	int                      lds_storage   = -1; // -1: as recorded; 2: the live rule (LDS > limit)
	uint32_t                 max_lds_bytes = 0;  // 0: from the device (or 48 KiB)
	uint32_t                 repeat        = 1;
	uint32_t                 jobs          = 1;
	uint32_t                 limit         = UINT32_MAX;
	// Every bindless 2D / 2D-array image specialized as this integer class, as a heap of integer
	// T#s specializes it (Float: as recorded).
	Prospero::TextureNumericClass bindless_class = Prospero::TextureNumericClass::Float;
};

void PrintUsage() {
	std::printf(
	    "kyty_emulator --shader-bench <journal.shaders> [options]\n"
	    "  --hash <hex[,hex...]>   only records whose shader hash starts with one of these\n"
	    "  --stage cs|ps|vs|...    only records of this stage\n"
	    "  --out <dir>             write spv_<hash>_r<record>.spv per record\n"
	    "  --time                  time vkCreateComputePipelines per compute module (no cache)\n"
	    "  --validate              run spirv-val (Vulkan 1.3) on each module\n"
	    "  --repeat <n>            translate each record n times; report differing SPIR-V\n"
	    "  --jobs <n>              translation threads (default 1; >1 skews translate ms)\n"
	    "  --limit <n>             stop after n matching records\n"
	    "  --bindless 0|1          bindless images option (default 1, as for PPSA26344)\n"
	    "  --bindless-class uint|sint  specialize every bindless 2D image as an integer image\n"
	    "  --hw-bounds 0|1         hardware storage buffer bounds (default: from the device)\n"
	    "  --lds-storage 0|1|live  LDS in a storage buffer (default: as recorded; live: LDS above\n"
	    "                          the device limit, as GetComputeProgram decides)\n"
	    "  --max-lds <bytes>       maxComputeSharedMemorySize for --lds-storage live\n"
	    "Environment switches of the recompiler (KYTY_DEBUG_*) apply as in the emulator.\n");
}

bool ParseU32(std::string_view text, uint32_t& value) {
	const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
	return error == std::errc {} && end == text.data() + text.size();
}

std::optional<ShaderType> ParseStage(std::string_view name) {
	if (name == "cs") return ShaderType::Compute;
	if (name == "ps") return ShaderType::Pixel;
	if (name == "vs") return ShaderType::Vertex;
	if (name == "ms" || name == "gs") return ShaderType::Mesh;
	if (name == "ls") return ShaderType::Local;
	if (name == "hs") return ShaderType::TessellationControl;
	if (name == "ds") return ShaderType::TessellationEvaluation;
	return std::nullopt;
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return "cs";
		case ShaderType::Pixel: return "ps";
		case ShaderType::Vertex: return "vs";
		case ShaderType::Mesh: return "ms";
		case ShaderType::Local: return "ls";
		case ShaderType::TessellationControl: return "hs";
		case ShaderType::TessellationEvaluation: return "ds";
		default: return "??";
	}
}

bool ParseArgs(int argc, char* argv[], Options& o) {
	if (argc < 2) return false;
	o.journal = argv[1];
	for (int i = 2; i < argc; i++) {
		const std::string_view arg = argv[i];
		const bool             has_value = i + 1 < argc;
		const auto             value     = has_value ? std::string_view(argv[i + 1]) : "";
		uint32_t               number    = 0;
		if (arg == "--time") {
			o.time = true;
		} else if (arg == "--validate") {
			o.validate = true;
		} else if (arg == "--out" && has_value) {
			o.out_dir = argv[++i];
		} else if (arg == "--hash" && has_value) {
			size_t start = 0;
			while (start <= value.size()) {
				const auto comma = value.find(',', start);
				const auto part  = value.substr(start, comma == std::string_view::npos
				                                           ? std::string_view::npos
				                                           : comma - start);
				if (!part.empty()) o.hashes.emplace_back(part);
				if (comma == std::string_view::npos) break;
				start = comma + 1;
			}
			i++;
		} else if (arg == "--stage" && has_value) {
			o.stage = ParseStage(value);
			if (!o.stage) return false;
			i++;
		} else if ((arg == "--repeat" || arg == "--jobs" || arg == "--limit" ||
		            arg == "--max-lds") &&
		           has_value && ParseU32(value, number)) {
			(arg == "--repeat" ? o.repeat
			 : arg == "--jobs" ? o.jobs
			 : arg == "--limit" ? o.limit
			                    : o.max_lds_bytes) = number;
			i++;
		} else if (arg == "--bindless-class" && (value == "uint" || value == "sint")) {
			o.bindless_class = value == "uint" ? Prospero::TextureNumericClass::Uint
			                                   : Prospero::TextureNumericClass::Sint;
			i++;
		} else if (arg == "--lds-storage" && value == "live") {
			o.lds_storage = 2;
			i++;
		} else if ((arg == "--bindless" || arg == "--hw-bounds" || arg == "--lds-storage") &&
		           has_value && (value == "0" || value == "1")) {
			const int flag = value == "1" ? 1 : 0;
			if (arg == "--bindless")
				o.bindless = flag != 0;
			else if (arg == "--hw-bounds")
				o.hw_bounds = flag;
			else
				o.lds_storage = flag;
			i++;
		} else {
			std::fprintf(stderr, "shader-bench: bad argument %s\n", argv[i]);
			return false;
		}
	}
	o.repeat = std::max(o.repeat, 1u);
	o.jobs   = std::max(o.jobs, 1u);
	return true;
}

bool Matches(const Options& o, const PermutationRecord& r) {
	if (o.stage && *o.stage != r.stage) return false;
	if (o.hashes.empty()) return true;
	const auto hex = fmt::format("{:016x}", r.hash);
	return std::ranges::any_of(o.hashes, [&](const std::string& prefix) {
		return hex.starts_with(prefix);
	});
}

struct SpirvStats {
	uint32_t instructions = 0;
	uint32_t blocks       = 0;
	uint32_t bitcasts     = 0;
	uint32_t phis         = 0;
	uint32_t selects      = 0;
	uint32_t functions    = 0;
};

SpirvStats CountSpirv(const std::vector<uint32_t>& words) {
	SpirvStats s;
	for (size_t i = 5; i < words.size();) {
		const uint32_t op = words[i] & 0xffffu, length = words[i] >> 16u;
		if (length == 0) break;
		s.instructions++;
		s.blocks += op == 248 ? 1u : 0u;    // OpLabel
		s.bitcasts += op == 124 ? 1u : 0u;  // OpBitcast
		s.phis += op == 245 ? 1u : 0u;      // OpPhi
		s.selects += op == 169 ? 1u : 0u;   // OpSelect
		s.functions += op == 54 ? 1u : 0u;  // OpFunction
		i += length;
	}
	return s;
}

struct Result {
	size_t                record = 0;
	bool                  done   = false;
	bool                  skipped = false;
	bool                  lds_storage = false;
	uint32_t              wave_size = 0;
	double                translate_ms = 0;
	double                emit_ms      = 0;
	std::vector<uint32_t> spirv;
	uint64_t              spirv_hash = 0;
	uint32_t              repeat_diffs = 0;
	uint32_t              ir_blocks    = 0;
	uint32_t              ir_phis      = 0;
	uint32_t              ir_selects   = 0;
	// The shader-data block: push start (NoStart: storage), DWORDs, stride and format words.
	uint32_t              data_start = 0, data_dwords = 0, data_strides = 0, data_formats = 0;
	// Per dynamic-stride (s) / generic-format (f) buffer: R read, W stored, A atomic, F formatted, U raw.
	std::string           slot_flags;
	SpirvStats            stats;
	std::string           validation;
	double                driver_s      = -1;
	int                   driver_result = 0;
	uint32_t              forced_images = 0; // --bindless-class
};

// Mirrors ReplayPrecompiled's options_for.
void Translate(const Options& o, PermutationRecord record, uint32_t max_lds_bytes, Result& out) {
	ShaderParams params;
	params.code            = record.code;
	params.back_code       = record.back_code;
	params.hash            = record.hash;
	params.user_data_count = record.user_data_count;
	ShaderRecompiler::CompileOptions options;
	options.stage           = record.stage;
	options.shader_hash     = record.hash;
	options.user_data       = std::span(params.user_data).first(params.user_data_count);
	options.back_code       = params.back_code;
	options.user_data_base  = record.user_data_base;
	options.wave_size       = record.wave_size;
	options.dump_ir         = false;
	options.early_dump      = false;
	options.dump_label      = "ShaderBench";
	options.bindless_images = o.bindless;
	std::visit(
	    [&](auto& info) {
		    using Info = std::decay_t<decltype(info)>;
		    if constexpr (std::is_same_v<Info, ShaderVertexInputInfo>) {
			    options.input_info.vertex = &info;
		    } else if constexpr (std::is_same_v<Info, ShaderPixelInputInfo>) {
			    options.input_info.pixel = &info;
		    } else {
			    if (o.lds_storage == 2) {
				    info.lds_storage = info.lds_size_dwords * 4u > max_lds_bytes;
			    } else if (o.lds_storage >= 0) {
				    info.lds_storage = o.lds_storage != 0;
			    }
			    out.lds_storage = info.lds_storage;
			    options.input_info.compute = &info;
		    }
	    },
	    record.info);

	for (uint32_t pass = 0; pass < o.repeat; pass++) {
		const auto t0         = std::chrono::steady_clock::now();
		auto       translated = ShaderRecompiler::TranslateProgram(params.code, options);
		const auto t1         = std::chrono::steady_clock::now();
		if (translated.skip_dispatch || !translated.program.resource_tracking_complete ||
		    translated.program.info.buffers.size() != record.specialization.buffers.size() ||
		    translated.program.info.images.size() > record.specialization.images.size()) {
			out.skipped = true;
			out.done    = true;
			return;
		}
		auto specialization = record.specialization;
		if (o.bindless_class != Prospero::TextureNumericClass::Float) {
			// As materialization: integer arrays for sampled 2D and 2D-array images only.
			using ShaderRecompiler::Decoder::ImageDimension;
			const auto& base = translated.program.info.images;
			uint32_t    forced = 0;
			for (uint32_t i = 0; i < specialization.images.size(); i++) {
				auto&       image = specialization.images[i];
				const auto* info  = i < base.size() ? &base[i] : nullptr;
				if (image.bindless && !image.cube &&
				    (image.dimension == ImageDimension::Dim2D ||
				     image.dimension == ImageDimension::Dim2DArray) &&
				    (info == nullptr ||
				     (!info->depth_compare && !info->atomic &&
				      info->resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled))) {
					image.numeric_class = o.bindless_class;
					forced++;
				}
			}
			out.forced_images = forced;
		}
		auto       compiled = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                                       specialization,
		                                                       record.push_data_start_dword);
		const auto t2       = std::chrono::steady_clock::now();
		const auto hash     = XXH3_64bits(compiled.spirv.data(), compiled.spirv.size() * 4u);
		if (pass == 0) {
			out.translate_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
			out.emit_ms      = std::chrono::duration<double, std::milli>(t2 - t1).count();
			out.wave_size    = compiled.program.wave_size;
			out.ir_blocks    = static_cast<uint32_t>(compiled.program.blocks.size());
			out.data_start   = compiled.program.bindings.push_data_start_dword;
			out.data_dwords  = compiled.program.bindings.ShaderDataDwords();
			out.data_strides = static_cast<uint32_t>(compiled.program.bindings.buffer_strides.size());
			out.data_formats = static_cast<uint32_t>(compiled.program.bindings.buffer_formats.size());
			for (uint32_t i = 0; i < compiled.program.info.buffers.size(); i++) {
				const auto& buf = compiled.program.info.buffers[i];
				const bool dyn = (buf.packed_stride & ShaderRecompiler::IR::DynamicBufferStride) != 0u;
				const bool gen = buf.descriptor_format == ShaderRecompiler::IR::GenericBufferFormat;
				if (dyn || gen) {
					out.slot_flags += fmt::format(" b{}{}{}:{}{}{}{}{}", i, dyn ? "s" : "", gen ? "f" : "",
					                             buf.read ? "R" : "", buf.stored ? "W" : "", buf.atomic ? "A" : "",
					                             buf.formatted ? "F" : "", buf.unformatted ? "U" : "");
				}
			}
			for (const auto* block: compiled.program.blocks) {
				for (const auto& inst: *block) {
					const auto op = inst.GetOpcode();
					out.ir_phis += op == ShaderRecompiler::IR::ValueOpcode::Phi ? 1u : 0u;
					out.ir_selects += op == ShaderRecompiler::IR::ValueOpcode::SelectU32 ||
					                          op == ShaderRecompiler::IR::ValueOpcode::SelectF32 ||
					                          op == ShaderRecompiler::IR::ValueOpcode::SelectU1
					                      ? 1u
					                      : 0u;
				}
			}
			out.spirv_hash   = hash;
			out.spirv        = std::move(compiled.spirv);
		} else if (hash != out.spirv_hash) {
			out.repeat_diffs++;
		}
	}
	out.stats = CountSpirv(out.spirv);
	if (o.validate) {
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
		std::string          messages;
		tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&,
		                             const char* message) {
			if (messages.empty()) messages = message;
		});
		out.validation = tools.Validate(out.spirv) ? "ok" : "FAIL: " + messages;
	}
	out.done = true;
}

// ---- Driver timing (spvtime.cpp) ----

struct Device {
	VkInstance                 instance = VK_NULL_HANDLE;
	VkPhysicalDevice           physical = VK_NULL_HANDLE;
	VkDevice                   device   = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties properties {};
	uint32_t                   min_subgroup = 0;
	uint32_t                   max_subgroup = 0;
	bool                       subgroup_control = false;
	bool                       hw_bounds        = false;
	bool                       native_float16   = false;
};

#define BENCH_VK(x)                                                                         \
	do {                                                                                    \
		VkResult r_ = (x);                                                                  \
		if (r_ != VK_SUCCESS) {                                                             \
			std::fprintf(stderr, "shader-bench: %s failed: %d\n", #x, static_cast<int>(r_)); \
			return false;                                                                   \
		}                                                                                   \
	} while (0)

PFN_vkGetInstanceProcAddr LoadVulkan() {
#if defined(_WIN32)
	HMODULE library = LoadLibraryA("vulkan-1.dll");
	return library == nullptr ? nullptr
	                          : reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	                                GetProcAddress(library, "vkGetInstanceProcAddr"));
#else
	void* library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	return library == nullptr ? nullptr
	                          : reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	                                dlsym(library, "vkGetInstanceProcAddr"));
#endif
}

bool CreateDevice(Device& out) {
	auto gipa = LoadVulkan();
	if (gipa == nullptr) {
		std::fprintf(stderr, "shader-bench: cannot load the Vulkan loader\n");
		return false;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	d.init(gipa);
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo ici {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	BENCH_VK(d.vkCreateInstance(&ici, nullptr, &out.instance));
	d.init(vk::Instance(out.instance));
	uint32_t count = 0;
	d.vkEnumeratePhysicalDevices(out.instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	d.vkEnumeratePhysicalDevices(out.instance, &count, devices.data());
	if (devices.empty()) return false;
	out.physical = devices[0];
	for (auto p: devices) {
		VkPhysicalDeviceProperties pr;
		d.vkGetPhysicalDeviceProperties(p, &pr);
		if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) out.physical = p;
	}

	VkPhysicalDeviceFloatControlsProperties float_controls {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
	VkPhysicalDeviceRobustness2PropertiesEXT robust_props {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT};
	robust_props.pNext = &float_controls;
	VkPhysicalDeviceSubgroupSizeControlProperties subgroup_props {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
	subgroup_props.pNext = &robust_props;
	VkPhysicalDeviceProperties2 props2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
	props2.pNext = &subgroup_props;
	d.vkGetPhysicalDeviceProperties2(out.physical, &props2);
	out.properties   = props2.properties;
	out.min_subgroup = subgroup_props.minSubgroupSize;
	out.max_subgroup = subgroup_props.maxSubgroupSize;

	VkPhysicalDeviceRobustness2FeaturesEXT robust {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
	VkPhysicalDeviceVulkan13Features f13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	f13.pNext = &robust;
	VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	f12.pNext = &f13;
	VkPhysicalDeviceVulkan11Features f11 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
	f11.pNext = &f12;
	VkPhysicalDeviceFeatures2 f2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	f2.pNext = &f11;
	d.vkGetPhysicalDeviceFeatures2(out.physical, &f2);
	out.subgroup_control = f13.subgroupSizeControl == VK_TRUE;
	out.hw_bounds = robust.robustBufferAccess2 == VK_TRUE && robust.nullDescriptor == VK_TRUE &&
	                robust_props.robustStorageBufferAccessSizeAlignment <= sizeof(uint32_t);
	// As vulkanWindow.cpp decides it.
	out.native_float16 = f12.shaderFloat16 == VK_TRUE &&
	                     float_controls.shaderDenormPreserveFloat16 == VK_TRUE &&
	                     float_controls.shaderSignedZeroInfNanPreserveFloat16 == VK_TRUE;

	// Every supported feature on, as spvtime does; robustness2 needs its extension.
	const char*             extensions[] = {VK_EXT_ROBUSTNESS_2_EXTENSION_NAME};
	float                   priority     = 1.0f;
	VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueCount       = 1;
	qci.pQueuePriorities = &priority;
	VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.pNext                   = &f2;
	dci.queueCreateInfoCount    = 1;
	dci.pQueueCreateInfos       = &qci;
	dci.enabledExtensionCount   = 1;
	dci.ppEnabledExtensionNames = extensions;
	if (d.vkCreateDevice(out.physical, &dci, nullptr, &out.device) != VK_SUCCESS) {
		f13.pNext                 = nullptr;
		dci.enabledExtensionCount = 0;
		BENCH_VK(d.vkCreateDevice(out.physical, &dci, nullptr, &out.device));
	}
	d.init(vk::Device(out.device));
	return true;
}

struct Binding {
	VkDescriptorType type;
	uint32_t         count;
};

std::vector<uint32_t> WithCacheBuster(std::vector<uint32_t> words) {
	static std::mt19937_64 random {std::random_device {}()};
	const auto             text = fmt::format("shader-bench-{:016x}", random());
	std::vector<uint32_t>  str((text.size() + 4) / 4, 0);
	std::memcpy(str.data(), text.data(), text.size());
	size_t i = 5;
	while (i < words.size()) {
		const uint32_t op = words[i] & 0xffffu, length = words[i] >> 16u;
		if (op == 71 || op == 72 || op == 332 || op == 73 || op == 74 || op == 19 || op == 21 ||
		    op == 22 || length == 0)
			break;
		i += length;
	}
	std::vector<uint32_t> instruction {(static_cast<uint32_t>(1 + str.size()) << 16u) | 330u};
	instruction.insert(instruction.end(), str.begin(), str.end());
	words.insert(words.begin() + static_cast<std::ptrdiff_t>(i), instruction.begin(),
	             instruction.end());
	return words;
}

// Descriptor sets from the module's own decorations (spvtime.cpp's reflection).
std::map<uint32_t, std::map<uint32_t, Binding>> ReflectSets(const std::vector<uint32_t>& words) {
	std::map<uint32_t, uint32_t> dset, dbind, constants;
	std::set<uint32_t>           bufferblock;
	struct Ty {
		uint32_t              op;
		std::vector<uint32_t> a;
	};
	std::map<uint32_t, Ty>                                          types;
	std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> vars;
	for (size_t i = 5; i < words.size();) {
		const uint32_t  op = words[i] & 0xffffu, len = words[i] >> 16u;
		const uint32_t* o  = &words[i + 1];
		if (len == 0) break;
		if (op == 71) {
			if (o[1] == 34) dset[o[0]] = o[2];
			if (o[1] == 33) dbind[o[0]] = o[2];
			if (o[1] == 3) bufferblock.insert(o[0]);
		} else if (op == 43 && len >= 4) {
			constants[o[1]] = o[2];
		} else if ((op >= 19 && op <= 39) || op == 5341) {
			types[o[0]] = Ty {op, std::vector<uint32_t>(o + 1, o + len - 1)};
		} else if (op == 59) {
			vars.push_back({o[1], {o[0], o[2]}});
		}
		i += len;
	}
	std::map<uint32_t, std::map<uint32_t, Binding>> sets;
	for (auto& v: vars) {
		const uint32_t id = v.first;
		if (!dset.contains(id) || !dbind.contains(id)) continue;
		const uint32_t sc  = v.second.second;
		uint32_t       cnt = 1;
		uint32_t       t   = types[v.second.first].a[1];
		while (types[t].op == 28 || types[t].op == 29) {
			cnt = types[t].op == 28 ? cnt * constants[types[t].a[1]] : 4096;
			t   = types[t].a[0];
		}
		VkDescriptorType dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		const Ty&        ty = types[t];
		if (sc == 12) {
			dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		} else if (sc == 2) {
			dt = bufferblock.contains(t) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
			                             : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		} else if (ty.op == 25) {
			const bool buf = ty.a[1] == 5;
			const bool st  = ty.a[5] == 2;
			dt = buf ? (st ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
			               : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
			         : (st ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
		} else if (ty.op == 26) {
			dt = VK_DESCRIPTOR_TYPE_SAMPLER;
		} else if (ty.op == 27) {
			dt = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		} else if (ty.op == 5341) {
			dt = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		}
		sets[dset[id]][dbind[id]] = Binding {dt, cnt};
	}
	return sets;
}

bool TimeCompute(const Device& dev, const std::vector<uint32_t>& spirv, uint32_t wave_size,
                 Result& out) {
	auto&      d     = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto words = WithCacheBuster(spirv);
	auto       sets  = ReflectSets(words);
	uint32_t   max_set = 0;
	for (auto& s: sets) max_set = std::max(max_set, s.first);
	std::vector<VkDescriptorSetLayout> layouts;
	for (uint32_t s = 0; s <= max_set; s++) {
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		std::vector<VkDescriptorBindingFlags>     flags;
		for (auto& b: sets[s]) {
			VkDescriptorSetLayoutBinding lb {};
			lb.binding         = b.first;
			lb.descriptorType  = b.second.type;
			lb.descriptorCount = b.second.count;
			lb.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
			bindings.push_back(lb);
			flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
		}
		VkDescriptorSetLayoutBindingFlagsCreateInfo bfi {
		    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
		bfi.bindingCount  = static_cast<uint32_t>(flags.size());
		bfi.pBindingFlags = flags.data();
		VkDescriptorSetLayoutCreateInfo lci {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		lci.pNext        = &bfi;
		lci.bindingCount = static_cast<uint32_t>(bindings.size());
		lci.pBindings    = bindings.data();
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		BENCH_VK(d.vkCreateDescriptorSetLayout(dev.device, &lci, nullptr, &layout));
		layouts.push_back(layout);
	}
	VkPushConstantRange pcr {VK_SHADER_STAGE_COMPUTE_BIT, 0,
	                         ShaderRecompiler::IR::NativePushConstantSize};
	VkPipelineLayoutCreateInfo plci {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	plci.setLayoutCount         = static_cast<uint32_t>(layouts.size());
	plci.pSetLayouts            = layouts.data();
	plci.pushConstantRangeCount = 1;
	plci.pPushConstantRanges    = &pcr;
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	BENCH_VK(d.vkCreatePipelineLayout(dev.device, &plci, nullptr, &pipeline_layout));

	VkShaderModuleCreateInfo smci {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	smci.codeSize = words.size() * 4u;
	smci.pCode    = words.data();
	VkShaderModule module = VK_NULL_HANDLE;
	BENCH_VK(d.vkCreateShaderModule(dev.device, &smci, nullptr, &module));

	// As PrepareComputePipeline: the program's wave size is required when the device can.
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss {
	    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
	rss.requiredSubgroupSize = wave_size;
	VkComputePipelineCreateInfo cpi {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	cpi.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	cpi.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	cpi.stage.module = module;
	cpi.stage.pName  = "main";
	if (dev.subgroup_control && wave_size >= dev.min_subgroup && wave_size <= dev.max_subgroup) {
		cpi.stage.pNext = &rss;
	}
	cpi.layout             = pipeline_layout;
	cpi.basePipelineIndex  = -1;
	VkPipeline pipeline    = VK_NULL_HANDLE;
	const auto t0          = std::chrono::steady_clock::now();
	out.driver_result      = static_cast<int>(
        d.vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));
	out.driver_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	if (pipeline != VK_NULL_HANDLE) d.vkDestroyPipeline(dev.device, pipeline, nullptr);
	d.vkDestroyShaderModule(dev.device, module, nullptr);
	d.vkDestroyPipelineLayout(dev.device, pipeline_layout, nullptr);
	for (auto layout: layouts) d.vkDestroyDescriptorSetLayout(dev.device, layout, nullptr);
	return true;
}

// Format 7 journals (before compute float_mode/lds_storage and buffer indirect fields were
// recorded): their compute records only, read without the codec, missing fields at their
// defaults (float_mode 0xc0, no indirect buffers; lds_storage per --lds-storage, default the live
// rule). Checksums are not verified. Lets the bench use journals recorded by older builds.
std::vector<PermutationRecord> ReadLegacyComputeRecords(const std::filesystem::path& path,
                                                        std::string& key) {
	std::vector<PermutationRecord> records;
	std::ifstream                  in(path, std::ios::binary);
	std::vector<uint8_t> blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	size_t     offset = 0;
	bool       ok     = true;
	const auto word   = [&]() -> uint32_t {
		if (offset + 4u > blob.size()) {
			ok = false;
			return 0;
		}
		uint32_t value = 0;
		std::memcpy(&value, blob.data() + offset, 4);
		offset += 4;
		return value;
	};
	if (blob.size() < 16 || std::memcmp(blob.data(), "KYTYSHDR", 8) != 0) return records;
	offset = 8;
	if (word() != 7u) return records;
	const auto key_length = word();
	if (key_length > blob.size() - offset) return records;
	key.assign(reinterpret_cast<const char*>(blob.data() + offset), key_length);
	offset += key_length;
	while (ok && offset + 12u <= blob.size()) {
		const auto size = word();
		offset += 8; // checksum
		const auto end = offset + size;
		if (size == 0 || end > blob.size()) break;
		PermutationRecord r;
		r.stage = static_cast<ShaderType>(word());
		r.hash  = word();
		r.hash |= static_cast<uint64_t>(word()) << 32u;
		r.push_data_start_dword = word();
		r.user_data_base        = word();
		r.user_data_count       = word();
		r.wave_size             = word();
		for (auto* code: {&r.code, &r.back_code}) {
			const auto count = word();
			if (count > (end - offset) / 4u) {
				ok = false;
				break;
			}
			code->resize(count);
			for (auto& w: *code) w = word();
		}
		if (ok && r.stage == ShaderType::Compute) {
			r.specialization.buffers.resize(word());
			for (auto& b: r.specialization.buffers) {
				b.packed_stride      = word();
				b.descriptor_format  = static_cast<Prospero::BufferFormat>(word());
				b.descriptor_swizzle = word();
				b.zero_stride_oob    = word() != 0;
			}
			r.specialization.images.resize(word());
			for (auto& i: r.specialization.images) {
				i.numeric_class              = static_cast<Prospero::TextureNumericClass>(word());
				i.dimension                  = static_cast<ShaderRecompiler::Decoder::ImageDimension>(word());
				i.mip_count                  = word();
				i.conversion_format          = static_cast<Prospero::BufferFormat>(word());
				i.shader_swizzle             = word();
				i.indirect_root              = word();
				i.indirect_mapping_offset    = word();
				i.indirect_search_iterations = word();
				i.cube                       = word() != 0;
				i.fmask                      = word() != 0;
				i.bindless                   = word() != 0;
			}
			ShaderComputeInputInfo info;
			for (auto& t: info.threads_num) t = word();
			info.lds_size_dwords     = word();
			info.scratch_size_dwords = word();
			info.host_subgroup_size  = word();
			info.wave_size           = word();
			for (auto& g: info.group_id) g = word() != 0;
			info.dispatch_thread_dimensions = word() != 0;
			info.thread_ids_num             = static_cast<int>(word());
			info.workgroup_register         = static_cast<int>(word());
			info.tg_size_en                 = word() != 0;
			r.info                          = info;
			if (ok && offset == end) records.push_back(std::move(r));
		}
		offset = end;
	}
	return records;
}

std::string Describe(const PermutationRecord& r) {
	std::string text;
	if (const auto* c = std::get_if<ShaderComputeInputInfo>(&r.info)) {
		text = fmt::format("lds={} host_sg={} threads={}x{}x{}", c->lds_size_dwords * 4u,
		                   c->host_subgroup_size, c->threads_num[0], c->threads_num[1],
		                   c->threads_num[2]);
	}
	return fmt::format("{} bufs={} imgs={} push={}", text, r.specialization.buffers.size(),
	                   r.specialization.images.size(), r.push_data_start_dword);
}

} // namespace

int Run(int argc, char* argv[]) {
	Options o;
	if (!ParseArgs(argc, argv, o)) {
		PrintUsage();
		return 1;
	}

	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction = Config::LogDirection::Silent;
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();

	std::string key;
	auto        records = ShaderPrecompile::ReadJournal(o.journal, &key);
	if (records.empty()) {
		records = ReadLegacyComputeRecords(o.journal, key);
		if (!records.empty()) {
			std::printf("journal: format 7, compute records only\n");
			if (o.lds_storage < 0) o.lds_storage = 2;
		}
	}
	if (records.empty()) {
		std::fprintf(stderr, "shader-bench: no readable records in %s\n",
		             o.journal.string().c_str());
		return 1;
	}
	std::printf("journal: %zu records, key %s\n", records.size(), key.c_str());

	Device                device;
	const bool            have_device = CreateDevice(device);
	if (o.time && !have_device) return 1;
	const uint32_t max_lds = o.max_lds_bytes != 0 ? o.max_lds_bytes
	                         : have_device        ? device.properties.limits.maxComputeSharedMemorySize
	                                              : 48u * 1024u;
	const bool hw_bounds = o.hw_bounds >= 0 ? o.hw_bounds != 0 : !have_device || device.hw_bounds;
	ShaderRecompiler::SetHardwareStorageBufferBounds(hw_bounds);
	ShaderRecompiler::SetNativeFloat16(have_device && device.native_float16);
	ShaderRecompiler::SetTableMode(std::getenv("KYTY_TABLE_MODE") != nullptr &&
	                                 std::string_view(std::getenv("KYTY_TABLE_MODE")) == "1");
	std::printf("device: %s, max LDS %u, hw bounds %s, bindless %s, native fp16 %s\n",
	            have_device ? device.properties.deviceName : "(none)", max_lds,
	            hw_bounds ? "on" : "off", o.bindless ? "on" : "off",
	            ShaderRecompiler::NativeFloat16() ? "on" : "off");

	std::vector<size_t> selected;
	for (size_t i = 0; i < records.size() && selected.size() < o.limit; i++) {
		if (Matches(o, records[i])) selected.push_back(i);
	}
	std::printf("selected: %zu records\n", selected.size());
	if (!o.out_dir.empty()) std::filesystem::create_directories(o.out_dir);

	std::vector<Result> results(selected.size());
	std::atomic_size_t  next {0};
	const auto          work = [&] {
        for (size_t n = next.fetch_add(1); n < selected.size(); n = next.fetch_add(1)) {
            results[n].record = selected[n];
            Translate(o, records[selected[n]], max_lds, results[n]);
        }
	};
	{
		std::vector<std::jthread> helpers;
		for (uint32_t i = 1; i < o.jobs; i++) helpers.emplace_back(work);
		work();
	}

	std::printf("%-6s %-16s %-2s %-3s %-3s %9s %8s %8s %7s %7s %7s %7s %-16s %9s %s\n", "rec",
	            "hash", "st", "wv", "lds", "transl_ms", "emit_ms", "insts", "blocks", "bitcast",
	            "phi", "select", "spirv_xxh3", "driver_s", "info");
	double total_translate = 0, total_driver = 0;
	for (auto& r: results) {
		const auto& record = records[r.record];
		if (r.skipped) {
			std::printf("%-6zu %016llx %-2s skipped by replay checks\n", r.record,
			            static_cast<unsigned long long>(record.hash), StageName(record.stage));
			continue;
		}
		if (o.time && record.stage == ShaderType::Compute) {
			(void)TimeCompute(device, r.spirv, r.wave_size, r);
			total_driver += r.driver_s;
		}
		total_translate += r.translate_ms + r.emit_ms;
		if (!o.out_dir.empty()) {
			const auto    path = o.out_dir / fmt::format("spv_{:016x}_r{}.spv", record.hash, r.record);
			std::ofstream file(path, std::ios::binary);
			file.write(reinterpret_cast<const char*>(r.spirv.data()),
			           static_cast<std::streamsize>(r.spirv.size() * 4u));
		}
		std::string extra = Describe(record) + fmt::format(" ir_blocks={} ir_phis={} ir_selects={}",
		                                                   r.ir_blocks, r.ir_phis, r.ir_selects);
		extra += fmt::format(" sdata={}+{} strides={} formats={}",
		                     r.data_start == ShaderRecompiler::IR::PushData::NoStart
		                         ? std::string("storage")
		                         : std::to_string(r.data_start),
		                     r.data_dwords, r.data_strides, r.data_formats);
		extra += r.slot_flags;
		if (r.forced_images != 0) extra += fmt::format(" bindless_int={}", r.forced_images);
		if (r.repeat_diffs != 0) extra += fmt::format(" NONDETERMINISTIC({}/{})", r.repeat_diffs, o.repeat - 1);
		if (!r.validation.empty()) extra += " val=" + r.validation;
		if (r.driver_result != 0) extra += fmt::format(" vkresult={}", r.driver_result);
		const auto driver = r.driver_s >= 0 ? fmt::format("{:9.2f}", r.driver_s) : std::string(9, '-');
		std::printf("%-6zu %016llx %-2s %-3u %-3s %9.1f %8.1f %8u %7u %7u %7u %7u %016llx %s %s\n",
		            r.record, static_cast<unsigned long long>(record.hash), StageName(record.stage),
		            r.wave_size, r.lds_storage ? "buf" : "wg", r.translate_ms, r.emit_ms,
		            r.stats.instructions, r.stats.blocks, r.stats.bitcasts, r.stats.phis,
		            r.stats.selects, static_cast<unsigned long long>(r.spirv_hash), driver.c_str(),
		            extra.c_str());
		std::fflush(stdout);
	}
	std::printf("total: translate+emit %.1f ms, driver %.2f s\n", total_translate, total_driver);
	if (have_device) {
		auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
		d.vkDestroyDevice(device.device, nullptr);
		d.vkDestroyInstance(device.instance, nullptr);
	}
	return 0;
}

} // namespace Libs::Graphics::ShaderBench
