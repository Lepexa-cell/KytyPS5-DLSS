#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/switches.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/dispatchThreadGroups.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {
void DebugRegisterDumpAddress(uint64_t address);

// A BVH traversal reads its nodes, instance records and bottom-level BVHs by address. A page
// missing from the BDA page table reads zero until the fault is processed, a frame later; Astro
// Bot relights each probe once, so the rays of those first dispatches all missed and the probes
// stayed black. The SDK's BVHs start with "PSR_BVHL" and give their size at +16: this finds the
// ones a compute shader's SRT points to and has the buffer cache hold them (and the BVHs their
// instance records name) before the dispatch.
static void PrefetchBvhs(BufferCache& buffers,
                         const ShaderRecompiler::IR::ResourceSnapshot& resources) {
	constexpr uint32_t MagicLow    = 0x5f525350u; // "PSR_"
	constexpr uint32_t MagicHigh   = 0x4c485642u; // "BVHL"
	constexpr uint64_t MaxBvhBytes = 1ull << 30u;
	if (resources.user_data.size() < 2u) {
		return;
	}
	const auto read = [](uint64_t address, uint32_t* words, uint32_t count) {
		return address >= 0x100000u && (address & 3u) == 0u &&
		       LibKernel::Memory::TryReadBacking(address, words, count * sizeof(uint32_t));
	};
	// Returns the BVH's size when `header` starts one and all of it is mapped.
	const auto bvh_size = [&](uint64_t header, std::array<uint32_t, 24>& head) -> uint64_t {
		if (!read(header, head.data(), 24) || head[0] != MagicLow || head[1] != MagicHigh) {
			return 0;
		}
		const uint64_t size = head[4] | (uint64_t {head[5]} << 32u);
		uint32_t       last = 0;
		if (size < 256u || size > MaxBvhBytes || !read((header + size - 4u) & ~uint64_t {3}, &last, 1)) {
			return 0;
		}
		return size;
	};
	const uint64_t srt = resources.user_data[0] | (uint64_t {resources.user_data[1]} << 32u);
	std::array<uint32_t, 24> table {};
	if (!read(srt, table.data(), 24)) {
		return;
	}
	static std::unordered_map<uint64_t, uint64_t> held; // Bottom-level header -> size.
	for (uint32_t word = 0; word + 1u < table.size(); word += 2u) {
		const uint64_t header = table[word] | (uint64_t {table[word + 1u]} << 32u);
		std::array<uint32_t, 24> head {};
		const auto size = bvh_size(header, head);
		if (size == 0) {
			continue;
		}
		(void)buffers.FindBuffer(header, size);
		// Instance records (128 bytes, the bottom-level pointer / 8 in dwords 12 and 13) follow
		// the nodes.
		const uint64_t nodes     = (uint64_t {head[22]} + 1u) << 6u;
		const uint64_t instances = head[12] | (uint64_t {head[13]} << 32u);
		if (instances < 0x100u || instances >= size || nodes > size) {
			continue;
		}
		const auto count = std::min<uint64_t>((size - instances) >> 7u, 1u << 16u);
		for (uint64_t index = 0; index < count; index++) {
			std::array<uint32_t, 2> pointer {};
			if (!read(header + instances + (index << 7u) + 48u, pointer.data(), 2)) {
				break;
			}
			const uint64_t bottom =
			    (pointer[0] | (uint64_t {pointer[1] & 0x003fffffu} << 32u)) << 3u;
			if (bottom == 0 || held.contains(bottom)) {
				continue;
			}
			std::array<uint32_t, 24> bottom_head {};
			const auto bottom_size = bvh_size(bottom, bottom_head);
			held.emplace(bottom, bottom_size);
			if (bottom_size != 0) {
				(void)buffers.FindBuffer(bottom, bottom_size);
			}
		}
	}
}
// Asked per dispatch: reading the environment each time cost about 1% of the GPU thread.
static bool BvhCheckEnabled() {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_BVH_CHECK") != nullptr;
	return enabled;
}

// KYTY_DEBUG_BVH_CHECK=<hash>: walks, on the CPU, the BVH that compute shader traverses (its
// header pointer is at SRT+40) and reports its shape: a cycle or garbage there makes the
// shader's traversal loop endless.
static void DebugCheckBvh(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                          const ShaderRecompiler::IR::ResourceSnapshot& resources,
                          const uint32_t* groups) {
	static const uint64_t hash = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_BVH_CHECK");
		return text != nullptr ? std::strtoull(text, nullptr, 16) : 0ull;
	}();
	if (hash == 0 || program.shader_hash != hash) {
		return;
	}
	static std::atomic<uint32_t> calls {0};
	const auto call = calls.fetch_add(1, std::memory_order_relaxed);
	if (call >= 6u && call % 200u != 0u) {
		return;
	}
	const auto read = [](uint64_t address, uint32_t* words, uint32_t count) {
		return LibKernel::Memory::TryReadBacking(address, words, count * sizeof(uint32_t));
	};
	std::printf("bvh-check: call=%u groups=%ux%ux%u user_data_base=%u user_data=%zu\n", call,
	            groups[0], groups[1], groups[2], program.user_data_base,
	            resources.user_data.size());
	for (size_t i = 0; i < resources.images.size(); i++) {
		const auto&    image   = resources.images[i];
		const uint64_t address = ((uint64_t {image.dwords[1] & 0xffu} << 32u) | image.dwords[0]) << 8u;
		DebugRegisterDumpAddress(address);
		if (call < 2u) {
			std::printf("bvh-check: image %zu address=0x%" PRIx64 " : %08x %08x %08x %08x %08x %08x %08x %08x\n",
			            i, address, image.dwords[0], image.dwords[1], image.dwords[2],
			            image.dwords[3], image.dwords[4], image.dwords[5], image.dwords[6],
			            image.dwords[7]);
		}
	}
	if (resources.user_data.size() < 2u) {
		std::fflush(stdout);
		return;
	}
	const uint64_t srt = resources.user_data[0] | (uint64_t {resources.user_data[1]} << 32u);
	std::array<uint32_t, 24> table {};
	if (!read(srt, table.data(), 24)) {
		std::printf("bvh-check: srt=0x%016" PRIx64 " unreadable\n", srt);
		std::fflush(stdout);
		return;
	}
	std::printf("bvh-check: srt=0x%016" PRIx64 " :", srt);
	for (const auto word: table) {
		std::printf(" %08x", word);
	}
	const uint64_t header = table[10] | (uint64_t {table[11]} << 32u);
	std::array<uint32_t, 32> head {};
	if (!read(header, head.data(), 32)) {
		std::printf("\nbvh-check: header=0x%016" PRIx64 " unreadable\n", header);
		std::fflush(stdout);
		return;
	}
	std::printf("\nbvh-check: header=0x%016" PRIx64 " :", header);
	for (const auto word: head) {
		std::printf(" %08x", word);
	}
	const uint64_t base      = header & ~uint64_t {0xff};
	const uint64_t instances = header + (head[12] | (uint64_t {head[13]} << 32u));
	const uint64_t last      = head[22] | (uint64_t {head[23] & 0x3ffu} << 32u);
	std::printf("\nbvh-check: instances=0x%016" PRIx64 " last_index=%" PRIu64 "\n", instances, last);
	std::fflush(stdout);

	std::vector<uint32_t>        stack {37u};
	std::unordered_map<uint32_t, uint32_t> seen;
	uint32_t boxes = 0, leaves = 0, revisits = 0, out_of_range = 0, unreadable = 0, zero = 0;
	uint32_t printed = 0, steps = 0, odd_kind = 0, dead_instances = 0;
	while (!stack.empty() && steps < 40000u) {
		const auto id = stack.back();
		stack.pop_back();
		steps++;
		if (seen[id]++ != 0u) {
			revisits++;
			continue;
		}
		const auto kind  = id & 7u;
		const auto index = id >> 3u;
		if (kind == 4u || kind == 5u) {
			std::array<uint32_t, 16> node {};
			if (index + (kind == 5u ? 1u : 0u) > last) {
				out_of_range++;
				continue;
			}
			if (!read(base + (uint64_t {index} << 6u), node.data(), 16)) {
				unreadable++;
				continue;
			}
			boxes++;
			if (std::ranges::all_of(node, [](uint32_t word) { return word == 0u; })) {
				zero++;
			}
			if (printed < 6u) {
				printed++;
				std::printf("bvh-check: box id=0x%x children=%08x %08x %08x %08x w4=%08x w5=%08x\n",
				            id, node[0], node[1], node[2], node[3], node[4], node[5]);
				std::array<float, 32> wide {};
				if (read(base + (uint64_t {index} << 6u), reinterpret_cast<uint32_t*>(wide.data()),
				         32)) {
					std::printf("bvh-check: box floats:");
					for (uint32_t w = 4; w < 32; w++) {
						std::printf(" %g", static_cast<double>(wide[w]));
					}
					std::printf("\n");
				}
			}
			for (uint32_t child = 0; child < 4u; child++) {
				if (node[child] != ~0u) {
					stack.push_back(node[child]);
				}
			}
		} else if (kind == 6u) {
			odd_kind++;
		} else {
			std::array<uint32_t, 16> record {};
			const auto address = kind == 7u ? header + 0x80u : instances + (uint64_t {index} << 7u);
			if (!read(address, record.data(), 16)) {
				unreadable++;
				continue;
			}
			leaves++;
			if (record[15] <= 0x00ffffffu) {
				dead_instances++;
			}
			if (leaves <= 4u) {
				std::printf("bvh-check: leaf id=0x%x record=0x%016" PRIx64 " :", id, address);
				for (const auto word: record) {
					std::printf(" %08x", word);
				}
				std::printf("\n");
			}
		}
	}
	std::printf("bvh-check: steps=%u boxes=%u leaves=%u dead_instances=%u revisits=%u "
	            "out_of_range=%u unreadable=%u zero_boxes=%u kind6=%u left=%zu\n",
	            steps, boxes, leaves, dead_instances, revisits, out_of_range, unreadable, zero,
	            odd_kind, stack.size());
	std::fflush(stdout);
}

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

void RenderExecutor::SyncHtileReads(const ShaderComputeInputInfo& input) {
	auto& cache = m_context.GetTextureCache();
	if (!cache.HasStaleHtile()) {
		return;
	}
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto  count     = std::min(program.info.buffers.size(), resources.buffers.size());
	// KYTY_NO_HTILE_SYNTH=1 (diagnostic, the title black tiles at resolution steps): reads never synthesize
	// HTile from depth. KYTY_DEBUG_HTILE_SYNTH=1 names the first 300 reads that may synthesize.
	static const bool no_synth  = KYTY_GETENV("KYTY_NO_HTILE_SYNTH") != nullptr;
	static const bool log_synth = KYTY_GETENV("KYTY_DEBUG_HTILE_SYNTH") != nullptr;
	for (size_t i = 0; i < count && !no_synth; i++) {
		if (!program.info.buffers[i].read) {
			continue;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		const auto size       = descriptor.GetSize();
		if (descriptor.Base48() != 0 && size != 0) {
			if (log_synth) [[unlikely]] {
				static std::atomic<uint32_t> logs {0};
				if (logs.fetch_add(1, std::memory_order_relaxed) < 300) {
					std::printf("htile-synth-read: cs %016" PRIx64 " buffer %zu base=0x%" PRIx64 " size=0x%" PRIx64
					            " written=%d\n",
					            program.shader_hash, i, descriptor.Base48(), static_cast<uint64_t>(size),
					            program.info.buffers[i].written ? 1 : 0);
					std::fflush(stdout);
				}
			}
			cache.SyncHtile(descriptor.Base48(), size);
		}
	}
	// Then the written ranges: what the dispatch writes over a stale surface's HTile is the guest's
	// own data from now on (a read later must not synthesize HTile over it).
	// Only ranges of a real buffer: a write through a window over a large part of memory (BDA-style
	// descriptors) says nothing about which bytes changed, and Yotei's cascade binning needs the
	// synthesis.
	constexpr uint64_t MaxWrittenRange = 16ull << 20u;
	for (size_t i = 0; i < count && cache.HasStaleHtile(); i++) {
		if (!program.info.buffers[i].written) {
			continue;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		const auto size       = descriptor.GetSize();
		if (descriptor.Base48() != 0 && size != 0 && size <= MaxWrittenRange) {
			cache.DropStaleHtile(descriptor.Base48(), size, "a compute dispatch");
		}
	}
}

void RenderExecutor::RecordHtileRewriteClear(const ShaderComputeInputInfo& input) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (program.info.has_bitwise_xor) {
		return;
	}
	auto&      cache = m_context.GetTextureCache();
	const auto count = std::min(program.info.buffers.size(), resources.buffers.size());
	uint64_t   meta  = 0;
	for (size_t i = 0; i < count; i++) {
		if (!program.info.buffers[i].written) {
			continue;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		if (!cache.IsMeta(descriptor.Base48())) {
			continue;
		}
		if (meta != 0 || !program.info.buffers[i].read) {
			return; // Several metadata surfaces, or a write-only clear handled by consumption.
		}
		meta = descriptor.Base48();
	}
	if (meta != 0) {
		(void)cache.ClearMeta(meta);
	}
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer& buffer, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode,
                                                bool consume) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	// KYTY_NO_COMPUTE_META_CLEAR=1 (diagnostic, the title black tiles): a dispatch writing a metadata
	// address always runs instead of being replaced by a metadata clear.
	static const bool no_meta_clear = KYTY_GETENV("KYTY_NO_COMPUTE_META_CLEAR") != nullptr;
	if (no_meta_clear) {
		return false;
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}
	// Only an HTile writer becomes a metadata clear: a dispatch writing exactly one buffer and no image,
	// that buffer exactly a known HTile range (the title clears cs 89bdd959103d4835, HTile init cs 6557023d /
	// 291b5b44). A dispatch writing other bytes over a metadata address reuses that memory for its own
	// data (transient pools): replacing it drops those writes. The title cs 98b52e5c writes five pool buffers
	// and in replays one of them lands on a depth target's HTile; replaced, it wrote no fighter box count
	// and cs b092a36d looped on the garbage there until the GPU hung; single-buffer pool
	// shaders (b80af84a, d793e734, cda58bd5, cloth 99c47b82 ...) lost their output the same way.
	// KYTY_META_CLEAR_LOOSE=1 replaces every dispatch that writes a metadata address, as before.
	static const bool loose = KYTY_GETENV("KYTY_META_CLEAR_LOOSE") != nullptr;
	uint32_t written_buffers = 0;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (program.info.buffers[i].written &&
		    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]).Base48() != 0) {
			written_buffers++;
		}
	}
	const bool writes_image =
	    std::ranges::any_of(program.info.images, [](const auto& image) { return image.written; });

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				// KYTY_DEBUG_HTILE_FORCE_PENDING=1 (test of KYTY_HTILE_LATE_CLEAR): HTile fills
				// are not consumed here, so each goes through the pending note and late take.
				static const bool force_pending =
				    KYTY_GETENV("KYTY_DEBUG_HTILE_FORCE_PENDING") != nullptr;
				if (force_pending && cache.IsHtileMeta(descriptor.Base48())) {
					return false;
				}
				if (!cache.IsMeta(descriptor.Base48())) {
					continue;
				}
				// A proven uniform fill of exactly this buffer: its HTile word says whether the
				// stencil is cleared too (SMEM, bits 9:8, is 0). The title cs 89bdd959103d4835 fills 0.
				ShaderBufferResource fill_descriptor {};
				uint32_t             fill_value = 0;
				uint64_t             fill_size  = 0;
				const bool           fill =
				    ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, fill_descriptor,
				                             fill_value, fill_size) &&
				    fill_descriptor.Base48() == descriptor.Base48();
				const bool stencil_cleared = fill && ((fill_value >> 8u) & 3u) == 0;
				const bool single = written_buffers == 1 && !writes_image;
				if (!loose && !(single && cache.IsExactHtileRange(descriptor.Base48(), descriptor.GetSize()))) {
					// A proven fill with the "cleared" HTile word (ZMASK 0) over an HTile base whose
					// registered range has another size (a depth target of another size is bound there
					// next, the title cs 89bdd959 once per run) clears it and writes its bytes as well, which
					// the PS5 does too; a pool zero fill over a retired HTile (cs 7e80065d) the same.
					const bool clears = single && fill && (fill_value & 0xfu) == 0;
					if (clears) {
						(void)cache.ClearMeta(descriptor.Base48(), stencil_cleared);
					}
					static const bool log_refused = KYTY_GETENV("KYTY_DEBUG_META_CONSUME") != nullptr;
					if (log_refused) [[unlikely]] {
						static std::mutex                             refused_mutex;
						static std::unordered_map<uint64_t, uint32_t> refused_counts;
						std::scoped_lock                              refused_lock(refused_mutex);
						const uint32_t n = ++refused_counts[program.shader_hash];
						if (n <= 4 || (n & (n - 1)) == 0) {
							std::printf("meta-clear-refused: cs %016" PRIx64 " #%u buffer %u base=0x%" PRIx64
							            " size=0x%" PRIx64 " written=%u image=%d fill=%d value=%08x: runs%s\n",
							            program.shader_hash, n, i, descriptor.Base48(),
							            static_cast<uint64_t>(descriptor.GetSize()), written_buffers,
							            writes_image ? 1 : 0, fill ? 1 : 0, fill_value,
							            clears ? ", clear recorded" : "");
							std::fflush(stdout);
						}
					}
					return false;
				}
				if (cache.ClearMeta(descriptor.Base48(), stencil_cleared)) {
					// KYTY_DEBUG_META_CONSUME=1: every shader whose dispatch became a metadata clear, the
					// first 16 per shader and then each power-of-two count (the game may reuse a retired
					// HTile's memory as an ordinary buffer: the replaced dispatch then never writes it).
					static const bool log_consume = KYTY_GETENV("KYTY_DEBUG_META_CONSUME") != nullptr;
					if (log_consume) [[unlikely]] {
						static std::mutex                             consume_mutex;
						static std::unordered_map<uint64_t, uint32_t> consume_counts;
						std::scoped_lock                              consume_lock(consume_mutex);
						const uint32_t n = ++consume_counts[program.shader_hash];
						if (n <= 16 || (n & (n - 1)) == 0) {
							std::printf("meta-consume: cs %016" PRIx64 " #%u buffer %u base=0x%" PRIx64
							            " size=0x%" PRIx64 " groups=%ux%ux%u fill=%d value=%08x consume=%d\n",
							            program.shader_hash, n, i, descriptor.Base48(),
							            static_cast<uint64_t>(descriptor.GetSize()), group_x, group_y, group_z,
							            fill ? 1 : 0, fill_value, consume ? 1 : 0);
							std::fflush(stdout);
						}
					}
					static const bool log_clear = KYTY_GETENV("KYTY_DEBUG_HTILE_SYNTH") != nullptr;
					if (log_clear) [[unlikely]] {
						static std::atomic<uint32_t> logs {0};
						if (logs.fetch_add(1, std::memory_order_relaxed) < 300) {
							std::printf("compute-meta-clear: cs %016" PRIx64 " buffer %u base=0x%" PRIx64
							            " size=0x%" PRIx64 " consume=%d fill=%d value=%08x stencil=%d\n",
							            program.shader_hash, i, descriptor.Base48(),
							            static_cast<uint64_t>(descriptor.GetSize()), consume ? 1 : 0,
							            fill ? 1 : 0, fill_value, stencil_cleared ? 1 : 0);
							std::fflush(stdout);
						}
					}
					return consume;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer || !fill.uniform) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

// A proven uniform fill that launches fewer threads than its buffer has records writes only the
// first `invocations` elements (thread i writes element i; the hardware launches no more). The title
// fills 0x88 bytes of a 0x400-byte V# (cs 7e80065d6f78525b) whose tail holds a SET_UCONFIG_REG_INDIRECT
// pair the CP reads next frame: marking the whole V# as GPU-written made that read fault and drain
// the GPU, ~8 ms a frame in the fight. Returns the written size, or 0 when the whole binding counts.
// KYTY_FILL_WRITE_EXACT=0 marks the whole binding as before.
static uint64_t PartialFillWriteSize(const ShaderComputeInputInfo& input, uint32_t group_x,
                                     uint32_t group_y, uint32_t group_z, uint32_t mode) {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_FILL_WRITE_EXACT");
		return text == nullptr || text[0] != '0';
	}();
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (!enabled || fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer || !fill.uniform ||
	    fill.words == 0 || fill.words > 4) {
		return 0;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor   = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Stride 0 (what the title actually binds: records=1024, size 0x400): every thread's element index
	// times the stride is 0, so all of them store the one element at the base; num_records is then
	// a byte bound, checked against offset 0.
	if (descriptor.Stride() == 0 && descriptor.Format() == formats[fill.words - 1] &&
	    !descriptor.SwizzleEnabled() && descriptor.IndexStride() == 0 && !descriptor.AddTid() &&
	    descriptor.Base48() != 0 && descriptor.NumRecords() >= element_size &&
	    descriptor.GetSize() > element_size) {
		return element_size;
	}
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return 0;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return 0;
	}
	if (input.dispatch_thread_dimensions &&
	    (input.dispatch_threads_num[0] != group_x || input.dispatch_threads_num[1] != 1 ||
	     input.dispatch_threads_num[2] != 1)) {
		return 0;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	if (invocations >= descriptor.NumRecords()) {
		return 0;
	}
	return invocations * element_size;
}

// DCC comp-to-single clear (GFX10+): the game sets every DCC key of a color surface to 0x10
// ("the block's color is stored in the image") and then writes the clear color into the first
// element of every 256-byte block, one thread per block. The hardware then reads the whole block
// as that color; the host keeps no DCC, so only one texel in 64 got the color. Ghost of Yotei
// clears its motion vectors to NaN this way ("no vector yet"), and the pass that fills in the
// camera motion only touches NaN pixels: without this, static geometry and the sky kept zero
// motion and TAA smeared buildings and characters behind every camera move.
bool RenderExecutor::TryConsumeComputeDccSingleClear(const ShaderComputeInputInfo& input,
                                                     CommandBuffer& command, uint32_t group_x,
                                                     uint32_t group_y, uint32_t group_z,
                                                     uint32_t mode) {
	constexpr uint32_t DccBlockBytes = 256;
	const auto&        resources     = *input.stage.resources;
	const auto&        fill          = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	const auto format = descriptor.Format();
	// One 32-bit element per block: the color of a 4-byte-per-texel surface.
	if (descriptor.Stride() != DccBlockBytes ||
	    (format != Prospero::BufferFormat::k32UInt && format != Prospero::BufferFormat::k32SInt &&
	     format != Prospero::BufferFormat::k32Float) ||
	    descriptor.DstSelX() != 4 || descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 ||
	    descriptor.AddTid() || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || input.dispatch_thread_dimensions || mode != 0x41u) {
		return false;
	}
	const uint64_t invocations = static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    !FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource)) {
		return false;
	}
	auto& cache = command.GetContext().GetTextureCache();
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, fill.value,
	                                sizeof(uint32_t))) {
		return false;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::printf("DCC single clear: shader=%016" PRIx64 " addr=0x%012" PRIx64 " size=0x%" PRIx64
		            " color=0x%08" PRIx32 "\n",
		            input.stage.program->shader_hash, descriptor.Base48(), size, fill.value);
		std::fflush(stdout);
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (!fill.uniform || mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return TryConsumeComputeDccSingleClear(input, command, group_x, group_y, group_z, mode);
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

// KYTY_DEBUG_DISPATCH_INPUTS=<shader hash>: prints that compute shader's inputs for its first 6
// dispatches and every 200th after: each buffer (base, size, whether the GPU wrote it, its first
// dwords as the CPU sees them) and the first flattened SRT constants, to find bad input data
// (e.g. a loop count).
extern std::atomic<uint64_t> g_debug_submission_seq;
extern std::atomic<uint32_t> g_debug_submission_queue;
extern uint64_t              g_debug_cp_indirect_args;
// The current dispatch's group counts (direct) or arguments address (indirect), for the print.
static uint32_t s_debug_groups[3] {};
static uint64_t s_debug_args = 0;

// KYTY_DEBUG_DISPATCH_REPEAT=1: how often a direct dispatch's materialized inputs (user data,
// descriptors, flattened SRT) equal those of the previous dispatch of the same shader, every 5 s:
// the ceiling of a cross-dispatch program/resource memo.
static void NoteDispatchRepeat(uint64_t shader_hash, const ShaderRecompiler::IR::ResourceSnapshot& snapshot) {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_DISPATCH_REPEAT") != nullptr;
	if (!enabled) [[likely]] {
		return;
	}
	const auto hash_of = [](const auto& values) {
		return values.empty() ? uint64_t {0}
		                      : XXH3_64bits(values.data(), values.size() * sizeof(values[0]));
	};
	const uint64_t user = hash_of(snapshot.user_data);
	const uint64_t all  = user ^ (hash_of(snapshot.buffers) * 3u) ^ (hash_of(snapshot.images) * 5u) ^
	                     (hash_of(snapshot.samplers) * 7u) ^ (hash_of(snapshot.flattened_srt) * 11u);
	static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> last;
	static uint64_t total = 0, same_user = 0, same_all = 0;
	static auto     start = std::chrono::steady_clock::now();
	auto [entry, inserted] = last.try_emplace(shader_hash, user, all);
	total++;
	if (!inserted) {
		same_user += entry->second.first == user ? 1u : 0u;
		same_all += entry->second.second == all ? 1u : 0u;
		entry->second = {user, all};
	}
	if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(5)) {
		std::printf("dispatch-repeat: %" PRIu64 " dispatches, same user data %.1f %%, same inputs %.1f %%, "
		            "%zu shaders\n",
		            total, 100.0 * static_cast<double>(same_user) / static_cast<double>(std::max<uint64_t>(total, 1)),
		            100.0 * static_cast<double>(same_all) / static_cast<double>(std::max<uint64_t>(total, 1)),
		            last.size());
		std::fflush(stdout);
		total = same_user = same_all = 0;
		start = std::chrono::steady_clock::now();
	}
}

static void DebugDumpDispatchInputs(RenderContext& context, const ShaderStageRuntime& stage) {
	// Several hashes may be given, separated by commas; each has its own count.
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		const char*           text = KYTY_GETENV("KYTY_DEBUG_DISPATCH_INPUTS");
		while (text != nullptr && *text != 0) {
			char* end = nullptr;
			if (const auto value = std::strtoull(text, &end, 16); value != 0) list.push_back(value);
			if (end == text) break;
			text = (*end == ',') ? end + 1 : end;
		}
		return list;
	}();
	if (hashes.empty()) [[likely]] {
		return;
	}
	const auto found = std::ranges::find(hashes, stage.program->shader_hash);
	if (found == hashes.end()) {
		return;
	}
	const auto hash = *found;
	// KYTY_DEBUG_DISPATCH_INPUTS_FILE=<flag>: a time window instead of the first dispatches. When the
	// flag file appears (checked every 50 ms) it is deleted and a window of
	// KYTY_DEBUG_DISPATCH_INPUTS_WINDOW_MS (default 4000) opens: every watched dispatch in it is
	// printed and, with _SAVE (+_READBACK), saved as _dispatch_dumps\<seq>_<hash>_b<i>.bin in
	// dispatch order (seq). Finds the first dispatch whose input is broken in a one-off build that a
	// timed capture misses (the title in Pro mode: LBVH key sort at the hub load).
	static const char* const window_flag = KYTY_GETENV("KYTY_DEBUG_DISPATCH_INPUTS_FILE");
	static std::atomic<uint64_t> window_seq {0};
	uint64_t seq = 0;
	if (window_flag != nullptr) {
		static const int64_t window_ms = [] {
			const char* text = KYTY_GETENV("KYTY_DEBUG_DISPATCH_INPUTS_WINDOW_MS");
			return text != nullptr ? std::strtoll(text, nullptr, 10) : 4000ll;
		}();
		static std::atomic<int64_t> window_end {0};
		static std::atomic<int64_t> next_check {0};
		const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
		                     std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
		if (now >= next_check.load(std::memory_order_relaxed)) {
			next_check.store(now + 50, std::memory_order_relaxed);
			std::error_code error;
			if (std::filesystem::exists(window_flag, error)) {
				std::filesystem::remove(window_flag, error);
				window_end.store(now + window_ms, std::memory_order_relaxed);
				std::printf("dispatch-inputs: window open for %lld ms\n", static_cast<long long>(window_ms));
				std::fflush(stdout);
			}
		}
		if (now >= window_end.load(std::memory_order_relaxed)) {
			return;
		}
		seq = window_seq.fetch_add(1, std::memory_order_relaxed);
	}
	static std::array<uint64_t, 16> counts {};
	const auto index = counts[static_cast<size_t>(found - hashes.begin()) % counts.size()]++;
	const auto& snapshot = *stage.resources;
	const auto& info     = stage.program->info;
	std::string text     = fmt::format("dispatch-inputs: cs {:016x} #{} buffers={} flattened={} queue={} "
	                                   "submission={} seq={}\n",
	                                   hash, index, snapshot.buffers.size(),
	                                   snapshot.flattened_srt.size(),
	                                   g_debug_submission_queue.load(std::memory_order_relaxed),
	                                   g_debug_submission_seq.load(std::memory_order_relaxed), seq);
	{
		// Where the group counts come from: the CP's read (direct, or an indirect one the CP turned
		// into a direct one) or the GPU's (indirect). "cpu" = the bytes the CP saw, "gpu" = after a
		// readback of GPU-written bytes, "dirty" = the GPU wrote them and the CPU copy may be stale.
		const uint64_t args = s_debug_args != 0 ? s_debug_args : g_debug_cp_indirect_args;
		text += fmt::format("  groups={}x{}x{} source={}", s_debug_groups[0], s_debug_groups[1],
		                    s_debug_groups[2],
		                    s_debug_args != 0 ? "indirect" : (args != 0 ? "cp-indirect" : "direct"));
		if (args != 0) {
			uint32_t   cpu[3] {};
			uint32_t   gpu[3] {};
			const bool dirty = context.GetBufferCache().HasGpuDirtyBytes(args, sizeof(cpu));
			LibKernel::Memory::TryReadBacking(args, cpu, sizeof(cpu));
			context.GetBufferCache().ReadMemory(args, sizeof(gpu), false);
			LibKernel::Memory::TryReadBacking(args, gpu, sizeof(gpu));
			text += fmt::format(" args=0x{:x} dirty={} cpu={}x{}x{} gpu={}x{}x{}", args,
			                    dirty ? 1 : 0, cpu[0], cpu[1], cpu[2], gpu[0], gpu[1], gpu[2]);
		}
		text += "\n";
	}
	for (size_t i = 0; i < snapshot.buffers.size(); i++) {
		if (snapshot.buffers[i].dword_count < 4) {
			continue;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
		const auto base       = descriptor.Base48();
		const auto size       = descriptor.GetSize();
		std::string words;
		for (uint64_t w = 0; base != 0 && w < std::min<uint64_t>(size / 4, 16); w++) {
			uint32_t value = 0;
			std::memcpy(&value, reinterpret_cast<const void*>(base + w * 4), sizeof(value));
			words += fmt::format(" {:08x}", value);
		}
		// The largest dword of a scalar-read buffer up to 8 MiB (loop bounds: a garbage count is a
		// GPU hang, the title cs 2ef93a49).
		if (base != 0 && i < info.buffers.size() && info.buffers[i].scalar && size <= (8u << 20)) {
			uint32_t max_value = 0;
			uint64_t big       = 0;
			for (uint64_t w = 0; w < size / 4; w++) {
				uint32_t value = 0;
				std::memcpy(&value, reinterpret_cast<const void*>(base + w * 4), sizeof(value));
				max_value = std::max(max_value, value);
				big += value >= 0x100000u ? 1u : 0u;
			}
			words += fmt::format(" | max={:#x} >=0x100000:{}", max_value, big);
		}
		const bool gpu = base != 0 && size != 0 &&
		                 context.GetBufferCache().IsRegionGpuModified(base, std::min<uint64_t>(size, 64));
		// The whole range's write state (LabelWriteFlags): 1 tracker GPU-modified, 2 GPU-dirty bytes,
		// 4 GPU-written ranges, 8 download in flight, 16 an image over it.
		const uint32_t state =
		    base != 0 && size != 0 ? context.GetBufferCache().LabelWriteFlags(base, size) : 0u;
		text += fmt::format("  buffer {} first_use_pc=0x{:x} scalar={} written={} base=0x{:x} "
		                    "size=0x{:x} gpu={} state={:x}:{}\n",
		                    i, i < info.buffers.size() ? info.buffers[i].first_use_pc : 0,
		                    i < info.buffers.size() && info.buffers[i].scalar ? 1 : 0,
		                    i < info.buffers.size() && info.buffers[i].written ? 1 : 0, base, size,
		                    gpu ? 1 : 0, state, words);
		// NaN dwords over the whole buffer (read like the words above), and the first ones as float3
		// element indices: where does a NaN state enter (the title cloth)?
		if (base != 0 && size >= 4 && size <= (1u << 20)) {
			// KYTY_DEBUG_DISPATCH_INPUTS_READBACK=1: take the GPU's bytes first (a drain per buffer),
			// so the counts are the GPU state at this dispatch, not the CPU copy.
			static const bool readback = KYTY_GETENV("KYTY_DEBUG_DISPATCH_INPUTS_READBACK") != nullptr;
			if (readback) {
				context.GetBufferCache().ReadMemory(base, size, false);
			}
			uint32_t              nan_count = 0;
			std::string           first;
			std::vector<uint32_t> data(size / 4);
			std::memcpy(data.data(), reinterpret_cast<const void*>(base), data.size() * 4);
			// KYTY_DEBUG_DISPATCH_INPUTS_SAVE=1 (with _READBACK): the whole buffer to
			// _dispatch_dumps\<hash>_<n>_b<i>.bin, the first 24 dispatches of each shader (offline
			// checks of a whole structure, e.g. the sort order of the title's PS5 Pro LBVH keys).
			static const bool save = KYTY_GETENV("KYTY_DEBUG_DISPATCH_INPUTS_SAVE") != nullptr;
			if (save && (window_flag != nullptr ? seq < 6000 : index < 24)) {
				std::filesystem::create_directories("_dispatch_dumps");
				const auto path =
				    window_flag != nullptr
				        ? fmt::format("_dispatch_dumps/{:05}_{:016x}_b{}.bin", seq, hash, i)
				        : fmt::format("_dispatch_dumps/{:016x}_{}_b{}.bin", hash, index, i);
				if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
					std::fwrite(data.data(), 4, data.size(), file);
					std::fclose(file);
				}
			}
			for (size_t w = 0; w < data.size(); w++) {
				if ((data[w] & 0x7f800000u) == 0x7f800000u && (data[w] & 0x7fffffu) != 0) {
					if (nan_count++ < 8) {
						first += fmt::format(" {}", w / 3);
					}
				}
			}
			if (nan_count != 0) {
				text += fmt::format("  buffer {} nan dwords {} of {}, first float3 elements{}\n", i,
				                    nan_count, data.size(), first);
			}
		}
	}
	// KYTY_DEBUG_DISPATCH_PROBE=s<n>+<table offset>:<byte offset> (hex offsets): follows the
	// 64-bit pointer in user data SGPRs n, n+1 to the buffer descriptor at the table offset and
	// reads the dword at the byte offset in that buffer, as an S_BUFFER_LOAD would.
	if (const char* probe = KYTY_GETENV("KYTY_DEBUG_DISPATCH_PROBE"); probe != nullptr) {
		unsigned sgpr = 0, table = 0, byte = 0;
		if (std::sscanf(probe, "s%u+%x:%x", &sgpr, &table, &byte) == 3 &&
		    sgpr + 1 < snapshot.user_data.size()) {
			const uint64_t pointer = snapshot.user_data[sgpr] |
			                         (uint64_t {snapshot.user_data[sgpr + 1]} << 32u);
			ShaderBufferResource descriptor {};
			std::memcpy(&descriptor, reinterpret_cast<const void*>(pointer + table),
			            sizeof(descriptor));
			const auto base = descriptor.Base48();
			uint32_t   word = 0;
			if (base != 0) {
				std::memcpy(&word, reinterpret_cast<const void*>(base + byte), sizeof(word));
			}
			const bool table_gpu = context.GetBufferCache().IsRegionGpuModified(pointer + table, 16);
			const bool word_gpu  = base != 0 && context.GetBufferCache().IsRegionGpuModified(base + byte, 4);
			text += fmt::format("  probe s{}+0x{:x}:0x{:x}: table=0x{:x} (gpu {}) buffer base=0x{:x} "
			                    "size=0x{:x} word=0x{:08x} ({}) (gpu {})\n",
			                    sgpr, table, byte, pointer, table_gpu ? 1 : 0, base,
			                    descriptor.GetSize(), word, word, word_gpu ? 1 : 0);
		}
	}
	text += "  flattened:";
	for (size_t w = 0; w < snapshot.flattened_srt.size(); w++) {
		text += fmt::format("{}{:08x}", w % 16 == 0 ? "\n   " : " ", snapshot.flattened_srt[w]);
	}
	text += "\n";
	// The last dispatch before a hang survives in its own file.
	if (std::FILE* file = std::fopen("_dispatch_inputs_last.txt", "wb"); file != nullptr) {
		std::fwrite(text.data(), 1, text.size(), file);
		std::fclose(file);
	}
	// KYTY_DEBUG_DISPATCH_INPUTS_ALL=1: every dispatch (shaders that run rarely, e.g. The title b092a36d).
	static const bool all = KYTY_GETENV("KYTY_DEBUG_DISPATCH_INPUTS_ALL") != nullptr;
	if (all || window_flag != nullptr || index < 40 || index % 200 == 0) {
		std::fwrite(text.data(), 1, text.size(), stdout);
		std::fflush(stdout);
	}
}

// KYTY_DEBUG_BASE_CHANGE=<cs hash>[:<buffer index>],..: cheap enough to play with (the title headless
// opponent: the game moves a per-frame ping-pong state buffer and a fresh one must start zeroed). Acts
// only when a buffer (>= 4 KiB; with an index only that one) gets a base not among the last 16 distinct
// bases at that index (the two fighters alternate, constants come from a ring): first the write state
// the emulator keeps for the range, then one readback and the NaN count. The readback downloads only
// GPU-dirty bytes (into guest memory): with state=0 dirty=0 the "gpu" count is the CPU copy again.
// Every _EVERY-th dispatch (default 600) the same readback runs as a heartbeat and names the buffers
// that hold NaNs. A NaN here is an all-ones exponent with a non-zero mantissa; "qnan" counts only the
// canonical 0x7fc00000/0xffc00000 the GPU produces (negative integers are not NaNs).
static void DebugWatchBaseChanges(RenderContext& context, const ShaderStageRuntime& stage) {
	struct Watch {
		uint64_t hash  = 0;
		uint32_t index = UINT32_MAX;
	};
	static const std::vector<Watch> watches = [] {
		std::vector<Watch> list;
		const char*        text = KYTY_GETENV("KYTY_DEBUG_BASE_CHANGE");
		while (text != nullptr && *text != 0) {
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) break;
			Watch watch {.hash = value};
			if (*end == ':') {
				text        = end + 1;
				watch.index = static_cast<uint32_t>(std::strtoul(text, &end, 10));
			}
			if (value != 0) list.push_back(watch);
			text = (*end == ',') ? end + 1 : end;
		}
		return list;
	}();
	if (watches.empty()) [[likely]] {
		return;
	}
	const auto hash  = stage.program->shader_hash;
	const auto found = std::ranges::find(watches, hash, &Watch::hash);
	if (found == watches.end()) {
		return;
	}
	const auto only = found->index;
	// KYTY_DEBUG_BASE_CHANGE_WORDS=1 (the title hang cs b092a36d): buffers of any size, and every
	// heartbeat prints the first words of each watched buffer (a loop bound read from word 0 shows its
	// course before the GPU hangs on it).
	static const bool words_mode = KYTY_GETENV("KYTY_DEBUG_BASE_CHANGE_WORDS") != nullptr;
	static const uint64_t every = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_BASE_CHANGE_EVERY");
		const auto  value = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
		return value != 0 ? value : uint64_t {600};
	}();
	struct Seen {
		std::vector<std::array<uint64_t, 16>> bases;
		std::vector<uint32_t>                 next;
		uint64_t                              dispatches = 0;
	};
	static std::mutex                         mutex;
	static std::unordered_map<uint64_t, Seen> seen;
	std::scoped_lock                          lock(mutex);
	auto&                                     state = seen[hash];
	const auto&                               snapshot = *stage.resources;
	const auto&                               info     = stage.program->info;
	const uint64_t                            dispatch = state.dispatches++;
	const bool                                heartbeat = dispatch != 0 && dispatch % every == 0;
	if (state.bases.size() < snapshot.buffers.size()) {
		state.bases.resize(snapshot.buffers.size(), {});
		state.next.resize(snapshot.buffers.size(), 0);
	}
	const auto count_nans = [](uint64_t base, uint64_t size, uint32_t& nan, uint32_t& qnan) {
		nan  = 0;
		qnan = 0;
		std::vector<uint32_t> data(size / 4);
		std::memcpy(data.data(), reinterpret_cast<const void*>(base), data.size() * 4);
		for (const auto word: data) {
			nan += (word & 0x7f800000u) == 0x7f800000u && (word & 0x7fffffu) != 0 ? 1u : 0u;
			qnan += (word & 0x7fffffffu) == 0x7fc00000u ? 1u : 0u;
		}
	};
	std::string text;
	for (size_t i = 0; i < snapshot.buffers.size(); i++) {
		if (snapshot.buffers[i].dword_count < 4 || (only != UINT32_MAX && i != only)) {
			continue;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
		const auto base       = descriptor.Base48();
		const auto size       = std::min<uint64_t>(descriptor.GetSize(), 1u << 20);
		if (base == 0 || (size < 4096 && !words_mode)) {
			continue;
		}
		auto&      recent  = state.bases[i];
		const bool changed = std::ranges::find(recent, base) == recent.end();
		if (changed) {
			recent[state.next[i]++ % recent.size()] = base;
		}
		if (!changed && !heartbeat) {
			continue;
		}
		const bool written = i < info.buffers.size() && info.buffers[i].written;
		// The emulator's view before the readback changes it.
		const uint32_t flags = context.GetBufferCache().LabelWriteFlags(base, size);
		const bool     dirty = context.GetBufferCache().HasGpuDirtyBytes(base, size);
		uint32_t       cpu_nan = 0, cpu_qnan = 0;
		count_nans(base, size, cpu_nan, cpu_qnan);
		context.GetBufferCache().ReadMemory(base, size, false);
		uint32_t gpu_nan = 0, gpu_qnan = 0;
		count_nans(base, size, gpu_nan, gpu_qnan);
		if (!changed && gpu_nan == 0 && !words_mode) {
			continue;
		}
		uint32_t words[4] {};
		std::memcpy(words, reinterpret_cast<const void*>(base), std::min<uint64_t>(size, sizeof(words)));
		text += fmt::format("base-change: cs {:016x} #{} {} buffer {} written={} base=0x{:x} size=0x{:x} "
		                    "state={:x} dirty={} cpu nan={} qnan={} | gpu nan={} qnan={} of {}: "
		                    "{:08x} {:08x} {:08x} {:08x}\n",
		                    hash, dispatch, changed ? "new" : "beat", i, written ? 1 : 0, base, size, flags,
		                    dirty ? 1 : 0, cpu_nan, cpu_qnan, gpu_nan, gpu_qnan, size / 4, words[0], words[1],
		                    words[2], words[3]);
	}
	if (!text.empty()) {
		std::fwrite(text.data(), 1, text.size(), stdout);
		std::fflush(stdout);
	}
}

// An indirect dispatch whose group count is not checked against its LDS binding, once per shader.
static void NoteUncheckedLds(uint64_t shader_hash, uint64_t groups) {
	static std::array<uint64_t, 32> named {};
	static size_t                   count = 0;
	if (count == named.size() || std::find(named.begin(), named.begin() + count, shader_hash) != named.begin() + count) {
		return;
	}
	named[count++] = shader_hash;
	std::printf("lds-unchecked: indirect dispatch of %016" PRIx64 " with LDS, room for %" PRIu64
	            " groups, count not checked\n",
	            shader_hash, groups);
	std::fflush(stdout);
}

// The groups an indirect dispatch's LDS binding holds (UINT32_MAX: no LDS binding).
static uint32_t LdsGroups(const ShaderComputeInputInfo& input, const PreparedBindings& bindings) {
	const uint64_t group_bytes = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
	if (bindings.shared_memory.buffer == nullptr || group_bytes == 0) {
		return UINT32_MAX;
	}
	return static_cast<uint32_t>(std::min<uint64_t>(bindings.shared_memory.range / group_bytes, UINT32_MAX));
}

// `indirect`: the group counts are in GPU memory. Reading them made the GPU thread wait for the GPU
// (the title in Pro mode, 2026-10-30), so the binding takes as many groups' LDS as the buffer holds (LdsGroups);
// thread-count dispatches are cut down to that on the GPU (DispatchThreadGroups), the others are
// named once per shader (they would read and write LDS past the binding with more groups).
static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, bool indirect = false, bool groups_on_gpu = false) {
	if (ShaderRecompiler::IR::FindBinding(input.stage.program->bindings,
	        ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory) == nullptr) {
		return;
	}
	auto& cache = context.GetBufferCache();
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	uint64_t size = sizeof(uint32_t);
	if (indirect) {
		const uint64_t group_bytes = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(group_bytes == 0 || group_bytes > limit);
		size = limit / group_bytes * group_bytes;
		if (!groups_on_gpu) {
			NoteUncheckedLds(input.stage.program->shader_hash, size / group_bytes);
		}
	} else if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

// KYTY_INDIRECT_IN_CHAIN=1: an indirect dispatch that continues a compute chain records no barrier for
// its arguments while the last one still covers them (CommandBuffer::IndirectArgsVisible). That barrier
// (all earlier commands -> draw-indirect) made every chained dispatch wait for the previous one to
// finish: the title in Pro mode's 5429ea1e runs 14 indirect dispatches in a row per cycle, each ~100 us whatever its
// size (one workgroup's latency), 1.17 ms a fight frame (run bd1, RenderDoc rd1). The guest separated
// them by no flush or wait, so on the console they overlap too. KYTY_INDIRECT_IN_CHAIN_OFF_FILE=<path>:
// the barrier every time while that file exists (in-run A/B, FileSwitch).
static bool IndirectInChainOn() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_INDIRECT_IN_CHAIN");
		return text != nullptr && text[0] == '1';
	}();
	if (!enabled) {
		return false;
	}
	static FileSwitch off(KYTY_GETENV("KYTY_INDIRECT_IN_CHAIN_OFF_FILE"), "indirect-in-chain");
	return !off.Exists();
}

// Why indirect dispatches kept their argument barrier (KYTY_INDIRECT_IN_CHAIN), printed every 50000.
enum class InChain { Skipped, ThreadCounts, NotChained, Off, NotCovered, Count };
static uint64_t g_chain_unplaced_writes = 0; // Chained dispatches with writes the host cannot place.

static void NoteIndirectInChain(InChain reason) {
	static std::array<uint64_t, static_cast<size_t>(InChain::Count)> counts {};
	static uint64_t                                                   total = 0;
	total++;
	counts[static_cast<size_t>(reason)]++;
	if (total % 50000u == 0u && KYTY_GETENV("KYTY_INDIRECT_IN_CHAIN") != nullptr) {
		std::printf("indirect-in-chain: argument barrier left out for %" PRIu64 " of %" PRIu64
		            " indirect dispatches (kept: thread counts %" PRIu64 ", not chained %" PRIu64
		            ", switch off %" PRIu64 ", no covering barrier or overlap %" PRIu64
		            "); chained dispatches with unplaced writes %" PRIu64 "\n",
		            counts[0], total, counts[1], counts[2], counts[3], counts[4],
		            g_chain_unplaced_writes);
		std::fflush(stdout);
		counts                  = {};
		total                   = 0;
		g_chain_unplaced_writes = 0;
	}
}

// What a chained dispatch writes, for the argument check of the dispatches after it: the ranges of its
// written host-evaluated buffer descriptors. Writes the host cannot place (GPU-selected V#s, address
// writes, DMA, storage images, swizzled buffers) are only counted: as on the console, where the command
// processor reads a dispatch's arguments without waiting for the dispatches before it, a guest whose
// dispatch writes the next one's arguments separates them by a flush or wait, which ends the chain.
static void NoteChainWrites(const CommandBuffer& buffer, const ShaderStageRuntime& runtime) {
	const auto& program   = *runtime.program;
	const auto& resources = *runtime.resources;
	bool        unplaced  = program.has_address_writes || program.info.uses_dma ||
	                resources.buffers.size() != program.info.buffers.size() ||
	                std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		                return image.written &&
		                       image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	                });
	const auto count = std::min(program.info.buffers.size(), resources.buffers.size());
	for (size_t i = 0; i < count; i++) {
		const auto& info = program.info.buffers[i];
		if (!info.written) {
			continue;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		if (info.indirect_root != ShaderRecompiler::IR::BufferResource::NoIndirectBuffer ||
		    descriptor.SwizzleEnabled()) {
			unplaced = true;
			continue;
		}
		const uint64_t base = descriptor.Base48();
		const uint64_t size = descriptor.GetSize();
		if (base != 0 && size != 0) {
			buffer.NoteChainWrite(base, base + size);
		}
	}
	g_chain_unplaced_writes += unplaced ? 1u : 0u;
}

// KYTY_DEBUG_DISPATCH_BINDINGS=1: for the dispatches of the shaders at the early device losses in
// Ghost of Yotei, one line per image binding element as the dispatch records it: the image, its
// view, its tracked layout and memory, and the tick the recent-image ring saw it created.
static bool DispatchBindingsEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_DISPATCH_BINDINGS");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

static void LogDispatchBindings(RenderContext& context,
                                const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                const PreparedBindings& bindings) {
	static constexpr std::array<uint64_t, 4> Watched {0xc427e96d8e8678cdull, 0x6b264377af50d025ull,
	                                                   0x79b9dff52896199dull, 0x8561056a850df874ull};
	if (std::find(Watched.begin(), Watched.end(), program.shader_hash) == Watched.end()) {
		return;
	}
	static uint64_t tick_seen  = 0;
	static uint32_t tick_index = 0;
	const auto      tick       = context.GetCommandScheduler().CurrentTick();
	tick_index                 = tick == tick_seen ? tick_index + 1u : 0u;
	tick_seen                  = tick;
	auto& cache = context.GetTextureCache();
	for (const auto& binding: program.bindings.descriptors) {
		const auto number = static_cast<uint32_t>(binding.kind);
		if (number < ShaderRecompiler::IR::FirstImageBinding ||
		    number >= ShaderRecompiler::IR::FirstImageBinding + ShaderRecompiler::IR::ImageBindingCount) {
			continue;
		}
		for (size_t element = 0; element < binding.resources.size(); ++element) {
			const auto resource = binding.resources[element];
			if (resource >= bindings.images.size()) {
				std::printf("dispatch-binding: tick=%" PRIu64 " index=%u shader=%016" PRIx64
				            " binding=%u[%zu] resource=%u not prepared\n",
				            tick, tick_index, program.shader_hash, number, element, resource);
				continue;
			}
			const auto& texture = bindings.images[resource];
			const auto* image   = cache.PeekImage(texture.image_id);
			if (image == nullptr) {
				std::printf("dispatch-binding: tick=%" PRIu64 " index=%u shader=%016" PRIx64
				            " binding=%u[%zu] image_id=%u NOT IN CACHE view=0x%" PRIx64 "\n",
				            tick, tick_index, program.shader_hash, number, element,
				            static_cast<uint32_t>(texture.image_id.index),
				            reinterpret_cast<uint64_t>(static_cast<VkImageView>(texture.image_view)));
				continue;
			}
			const auto vk_image = reinterpret_cast<uint64_t>(static_cast<VkImage>(image->backing.image));
			const char* memory  = image->backing.image == nullptr ? "no-image"
			                      : image->backing.allocation != nullptr ? "bound"
			                                                              : "not-bound";
			const auto created = RecentImageCreatedTick(vk_image);
			std::printf("dispatch-binding: tick=%" PRIu64 " index=%u shader=%016" PRIx64
			            " binding=%u[%zu] %s guest=0x%016" PRIx64 " size=0x%" PRIx64
			            " extent=%ux%ux%u layers=%u mips=%u guest_format=%u vkformat=%u image=0x%" PRIx64
			            " view=0x%" PRIx64 " bound_layout=%d tracked_layout=%d memory=%s registered=%d"
			            " created_tick=%s%" PRIu64 " last_access_tick=%" PRIu64
			            " barrier_this_tick=unknown\n",
			            tick, tick_index, program.shader_hash, number, element,
			            texture.desc.type == TextureCache::BindingType::Storage ? "storage" : "sampled",
			            image->info.data.address, image->info.data.size, image->info.extent.width,
			            image->info.extent.height, image->info.extent.depth,
			            image->info.resources.layers, image->info.resources.levels,
			            static_cast<uint32_t>(image->info.guest_format),
			            static_cast<uint32_t>(image->backing.format), vk_image,
			            reinterpret_cast<uint64_t>(static_cast<VkImageView>(texture.image_view)),
			            static_cast<int>(texture.layout), static_cast<int>(image->backing.state.layout),
			            memory, image->registered ? 1 : 0, created == 0 ? "unknown/" : "", created,
			            image->tick_accessed_last);
		}
	}
	std::fflush(stdout);
}

bool RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool may_defer,
                                    bool may_defer_reads) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations(false);
	m_context.GetBufferCache().StampGpuWrites();
	m_context.GetTextureCache().BeginBindings();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);
	s_debug_groups[0] = thread_group_x;
	s_debug_groups[1] = thread_group_y;
	s_debug_groups[2] = thread_group_z;
	s_debug_args      = 0;

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return true;
	}

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return true;
	}

	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		return true;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	// KYTY_DEFER_DIRTY_READS: the program lookup reads the shader's tables; bytes a dispatch wrote
	// on the GPU are waited for here by deferring, not by draining in the read's fault.
	if (may_defer_reads && m_context.GetBufferCache().DeferDirtyReads(cs_regs.cs_regs.data_addr) != 0) {
		return false;
	}
	const BufferCache::ReadSiteScope read_site(may_defer_reads ? cs_regs.cs_regs.data_addr : 0);
	const auto compute_program = [&] {
		DrainStats::SlowLookupTimer compile_timer(DrainStats::Kind::ShaderCompile);
		return m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	}();
	if (!compute_program) {
		// Temporary until RT is implemented.
		return true;
	}
	// Nothing is recorded or bound before this point, so the packet can run again later.
	if (may_defer && !m_context.GetPipelineCache().ComputePipelineReady(input_info, compute_program)) {
		return false;
	}
	if (DebugSkipShader(input_info.stage.program->shader_hash, DebugShaderKind::Compute))
	    [[unlikely]] {
		return true;
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	NoteDispatchRepeat(program.shader_hash, resources);
	{
		const std::array groups {thread_group_x, thread_group_y, thread_group_z};
		DebugCheckBvh(program, resources, groups.data());
		if (StampTraceEnabled()) [[unlikely]] {
			StampTraceDispatch(program.shader_hash);
		}
	}
	// Ghost of Yotei clears each shadow map's HTile with a dispatch that reads and rewrites every
	// HTile word in place (no address swizzle, one buffer exactly on the metadata). It is not a
	// proven full overwrite, so it is not consumed, but the clear is recorded for the next draw:
	// without it the depth target was never cleared and every caster failed the depth test.
	RecordHtileRewriteClear(input_info);
	// A fill consumed as a metadata clear may also cover color images over exactly its memory:
	// Ghost of Yotei's 8-bit shadow mask lives at an address a depth target used once as its HTile
	// base, so every later "Clear G-Buffer" fill of the mask became an HTile clear only and the
	// mask kept stale contents. The fill writes the same bytes under both, so both get the clear.
	// KYTY_META_DISPATCH_RUN=1 (upstream 01ce9738): a metadata write is recorded as a clear but the
	// dispatch still runs, since cached metadata addresses can be reused as ordinary buffers. Off by
	// default: with it ASTRO BOT's bot gallery showed empty grey tiles (2026-10-14).
	static const bool meta_dispatch_run = KYTY_GETENV("KYTY_META_DISPATCH_RUN") != nullptr;
	// KYTY_META_CLEAR_FILL_ONLY=1 (the title black tiles at resolution steps): a dispatch is replaced by a
	// metadata clear only when it is a proven uniform fill (ResolveComputeBufferFill). Any other
	// dispatch writing an address the cache holds as metadata (a retired depth target's HTile the game
	// now uses for its own data, e.g. a tile list) runs. KYTY_DEBUG_HTILE_SYNTH=1 names those.
	static const bool meta_fill_only = [] {
		const char* text = KYTY_GETENV("KYTY_META_CLEAR_FILL_ONLY");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	const auto meta_clear_allowed = [&] {
		if (!meta_fill_only) {
			return true;
		}
		ShaderBufferResource fill_descriptor {};
		uint32_t             fill_value = 0;
		uint64_t             fill_size  = 0;
		if (ResolveComputeBufferFill(input_info, thread_group_x, thread_group_y, thread_group_z, mode,
		                             fill_descriptor, fill_value, fill_size)) {
			return true;
		}
		static const bool log = KYTY_GETENV("KYTY_DEBUG_HTILE_SYNTH") != nullptr;
		if (log) [[unlikely]] {
			auto&      cache = m_context.GetTextureCache();
			const auto count = std::min(program.info.buffers.size(), resources.buffers.size());
			for (size_t i = 0; i < count; i++) {
				const auto d = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (program.info.buffers[i].written && cache.IsMeta(d.Base48())) {
					static std::atomic<uint32_t> logs {0};
					if (logs.fetch_add(1, std::memory_order_relaxed) < 2000) {
						std::printf("meta-clear-refused: cs %016" PRIx64 " buffer %zu base=0x%" PRIx64
						            " size=0x%" PRIx64 " read=%d groups=%ux%ux%u\n",
						            program.shader_hash, i, d.Base48(), static_cast<uint64_t>(d.GetSize()),
						            program.info.buffers[i].read ? 1 : 0, thread_group_x, thread_group_y,
						            thread_group_z);
						std::fflush(stdout);
					}
				}
			}
		}
		return false;
	};
	const auto consume_fill = [&] {
		if (meta_dispatch_run) {
			(void)TryConsumeComputeMetaClear(input_info, buffer, thread_group_x, thread_group_y,
			                                 thread_group_z, mode, false);
		} else if (meta_clear_allowed() &&
		           TryConsumeComputeMetaClear(input_info, buffer, thread_group_x, thread_group_y,
		                                      thread_group_z, mode)) {
			(void)TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
			                                  thread_group_z, mode);
			return true;
		}
		return TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
		                                   thread_group_z, mode);
	};
	// KYTY_DEBUG_META_FILL=1: while KYTY_DEBUG_FRAME_PASSES records, every dispatch that writes a
	// buffer and uses no image prints each buffer, whether the cache knows it as metadata, and
	// whether the dispatch was consumed as a clear (an HTile fast clear the cache does not
	// recognize runs as a plain fill) or remembered as a late HTile clear.
	static const bool meta_fill_log = KYTY_GETENV("KYTY_DEBUG_META_FILL") != nullptr;
	const bool        meta_fill_note = meta_fill_log && FramePassesRecording() &&
	                            program.info.images.empty() &&
	                            std::ranges::any_of(program.info.buffers,
	                                                [](const auto& b) { return b.written; });
	// Metadata state before consumption: a consumed clear does not change it, but the log
	// should show what the decision saw.
	std::string meta_fill_text;
	if (meta_fill_note) [[unlikely]] {
		auto&      cache = m_context.GetTextureCache();
		const auto count = std::min(program.info.buffers.size(), resources.buffers.size());
		for (size_t i = 0; i < count; i++) {
			const auto d = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			meta_fill_text += fmt::format(" b{}{}{} 0x{:x}+0x{:x} meta={}", i,
			                              program.info.buffers[i].read ? "r" : "",
			                              program.info.buffers[i].written ? "w" : "", d.Base48(),
			                              d.GetSize(), cache.IsMeta(d.Base48()) ? 1 : 0);
		}
	}
	const bool consumed = resources.specialization_reads.empty() && consume_fill();
	// KYTY_HTILE_LATE_CLEAR=1 (see TextureCache::NotePendingHtileFill): a proven uniform fill
	// that leaves every HTile word cleared (ZMASK 0), at an address not known as HTile, is
	// remembered; it still runs as the buffer write it is. An HTile base is 32 KiB aligned
	// (depthRenderTarget.cpp rejects anything else). Every other write of a dispatch that was
	// not consumed drops the pending fills it overlaps (the HTile no longer says "cleared").
	static const bool late_clear     = KYTY_GETENV("KYTY_HTILE_LATE_CLEAR") != nullptr;
	static const bool force_pending  = KYTY_GETENV("KYTY_DEBUG_HTILE_FORCE_PENDING") != nullptr;
	int               fill_state     = -1; // -1 not checked, 0 no uniform fill, 1 fill, 2 pending
	uint32_t          fill_value     = 0;
	if (!consumed && (late_clear || meta_fill_note)) {
		auto& cache = m_context.GetTextureCache();
		if (resources.specialization_reads.empty()) {
			ShaderBufferResource fill_descriptor {};
			uint64_t             fill_size = 0;
			fill_state = ResolveComputeBufferFill(input_info, thread_group_x, thread_group_y,
			                                      thread_group_z, mode, fill_descriptor,
			                                      fill_value, fill_size)
			                 ? 1
			                 : 0;
			const auto base = fill_descriptor.Base48();
			if (late_clear && fill_state == 1 && (fill_value & 0xfu) == 0 &&
			    (base & 0x7fffu) == 0 &&
			    (!cache.IsMeta(base) || (force_pending && cache.IsHtileMeta(base)))) {
				cache.NotePendingHtileFill(base, fill_size, fill_value);
				fill_state = 2;
			}
		}
		if (late_clear && fill_state != 2 && cache.HasPendingHtileFills()) {
			const auto count = std::min(program.info.buffers.size(), resources.buffers.size());
			for (size_t i = 0; i < count; i++) {
				if (program.info.buffers[i].written) {
					const auto d = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
					cache.DropPendingHtileFills(d.Base48(), std::max<uint64_t>(d.GetSize(), 1));
				}
			}
		}
	}
	if (meta_fill_note) [[unlikely]] {
		std::printf("frame-pass meta-fill cs=%016" PRIx64 " groups=%ux%ux%u spec=%zu xor=%d "
		            "consumed=%d fill=%d value=%08x%s\n",
		            program.shader_hash, thread_group_x, thread_group_y, thread_group_z,
		            resources.specialization_reads.size(), program.info.has_bitwise_xor ? 1 : 0,
		            consumed ? 1 : 0, fill_state, fill_value, meta_fill_text.c_str());
		std::fflush(stdout);
	}
	if (consumed) {
		ResetBindings();
		return true;
	}

	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	if (use_thread_dimensions) {
		auto groups_from_threads = [](uint32_t threads, uint32_t group_size) {
			return (threads == 0
			            ? 0u
			            : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
		};

		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = groups_from_threads(thread_group_x, cs_regs.cs_regs.num_thread_x);
		thread_group_y       = groups_from_threads(thread_group_y, cs_regs.cs_regs.num_thread_y);
		thread_group_z       = groups_from_threads(thread_group_z, cs_regs.cs_regs.num_thread_z);

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	NoteDispatchPassBreak(buffer.IsRendering());
	buffer.EndRendering();
	SyncHtileReads(input_info);
	auto& pipeline = [&]() -> PipelineCache::Pipeline& {
		DrainStats::SlowLookupTimer create_timer(DrainStats::Kind::PipelineCreate);
		return m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	}();
	auto& bindings = m_compute_bindings;
	DrainStats::t_binding_shader = input_info.stage.program->shader_hash;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	FindBuffers(bindings);
	// KYTY_DEBUG_FILL_PROBE=<cs hash>: the fill analysis and dispatch shape of that shader (first 6).
	static const uint64_t fill_probe = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_FILL_PROBE");
		return text != nullptr ? std::strtoull(text, nullptr, 16) : 0ull;
	}();
	if (fill_probe != 0 && program.shader_hash == fill_probe && !resources.buffers.empty() &&
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[0]).GetSize() <= 0x1000) [[unlikely]] {
		static uint32_t probes = 0;
		if (probes++ < 6) {
			const auto& f = resources.uniform_fill;
			std::printf("fill-probe: cs=%016" PRIx64 " groups=%ux%ux%u mode=0x%x threads=%ux%ux%u dims=%d "
			            "dthreads=%ux%ux%u fill kind=%d uniform=%d words=%u stride0=%u res=%u written=0x%" PRIx64 "\n",
			            program.shader_hash, thread_group_x, thread_group_y, thread_group_z, mode,
			            input_info.threads_num[0], input_info.threads_num[1], input_info.threads_num[2],
			            input_info.dispatch_thread_dimensions ? 1 : 0, input_info.dispatch_threads_num[0],
			            input_info.dispatch_threads_num[1], input_info.dispatch_threads_num[2],
			            static_cast<int>(f.kind), f.uniform ? 1 : 0, f.words, f.group_stride[0], f.resource,
			            PartialFillWriteSize(input_info, thread_group_x, thread_group_y, thread_group_z, mode));
			for (size_t i = 0; i < resources.buffers.size(); i++) {
				const auto d = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				std::printf("fill-probe:   b%zu 0x%016" PRIx64 " stride=%u records=%u fmt=%u swz=%d istride=%u tid=%d size=0x%" PRIx64 "\n",
				            i, d.Base48(), static_cast<uint32_t>(d.Stride()), d.NumRecords(),
				            static_cast<uint32_t>(d.Format()), d.SwizzleEnabled() ? 1 : 0,
				            static_cast<uint32_t>(d.IndexStride()), d.AddTid() ? 1 : 0, d.GetSize());
			}
			std::fflush(stdout);
		}
	}
	if (const auto written = PartialFillWriteSize(input_info, thread_group_x, thread_group_y,
	                                              thread_group_z, mode);
	    written != 0) {
		auto& source = bindings.buffer_sources[resources.uniform_fill.resource];
		if (source.size > written) {
			static std::atomic<uint32_t> logs {0};
			if (logs.fetch_add(1, std::memory_order_relaxed) < 8) {
				std::printf("fill-write-exact: cs=%016" PRIx64 " groups=%ux%ux%u mode=0x%x 0x%016" PRIx64
				            " bound 0x%" PRIx64 " -> written 0x%" PRIx64 "\n",
				            program.shader_hash, thread_group_x, thread_group_y, thread_group_z, mode,
				            source.address, source.size, written);
				std::fflush(stdout);
			}
			source.size = written;
			// Not kept: the next dispatch with this V# may launch more threads.
			source.map_generation = 0;
			source.found_kept     = false;
		}
	}
	DebugDumpDispatchInputs(m_context, input_info.stage);
	DebugWatchBaseChanges(m_context, input_info.stage);
	if (program.info.uses_dma) {
		PrefetchBvhs(m_context.GetBufferCache(), *input_info.stage.resources);
		m_context.PrepareBda();
		NoteBdaShader(program.shader_hash);
	}
	if (program.shader_hash == 0x380bb9d636390baeull && BvhCheckEnabled()) [[unlikely]] {
		static uint32_t described = 0;
		const auto&     user      = resources.user_data;
		std::array<uint32_t, 12> table {};
		if (described++ % 25u == 0u && user.size() >= 2u &&
		    LibKernel::Memory::TryReadBacking(user[0] | (uint64_t {user[1]} << 32u), table.data(),
		                                      sizeof(table))) {
			m_context.GetBufferCache().DebugDescribePage((table[10] | (uint64_t {table[11]} << 32u)) +
			                                             0x100u);
		}
	}
	// KYTY_DEBUG_READBACK_SCAN=<cs hash>:<buffer index>: before each of that shader's dispatches
	// (first 60 per range) the buffer binding is read back from the GPU and its NaN/Inf and
	// |x| > 100 floats are counted: which frame after a structure's creation first holds garbage
	// (the title cloth particle state). Waits for the GPU each time: diagnostics only.
	static const auto readback_scan = [] {
		std::pair<uint64_t, uint32_t> spec {0, 0};
		if (const char* text = KYTY_GETENV("KYTY_DEBUG_READBACK_SCAN"); text != nullptr) {
			char* end   = nullptr;
			spec.first  = std::strtoull(text, &end, 16);
			spec.second = *end == ':' ? static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10)) : 0u;
		}
		return spec;
	}();
	// The armed shader's binding range is remembered; afterwards every dispatch of any shader that
	// binds an overlapping range reads it back first (40 per range), so the first dispatch seeing
	// the garbage names the one before it as its writer.
	struct ScanRange {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint32_t scans   = 0;
	};
	static std::vector<ScanRange> scan_ranges;
	if (readback_scan.first != 0) [[unlikely]] {
		const auto hash = input_info.stage.program->shader_hash;
		if (hash == readback_scan.first && readback_scan.second < bindings.buffer_sources.size()) {
			const auto& source = bindings.buffer_sources[readback_scan.second];
			if (source.size != 0 && source.size <= (1u << 20u) && scan_ranges.size() < 64 &&
			    std::none_of(scan_ranges.begin(), scan_ranges.end(),
			                 [&](const ScanRange& r) { return r.address == source.address; })) {
				scan_ranges.push_back({source.address, source.size, 0});
			}
		}
		for (auto& range: scan_ranges) {
			if (range.scans >= 40) {
				continue;
			}
			int binding = -1;
			for (size_t i = 0; i < bindings.buffer_sources.size(); i++) {
				const auto& b = bindings.buffer_sources[i];
				if (b.size != 0 && b.address < range.address + range.size && range.address < b.address + b.size) {
					binding = static_cast<int>(i);
					break;
				}
			}
			if (binding < 0) {
				continue;
			}
			range.scans++;
			m_context.GetBufferCache().ReadMemory(range.address, range.size, false);
			std::vector<float> values(range.size / 4);
			if (LibKernel::Memory::TryReadBacking(range.address, values.data(), values.size() * 4)) {
				uint32_t nan = 0, big = 0;
				for (uint32_t i = 1; i < values.size(); i++) {
					const float v = values[i];
					nan += std::isfinite(v) ? 0u : 1u;
					big += std::isfinite(v) && std::fabs(v) > 100.0f ? 1u : 0u;
				}
				std::printf("readback-scan: [0x%016" PRIx64 "+0x%llx] #%u before cs=%016llx binding %d "
				            "groups=%ux%ux%u nan=%u big=%u\n",
				            range.address, static_cast<unsigned long long>(range.size), range.scans,
				            static_cast<unsigned long long>(hash), binding, s_debug_groups[0],
				            s_debug_groups[1], s_debug_groups[2], nan, big);
				std::fflush(stdout);
			}
		}
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings);
	if (ImageUsersEnabled()) [[unlikely]] {
		PreparedBindings* const compute_stages[] {&bindings};
		NoteImageUsers(m_context.GetTextureCache(), compute_stages, {}, nullptr, 0,
		               input_info.stage.program->shader_hash);
	}
	RebindBuffers(bindings);
	if (DrawResourcesEnabled()) [[unlikely]] {
		PreparedBindings* const compute_stages[] {&bindings};
		NoteDrawResources(m_context.GetTextureCache(), compute_stages, {}, nullptr, nullptr, 0,
		                  input_info.stage.program->shader_hash,
		                  m_context.GetCommandScheduler().CurrentTick());
	}

	// The large shared memory buffer is reused by every dispatch: no chaining past one.
	const bool chain = ComputeChainEnabled() && bindings.shared_memory.buffer == nullptr;
	auto              vk_buffer        = chain ? buffer.ChainHandle() : buffer.Handle();
	PreparedBindings* descriptor_stage = &bindings;
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u}, chain);
	if (DispatchBindingsEnabled()) [[unlikely]] {
		LogDispatchBindings(m_context, program, bindings);
	}
	// KYTY_DEBUG_DISPATCH_BUFFERS=<hex hash>: the storage buffers this shader's dispatches really
	// bind (host handle null or not, offset, range, guest source), first 400 dispatches.
	{
		static const uint64_t watch = [] {
			const char* text = KYTY_GETENV("KYTY_DEBUG_DISPATCH_BUFFERS");
			return text != nullptr ? std::strtoull(text, nullptr, 16) : 0ull;
		}();
		static uint32_t printed = 0;
		if (watch != 0 && program.shader_hash == watch && printed++ < 400) [[unlikely]] {
			std::string text = fmt::format("dispatch-buffers: cs {:016x} groups={}x{}x{} buffers={} "
			                               "wave={} submission={}\n",
			                               watch, thread_group_x, thread_group_y, thread_group_z,
			                               bindings.buffers.size(), program.wave_size,
			                               g_debug_submission_seq.load(std::memory_order_relaxed));
			for (size_t i = 0; i < bindings.buffers.size(); i++) {
				const auto& info   = bindings.buffers[i];
				const auto  source = i < bindings.buffer_sources.size() ? bindings.buffer_sources[i]
				                                                       : PreparedBindings::BufferSource {};
				text += fmt::format("  [{}] handle={} offset=0x{:x} range=0x{:x} guest=0x{:x}+0x{:x}\n", i,
				                    info.buffer ? "set" : "NULL", info.offset, info.range, source.address,
				                    source.size);
			}
			std::fwrite(text.data(), 1, text.size(), stdout);
			std::fflush(stdout);
		}
	}
	// Still pending: nothing but dispatches since the previous one, and no guest synchronization.
	const bool continues_chain = chain && buffer.ComputeChainPending();
	if (chain) {
		NoteComputeChain(continues_chain);
	}
	bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	if (has_storage_writes && !continues_chain) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (DebugFullBarriers()) [[unlikely]] {
		m_context.GetCommandScheduler().EndRendering();
		RecordFullBarrier(vk_buffer);
	}
	GpuZones::Mark(vk_buffer,
	               DrainStats::t_async_compute ? DrainStats::Zone::GameDispatchAsync
	                                           : DrainStats::Zone::GameDispatch,
	               input_info.stage.program->shader_hash);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);

	// The removed host fence also ordered read-only dispatches before later writers.
	if (chain) {
		buffer.ContinueComputeChain();
		NoteChainWrites(buffer, input_info.stage);
	} else {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	m_context.GetBufferCache().OnCommandRecorded();
	// KYTY_DEBUG_DUMP_AT_CS=<hash> (the title black tiles): the KYTY_DEBUG_DUMP_IMAGES list is written right
	// after this shader's dispatch in filmed frames, while the game's reused memory still holds them.
	static const std::vector<uint64_t> dump_at_cs = [] {
		std::vector<uint64_t> list;
		for (const char* text = KYTY_GETENV("KYTY_DEBUG_DUMP_AT_CS"); text != nullptr && *text != '\0';) {
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) break;
			list.push_back(value);
			text = *end == ',' ? end + 1 : nullptr;
		}
		return list;
	}();
	if (!dump_at_cs.empty() && std::ranges::find(dump_at_cs, input_info.stage.program->shader_hash) != dump_at_cs.end())
	    [[unlikely]] {
		static std::atomic<uint32_t> hits {0};
		if (hits.fetch_add(1, std::memory_order_relaxed) < 6) {
			std::printf("dump-at-cs hit: cs %016" PRIx64 "\n", input_info.stage.program->shader_hash);
			std::fflush(stdout);
		}
		{
			std::vector<ImageId> bound;
			for (const auto& b: m_compute_bindings.images) {
				bound.push_back(b.image_id);
			}
			m_context.GetTextureCache().DebugDumpListedNow(bound);
		}
	}
	ResetBindings();
	return true;
}

bool RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode, bool may_defer,
                                      bool may_defer_reads) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0);
	// Arguments in threads: a one-thread pass turns them into group counts on the GPU
	// (DispatchThreadGroups) and the dispatch reads those.
	const bool thread_dimensions = (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	m_context.GetCommandScheduler().PopPendingOperations(false);
	m_context.GetBufferCache().StampGpuWrites();
	m_context.GetTextureCache().BeginBindings();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	s_debug_args = args_addr;
	Common::LockGuard lock(m_context.GetMutex());
	// KYTY_SKIP_EMPTY_INDIRECT=1: arguments the CPU wrote (no GPU-written byte) that ask for no
	// groups run nothing, so neither the program nor its bindings are prepared (upstream
	// be5915c24). Off by default: it trusts the GPU-written tracking of the arguments, and the
	// first the title run with it (prof1, user fighting) showed exploding vertices on a fighter's
	// shorts (cloth); the cause is not proven.
	static const bool skip_empty = [] {
		const char* text = KYTY_GETENV("KYTY_SKIP_EMPTY_INDIRECT");
		return text != nullptr && text[0] == '1';
	}();
	if (vk::DispatchIndirectCommand args {}; skip_empty &&
	    !m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return true;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return true;
	}
	ShaderComputeInputInfo input_info {};
	input_info.dispatch_thread_dimensions = thread_dimensions;
	// KYTY_DEFER_DIRTY_READS: as in DispatchDirect (the title in Pro mode's one drain a fight frame: this
	// lookup read +0x16c8 of a 16 MiB object block, written by the dispatch just before).
	if (may_defer_reads && m_context.GetBufferCache().DeferDirtyReads(cs_regs.cs_regs.data_addr) != 0) {
		return false;
	}
	const BufferCache::ReadSiteScope read_site(may_defer_reads ? cs_regs.cs_regs.data_addr : 0);
	const auto compute_program = [&] {
		DrainStats::SlowLookupTimer compile_timer(DrainStats::Kind::ShaderCompile);
		return m_context.GetPipelineCache().GetComputeProgram(
		    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	}();
	if (!compute_program) {
		// Temporary until RT is implemented.
		return true;
	}
	// Nothing is recorded or bound before this point, so the packet can run again later.
	if (may_defer && !m_context.GetPipelineCache().ComputePipelineReady(input_info, compute_program)) {
		return false;
	}
	if (DebugSkipShader(input_info.stage.program->shader_hash, DebugShaderKind::Compute))
	    [[unlikely]] {
		return true;
	}
	// The group counts' slot, reserved before recording starts: the ring can wait and restart the
	// scheduler (as for the mesh-draw records in ExecutePreparedDraw).
	auto&    records      = m_context.GetBufferCache().GetDrawRecordBuffer();
	uint64_t groups_slot  = 0;
	if (thread_dimensions) {
		const auto [mapped, offset] = records.Map(DispatchThreadGroups::OutputSize, 64);
		EXIT_IF(mapped == nullptr);
		records.Commit();
		groups_slot = offset;
		if (m_dispatch_groups == nullptr) {
			m_dispatch_groups = std::make_unique<DispatchThreadGroups>(m_context.GetGraphics(),
			                                                           m_context.GetCommandScheduler());
		}
	}
	NoteDispatchPassBreak(buffer.IsRendering());
	buffer.EndRendering();
	SyncHtileReads(input_info);
	auto& pipeline = [&]() -> PipelineCache::Pipeline& {
		DrainStats::SlowLookupTimer create_timer(DrainStats::Kind::PipelineCreate);
		return m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	}();
	auto& bindings = m_compute_bindings;
	DrainStats::t_binding_shader = input_info.stage.program->shader_hash;
	PrepareBindings(input_info.stage, bindings);
	FindBuffers(bindings);
	DebugDumpDispatchInputs(m_context, input_info.stage);
	DebugWatchBaseChanges(m_context, input_info.stage);
	const auto& program = *input_info.stage.program;
	{
		std::array<uint32_t, 3> groups {};
		LibKernel::Memory::TryReadBacking(args_addr, groups.data(), sizeof(groups));
		DebugCheckBvh(program, *input_info.stage.resources, groups.data());
		if (StampTraceEnabled()) [[unlikely]] {
			StampTraceDispatch(program.shader_hash);
		}
	}
	if (program.info.uses_dma) {
		PrefetchBvhs(m_context.GetBufferCache(), *input_info.stage.resources);
		m_context.PrepareBda();
		NoteBdaShader(program.shader_hash);
	}
	if (program.shader_hash == 0x380bb9d636390baeull && BvhCheckEnabled()) [[unlikely]] {
		static uint32_t described = 0;
		const auto&     user      = input_info.stage.resources->user_data;
		std::array<uint32_t, 12> table {};
		if (described++ % 25u == 0u && user.size() >= 2u &&
		    LibKernel::Memory::TryReadBacking(user[0] | (uint64_t {user[1]} << 32u), table.data(),
		                                      sizeof(table))) {
			m_context.GetBufferCache().DebugDescribePage((table[10] | (uint64_t {table[11]} << 32u)) +
			                                             0x100u);
		}
	}
	BindSharedMemory(m_context, input_info, bindings, true, thread_dimensions);
	RebindImages(bindings);
	if (ImageUsersEnabled()) [[unlikely]] {
		PreparedBindings* const compute_stages[] {&bindings};
		NoteImageUsers(m_context.GetTextureCache(), compute_stages, {}, nullptr, 0,
		               input_info.stage.program->shader_hash);
	}
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	RebindBuffers(bindings);
	if (thread_dimensions) {
		// The pass also writes the thread counts into the uploaded shader data, as DispatchDirect
		// fills the dispatch thread extent (shader data with an extent is never push data).
		const auto extent_dword = program.bindings.dispatch_thread_dword;
		const bool with_extent  = extent_dword != ShaderRecompiler::IR::PushData::NoStart;
		EXIT_IF(with_extent &&
		        (program.bindings.UsesPushData() || bindings.shader_data_buffer.buffer == nullptr));
		// Before the dispatch's bindings: the pass binds its own pipeline and pushed set. With
		// KYTY_GPU_ZONES its GPU time is zone mesh-args, key 7d1.
		GpuZones::Mark(buffer.Handle(), DrainStats::Zone::MeshArgs, 0x7d1);
		m_dispatch_groups->Record(buffer.Handle(), *args_buffer, args_offset, records, groups_slot,
		                          {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                           cs_regs.cs_regs.num_thread_z},
		                          with_extent ? bindings.shader_data_buffer : vk::DescriptorBufferInfo {},
		                          extent_dword, LdsGroups(input_info, bindings), program.shader_hash);
	}
	if (DrawResourcesEnabled()) [[unlikely]] {
		PreparedBindings* const compute_stages[] {&bindings};
		NoteDrawResources(m_context.GetTextureCache(), compute_stages, {}, nullptr, nullptr, 0,
		                  input_info.stage.program->shader_hash,
		                  m_context.GetCommandScheduler().CurrentTick());
	}
	PreparedBindings* descriptor_stage = &bindings;
	const bool chain = ComputeChainEnabled() && bindings.shared_memory.buffer == nullptr;
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u}, chain);
	if (DispatchBindingsEnabled()) [[unlikely]] {
		LogDispatchBindings(m_context, program, bindings);
	}
	const bool continues_chain = chain && buffer.ComputeChainPending();
	if (chain) {
		NoteComputeChain(continues_chain);
	}
	const auto vk_buffer = chain ? buffer.ChainHandle() : buffer.Handle();
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	if (has_storage_writes && !continues_chain) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	// The dispatch reads its group counts from the conversion pass's slot when the arguments are
	// thread counts; the pass's own barrier already made them visible to indirect reads. With
	// KYTY_LATER_ACCESS barriers to all commands leave indirect-argument reads out, so this one
	// covers every earlier write (whether or not the in-run A/B has the masks off); thread-count
	// dispatches, whose slot only the pass writes, then go without it.
	const bool later = LaterAccessEnabled();
	// KYTY_INDIRECT_IN_CHAIN: the barrier recorded before an earlier dispatch of this chain still
	// covers these arguments.
	const auto in_chain = thread_dimensions    ? InChain::ThreadCounts
	                      : !continues_chain    ? InChain::NotChained
	                      : !IndirectInChainOn() ? InChain::Off
	                      : !buffer.IndirectArgsVisible(args_addr, sizeof(vk::DispatchIndirectCommand))
	                          ? InChain::NotCovered
	                          : InChain::Skipped;
	const bool args_covered = in_chain == InChain::Skipped;
	NoteIndirectInChain(in_chain);
	if (!(thread_dimensions && later) && !args_covered) {
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = later ? vk::AccessFlags {vk::AccessFlagBits::eMemoryWrite}
		                              : vk::AccessFlagBits::eShaderWrite |
		                                    vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(later ? vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands}
		                                : vk::PipelineStageFlagBits::eAllGraphics |
		                                      vk::PipelineStageFlagBits::eComputeShader |
		                                      vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {},
		                          1, &barrier, 0, nullptr, 0, nullptr);
		// Only a chain keeps the mark: a dispatch outside it leaves no pending barrier, so the
		// game's next flush point would not end the mark.
		if (chain) {
			buffer.SetIndirectArgsVisible();
		}
	}
	if (DebugFullBarriers()) [[unlikely]] {
		m_context.GetCommandScheduler().EndRendering();
		RecordFullBarrier(vk_buffer);
	}
	GpuZones::Mark(vk_buffer,
	               DrainStats::t_async_compute ? DrainStats::Zone::GameDispatchAsync
	                                           : DrainStats::Zone::GameDispatch,
	               input_info.stage.program->shader_hash);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	if (thread_dimensions) {
		vk_buffer.dispatchIndirect(records.Handle(), groups_slot);
	} else {
		vk_buffer.dispatchIndirect(args_buffer->Handle(), args_offset);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	if (chain) {
		buffer.ContinueComputeChain();
		NoteChainWrites(buffer, input_info.stage);
	} else {
		ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	m_context.GetBufferCache().OnCommandRecorded();
	// KYTY_DEBUG_DUMP_AT_CS=<hash> (the title black tiles): the KYTY_DEBUG_DUMP_IMAGES list is written right
	// after this shader's dispatch in filmed frames, while the game's reused memory still holds them.
	static const std::vector<uint64_t> dump_at_cs = [] {
		std::vector<uint64_t> list;
		for (const char* text = KYTY_GETENV("KYTY_DEBUG_DUMP_AT_CS"); text != nullptr && *text != '\0';) {
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) break;
			list.push_back(value);
			text = *end == ',' ? end + 1 : nullptr;
		}
		return list;
	}();
	if (!dump_at_cs.empty() && std::ranges::find(dump_at_cs, input_info.stage.program->shader_hash) != dump_at_cs.end())
	    [[unlikely]] {
		static std::atomic<uint32_t> hits {0};
		if (hits.fetch_add(1, std::memory_order_relaxed) < 6) {
			std::printf("dump-at-cs hit: cs %016" PRIx64 "\n", input_info.stage.program->shader_hash);
			std::fflush(stdout);
		}
		{
			std::vector<ImageId> bound;
			for (const auto& b: m_compute_bindings.images) {
				bound.push_back(b.image_id);
			}
			m_context.GetTextureCache().DebugDumpListedNow(bound);
		}
	}
	ResetBindings();
	return true;
}

} // namespace Libs::Graphics
