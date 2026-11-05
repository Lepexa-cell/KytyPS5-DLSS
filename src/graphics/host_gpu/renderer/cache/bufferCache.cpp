#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/switches.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/readbackQueue.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <fmt/format.h>
#include <xxhash.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// Buffers registered in extended memory (1 TiB page-table hole diagnostic, see ChangeRegister).
std::atomic<uint64_t> g_extended_registers {0};

// The GPU-written bindings for KYTY_DEBUG_GPU_WRITE_WATCH=0: a ring of ranges with their code path.
struct BlameEntry {
	uint64_t              begin = 0;
	uint64_t              end   = 0;
	uint32_t              op    = 0;
	uint32_t              count = 0;
	std::array<void*, 10> frames {};
	uint64_t              shader = 0;
};
std::mutex                g_blame_mutex;
std::vector<BlameEntry>   g_blame; // 1 << 19 entries once the first binding is noted
size_t                    g_blame_next = 0;

void NoteBlame(uint64_t vaddr, uint64_t size) {
	BlameEntry entry {vaddr, vaddr + size, DrainStats::t_pm4_op, 0, {}, DrainStats::t_binding_shader};
#ifdef _WIN32
	entry.count = RtlCaptureStackBackTrace(2, static_cast<DWORD>(entry.frames.size()),
	                                       entry.frames.data(), nullptr);
#endif
	std::lock_guard lock(g_blame_mutex);
	if (g_blame.empty()) {
		g_blame.resize(1u << 19u);
	}
	g_blame[g_blame_next++ % g_blame.size()] = entry;
}

// A CPU write on the GPU thread found `vaddr` GPU-dirty: print the latest recorded binding that
// covers it, once per code path.
void BlameGpuWrite(uint64_t vaddr) {
#ifdef _WIN32
	const auto base = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
#else
	const uint64_t base = 0;
#endif
	static std::vector<uint64_t> seen;
	std::lock_guard              lock(g_blame_mutex);
	const size_t                 total = std::min(g_blame_next, g_blame.size());
	// First a binding over the written bytes themselves, then one that only shares the 4 KiB tracker
	// page (dirty state is per page).
	size_t found = total;
	for (int pass = 0; pass < 2 && found == total; pass++) {
		for (size_t i = 0; i < total; i++) {
			const auto& entry = g_blame[(g_blame_next - 1 - i) % g_blame.size()];
			const auto  begin = pass == 0 ? vaddr : (vaddr & ~uint64_t {0xfff});
			const auto  end   = pass == 0 ? vaddr + 1 : begin + 0x1000;
			if (end > entry.begin && begin < entry.end) {
				found = i;
				break;
			}
		}
	}
	for (size_t i = found; i < total; i = total) {
		const auto& entry = g_blame[(g_blame_next - 1 - i) % g_blame.size()];
		uint64_t key = (entry.end - entry.begin) ^ entry.shader;
		for (uint32_t f = 0; f < entry.count; f++) {
			key = key * 1099511628211ull ^ reinterpret_cast<uint64_t>(entry.frames[f]);
		}
		if (std::find(seen.begin(), seen.end(), key) != seen.end() || seen.size() >= 64) {
			return;
		}
		seen.push_back(key);
		std::printf("gpu-write-blame: cpu write 0x%016" PRIx64 " in gpu-written [0x%016" PRIx64
		            ", 0x%016" PRIx64 ") %zu bindings ago, op=0x%x shader=%016" PRIx64 " inside=%d stack",
		            vaddr, entry.begin, entry.end, i, entry.op, entry.shader,
		            vaddr >= entry.begin && vaddr < entry.end ? 1 : 0);
		for (uint32_t f = 0; f < entry.count; f++) {
			std::printf(" %" PRIx64, reinterpret_cast<uint64_t>(entry.frames[f]) - base);
		}
		std::printf("\n");
		std::fflush(stdout);
		return;
	}
	static uint32_t unknown = 0;
	if (unknown++ < 32) {
		std::printf("gpu-write-blame: cpu write 0x%016" PRIx64 " GPU-dirty, no recorded binding\n",
		            vaddr);
		std::fflush(stdout);
	}
}

// KYTY_DEBUG_GPU_WRITE_WATCH=<hex addr>[,<hex addr>...]: prints each distinct code path that binds
// a GPU-written buffer range covering one of the addresses (range, PM4 op, return addresses as
// image offsets for the linker map), to find why a page the CPU writes is GPU-dirty.
void WatchGpuWrite(uint64_t vaddr, uint64_t size) {
	static const auto watch = [] {
		std::vector<uint64_t> list;
		for (const char* text = KYTY_GETENV("KYTY_DEBUG_GPU_WRITE_WATCH"); text != nullptr && *text != '\0';) {
			char* end = nullptr;
			list.push_back(std::strtoull(text, &end, 16));
			if (end == text) break;
			text = *end == ',' ? end + 1 : end;
		}
		return list;
	}();
	if (watch.empty()) {
		return;
	}
	if (watch.front() == 0) {
		// KYTY_DEBUG_GPU_WRITE_WATCH=0: remember every GPU-written binding (BlameGpuWrite).
		NoteBlame(vaddr, size);
		return;
	}
	for (const auto address: watch) {
		if (address < vaddr || address >= vaddr + size) {
			continue;
		}
		std::array<void*, 10> frames {};
#ifdef _WIN32
		const auto count = RtlCaptureStackBackTrace(1, static_cast<DWORD>(frames.size()), frames.data(),
		                                            nullptr);
		const auto base  = reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
#else
		const uint32_t count = 0;
		const uint64_t base  = 0;
#endif
		uint64_t key = vaddr ^ (size << 20u);
		for (uint32_t i = 0; i < count; i++) {
			key = key * 1099511628211ull ^ reinterpret_cast<uint64_t>(frames[i]);
		}
		static std::mutex             mutex;
		static std::vector<uint64_t>  seen;
		std::lock_guard               lock(mutex);
		if (std::find(seen.begin(), seen.end(), key) != seen.end() || seen.size() >= 64) {
			return;
		}
		seen.push_back(key);
		std::printf("gpu-write-watch: 0x%016" PRIx64 " in [0x%016" PRIx64 ", 0x%016" PRIx64
		            ") op=0x%x stack", address, vaddr, vaddr + size,
		            static_cast<unsigned>(DrainStats::t_pm4_op));
		for (uint32_t i = 0; i < count; i++) {
			std::printf(" %" PRIx64, reinterpret_cast<uint64_t>(frames[i]) - base);
		}
		std::printf("\n");
		std::fflush(stdout);
		return;
	}
}

// copyBuffer is not bounds-checked by the device: a region past either VkBuffer reads or writes
// past its allocation. Such regions are shortened to end inside both (dropped when nothing is
// left) and logged. A size of 0 for `source_size` skips the source check (unknown buffer).
void ClampBufferCopies(std::vector<vk::BufferCopy>& copies, uint64_t source_size,
                       uint64_t destination_size, const char* operation, uint64_t guest) {
	bool dropped = false;
	for (auto& copy: copies) {
		uint64_t limit = destination_size > copy.dstOffset ? destination_size - copy.dstOffset : 0;
		if (source_size != 0) {
			limit = std::min(limit, source_size > copy.srcOffset ? source_size - copy.srcOffset : 0);
		}
		if (copy.size <= limit) {
			continue;
		}
		std::printf("copy-overrun-clamped: %s guest=0x%016" PRIx64 " src_offset=0x%" PRIx64
		            " dst_offset=0x%" PRIx64 " size=0x%" PRIx64 " source_size=0x%" PRIx64
		            " destination_size=0x%" PRIx64 " kept=0x%" PRIx64 "\n",
		            operation, guest, static_cast<uint64_t>(copy.srcOffset),
		            static_cast<uint64_t>(copy.dstOffset), static_cast<uint64_t>(copy.size),
		            source_size, destination_size, limit);
		std::fflush(stdout);
		copy.size = limit;
		dropped |= limit == 0;
	}
	if (dropped) {
		std::erase_if(copies, [](const vk::BufferCopy& copy) { return copy.size == 0; });
	}
}

} // namespace

// KYTY_DEBUG_BDA_STATS=1: every 5 s, how the GPU thread's buffer synchronization spends its time,
// for PrepareBda's dirty-page passes and for the other callers (mostly bound buffers): the tracker
// walk, the copies into the staging buffer and the recording of the copy commands, and how many
// uploads ended a render pass.
namespace {
struct SyncStats {
	static bool Enabled() {
		static const bool enabled = KYTY_GETENV("KYTY_DEBUG_BDA_STATS") != nullptr;
		return enabled;
	}
	static uint64_t Now() {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                 std::chrono::steady_clock::now().time_since_epoch())
		                                 .count());
	}
	uint64_t passes  = 0;
	uint64_t pass_ns = 0;
	// [0]: PrepareBda's passes, [1]: other callers.
	std::array<uint64_t, 2> calls {};
	std::array<uint64_t, 2> uploads {};
	std::array<uint64_t, 2> copies {};
	std::array<uint64_t, 2> bytes {};
	std::array<uint64_t, 2> walk_ns {};
	std::array<uint64_t, 2> copy_ns {};
	std::array<uint64_t, 2> record_ns {};
	std::array<uint64_t, 2> breaks {};
	uint64_t                window_start = 0;
};
thread_local SyncStats t_sync_stats;
// Guest-thread side: InvalidateMemory calls (write faults and kernel writes) and their time.
std::atomic<uint64_t> g_invalidate_calls {0};
std::atomic<uint64_t> g_invalidate_ns {0};
} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	NoteAddrWatchUpload(address, size, "write-data (PM4 DMA/WRITE_DATA)");
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask =
		    LaterAccessOr(LaterSite::BufferCache, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	m_buffer_set_generation++;
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		// Diagnostic for the 1 TiB page-table hole: buffers in extended memory (their page-table
		// slots alias guest addresses in [1 TiB, 1.5 TiB)); logged for the first few.
		if (buffer.CpuAddress() >= LibKernel::Memory::kExtendedMemoryBase) {
			const auto count = g_extended_registers.fetch_add(1, std::memory_order_relaxed) + 1;
			if (count <= 16 || (count % 1024) == 0) {
				std::printf("bda-extended-buffer: #%" PRIu64 " guest=0x%016" PRIx64 " size=0x%" PRIx64
				            "\n",
				            count, buffer.CpuAddress(), buffer.Size());
			}
		}
		Common::DebugCounters::Adjust(Common::DebugCounters::Gauge::BufferCacheBytes,
		                              static_cast<int64_t>(buffer.Size()));
		buffer.lru_id   = m_lru_cache.Insert(id, m_gc_tick);
		buffer.lru_tick = m_gc_tick;
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		// Publish the full resulting owner, including pages newly reachable after a merge. The new
		// page-table entries make it readable through addresses before its next upload.
		m_memory_tracker.PublishBdaHints(buffer.CpuAddress(), buffer.Size());
		m_scheduler.Context().AdvanceBdaEpoch();
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		Common::DebugCounters::Adjust(Common::DebugCounters::Gauge::BufferCacheBytes,
		                              -static_cast<int64_t>(buffer.Size()));
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	// Draws touch the same buffers many times per GC tick: the LRU holds this tick already.
	if (!buffer.is_deleted && buffer.lru_tick != m_gc_tick) {
		buffer.lru_tick = m_gc_tick;
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

static const BufferCache* g_describe_cache = nullptr;

// A binding served from the stream ring still uses the owner's memory: it ages the owner like
// any other binding (the upload-age counter and the GC key on lru_tick).
void BufferCache::TouchStreamOwner(BufferId id, uint64_t vaddr, uint64_t size) {
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		const auto* owner = m_page_table.Find(vaddr >> CACHING_PAGEBITS);
		if (owner == nullptr || !*owner) {
			return;
		}
		id = *owner;
	}
	auto* buffer = m_slot_buffers.try_get(id);
	if (buffer == nullptr || buffer->is_deleted) {
		return;
	}
	TouchBuffer(*buffer);
	buffer->stream_tick = m_gc_tick;
}

void BufferCache::DescribeDeviceAddress(uint64_t address) const {
	// Also the 64 KiB past a buffer: a window or element run that walks off its last page lands
	// there (the next guest page is not necessarily the next device page).
	const auto within = [address](uint64_t base, uint64_t size) {
		return address >= base && address < base + size + 0x10000;
	};
	bool found = false;
	m_slot_buffers.ForEach([&](BufferId id, const Buffer& buffer) {
		if (buffer.HasDeviceAddress() && within(buffer.BufferDeviceAddress(), buffer.Size())) {
			std::printf("device-fault-owner: live buffer id=%u guest=0x%016llx size=0x%llx deleted=%d "
			            "offset=0x%llx%s\n",
			            static_cast<uint32_t>(id.index),
			            static_cast<unsigned long long>(buffer.CpuAddress()),
			            static_cast<unsigned long long>(buffer.Size()), buffer.is_deleted ? 1 : 0,
			            static_cast<unsigned long long>(address - buffer.BufferDeviceAddress()),
			            address >= buffer.BufferDeviceAddress() + buffer.Size() ? " (PAST THE END)" : "");
			found = true;
		}
	});
	for (const auto& freed: m_freed_buffers) {
		if (freed.bda != 0 && freed.size != 0 && within(freed.bda, freed.size)) {
			std::printf("device-fault-owner: freed buffer guest=0x%016llx size=0x%llx deleted at tick %llu "
			            "(completed %llu) offset=0x%llx%s\n",
			            static_cast<unsigned long long>(freed.cpu),
			            static_cast<unsigned long long>(freed.size),
			            static_cast<unsigned long long>(freed.tick),
			            static_cast<unsigned long long>(m_scheduler.GetMasterSemaphore().KnownGpuTick()),
			            static_cast<unsigned long long>(address - freed.bda),
			            address >= freed.bda + freed.size ? " (PAST THE END)" : "");
			found = true;
		}
	}
	// The cache's own buffers: a store past the end of a ring lands just behind one of these.
	const std::pair<const char*, const Buffer*> internal[] = {
	    {"device buffer", &m_device_buffer},     {"stream buffer", &m_stream_buffer},
	    {"staging buffer", &m_staging_buffer},   {"download buffer", &m_download_buffer},
	    {"draw record buffer", &m_draw_record_buffer}, {"GDS buffer", &m_gds_buffer},
	    {"fault buffer", m_fault_manager.GetFaultBuffer()},
	    {"BDA zero page", &m_bda_zero_page},
	};
	for (const auto& [name, buffer]: internal) {
		if (buffer->HasDeviceAddress() && within(buffer->BufferDeviceAddress(), buffer->Size())) {
			std::printf("device-fault-owner: %s size=0x%llx offset=0x%llx%s\n", name,
			            static_cast<unsigned long long>(buffer->Size()),
			            static_cast<unsigned long long>(address - buffer->BufferDeviceAddress()),
			            address >= buffer->BufferDeviceAddress() + buffer->Size() ? " (PAST THE END)" : "");
			found = true;
		}
	}
	if (m_bda_pagetable_buffer.HasDeviceAddress() &&
	    within(m_bda_pagetable_buffer.BufferDeviceAddress(), m_bda_pagetable_buffer.Size())) {
		std::printf("device-fault-owner: BDA page table\n");
		found = true;
	}
	if (!found) {
		std::printf("device-fault-owner: no game buffer (page table, stream or image memory)\n");
	}
	std::printf("device-fault-owner: %" PRIu64
	            " buffers registered at or above the extended memory base\n",
	            g_extended_registers.load(std::memory_order_relaxed));
	std::fflush(stdout);
	m_texture_cache.DescribeRecentImages();
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	{
		const auto& buffer = m_slot_buffers[id];
		auto&       freed  = m_freed_buffers[m_freed_next++ % m_freed_buffers.size()];
		freed = {buffer.HasDeviceAddress() ? buffer.BufferDeviceAddress() : 0, buffer.Size(),
		         buffer.CpuAddress(), m_scheduler.Active() ? m_scheduler.CurrentTick() : 0};
	}
	Unregister(id);
	// Deferred twice, like images (TextureCache::DeleteImage): a draw that resolved this buffer
	// before a ring wrap submitted and completed the current tick still names it in the commands
	// it records after the wrap (push descriptors, replayed later by the submit thread). Destroyed
	// after one tick, the replay read a dead VkBuffer: the title lost the device with WriteInvalid at
	// address 0. KYTY_BUFFER_DELETE_ONCE=1 restores the single deferral.
	static const bool once = [] {
		const char* text = KYTY_GETENV("KYTY_BUFFER_DELETE_ONCE");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	if (m_scheduler.Active() && once) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] {
			m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
		});
	} else {
		m_slot_buffers.erase(id);
	}
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy>                copies;
	std::vector<std::pair<uint64_t, uint64_t>> pages;
	uint64_t                                   total_size     = 0;
	const auto                                 buffer_address = buffer.CpuAddress();
	// Pages armed by an earlier download already have their bytes in flight.
	m_memory_tracker.ForEachUnarmedDownloadRange(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    pages.emplace_back(address, bytes);
		    NoteAddrWatchUpload(address, bytes, "download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (pages.empty()) {
		return false;
	}
	// Arm after any ring wrap (which can submit and move the tick) and before queueing the
	// publication, which may run inline during shutdown. Waiters only wait on submitted ticks,
	// so they cannot observe an arm before its publication is queued.
	const auto token = ++m_readback_token;
	const auto arm   = [this, &pages, token] {
		for (const auto& [address, bytes]: pages) {
			m_memory_tracker.ArmReadback(address, bytes, token, m_scheduler.CurrentTick());
		}
	};
	if (copies.empty()) {
		// Dirty pages without recorded bytes carry nothing to copy; their publication only
		// ends their GPU ownership.
		arm();
		m_scheduler.DeferPriorityOperation([this, token, pages = std::move(pages)] {
			for (const auto& [address, bytes]: pages) {
				m_memory_tracker.FinalizeReadback(address, bytes, token);
			}
		});
		return true;
	}
	DrainStats::Record(DrainStats::Kind::Readback, total_size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	Buffer*                 download = &m_download_buffer;
	if (mapped == nullptr) {
		// Ring wraps already wait for retirement. A reservation larger than the ring needs
		// separate host staging, kept alive until the priority callback publishes the bytes.
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		download  = temporary.get();
		mapped    = download->Mapped().data();
		offset    = 0;
		EXIT_IF(mapped == nullptr);
	} else {
		m_download_buffer.Commit();
	}
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	GpuZones::Mark(native, DrainStats::Zone::BufferCopy, 1); // key 1: download
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	ClampBufferCopies(copies, buffer.Size(), download->Size(), "download", buffer.CpuAddress());
	if (!copies.empty()) {
		native.copyBuffer(buffer.Handle(), download->Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
	}

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download->Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	arm();
	PruneInFlight();
	m_gpu_dirty_generation++;
	auto& inflight = m_inflight_downloads.emplace_back();
	inflight.tick  = m_scheduler.CurrentTick();
	inflight.ranges.reserve(copies.size());
	for (const auto& copy: copies) {
		inflight.ranges.emplace_back(buffer_address + copy.srcOffset, copy.size);
	}
	m_scheduler.DeferPriorityOperation([this, token, pages = std::move(pages), download,
	                                    owner = std::move(temporary), mapped, offset, total_size,
	                                    buffer_address, copies = std::move(copies)] {
		download->Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
		(void)owner; // Retain dedicated staging through the final backing write.
		// The backing holds the GPU bytes now; pages still armed by this download become
		// accessible. A page re-dirtied meanwhile keeps its protection for the next download.
		for (const auto& [address, bytes]: pages) {
			m_memory_tracker.FinalizeReadback(address, bytes, token);
		}
	});
	return true;
}

// The stream ring takes the render thread's per-draw copies: small CPU-written buffers, flattened
// SRTs, shader data and index data. In cached host memory the copies stay in the CPU's caches. In
// write-combined device memory (ReBAR) they cost Sky Garden 0.32 us a draw and the overworld 0.26,
// mostly by stalling the memory accesses after them; the GPU reads host copies about as fast (1%
// more GPU time in the overworld, none measurable in Sky Garden). KYTY_DEBUG_STREAM_DEVICE=1
// keeps the ring in device memory.
static bool StreamRingInHostMemory() {
	return KYTY_GETENV("KYTY_DEBUG_STREAM_DEVICE") == nullptr;
}

// KYTY_STAGING_DEVICE=1: tracked-memory uploads up to KYTY_STAGING_DEVICE_MAX KiB (default 4096) stage
// in a second ring in write-combined device memory (ReBAR). The GPU's upload copies then read VRAM
// instead of host memory over PCIe (the title arena: ~4.7 ms of GPU time a frame in upload copies,
// ~60 MB; the title in Pro mode fight, run sd1 2026-10-31: 2.86 -> 1.17 ms, 45-48 -> 50-55 fps); the upload
// helper's writes cross PCIe instead. Larger uploads (loading: the whole ring there made the fight's
// loading crawl at 5 fps) and image and WRITE_DATA staging keep the host ring. Placement only.
// KYTY_STAGING_DEVICE_OFF_FILE=<path>: the host ring while that file exists (in-run A/B, FileSwitch).
static bool StagingDeviceEnabled() {
	static const bool enabled = KYTY_GETENV("KYTY_STAGING_DEVICE") != nullptr;
	return enabled;
}

static uint64_t StagingDeviceMax() {
	static const uint64_t bytes = [] {
		const char* text = KYTY_GETENV("KYTY_STAGING_DEVICE_MAX");
		return (text != nullptr ? std::strtoull(text, nullptr, 10) : 4096ull) * 1024ull;
	}();
	return bytes;
}

static bool StagingDeviceOff() {
	static FileSwitch off(KYTY_GETENV("KYTY_STAGING_DEVICE_OFF_FILE"), "staging-device");
	return off.Exists();
}

StreamBuffer& BufferCache::UploadStaging(uint64_t size) {
	if (m_staging_device == nullptr || size > StagingDeviceMax() || StagingDeviceOff()) {
		return m_staging_buffer;
	}
	return *m_staging_device;
}

uint64_t BufferCache::StagingCapacity(vk::Buffer source) const {
	if (source == m_staging_buffer.Handle()) {
		return m_staging_buffer.Size();
	}
	if (m_staging_device != nullptr && source == m_staging_device->Handle()) {
		return m_staging_device->Size();
	}
	return 0;
}

// KYTY_PAGE_HISTORY: the upload helper hashes the pages it copies into the tracker's page history.
static void EnableUploadHistory(MemoryTracker& tracker);

// KYTY_TABLE_MODE=1: table programs read stream copies by device address (their table block and
// the small buffers ObtainBuffer streams). KYTY_GPU_DIRTY_TRACK=1: tracked programs their track
// entries.
static vk::BufferUsageFlags StreamRingFlags() {
	const char* table = KYTY_GETENV("KYTY_TABLE_MODE");
	const char* track = KYTY_GETENV("KYTY_GPU_DIRTY_TRACK");
	return (table != nullptr && std::strcmp(table, "1") == 0) || (track != nullptr && track[0] == '1')
	           ? vk::BufferUsageFlags {vk::BufferUsageFlagBits::eShaderDeviceAddress}
	           : vk::BufferUsageFlags {};
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_bda_zero_page(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                      AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, CACHING_PAGESIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB, StreamRingFlags(),
                      StreamRingInHostMemory()),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_draw_record_buffer(graphics, scheduler, MemoryUsage::Stream, 4 * MiB,
                           vk::BufferUsageFlagBits::eShaderDeviceAddress),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	SetVulkanObjectNameF(m_graphics.device, m_bda_zero_page.Handle(), "BDA Zero Page");
	// Both live for the whole session; their addresses place device faults (DescribeDeviceAddress).
	std::printf("bda-page-table: device=0x%016llx size=0x%llx\n",
	            static_cast<unsigned long long>(m_bda_pagetable_buffer.HasDeviceAddress()
	                                                ? m_bda_pagetable_buffer.BufferDeviceAddress()
	                                                : 0),
	            static_cast<unsigned long long>(m_bda_pagetable_buffer.Size()));
	g_describe_cache                     = this;
	GpuZones::g_gds_words = reinterpret_cast<const volatile uint32_t*>(m_gds_buffer.Mapped().data());
	GpuZones::g_describe_device_address = [](uint64_t address) {
		if (g_describe_cache != nullptr) {
			g_describe_cache->DescribeDeviceAddress(address);
		}
	};
	m_scheduler.SetPreSubmitHook(
	    [](void* cache) { static_cast<BufferCache*>(cache)->FlushUploadBatch(); }, this);
	{
		// Storage buffer bounds facts for the device-loss hunt: whether range checks are left to
		// the hardware, the device limit, and the range the page table is bound with.
		const auto limit = m_graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
		std::printf("storage-bounds: hardware_storage_buffer_bounds=%d maxStorageBufferRange=0x%llx "
		            "pagetable-descriptor-range=0x%llx%s\n",
		            m_graphics.hardware_storage_buffer_bounds ? 1 : 0,
		            static_cast<unsigned long long>(limit),
		            static_cast<unsigned long long>(m_bda_pagetable_buffer.Size()),
		            m_bda_pagetable_buffer.Size() > limit ? " (EXCEEDS THE LIMIT)" : "");
		std::fflush(stdout);
		if (const char* probe = KYTY_GETENV("KYTY_DEBUG_PAGETABLE_PROBE"); probe != nullptr) {
			unsigned long long entry = 0;
			unsigned long long value = 0;
			if (std::sscanf(probe, "%llu:%llx", &entry, &value) == 2) {
				SetPageTableProbe(entry, value);
			}
		}
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (m_graphics.readback_queue != nullptr) {
		m_readback_queue = std::make_unique<ReadbackQueue>(m_graphics);
	}
	if (StagingDeviceEnabled()) {
		m_staging_device =
		    std::make_unique<StreamBuffer>(graphics, scheduler, MemoryUsage::Stream, 512 * MiB);
		// Vulkan memory property bits: 1 device-local, 2 host-visible, 4 coherent, 8 cached.
		std::printf("staging-device: device ring memory 0x%x coherent=%d, host ring 0x%x, up to %llu KiB\n",
		            m_staging_device->MemoryProperties(), m_staging_device->IsCoherent() ? 1 : 0,
		            m_staging_buffer.MemoryProperties(),
		            static_cast<unsigned long long>(StagingDeviceMax() / 1024u));
		std::fflush(stdout);
	}
	if (PageHistoryEnabled()) {
		EnableUploadHistory(m_memory_tracker);
	}
	if (AbSelected("streamhost")) {
		m_stream_device = std::make_unique<StreamBuffer>(graphics, scheduler, MemoryUsage::Stream,
		                                                 64 * MiB, StreamRingFlags(), false);
		// Vulkan memory property bits: 1 device-local, 2 host-visible, 4 coherent, 8 cached.
		std::printf("streamhost A/B: ring memory 0x%x, device ring memory 0x%x\n",
		            m_stream_buffer.MemoryProperties(), m_stream_device->MemoryProperties());
		std::fflush(stdout);
	}
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	m_scheduler.SetPreSubmitHook(nullptr, nullptr);
	GpuZones::g_describe_device_address = nullptr;
	g_describe_cache                     = nullptr;
	GpuZones::g_gds_words = nullptr;
	m_readback_queue.reset();
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	struct Timer {
		bool                                  on;
		std::chrono::steady_clock::time_point start;
		~Timer() {
			if (on) {
				g_invalidate_calls.fetch_add(1, std::memory_order_relaxed);
				g_invalidate_ns.fetch_add(
				    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                              std::chrono::steady_clock::now() - start)
				                              .count()),
				    std::memory_order_relaxed);
			}
		}
	} timer {SyncStats::Enabled(), SyncStats::Enabled() ? std::chrono::steady_clock::now()
	                                                     : std::chrono::steady_clock::time_point {}};
	NoteAddrWatchUpload(vaddr, size, size <= TRACKER_PAGE_SIZE ? "cpu-write-fault" : "cpu-invalidate");
	// A CPU write needs the whole range CPU-dirty. GPU-dirty pages are published first; a GPU
	// write recorded while this thread waits can re-dirty a page, so repeat until all are marked.
	for (;;) {
		bool gpu_dirty = false;
		m_memory_tracker.InvalidateRegion(vaddr, size, [&gpu_dirty] { gpu_dirty = true; });
		if (!gpu_dirty) {
			return;
		}
		if (static const bool blame = [] {
			    const char* text = KYTY_GETENV("KYTY_DEBUG_GPU_WRITE_WATCH");
			    return text != nullptr && std::strtoull(text, nullptr, 16) == 0;
		    }();
		    blame && GuestGpu::IsGpuThread() && DrainStats::t_eop_label_write) [[unlikely]] {
			BlameGpuWrite(vaddr);
		}
		ReadMemory(vaddr, size, true);
	}
}

void BufferCache::ExpandWriteFault(uint64_t vaddr) {
	static const uint64_t window = [] {
		const char* text  = KYTY_GETENV("KYTY_FAULT_AHEAD_PAGES");
		const auto  pages = text != nullptr ? std::strtoull(text, nullptr, 10) : 16ull;
		return std::has_single_bit(pages) && pages <= TRACKER_REGION_PAGES ? pages * TRACKER_PAGE_SIZE
		                                                                   : TRACKER_PAGE_SIZE;
	}();
	if (window > TRACKER_PAGE_SIZE) {
		m_memory_tracker.ExpandWriteFault(vaddr, window);
	}
}

void BufferCache::RecordReadback(uint64_t vaddr, uint64_t size, bool is_write) {
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		if (m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
			EXIT("BufferCache: GPU-dirty memory has no buffer owner, addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n",
			     vaddr, size);
		}
		return;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

	// Widen nearby CPU reads so they share one publication.
	constexpr uint64_t WindowSize   = 512 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
	if (!DownloadBufferMemory(buffer, window_begin, window_end - window_begin)) {
		DrainStats::Record(DrainStats::Kind::ReadbackClean, 0);
	}
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	if (!GuestGpu::IsGpuThread()) {
		ReadMemoryAsync(vaddr, size, is_write);
		return;
	}
	// Thread_Gpu consumes the bytes itself, so it waits here. Pages armed by an earlier
	// (already submitted) download only need their own tick, which is not a full drain.
	// Its reads do not mark pages hot: they read shader code sharing a page with GPU data, or
	// arguments written just before, where an eager download only delays the next wait.
	RecordReadback(vaddr, size, is_write);
	const auto state = m_memory_tracker.QueryReadback(vaddr, size);
	if (state.gpu_dirty) {
		EXIT_IF(state.unarmed);
		m_scheduler.Wait(state.tick);
		m_scheduler.WaitPriorityOperations(state.tick);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		m_scheduler.Context().AdvanceBdaEpoch();
	}
}

void BufferCache::NoteGpuWrite(uint64_t vaddr, uint64_t size) {
	// Starting, or no point between commands for very long: forget what is not stamped and
	// refuse to answer until everything recorded so far has completed. The log is complete from
	// the open command buffer's tick on (unlogged writes are in it or earlier). Left unset, only an
	// InflightWriteTick call dated it, and QueueReadback refused before ever calling one: the
	// readback queue never took a guest readback (KYTY_READBACK_QUEUE, the title in Pro mode runs cr1-cr3c,
	// 2026-10-31: working, the fight ran ~4 fps faster).
	constexpr size_t Limit = size_t {1} << 20u;
	if (!m_gpu_writes_on || m_gpu_writes.size() - m_gpu_writes_head >= Limit) {
		m_gpu_writes_on = true;
		m_gpu_writes.clear();
		m_gpu_writes_head = m_gpu_writes_stamped = 0;
		m_gpu_writes_from = m_scheduler.CurrentTick();
	}
	m_gpu_writes.push_back({vaddr, vaddr + size, 0});
}

void BufferCache::StampGpuWrites() noexcept {
	if (m_gpu_writes_stamped >= m_gpu_writes.size()) {
		return;
	}
	const uint64_t current = m_scheduler.CurrentTick();
	for (; m_gpu_writes_stamped < m_gpu_writes.size(); ++m_gpu_writes_stamped) {
		m_gpu_writes[m_gpu_writes_stamped].tick = current;
	}
}

bool BufferCache::DeferDirtyReadsEnabled() noexcept {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEFER_DIRTY_READS");
		return text != nullptr && text[0] == '1';
	}();
	return enabled;
}

// KYTY_DEFER_DIRTY_READS_OFF_FILE (FileSwitch).
static bool DeferDirtyReadsOff() {
	static FileSwitch off(KYTY_GETENV("KYTY_DEFER_DIRTY_READS_OFF_FILE"), "defer-reads");
	return off.Exists();
}

void BufferCache::NoteReadSiteFault(uint64_t vaddr) {
	const auto shader = t_read_site_shader;
	if (shader == 0) {
		return;
	}
	const auto block = Common::AlignDown(vaddr, ReadSiteBytes);
	for (const auto& site: m_read_sites) {
		if (site.shader == shader && site.vaddr == block) {
			return;
		}
	}
	m_read_sites[m_read_site_next] = {shader, block, m_read_site_period};
	m_read_site_next               = (m_read_site_next + 1) % m_read_sites.size();
	m_defer_reads.learned++;
}

uint64_t BufferCache::InFlightTick(uint64_t vaddr, uint64_t size) {
	PruneInFlight();
	const auto end  = vaddr + size;
	uint64_t   tick = 0;
	for (const auto& download: m_inflight_downloads) {
		for (const auto& [address, bytes]: download.ranges) {
			if (address < end && address + bytes > vaddr) {
				tick = std::max(tick, download.tick);
				break;
			}
		}
	}
	return tick;
}

uint64_t BufferCache::ArmGpuThreadRead(uint64_t vaddr, uint64_t size) {
	if (m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		return UINT64_MAX;
	}
	if (!HasGpuDirtyBytes(vaddr, size)) {
		return 0;
	}
	// Page by page, as RecordEagerReadbacks: the readback window of a fault (RecordReadback) would
	// also download unrelated pages. Pages armed already keep their download (DownloadBufferMemory
	// skips them); a page written again since is unarmed and downloaded anew.
	const auto end = vaddr + size;
	for (auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE); page < end; page += TRACKER_PAGE_SIZE) {
		if (m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
		    IsRegionRegistered(page, TRACKER_PAGE_SIZE)) {
			auto& buffer = m_slot_buffers[FindBuffer(page, TRACKER_PAGE_SIZE)];
			(void)DownloadBufferMemory(buffer, page, TRACKER_PAGE_SIZE);
		}
	}
	if (m_gpu_modified_ranges.Intersects(vaddr, size)) {
		return UINT64_MAX;
	}
	const auto tick = InFlightTick(vaddr, size);
	if (tick == 0) {
		return HasGpuDirtyBytes(vaddr, size) ? UINT64_MAX : 0;
	}
	if (tick >= m_scheduler.CurrentTick()) {
		// The writer and the copy start on the GPU now (the fault's drain submitted here too), with
		// the eager readbacks the CP's own flush records first (CommandProcessor::BufferFlush).
		RecordEagerReadbacks();
		m_scheduler.Flush();
		m_defer_reads.flushes++;
	}
	if (m_scheduler.IsFree(tick)) {
		// Only the copy into guest memory may still be queued on the priority thread.
		m_scheduler.WaitPriorityOperations(tick);
		return HasGpuDirtyBytes(vaddr, size) ? UINT64_MAX : 0;
	}
	return tick;
}

uint64_t BufferCache::DeferDirtyReads(uint64_t shader_addr) {
	auto& stats = m_defer_reads;
	stats.checks++;
	const auto now = std::chrono::steady_clock::now();
	if (now >= stats.next_print) {
		if (stats.next_print != std::chrono::steady_clock::time_point {}) {
			size_t sites = 0;
			for (const auto& site: m_read_sites) {
				sites += site.shader != 0 ? 1u : 0u;
			}
			std::printf("defer-reads 10 s: checks=%u deferred=%u flushes=%u unsupported=%u gave-up=%u "
			            "off=%u fallback=%u %.1f ms, interrupts while deferred=%u, sites=%zu (learned %u, "
			            "expired %u)\n",
			            stats.checks, stats.deferred, stats.flushes, stats.unsupported, stats.gave_up,
			            stats.off, stats.fallbacks, static_cast<double>(stats.fallback_ns) / 1e6,
			            stats.interrupts, sites, stats.learned, stats.expired);
			std::fflush(stdout);
		}
		m_read_site_period++;
		uint32_t expired = 0;
		for (auto& site: m_read_sites) {
			if (site.shader != 0 && m_read_site_period - site.period >= ReadSitePeriods) {
				site = {};
				expired++;
			}
		}
		const auto streak = stats.streak;
		stats             = {};
		stats.streak      = streak;
		stats.expired     = expired;
		stats.next_print  = now + std::chrono::seconds(10);
	}
	m_deferred_read_tick = 0;
	if (shader_addr == m_deferred_shader) {
		// The deferred dispatch is back: it is recorded now, or deferred again below.
		m_deferred_shader = 0;
	}
	if (DeferDirtyReadsOff()) {
		stats.off++;
		stats.streak = 0;
		return 0;
	}
	uint64_t wait = 0;
	for (const auto& site: m_read_sites) {
		if (site.shader != shader_addr || shader_addr == 0) {
			continue;
		}
		const auto tick = ArmGpuThreadRead(site.vaddr, ReadSiteBytes);
		if (tick == UINT64_MAX) {
			stats.unsupported++;
			continue;
		}
		wait = std::max(wait, tick);
	}
	if (wait == 0) {
		stats.streak = 0;
		return 0;
	}
	// Bytes written again while the dispatch waited are downloaded again; should that keep
	// happening, the dispatch takes the fault's wait as before.
	constexpr uint32_t MaxStreak = 8;
	if (++stats.streak > MaxStreak) {
		stats.gave_up++;
		stats.streak = 0;
		return 0;
	}
	stats.deferred++;
	m_deferred_read_tick = wait;
	m_deferred_shader    = shader_addr;
	return wait;
}

void BufferCache::WaitDeferredRead(uint64_t tick) {
	// ArmGpuThreadRead submitted the tick: a wait on the open one would submit it from outside a slice.
	EXIT_IF(tick >= m_scheduler.CurrentTick());
	DrainStats::ReasonScope reason(DrainStats::Reason::GpuThreadReadFault);
	const auto              start = std::chrono::steady_clock::now();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	m_defer_reads.fallbacks++;
	m_defer_reads.fallback_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
	        .count());
}

uint64_t BufferCache::InflightWriteTick(uint64_t begin, uint64_t end, uint64_t completed) {
	const uint64_t current = m_scheduler.CurrentTick();
	for (; m_gpu_writes_stamped < m_gpu_writes.size(); ++m_gpu_writes_stamped) {
		m_gpu_writes[m_gpu_writes_stamped].tick = current;
	}
	if (m_gpu_writes_from == UINT64_MAX) {
		m_gpu_writes_from = current;
	}
	// Stamps never decrease: the completed writes are a prefix.
	while (m_gpu_writes_head < m_gpu_writes.size() &&
	       m_gpu_writes[m_gpu_writes_head].tick <= completed) {
		++m_gpu_writes_head;
	}
	if (m_gpu_writes_head >= 4096 && m_gpu_writes_head * 2 >= m_gpu_writes.size()) {
		m_gpu_writes.erase(m_gpu_writes.begin(),
		                   m_gpu_writes.begin() + static_cast<std::ptrdiff_t>(m_gpu_writes_head));
		m_gpu_writes_stamped -= m_gpu_writes_head;
		m_gpu_writes_head = 0;
	}
	uint64_t tick = 0;
	for (size_t i = m_gpu_writes_head; i < m_gpu_writes.size(); ++i) {
		const auto& write = m_gpu_writes[i];
		if (write.begin < end && begin < write.end) {
			tick = std::max(tick, write.tick);
		}
	}
	return tick;
}

namespace {
// KYTY_DEBUG_READBACK_AGE=1: guest threads' waits for GPU-written bytes, by the state of the
// bytes' last GPU write when the download was recorded: finished, submitted, still recording,
// or unknown (the write log is incomplete), plus waits for an already recorded download.
struct ReadbackAgeStats {
	enum Class : uint32_t { Done, Submitted, Open, Unknown, Armed, Queued, Count };
	std::mutex                  mutex;
	std::array<uint64_t, Count> n {};
	std::array<double, Count>   ms {};
	uint64_t                    total = 0;
	void Add(uint32_t c, double wait_ms) {
		std::scoped_lock lock(mutex);
		n[c]++;
		ms[c] += wait_ms;
		if (++total % 500u == 0u) {
			std::printf("readback-age: %" PRIu64 " waits | done n=%" PRIu64 " %.0fms | submitted n=%" PRIu64
			            " %.0fms | open n=%" PRIu64 " %.0fms | unknown n=%" PRIu64
			            " %.0fms | armed n=%" PRIu64 " %.0fms | readback queue n=%" PRIu64
			            " %.0fms\n",
			            total, n[Done], ms[Done], n[Submitted], ms[Submitted], n[Open], ms[Open],
			            n[Unknown], ms[Unknown], n[Armed], ms[Armed], n[Queued], ms[Queued]);
			std::fflush(stdout);
		}
	}
};
ReadbackAgeStats g_readback_age;
} // namespace

struct BufferCache::QueuedReadback {
	vk::Buffer                                 source;
	std::unique_ptr<Buffer>                    download;
	std::vector<vk::BufferCopy>                copies;
	std::vector<std::pair<uint64_t, uint64_t>> pages;
	uint64_t                                   buffer_address = 0;
	uint64_t                                   token          = 0;
	uint64_t                                   total_size     = 0;
	uint64_t                                   wait_tick      = 0;
	uint64_t                                   queue_value    = 0;
	bool                                       published      = false;
	std::mutex                                 mutex;
};

std::shared_ptr<BufferCache::QueuedReadback> BufferCache::QueueReadback(uint64_t vaddr,
                                                                        uint64_t size,
                                                                        bool     is_write) {
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return nullptr;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	// The window RecordReadback would download.
	constexpr uint64_t WindowSize   = 512 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	auto window_end   = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);
	auto& master      = m_scheduler.GetMasterSemaphore();
	master.Refresh();
	const uint64_t completed = master.KnownGpuTick();
	// KYTY_DEBUG_READBACK_AGE: outcomes (queued window, queued pages, writes in flight, log
	// incomplete, nothing unarmed).
	static std::array<uint64_t, 5> outcomes {};
	const auto                     note = [](size_t outcome) {
        if (ReadbackAgeEnabled() && ++outcomes[outcome] % 1000u == 0u) {
            std::printf("readback queue: window=%" PRIu64 " pages=%" PRIu64 " in-flight=%" PRIu64
			                                " log=%" PRIu64 " clean=%" PRIu64 "\n",
			                                outcomes[0], outcomes[1], outcomes[2], outcomes[3], outcomes[4]);
            std::fflush(stdout);
        }
	};
	if (!GpuWriteLogComplete(completed)) {
		note(3);
		return nullptr;
	}
	size_t   outcome  = 0;
	uint64_t producer = InflightWriteTick(window_begin, window_end, completed);
	if (producer != 0) {
		// The window is still being written elsewhere: download only the requested pages, after
		// their last writer, when that one is submitted (the open command buffer's would need a
		// flush, which the graphics queue path does anyway).
		window_begin = std::max(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE), buffer_begin);
		window_end   = std::min(Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE), buffer_end);
		producer     = InflightWriteTick(window_begin, window_end, completed);
		if (producer >= m_scheduler.CurrentTick()) {
			note(2);
			return nullptr;
		}
		outcome = 1;
	}
	auto request            = std::make_shared<QueuedReadback>();
	request->buffer_address = buffer_begin;
	m_memory_tracker.ForEachUnarmedDownloadRange(
	    window_begin, window_end - window_begin, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "queued buffer download");
		    request->pages.emplace_back(address, bytes);
		    NoteAddrWatchUpload(address, bytes, "queued-download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    request->copies.emplace_back(start - buffer_begin, request->total_size, end - start);
			    request->total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (request->pages.empty()) {
		note(4);
		return nullptr;
	}
	note(outcome);
	if (!request->copies.empty()) {
		DrainStats::Record(DrainStats::Kind::Readback, request->total_size);
		request->download =
		    std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                             vk::BufferUsageFlagBits::eTransferDst, request->total_size);
		EXIT_IF(request->download->Mapped().empty());
		request->source = buffer.Handle();
	}
	request->wait_tick = std::max(completed, producer);
	request->token     = ++m_readback_token;
	for (const auto& [address, bytes]: request->pages) {
		m_memory_tracker.ArmReadback(address, bytes, request->token, m_scheduler.CurrentTick());
	}
	PruneInFlight();
	m_gpu_dirty_generation++;
	auto& inflight = m_inflight_downloads.emplace_back();
	inflight.tick  = m_scheduler.CurrentTick();
	inflight.ranges.reserve(request->copies.size());
	for (const auto& copy: request->copies) {
		inflight.ranges.emplace_back(buffer_begin + copy.srcOffset, copy.size);
	}
	m_scheduler.DeferPriorityOperation([this, request] { PublishQueuedReadback(*request); });
	return request;
}

void BufferCache::PublishQueuedReadback(QueuedReadback& request) {
	std::scoped_lock lock(request.mutex);
	if (request.published) {
		return;
	}
	if (!request.copies.empty()) {
		if (request.queue_value == 0) {
			request.queue_value = m_readback_queue->Copy(
			    request.source, request.download->Handle(), request.copies,
			    m_scheduler.GetMasterSemaphore().Handle(), request.wait_tick);
		}
		m_readback_queue->Wait(request.queue_value);
		request.download->Invalidate(0, request.total_size);
		const auto* mapped = request.download->Mapped().data();
		for (const auto& copy: request.copies) {
			Libs::LibKernel::Memory::WriteBacking(request.buffer_address + copy.srcOffset,
			                                      mapped + copy.dstOffset, copy.size);
		}
	}
	for (const auto& [address, bytes]: request.pages) {
		m_memory_tracker.FinalizeReadback(address, bytes, request.token);
	}
	request.published = true;
	request.download.reset();
}

void BufferCache::ReadMemoryAsync(uint64_t vaddr, uint64_t size, bool is_write) {
	// Thread_Gpu only records and submits the download; this thread waits for its own
	// publication while Thread_Gpu keeps working. The caller retries the access afterwards:
	// a page the GPU re-dirtied meanwhile simply faults again.
	auto     state = m_memory_tracker.QueryReadback(vaddr, size);
	uint64_t tick  = state.gpu_dirty ? state.tick : 0;
	if (!state.gpu_dirty) {
		return;
	}
	// Relaxed readback: a download already carries the GPU's new bytes, so the read may see the
	// previous ones meanwhile, as on hardware when the CPU reads before the GPU has written.
	if (!is_write && !state.unarmed && Config::RelaxedReadbackEnabled() &&
	    m_memory_tracker.GrantStaleRead(vaddr, size)) {
		DrainStats::Record(DrainStats::Kind::StaleRead, 1);
		return;
	}
	uint32_t                        age = ReadbackAgeStats::Armed;
	std::shared_ptr<QueuedReadback> queued;
	if (state.unarmed || tick >= m_scheduler.CurrentTick()) {
		const auto reason = DrainStats::CurrentReason();
		m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, reason, &tick,
		                                                &age, &queued] {
			DrainStats::ReasonScope reason_scope(reason);
			if (ReadbackAgeEnabled()) [[unlikely]] {
				auto& master = m_scheduler.GetMasterSemaphore();
				master.Refresh();
				const uint64_t completed = master.KnownGpuTick();
				const uint64_t producer  = InflightWriteTick(vaddr, vaddr + size, completed);
				age = !GpuWriteLogComplete(completed)        ? ReadbackAgeStats::Unknown
				      : producer == 0                         ? ReadbackAgeStats::Done
				      : producer < m_scheduler.CurrentTick() ? ReadbackAgeStats::Submitted
				                                              : ReadbackAgeStats::Open;
			}
			// KYTY_READBACK_QUEUE_OFF_FILE=<path>: guest readbacks take the old way while the file
			// exists (in-run A/B of the readback queue, looked at every 16 readbacks).
			static const char* queue_off_file = KYTY_GETENV("KYTY_READBACK_QUEUE_OFF_FILE");
			static uint32_t    queue_off_calls = 0;
			static bool        queue_off       = false;
			if (queue_off_file != nullptr && queue_off_calls++ % 16u == 0u) {
				std::error_code error;
				queue_off = std::filesystem::exists(queue_off_file, error);
			}
			if (m_readback_queue != nullptr && !queue_off) {
				queued = QueueReadback(vaddr, size, is_write);
				if (queued != nullptr) {
					if (!is_write) {
						MarkReadbackHot(vaddr);
					}
					tick = 0;
					return;
				}
			}
			if (!is_write) {
				MarkReadbackHot(vaddr);
			}
			RecordReadback(vaddr, size, is_write);
			const auto current = m_memory_tracker.QueryReadback(vaddr, size);
			if (!current.gpu_dirty) {
				tick = 0;
				return;
			}
			EXIT_IF(current.unarmed);
			tick = current.tick;
			// A foreign thread may only wait on submitted ticks: nothing else is guaranteed
			// to submit the open command buffer while guest work waits on this thread.
			if (tick >= m_scheduler.CurrentTick()) {
				m_scheduler.Flush();
			}
		});
	}
	if (queued != nullptr) {
		DrainStats::WaitTimer wait(DrainStats::Kind::TickWait);
		const auto            start = ReadbackAgeEnabled() ? std::chrono::steady_clock::now()
		                                                   : std::chrono::steady_clock::time_point {};
		PublishQueuedReadback(*queued);
		if (ReadbackAgeEnabled()) [[unlikely]] {
			g_readback_age.Add(ReadbackAgeStats::Queued,
			                   std::chrono::duration<double, std::milli>(
			                       std::chrono::steady_clock::now() - start)
			                       .count());
		}
		return;
	}
	if (tick == 0) {
		return;
	}
	{
		DrainStats::WaitTimer wait(DrainStats::Kind::TickWait);
		const auto            start = ReadbackAgeEnabled() ? std::chrono::steady_clock::now()
		                                                   : std::chrono::steady_clock::time_point {};
		m_scheduler.GetMasterSemaphore().Wait(tick);
		if (ReadbackAgeEnabled()) [[unlikely]] {
			g_readback_age.Add(age, std::chrono::duration<double, std::milli>(
			                            std::chrono::steady_clock::now() - start)
			                            .count());
		}
	}
	m_scheduler.WaitPriorityOperations(tick);
}

void BufferCache::MarkReadbackHot(uint64_t vaddr) {
	const auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto tick = m_scheduler.CurrentTick();
	if (auto it = m_hot_pages.find(page); it != m_hot_pages.end()) {
		it->second = tick;
		return;
	}
	if (m_hot_pages.size() >= HotReadbackPages) {
		// Replace the page whose latest fault is the oldest.
		const auto oldest = std::min_element(
		    m_hot_pages.begin(), m_hot_pages.end(),
		    [](const auto& a, const auto& b) { return a.second < b.second; });
		m_memory_tracker.SetReadbackHot(oldest->first, TRACKER_PAGE_SIZE, false);
		m_hot_pages.erase(oldest);
	}
	m_hot_pages.emplace(page, tick);
	m_memory_tracker.SetReadbackHot(page, TRACKER_PAGE_SIZE, true);
}

void BufferCache::QueueEagerReadback(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	for (const auto& [page, tick]: m_hot_pages) {
		(void)tick;
		if (page < end && page + TRACKER_PAGE_SIZE > vaddr) {
			m_eager_pending.push_back(page);
		}
	}
}

void BufferCache::OnCommandRecorded() {
	if (m_eager_pending.empty()) {
		return;
	}
	m_eager_ready.insert(m_eager_ready.end(), m_eager_pending.begin(), m_eager_pending.end());
	m_eager_pending.clear();
}

void BufferCache::RecordEagerReadbacks() {
	if (m_eager_ready.empty()) {
		return;
	}
	DrainStats::ReasonScope reason(DrainStats::Reason::EagerReadback);
	std::vector<uint64_t>   pages;
	pages.swap(m_eager_ready);
	std::sort(pages.begin(), pages.end());
	pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
	for (const auto page: pages) {
		// The written buffer may have been retired since; its pages then have no GPU owner.
		if (!IsRegionRegistered(page, TRACKER_PAGE_SIZE)) {
			continue;
		}
		// Buffers are CACHING_PAGESIZE-aligned, so one buffer owns the whole tracker page.
		auto& buffer = m_slot_buffers[FindBuffer(page, TRACKER_PAGE_SIZE)];
		(void)DownloadBufferMemory(buffer, page, TRACKER_PAGE_SIZE);
	}
}

void BufferCache::DebugDescribePage(uint64_t vaddr) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) {
		std::printf("bda-page: 0x%016llx has no cache buffer (table index %llu)\n",
		            static_cast<unsigned long long>(vaddr),
		            static_cast<unsigned long long>(PageIndex(vaddr)));
	} else {
		auto& buffer = m_slot_buffers[*owner];
		std::printf("bda-page: 0x%016llx owner cpu=0x%016llx size=0x%llx bda=0x%016llx table index "
		            "%llu (first %llu) deleted=%d\n",
		            static_cast<unsigned long long>(vaddr),
		            static_cast<unsigned long long>(buffer.CpuAddress()),
		            static_cast<unsigned long long>(buffer.Size()),
		            static_cast<unsigned long long>(buffer.BufferDeviceAddress()),
		            static_cast<unsigned long long>(PageIndex(vaddr)),
		            static_cast<unsigned long long>(PageIndex(buffer.CpuAddress())),
		            buffer.is_deleted ? 1 : 0);
	}
	std::fflush(stdout);
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			// Callers that only register a range for shader reads through the BDA page table
			// (BVHs, mesh-emulation indices) never bind it: their lookup is its use
			// (KYTY_RELEASE_UNBOUND keys on lru_tick).
			TouchBuffer(buffer);
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	new_buffer.bda_pinned = new_buffer.bda_pinned || overlap.bda_pinned;
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	g_allocation_counters.game_buffers_joined.fetch_add(1, std::memory_order_relaxed);
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);
	g_allocation_counters.game_buffers_created.fetch_add(1, std::memory_order_relaxed);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	NoteReleasedOverlap(overlap.begin, overlap.end - overlap.begin, false);
	return id;
}

static uint64_t ReleaseUnboundTicks() {
	static const uint64_t ticks = [] {
		const char* text = KYTY_GETENV("KYTY_RELEASE_UNBOUND");
		return text != nullptr ? std::strtoull(text, nullptr, 10) : uint64_t {0};
	}();
	return ticks;
}

void BufferCache::PinBda(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	m_slot_buffers[id].bda_pinned = true;
}

void BufferCache::NoteBdaFault(uint64_t vaddr, uint64_t size) {
	NoteReleasedOverlap(vaddr, size, true);
}

// Released ranges (KYTY_RELEASE_UNBOUND) met again: by a binding (a new buffer uploads the range
// in full) or by a shader through the BDA page table (it read zeros from the zero page: the
// release was wrong for that draw). Printed per 5 s with the release counts.
void BufferCache::NoteReleasedOverlap(uint64_t vaddr, uint64_t size, bool bda_fault) {
	if (m_released.empty()) {
		return;
	}
	auto it = m_released.upper_bound(vaddr);
	if (it != m_released.begin()) {
		--it;
	}
	while (it != m_released.end() && it->first < vaddr + size) {
		const auto& [address, released] = *it;
		if (address + released.size <= vaddr) {
			++it;
			continue;
		}
		if (bda_fault) {
			m_release_stats.refaults++;
			if (m_release_stats.refaults_printed++ < 64) {
				std::printf("release-refault: BDA read of released [0x%016" PRIx64 "+0x%" PRIx64
				            "] unbound %" PRIu64 " frames at release, released %" PRIu64 " frames ago\n",
				            address, released.size, released.age, m_gc_tick - released.tick);
				std::fflush(stdout);
			}
		} else {
			m_release_stats.rebound++;
			m_release_stats.rebound_bytes += released.size;
			const auto after = m_gc_tick - released.tick;
			m_release_stats.rebound_after[after < 300 ? 0 : after < 900 ? 1 : after < 1800 ? 2 : 3]++;
		}
		it = m_released.erase(it);
	}
}

// Upstream pull request 1809: a buffer otherwise stays registered until memory pressure retires
// it, so its pages stay write-protected after the game reuses the memory for CPU data: every
// write faults and every BDA pass uploads the page again (the title arena: ~1.2 GB/s into ~170
// buffers no draw bound for 100+ frames). KYTY_RELEASE_UNBOUND=<frames> releases up to 16 per
// frame that no draw or dispatch has bound for that many frames and that hold no GPU-written
// data. A BDA read of a released page reaches the fault buffer, which recreates and pins the
// buffer (it read zeros once: counted as release-refault).
void BufferCache::ReleaseUnboundBuffers() {
	const auto unbound = ReleaseUnboundTicks();
	if (unbound == 0 || m_release_tick == m_gc_tick || m_gc_tick <= unbound) {
		return;
	}
	m_release_tick = m_gc_tick;
	constexpr size_t MaxReleased = 16;
	constexpr size_t MaxVisited  = 256;
	std::vector<BufferId> released;
	std::vector<BufferId> kept;
	size_t                visited = 0;
	m_lru_cache.ForEachItemBelow(m_gc_tick - unbound, [&](BufferId id) {
		const auto& buffer = m_slot_buffers[id];
		if (!buffer.bda_pinned &&
		    !m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size()) &&
		    !m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) &&
		    !m_memory_tracker.HasArmedPages(buffer.CpuAddress(), buffer.Size())) {
			released.push_back(id);
		} else {
			kept.push_back(id);
		}
		return ++visited == MaxVisited || released.size() == MaxReleased;
	});
	// Kept buffers move to the young end: left at the old end, pinned and GPU-dirty buffers would
	// fill the visit budget and stop all later releases (review 2026-10-23).
	for (const auto id: kept) {
		TouchBuffer(m_slot_buffers[id]);
	}
	m_release_stats.kept += kept.size();
	for (const auto id: released) {
		const auto& buffer = m_slot_buffers[id];
		if (m_released.size() >= 65536) {
			m_released.clear();
		}
		m_released[buffer.CpuAddress()] = {buffer.Size(), m_gc_tick, m_gc_tick - buffer.lru_tick};
		m_release_stats.released++;
		m_release_stats.released_bytes += buffer.Size();
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		g_allocation_counters.game_buffers_collected.fetch_add(1, std::memory_order_relaxed);
		DeleteBuffer(id);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - m_release_stats.window >= std::chrono::seconds(5)) {
		const double seconds = std::chrono::duration<double>(now - m_release_stats.window).count();
		std::printf("release-unbound: released=%" PRIu64 " (%.1f MB) rebound=%" PRIu64
		            " (%.1f MB; after <300/<900/<1800/more frames: %" PRIu64 "/%" PRIu64 "/%" PRIu64
		            "/%" PRIu64 ") bda-refaults=%" PRIu64 " kept(pinned/gpu-dirty)=%" PRIu64
		            " in %.1f s, tracked released=%zu\n",
		            m_release_stats.released, static_cast<double>(m_release_stats.released_bytes) / 1e6,
		            m_release_stats.rebound, static_cast<double>(m_release_stats.rebound_bytes) / 1e6,
		            m_release_stats.rebound_after[0], m_release_stats.rebound_after[1],
		            m_release_stats.rebound_after[2], m_release_stats.rebound_after[3],
		            m_release_stats.refaults, m_release_stats.kept, seconds, m_released.size());
		std::fflush(stdout);
		const auto printed     = m_release_stats.refaults_printed;
		m_release_stats        = {};
		m_release_stats.window = now;
		m_release_stats.refaults_printed = printed;
	}
}


bool BufferCache::BdaStatsEnabled() {
	return SyncStats::Enabled();
}

void BufferCache::NoteBdaPass(uint64_t elapsed_ns) {
	auto& stats = t_sync_stats;
	stats.passes++;
	stats.pass_ns += elapsed_ns;
	const auto now = SyncStats::Now();
	if (stats.window_start == 0) {
		stats.window_start = now;
		return;
	}
	const auto window = now - stats.window_start;
	if (window < 5'000'000'000ull) {
		return;
	}
	const double seconds = static_cast<double>(window) / 1e9;
	const auto   per_s   = [&](uint64_t value) { return static_cast<double>(value) / seconds; };
	const auto   ms_s    = [&](uint64_t ns) { return static_cast<double>(ns) / 1e6 / seconds; };
	std::printf("bda-stats: passes/s=%.0f pass-ms/s=%.1f", per_s(stats.passes), ms_s(stats.pass_ns));
	for (int kind = 0; kind < 2; kind++) {
		std::printf(" | %s calls/s=%.0f uploads/s=%.0f copies/s=%.0f MB/s=%.1f walk-ms/s=%.1f "
		            "copy-ms/s=%.1f record-ms/s=%.1f breaks/s=%.0f",
		            kind == 0 ? "bda" : "other", per_s(stats.calls[kind]), per_s(stats.uploads[kind]),
		            per_s(stats.copies[kind]), per_s(stats.bytes[kind]) / 1e6,
		            ms_s(stats.walk_ns[kind]), ms_s(stats.copy_ns[kind]),
		            ms_s(stats.record_ns[kind]), per_s(stats.breaks[kind]));
	}
	const auto protect     = PageManager::TakeProtectStats();
	const auto invalidates = g_invalidate_calls.exchange(0, std::memory_order_relaxed);
	const auto invalid_ns  = g_invalidate_ns.exchange(0, std::memory_order_relaxed);
	std::printf(" | gpu-thread protects/s=%.0f protect-ms/s=%.1f | invalidates/s=%.0f "
	            "invalidate-ms/s=%.1f | hot MB/s=%.1f fresh MB/s=%.1f | history demoted/s=%.0f "
	            "window marked/s=%.0f skipped/s=%.0f hashed=%.0f%%\n",
	            per_s(protect.calls), ms_s(protect.ns), per_s(invalidates), ms_s(invalid_ns),
	            per_s(g_upload_hot_pages.exchange(0, std::memory_order_relaxed)) * TRACKER_PAGE_SIZE / 1e6,
	            per_s(g_upload_fresh_pages.exchange(0, std::memory_order_relaxed)) * TRACKER_PAGE_SIZE / 1e6,
	            per_s(g_history_demoted.exchange(0, std::memory_order_relaxed)),
	            per_s(g_history_window_marked.exchange(0, std::memory_order_relaxed)),
	            per_s(g_history_window_skipped.exchange(0, std::memory_order_relaxed)),
	            [] {
		            const auto copied = g_history_copied.exchange(0, std::memory_order_relaxed);
		            const auto hashed = g_history_hashed.exchange(0, std::memory_order_relaxed);
		            return copied != 0 ? 100.0 * static_cast<double>(hashed) / static_cast<double>(copied)
		                               : 0.0;
	            }());
	std::fflush(stdout);
	stats              = {};
	stats.window_start = now;
}

static bool DeferProtectsEnabled();

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	const bool     stats_on = SyncStats::Enabled();
	const int      kind     = t_upload_source == UploadSource::Bda ? 0 : 1;
	const uint64_t start    = stats_on ? SyncStats::Now() : 0;
	uint64_t       copy_ns  = 0;
	// A read-only upload leaves its write protections to the upload helper, which makes them
	// before it copies the pages (see PageManager::DeferProtects).
	const bool defer = !is_written && DeferProtectsEnabled() && m_staging_buffer.IsCoherent();
	if (is_written) {
		WatchGpuWrite(vaddr, size);
	}
	// KYTY_DEBUG_SYNC_SIZE=<hex size>: the first synchronization of each address with exactly that
	// binding size prints the tracker state (CPU/GPU modified), the bytes it uploads and the NaN
	// dwords of the guest copy: a structure the CPU (re)initialised but the GPU never received.
	static const uint64_t sync_size = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_SYNC_SIZE");
		return text != nullptr ? std::strtoull(text, nullptr, 16) : uint64_t {0};
	}();
	bool sync_trace = false;
	if (sync_size != 0 && size == sync_size) [[unlikely]] {
		static std::vector<uint64_t> seen;
		if (seen.size() < 256 && std::find(seen.begin(), seen.end(), vaddr) == seen.end()) {
			seen.push_back(vaddr);
			sync_trace = true;
			std::vector<uint32_t> words(size / 4);
			uint32_t              nan = 0;
			if (Libs::LibKernel::Memory::TryReadBacking(vaddr, words.data(), words.size() * 4)) {
				for (const auto w: words) {
					nan += (w & 0x7f800000u) == 0x7f800000u ? 1u : 0u;
				}
			}
			std::printf("sync-trace: [0x%016" PRIx64 "+0x%llx] written=%d cpu_mod=%d gpu_mod=%d "
			            "guest_nan=%u buffer=0x%016" PRIx64 "+0x%llx\n",
			            vaddr, static_cast<unsigned long long>(size), is_written ? 1 : 0,
			            m_memory_tracker.IsRegionCpuModified(vaddr, size) ? 1 : 0,
			            m_memory_tracker.IsRegionGpuModified(vaddr, size) ? 1 : 0, nan, buffer.CpuAddress(),
			            static_cast<unsigned long long>(buffer.Size()));
			std::printf("sync-trace:   aliases %s\n",
			            Libs::LibKernel::Memory::DebugDescribeAliases(vaddr, size).c_str());
#ifdef _WIN32
			// The host protection of each 4 KiB page next to the tracker's state: a writable page
			// the tracker calls clean or GPU-dirty lost its write watch.
			for (uint64_t page = vaddr & ~uint64_t {0xfff}; page < vaddr + size; page += 0x1000) {
				MEMORY_BASIC_INFORMATION info {};
				VirtualQuery(reinterpret_cast<const void*>(page), &info, sizeof(info));
				std::printf("sync-trace:   page 0x%016" PRIx64 " host_protect=0x%lx cpu_mod=%d gpu_mod=%d\n",
				            page, info.Protect, m_memory_tracker.IsRegionCpuModified(page, 0x1000) ? 1 : 0,
				            m_memory_tracker.IsRegionGpuModified(page, 0x1000) ? 1 : 0);
			}
#endif
			std::fflush(stdout);
		}
	}
	PageManager::DeferProtects(defer);
	const auto pushes = defer ? PageManager::DeferredPushes() : 0;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
		    NoteAddrWatchUpload(address, bytes,
		                        t_upload_source == UploadSource::Bda ? "upload-bda" : "upload");
	    },
	    [&]() noexcept {
		    const uint64_t copy_start = stats_on ? SyncStats::Now() : 0;
		    source = UploadCopies(buffer, copies, total_size,
		                          defer && PageManager::DeferredPushes() != pushes);
		    if (stats_on) {
			    copy_ns = SyncStats::Now() - copy_start;
		    }
	    },
	    m_scheduler.Context().CurrentBdaEpoch());
	PageManager::DeferProtects(false);
	if (sync_trace) [[unlikely]] {
		std::printf("sync-trace: [0x%016" PRIx64 "] uploaded 0x%llx bytes in %zu copies\n", vaddr,
		            static_cast<unsigned long long>(total_size), copies.size());
		std::fflush(stdout);
	}
	const uint64_t walked = stats_on ? SyncStats::Now() : 0;
	if (stats_on) {
		auto& stats = t_sync_stats;
		stats.calls[kind]++;
		stats.walk_ns[kind] += walked - start - copy_ns;
		stats.copy_ns[kind] += copy_ns;
		if (source) {
			stats.uploads[kind]++;
			stats.copies[kind] += copies.size();
			stats.bytes[kind] += total_size;
			// A batched upload ends rendering once, at the flush (counted there).
			if (!m_upload_batch) {
				stats.breaks[kind] += m_scheduler.Current().IsRendering() ? 1 : 0;
			}
		}
	}
	if (source) {
		DrainStats::NoteFrameUpload(total_size);
	}
	m_last_sync_uploaded = static_cast<bool>(source);
	// Only read-only, non-texel uploads may wait for the batch: a written range is written by the
	// caller's own command, and a texel buffer's image copies below would land before the upload.
	if (source && m_upload_batch && !is_written && !is_texel_buffer) {
		ClampBufferCopies(copies, StagingCapacity(source),
		                  buffer.Size(), "upload", buffer.CpuAddress());
		if (!copies.empty()) {
			QueueUpload(source, buffer.Handle(), std::move(copies));
		}
	} else if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		GpuZones::Mark(native, DrainStats::Zone::BufferCopy, 2); // key 2: upload
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		ClampBufferCopies(copies, StagingCapacity(source),
		                  buffer.Size(), "upload", buffer.CpuAddress());
		if (!copies.empty()) {
			native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
			                  copies.data());
		}
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask =
		    LaterAccessOr(LaterSite::BufferCache, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
		if (stats_on) {
			t_sync_stats.record_ns[kind] += SyncStats::Now() - walked;
		}
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

// KYTY_UPLOAD_BATCH=1: a BDA pass's uploads behind one barrier pair (off by default until measured
// on a GPU-bound fight; the the title 40-fps arena records ~380 uploads per frame, 6.5 ms GPU).
// KYTY_DEBUG_AB=uploadbatch: one barrier pair per upload again in every other window (in-run A/B).
static bool UploadBatchEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_UPLOAD_BATCH");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("uploadbatch");
	return enabled && !(ab && AbFeatureOff());
}

BufferCache::UploadBatchScope::UploadBatchScope(BufferCache& cache)
    : m_cache(cache), m_previous(cache.m_upload_batch) {
	m_cache.m_upload_batch = UploadBatchEnabled();
}

BufferCache::UploadBatchScope::~UploadBatchScope() {
	m_cache.m_upload_batch = m_previous;
	if (!m_previous) {
		m_cache.FlushUploadBatch();
	}
}

void BufferCache::QueueUpload(vk::Buffer source, vk::Buffer destination,
                              std::vector<vk::BufferCopy>&& copies) {
	vk::DeviceSize begin = std::numeric_limits<vk::DeviceSize>::max();
	vk::DeviceSize end   = 0;
	for (const auto& copy: copies) {
		begin = std::min(begin, copy.dstOffset);
		end   = std::max(end, copy.dstOffset + copy.size);
	}
	// Copies of one batch run unordered. Each destination keeps the span of its queued copies;
	// a new range inside it (a page written twice) records the batch first. The dirty runs of a
	// pass come in ascending order, so this rarely fires.
	const auto key  = static_cast<VkBuffer>(destination);
	auto       span = m_pending_spans.find(key);
	if (span != m_pending_spans.end() && begin < span->second.second && span->second.first < end) {
		FlushUploadBatch();
		span = m_pending_spans.end();
	}
	if (span == m_pending_spans.end()) {
		m_pending_spans.emplace(key, std::pair {begin, end});
	} else {
		span->second = {std::min(span->second.first, begin), std::max(span->second.second, end)};
	}
	if (!m_pending_uploads.empty() && m_pending_uploads.back().source == source &&
	    m_pending_uploads.back().destination == destination) {
		auto& queued = m_pending_uploads.back().copies;
		queued.insert(queued.end(), copies.begin(), copies.end());
		return;
	}
	m_pending_uploads.push_back({source, destination, std::move(copies)});
}

void BufferCache::FlushUploadBatch() {
	if (m_pending_uploads.empty()) {
		return;
	}
	const bool     stats_on = SyncStats::Enabled();
	const uint64_t start    = stats_on ? SyncStats::Now() : 0;
	auto&          command  = m_scheduler.Current();
	if (stats_on) {
		t_sync_stats.breaks[0] += command.IsRendering() ? 1 : 0;
	}
	command.EndRendering();
	const auto native = command.Handle();
	GpuZones::Mark(native, DrainStats::Zone::BufferCopy, 2); // key 2: upload
	// The scopes of SynchronizeBuffer's buffer barriers, for all memory: no weaker than one pair
	// per destination.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion, 1,
	                       &before, 0, nullptr, 0, nullptr);
	for (const auto& upload: m_pending_uploads) {
		native.copyBuffer(upload.source, upload.destination,
		                  static_cast<uint32_t>(upload.copies.size()), upload.copies.data());
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask =
	    LaterAccessOr(LaterSite::BufferCache, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eByRegion,
	                       1, &after, 0, nullptr, 0, nullptr);
	m_pending_uploads.clear();
	m_pending_spans.clear();
	if (stats_on) {
		t_sync_stats.record_ns[0] += SyncStats::Now() - start; // Batches come from BDA passes.
	}
}

// Large uploads (the BDA synchronization copies about 600 MiB a second in ASTRO BOT's Sky Garden,
// a few hundred KiB at a time) are copied into the staging buffer by a helper thread while the GPU
// thread goes on recording; CommandScheduler::Submit waits for them (JoinAsyncUploads) before the
// commands that read the staging memory reach the GPU. The helper reads the backing, which cannot
// fault. KYTY_ASYNC_UPLOADS=0 copies on the GPU thread as before.
namespace {
// The tracker whose page history the helper fills (KYTY_PAGE_HISTORY; set by the buffer cache).
std::atomic<MemoryTracker*> g_history_tracker {nullptr};
// The helper thread runs outside the guest fault handling: memory the game unmapped after the
// upload was queued (seen in Ghost of Yotei after the calligraphy scene) must not be touched.
// Readable pages are copied, the rest of the range reads as zero.
void CopyReadableOrZero(uint64_t address, uint8_t* destination, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	uint64_t done = 0;
	while (done < size) {
		MEMORY_BASIC_INFORMATION info {};
		const auto current = address + done;
		if (VirtualQuery(reinterpret_cast<const void*>(current), &info, sizeof(info)) == 0) {
			std::memset(destination + done, 0, static_cast<size_t>(size - done));
			return;
		}
		const auto region_end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		const auto chunk      = std::min(size - done, region_end - current);
		const bool readable =
		    info.State == MEM_COMMIT && (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0 &&
		    (info.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
		                     PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
		if (readable) {
			// The game can unmap the range between the query and the copy (Ghost of Yotei after the
			// first calligraphy name): ReadProcessMemory fails instead of faulting, and what it
			// could not read reads as zero.
			SIZE_T got = 0;
			if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(current),
			                      destination + done, static_cast<size_t>(chunk), &got) == 0 &&
			    got < chunk) {
				std::memset(destination + done + got, 0, static_cast<size_t>(chunk - got));
			}
		} else {
			std::memset(destination + done, 0, static_cast<size_t>(chunk));
		}
		done += chunk;
	}
#else
	std::memcpy(destination, reinterpret_cast<const void*>(address), size);
#endif
}

// KYTY_DEBUG_UPLOAD_RACE=1: each upload handed to the helper keeps a sample of the guest bytes as they
// were when the GPU thread queued it (16 dwords spread over the range); after the helper's copy the
// samples are compared. A difference = the game wrote the range between the command that reads it
// being recorded and the copy, so the GPU gets newer data than the command saw (the title haze). Prints
// the first 100 with address, size, offset, old/new dword, delay, consuming shader; then counts.
static bool UploadRaceCheck() {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_UPLOAD_RACE") != nullptr;
	return enabled;
}
struct UploadRaceSample {
	static constexpr uint32_t Count = 16;
	std::array<uint32_t, Count> values {};
	uint32_t                    taken    = 0;
	uint64_t                    queued   = 0; // steady_clock ns
	uint64_t                    shader   = 0;
	uint32_t                    source   = 0;
	uint64_t SampleOffset(uint64_t size, uint32_t i) const {
		const uint64_t dwords = size / 4u;
		return dwords <= Count ? uint64_t {i} * 4u : (dwords - 1u) * i / (Count - 1u) * 4u;
	}
};
static std::atomic<uint64_t> g_upload_race_hits {0};
static std::atomic<uint64_t> g_upload_race_checked {0};

struct AsyncUploads {
	struct Item {
		uint64_t address;
		uint8_t* destination;
		uint64_t size;
		UploadRaceSample race {};
	};
	std::mutex              mutex;
	std::condition_variable wake;
	std::vector<Item>       queue;
	std::atomic<uint64_t>   pending {0};
	// Items queued and copied since the start, in queue order (one worker copies them in order):
	// a ticket (`pushed` at some point) is done once `completed` reaches it.
	std::atomic<uint64_t>   pushed {0};
	std::atomic<uint64_t>   completed {0};
	std::atomic<bool>       hint {false};
	std::thread             thread;
	bool                    started = false;

	// KYTY_UPLOAD_HELPERS=2 (or more): other threads copy part of a large batch (pieces of at most
	// ChunkBytes). The GPU thread waits for the batch at its next submit (the title arena: ~7 % of it
	// spun there), so the copy time is on its path. A claim is (round << 32) | index: the helper
	// rebuilds `chunks` only after every claimed piece of the round was copied, and a claim of an
	// older round, or past the round's count, reads nothing.
	struct Chunk {
		uint64_t address;
		uint8_t* destination;
		uint64_t size;
	};
	static constexpr uint64_t ChunkBytes    = 256 * 1024;
	static constexpr uint64_t AssistMinimum = 512 * 1024;
	std::vector<Chunk>        chunks;
	std::atomic<uint64_t>     claim {0};
	std::atomic<uint64_t>     limit {0}; // (round << 32) | chunk count
	std::atomic<uint64_t>     copied {0};
	std::atomic<uint64_t>     round {0};
	bool                      assist_started = false;

	// KYTY_UPLOAD_HELPERS=<1..4>: the helper plus that many minus one assisting threads.
	static uint32_t Helpers() {
		static const uint32_t helpers = [] {
			const char* text = KYTY_GETENV("KYTY_UPLOAD_HELPERS");
			return text != nullptr
			           ? std::clamp<uint32_t>(static_cast<uint32_t>(std::strtoul(text, nullptr, 10)),
			                                  1u, 4u)
			           : 1u;
		}();
		return helpers;
	}

	// KYTY_UPLOAD_JOIN_ASSIST=1: a thread waiting for the copies (the GPU thread at its submit)
	// copies pieces of the current batch itself instead of spinning; no extra thread.
	static bool JoinAssist() {
		static const bool enabled = [] {
			const char* text = KYTY_GETENV("KYTY_UPLOAD_JOIN_ASSIST");
			return text != nullptr && std::strcmp(text, "0") != 0;
		}();
		return enabled;
	}

	// KYTY_DEBUG_AB=joinassist: the waiting thread only spins in every other window (safe to
	// toggle: only who copies changes).
	static bool JoinAssistAb() {
		static const bool ab = AbSelected("joinassist");
		return ab;
	}

	static void CopyChunk(const Chunk& chunk) {
		if (!Libs::LibKernel::Memory::TryReadBacking(chunk.address, chunk.destination, chunk.size)) {
			CopyReadableOrZero(chunk.address, chunk.destination, chunk.size);
		}
	}

	// KYTY_PAGE_HISTORY: reads the range in 64 KiB pieces and hashes each whole page into its
	// PageHistory entry (unchanged count, pass of the last change). Stops (false) as soon as new
	// uploads are queued (`busy`): the helper's copies come first.
	static bool HashIntoHistory(uint64_t base, uint64_t total, MemoryTracker& tracker,
	                            const std::atomic<bool>& busy) {
		constexpr uint64_t       Piece = 64 * 1024;
		alignas(64) static thread_local std::array<uint8_t, Piece> bounce;
		const auto pass = g_bda_pass_serial.load(std::memory_order_relaxed);
		for (uint64_t done = 0; done < total;) {
			if (busy.load(std::memory_order_acquire)) {
				return false;
			}
			const auto address = base + done;
			const auto size    = std::min(Piece, total - done);
			if (!Libs::LibKernel::Memory::TryReadBacking(address, bounce.data(), size)) {
				CopyReadableOrZero(address, bounce.data(), size);
			}
			for (uint64_t page = Common::AlignUp(address, TRACKER_PAGE_SIZE);
			     page + TRACKER_PAGE_SIZE <= address + size; page += TRACKER_PAGE_SIZE) {
				auto* history = tracker.FindHistory(page);
				if (history == nullptr) {
					continue;
				}
				g_history_hashed.fetch_add(1, std::memory_order_relaxed);
				auto hash = XXH3_64bits(bounce.data() + (page - address), TRACKER_PAGE_SIZE);
				hash      = hash != 0 ? hash : 1;
				const auto index =
				    static_cast<size_t>((page % TRACKER_REGION_SIZE) / TRACKER_PAGE_SIZE);
				if (history->hash[index].load(std::memory_order_relaxed) == hash) {
					const auto same = history->same[index].load(std::memory_order_relaxed);
					history->same[index].store(static_cast<uint8_t>(same < 255 ? same + 1 : same),
					                           std::memory_order_relaxed);
				} else {
					history->hash[index].store(hash, std::memory_order_relaxed);
					history->same[index].store(0, std::memory_order_relaxed);
					history->changed_pass[index].store(pass, std::memory_order_relaxed);
				}
			}
			done += size;
		}
		return true;
	}

	// All threads: copies pieces of the current round until none is left. The claim must come
	// before the limit load: only then is the limit seen at least the claim's round (a later limit
	// load could reject a valid piece, which nobody would copy, and the helper would wait forever).
	void CopyChunks() {
		for (;;) {
			const auto value = claim.fetch_add(1, std::memory_order_acq_rel);
			const auto lim   = limit.load(std::memory_order_acquire);
			if ((value >> 32u) != (lim >> 32u) || (value & 0xffffffffu) >= (lim & 0xffffffffu)) {
				return;
			}
			CopyChunk(chunks[value & 0xffffffffu]);
			copied.fetch_add(1, std::memory_order_acq_rel);
		}
	}

	void Assist() {
		KYTY_PROFILER_THREAD("UploadAssist");
		uint64_t seen = 0;
		for (;;) {
			for (uint32_t i = 0; i < 20000 && round.load(std::memory_order_acquire) == seen; i++) {
				_mm_pause();
			}
			round.wait(seen, std::memory_order_acquire);
			seen = round.load(std::memory_order_acquire);
			CopyChunks();
		}
	}

	// A waiting thread joins the current round only while it has unclaimed pieces (no claim churn
	// between rounds).
	void AssistIfOpen() {
		const auto lim = limit.load(std::memory_order_acquire);
		const auto now = claim.load(std::memory_order_acquire);
		if ((now >> 32u) == (lim >> 32u) && (now & 0xffffffffu) < (lim & 0xffffffffu)) {
			CopyChunks();
		}
	}

	// The helper: copies `taken` together with the assisting thread.
	void CopyAssisted(const std::vector<Item>& taken) {
		chunks.clear();
		for (const auto& item: taken) {
			for (uint64_t offset = 0; offset < item.size; offset += ChunkBytes) {
				chunks.push_back({item.address + offset, item.destination + offset,
				                  std::min(ChunkBytes, item.size - offset)});
			}
		}
		const auto next = round.load(std::memory_order_relaxed) + 1;
		copied.store(0, std::memory_order_relaxed);
		limit.store((next << 32u) | chunks.size(), std::memory_order_release);
		claim.store(next << 32u, std::memory_order_release);
		if (!assist_started) {
			assist_started = true;
			for (uint32_t i = 1; i < Helpers(); i++) {
				std::thread([this] { Assist(); }).detach();
			}
		}
		round.store(next, std::memory_order_release);
		round.notify_all();
		CopyChunks();
		while (copied.load(std::memory_order_acquire) < chunks.size()) {
			_mm_pause();
		}
	}

	static bool Enabled() {
		static const bool enabled = [] {
			const char* text = KYTY_GETENV("KYTY_ASYNC_UPLOADS");
			return text == nullptr || std::strcmp(text, "0") != 0;
		}();
		return enabled;
	}

	void Push(std::span<const Item> items) {
		pending.fetch_add(items.size(), std::memory_order_acq_rel);
		{
			std::scoped_lock lock {mutex};
			queue.insert(queue.end(), items.begin(), items.end());
			pushed.fetch_add(items.size(), std::memory_order_acq_rel);
			if (!started) {
				started = true;
				thread  = std::thread([this] { Run(); });
				thread.detach();
			}
		}
		hint.store(true, std::memory_order_release);
		wake.notify_one();
	}

	void Run() {
		KYTY_PROFILER_THREAD("UploadHelper");
		std::vector<Item> taken;
		for (;;) {
			// Spin briefly for the next upload: they come in bursts, a few per synchronization.
			// KYTY_DEBUG_AB=uploadspin sleeps at once in every other window (the spin shares cores
			// with the GPU thread).
			static const bool spin_ab = AbSelected("uploadspin");
			const uint32_t    spins   = spin_ab && AbFeatureOff() ? 0u : 20000u;
			for (uint32_t i = 0; i < spins && !hint.load(std::memory_order_acquire); i++) {
				_mm_pause();
			}
			{
				std::unique_lock lock {mutex};
				wake.wait(lock, [this] { return !queue.empty(); });
				taken.swap(queue);
				hint.store(false, std::memory_order_release);
			}
			// The protections of the pages first: a write landing before them is in the copy.
			PageManager::RunDeferredProtects();
			uint64_t total = 0;
			for (const auto& item: taken) {
				total += item.size;
			}
			// KYTY_DEBUG_AB=uploadhelpers: one thread copies in every other window.
			static const bool ab       = AbSelected("uploadhelpers");
			const bool        assisted = (Helpers() > 1 || JoinAssist()) && total >= AssistMinimum &&
			                      !(ab && AbFeatureOff());
			if (assisted) {
				CopyAssisted(taken);
			}
			for (const auto& item: taken) {
				if (!assisted) {
					CopyChunk({item.address, item.destination, item.size});
				}
				if (item.race.taken != 0) [[unlikely]] {
					CheckRace(item);
				}
				completed.fetch_add(1, std::memory_order_acq_rel);
				pending.fetch_sub(1, std::memory_order_acq_rel);
			}
			// KYTY_PAGE_HISTORY: hashed after the copies, while no new upload waits (nobody joins on
			// it; a page whose content changed since the copy only steers tracking).
			if (auto* history_tracker = g_history_tracker.load(std::memory_order_acquire);
			    history_tracker != nullptr && HistoryPoliciesOn()) {
				g_history_copied.fetch_add(total / TRACKER_PAGE_SIZE, std::memory_order_relaxed);
				for (const auto& item: taken) {
					if (!HashIntoHistory(item.address, item.size, *history_tracker, hint)) {
						break;
					}
				}
			}
			taken.clear();
		}
	}

	static void CheckRace(const Item& item) {
		const auto checked = g_upload_race_checked.fetch_add(1, std::memory_order_relaxed) + 1;
		for (uint32_t i = 0; i < item.race.taken; i++) {
			const auto offset = item.race.SampleOffset(item.size, i);
			uint32_t   now    = 0;
			std::memcpy(&now, item.destination + offset, sizeof(now));
			if (now == item.race.values[i]) {
				continue;
			}
			const auto hits = g_upload_race_hits.fetch_add(1, std::memory_order_relaxed) + 1;
			if (hits <= 100 || hits % 1000 == 0) {
				const auto delay_us =
				    (static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                               std::chrono::steady_clock::now().time_since_epoch())
				                               .count()) -
				     item.race.queued) /
				    1000u;
				std::printf("upload-race #%" PRIu64 " (of %" PRIu64 " checked): 0x%016" PRIx64
				            " size=0x%" PRIx64 " offset=0x%" PRIx64 " old=0x%08x new=0x%08x delay=%" PRIu64
				            "us shader=%016" PRIx64 " source=%u\n",
				            hits, checked, item.address, item.size, offset, item.race.values[i], now,
				            delay_us, item.race.shader, item.race.source);
				std::fflush(stdout);
			}
			return;
		}
	}

	// Asked once per join, not per spin: with KYTY_DEBUG_AB_OFF_FILE, AbFeatureOff() looked at the file
	// every 256 calls, which inside the spin loop slowed every join (run uh1).
	static bool AssistJoin() { return JoinAssist() && !(JoinAssistAb() && AbFeatureOff()); }

	void Join() noexcept {
		const bool assist = AssistJoin();
		for (uint32_t i = 1; pending.load(std::memory_order_acquire) != 0; i++) {
			if (assist) {
				AssistIfOpen();
			}
			_mm_pause();
			if (i % 4096 == 0) {
				std::this_thread::yield();
			}
		}
	}

	void Join(uint64_t ticket) noexcept {
		const bool assist = AssistJoin();
		for (uint32_t i = 1; completed.load(std::memory_order_acquire) < ticket; i++) {
			if (assist) {
				AssistIfOpen();
			}
			_mm_pause();
			if (i % 4096 == 0) {
				std::this_thread::yield();
			}
		}
	}
};
AsyncUploads g_async_uploads;
} // namespace

static void EnableUploadHistory(MemoryTracker& tracker) {
	g_history_tracker.store(&tracker, std::memory_order_release);
	std::printf("page-history: on (hot demote after %u unchanged uploads, window history %s)\n",
	            HotDemoteLimit(), KYTY_GETENV("KYTY_WINDOW_HISTORY") != nullptr ? "on" : "off");
	std::fflush(stdout);
}

uint64_t AsyncUploadTicket() noexcept {
	return g_async_uploads.pushed.load(std::memory_order_acquire);
}

uint64_t AsyncUploadsPending() noexcept {
	return g_async_uploads.pending.load(std::memory_order_acquire);
}

void JoinAsyncUploads(uint64_t ticket) noexcept {
	g_async_uploads.Join(ticket);
}

// KYTY_DEFER_PROTECTS=0 protects pages on the GPU thread as before.
static bool DeferProtectsEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEFER_PROTECTS");
		return (text == nullptr || std::strcmp(text, "0") != 0) && AsyncUploads::Enabled();
	}();
	return enabled;
}

void JoinAsyncUploads() noexcept {
	g_async_uploads.Join();
}

// KYTY_DEBUG_HOST_IMPORT_TEST=1: once, tries to import 64 KiB of guest memory at `address` as the
// memory of a storage buffer (VK_EXT_external_memory_host), both as a host allocation and as
// mapped foreign memory, and prints what the driver answers: whether guest memory could back
// buffers directly instead of being copied.
static void DebugHostImportTest(GraphicContext& graphics, uint64_t address) {
	static bool done = KYTY_GETENV("KYTY_DEBUG_HOST_IMPORT_TEST") == nullptr;
	if (done) {
		return;
	}
	done = true;
	constexpr uint64_t Size = 64 * 1024;
	// Pass 0: guest memory as it is. Pass 1: the same after making it writable. Pass 2: private
	// memory of the emulator, to see whether the driver imports anything at all.
	for (int pass = 0; pass < 3; pass++) {
	auto* pointer = reinterpret_cast<void*>(address & ~uint64_t {0xfff});
	if (pass == 1) {
		DWORD old = 0;
		(void)VirtualProtect(pointer, Size, PAGE_READWRITE, &old);
	} else if (pass == 2) {
		pointer = VirtualAlloc(nullptr, Size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
	}
	MEMORY_BASIC_INFORMATION info {};
	(void)VirtualQuery(pointer, &info, sizeof(info));
	std::printf("host-import: pass=%d protect=0x%lx type=0x%lx state=0x%lx\n", pass, info.Protect,
	            info.Type, info.State);
	// Variants: with and without a device address, and each memory type the driver offers.
	for (int variant = 0; variant < 4; variant++) {
		const auto type           = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
		const bool device_address = (variant & 1) == 0;
		const bool second_type    = (variant & 2) != 0;
		vk::MemoryHostPointerPropertiesEXT properties {};
		const auto queried =
		    graphics.device.getMemoryHostPointerPropertiesEXT(type, pointer, &properties);
		vk::ExternalMemoryBufferCreateInfo external {};
		external.handleTypes = type;
		vk::BufferCreateInfo create {};
		create.pNext = &external;
		create.size  = Size;
		create.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
		               vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
		if (device_address) {
			create.usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
		}
		graphics.ShareBuffer(create);
		vk::Buffer buffer;
		const auto created = graphics.device.createBuffer(&create, nullptr, &buffer);
		vk::Result allocated = vk::Result::eErrorUnknown;
		vk::Result bound     = vk::Result::eErrorUnknown;
		uint32_t   required  = 0;
		int        index     = -1;
		if (created == vk::Result::eSuccess) {
			const auto requirements = graphics.device.getBufferMemoryRequirements(buffer);
			required                = requirements.memoryTypeBits;
			const auto usable       = requirements.memoryTypeBits & properties.memoryTypeBits;
			if (queried == vk::Result::eSuccess && usable != 0) {
				index = std::countr_zero(usable);
				if (second_type && (usable & (usable - 1u)) != 0) {
					index = std::countr_zero(usable & (usable - 1u));
				}
				vk::ImportMemoryHostPointerInfoEXT import {};
				import.handleType   = type;
				import.pHostPointer = pointer;
				vk::MemoryAllocateFlagsInfo flags {};
				flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
				flags.pNext = &import;
				vk::MemoryAllocateInfo allocate {};
				allocate.pNext = device_address ? static_cast<const void*>(&flags)
				                                : static_cast<const void*>(&import);
				allocate.allocationSize  = Size;
				allocate.memoryTypeIndex = static_cast<uint32_t>(index);
				vk::DeviceMemory memory;
				allocated = graphics.device.allocateMemory(&allocate, nullptr, &memory);
				if (allocated == vk::Result::eSuccess) {
					bound = graphics.device.bindBufferMemory(buffer, memory, 0);
				}
			}
		}
		std::printf("host-import: variant=%d address=0x%016" PRIx64 " type=%s query=%s types=0x%x buffer=%s "
		            "required=0x%x index=%d allocate=%s bind=%s\n",
		            variant, reinterpret_cast<uint64_t>(pointer), vk::to_string(type).c_str(),
		            vk::to_string(queried).c_str(), properties.memoryTypeBits,
		            vk::to_string(created).c_str(), required, index, vk::to_string(allocated).c_str(),
		            vk::to_string(bound).c_str());
		std::fflush(stdout);
	}
	}
}

// KYTY_DEBUG_UPLOAD_AGE=1: upload bytes per 5 s by source (BDA pass or binding) and by the frames
// since a draw or dispatch last bound the buffer (bindings touch before they upload: age 0). Large
// old-age BDA shares = memory of buffers nothing binds any more, re-uploaded after CPU writes
// (upstream pull request 1809's finding).
static void DebugUploadAge(bool bda, uint64_t age, bool streamed, uint64_t bytes,
                           uint64_t buffer_address, uint64_t buffer_size) {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_UPLOAD_AGE") != nullptr;
	if (!enabled) {
		return;
	}
	static constexpr std::array<uint64_t, 5> Limits {1, 2, 10, 100, ~uint64_t {0}};
	static constexpr std::array<const char*, 5> Names {"0", "1", "2-9", "10-99", "100+"};
	struct State {
		std::mutex                                       mutex;
		std::array<std::array<uint64_t, 5>, 2>           bytes {};
		std::array<std::array<uint64_t, 5>, 2>           copies {};
		std::array<uint64_t, 2>                          streamed {}; // owner stream-bound this/last frame
		std::unordered_map<uint64_t, uint64_t>           old_buffers; // address -> size, age >= 100
		std::unordered_map<uint64_t, uint64_t>           all_buffers;
		std::chrono::steady_clock::time_point            window = std::chrono::steady_clock::now();
	};
	static State     state;
	std::scoped_lock lock {state.mutex};
	size_t           bucket = 0;
	while (age >= Limits[bucket]) {
		bucket++;
	}
	state.bytes[bda ? 0 : 1][bucket] += bytes;
	state.copies[bda ? 0 : 1][bucket]++;
	state.streamed[bda ? 0 : 1] += streamed ? bytes : 0;
	state.all_buffers[buffer_address] = buffer_size;
	if (bucket == 4) {
		state.old_buffers[buffer_address] = buffer_size;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - state.window < std::chrono::seconds(5)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - state.window).count();
	for (int kind = 0; kind < 2; kind++) {
		std::string line = fmt::format("upload-age: {} MB/s", kind == 0 ? "bda " : "bind");
		for (size_t i = 0; i < Limits.size(); i++) {
			line += fmt::format(" {}={:.1f}/{:.0f}c", Names[i],
			                    static_cast<double>(state.bytes[kind][i]) / 1e6 / seconds,
			                    static_cast<double>(state.copies[kind][i]) / seconds);
		}
		line += fmt::format(" | owner stream-bound={:.1f}", static_cast<double>(state.streamed[kind]) / 1e6 / seconds);
		std::printf("%s\n", line.c_str());
	}
	uint64_t old_bytes = 0;
	for (const auto& [address, size]: state.old_buffers) {
		old_bytes += size;
	}
	std::printf("upload-age: buffers uploaded=%zu old(>=100 frames unbound)=%zu old size=%.1f MB\n",
	            state.all_buffers.size(), state.old_buffers.size(), static_cast<double>(old_bytes) / 1e6);
	std::fflush(stdout);
	state.bytes  = {};
	state.copies = {};
	state.streamed = {};
	state.old_buffers.clear();
	state.all_buffers.clear();
	state.window = now;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size, bool deferred) {
	DebugHostImportTest(m_graphics, buffer.CpuAddress());
	if (!copies.empty()) {
		DebugUploadAge(t_upload_source == UploadSource::Bda, m_gc_tick - buffer.lru_tick,
		               buffer.stream_tick != 0 && m_gc_tick - buffer.stream_tick <= 1, total_size,
		               buffer.CpuAddress(), buffer.Size());
	}
	if (copies.empty()) {
		if (deferred) {
			// No copy will run them: a queued protection must not wait for some later upload.
			PageManager::RunDeferredProtects();
		}
		return nullptr;
	}
	RecordUpload(t_upload_source, buffer.CpuAddress() + copies.front().dstOffset, total_size);
	for (const auto& copy: copies) {
		RecordUploadPages(buffer.CpuAddress() + copy.dstOffset, copy.size);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	auto& staging = UploadStaging(total_size);
	auto [mapped, base_offset] = staging.Map(total_size, 4);
	constexpr uint64_t AsyncUploadMinimum = 64 * 1024;
	if (mapped != nullptr && (total_size >= AsyncUploadMinimum || deferred) &&
	    staging.IsCoherent() && AsyncUploads::Enabled()) {
		thread_local std::vector<AsyncUploads::Item> items;
		items.clear();
		for (auto& copy: copies) {
			items.push_back({buffer.CpuAddress() + copy.dstOffset, mapped + copy.srcOffset, copy.size});
			if (UploadRaceCheck()) [[unlikely]] {
				auto&      item  = items.back();
				const auto count = static_cast<uint32_t>(
				    std::min<uint64_t>(UploadRaceSample::Count, item.size / 4u));
				// One backing read per range (its lock once), then the samples from that copy.
				thread_local std::vector<uint8_t> snapshot;
				const bool whole = item.size <= 256u * 1024u;
				if (whole) {
					snapshot.resize(item.size);
				}
				const bool read = whole && Libs::LibKernel::Memory::TryReadBacking(
				                               item.address, snapshot.data(), item.size);
				for (uint32_t i = 0; i < count; i++) {
					const auto offset = item.race.SampleOffset(item.size, i);
					if (read) {
						std::memcpy(&item.race.values[i], snapshot.data() + offset, 4);
					} else {
						(void)Libs::LibKernel::Memory::TryReadBacking(item.address + offset,
						                                              &item.race.values[i], 4);
					}
				}
				item.race.taken  = count;
				item.race.queued = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(
				        std::chrono::steady_clock::now().time_since_epoch())
				        .count());
				item.race.shader = DrainStats::t_binding_shader;
				item.race.source = static_cast<uint32_t>(t_upload_source);
			}
			copy.srcOffset += base_offset;
		}
		g_async_uploads.Push(items);
		staging.Commit();
		return staging.Handle();
	}
	if (deferred) {
		PageManager::RunDeferredProtects();
	}
	if (mapped != nullptr) {
		// Copying on several threads measured no faster (A/B 2026-09-28): the staging memory's
		// write bandwidth, not one thread's latency, bounds it.
		DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::UploadCopy);
		// A range the game unmapped since the tracker saw it, or a PRT aperture page that was never
		// committed (Yotei, 0x50d3400000 on the first start): read as zero instead of faulting
		// outside the guest fault handler.
		// KYTY_BDA_BACKING_FIRST=1: the backing first, as the helper thread reads it (one copy,
		// no VirtualQuery + ReadProcessMemory per range: ~1 ms per frame in the the title arena).
		static const bool backing_first = KYTY_GETENV("KYTY_BDA_BACKING_FIRST") != nullptr;
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!backing_first || !Libs::LibKernel::Memory::TryReadBacking(
			                          address, mapped + copy.srcOffset, copy.size)) {
				CopyReadableOrZero(address, mapped + copy.srcOffset, copy.size);
			}
			copy.srcOffset += base_offset;
		}
		if (&staging == m_staging_device.get()) {
			// Write-combined memory: drain this thread's WC buffers before the copy can be submitted
			// (the helper threads' locked completion counters do that for their copies).
			_mm_sfence();
		}
		staging.Commit();
		return staging.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		CopyReadableOrZero(address, temporary->Mapped().data() + copy.srcOffset, copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

// KYTY_DEBUG_AB=streamhost: 'off' windows copy into a second ring in device memory. The tiler keeps
// the ring it got at startup.
StreamBuffer& BufferCache::ActiveStream() noexcept {
	return m_stream_device != nullptr && AbFeatureOff() ? *m_stream_device : m_stream_buffer;
}

// KYTY_DEBUG_STREAM_REUSE=0 copies every stream range every time; KYTY_DEBUG_AB=streamreuse
// alternates.
static bool StreamReuseEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_STREAM_REUSE");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("streamreuse");
	return enabled && !(ab && AbFeatureOff());
}

// With KYTY_VERIFY_STREAM_REUSE=1 a copy that would be bound again is compared with the guest
// memory it came from, and the range is copied again as without reuse. It reports how often the
// bytes changed within the epoch: the draws that reuse would have shown older CPU writes.
bool BufferCache::VerifyStreamReuse(const StreamCopy& copy) {
	static const bool verify = KYTY_GETENV("KYTY_VERIFY_STREAM_REUSE") != nullptr;
	if (!verify) {
		return false;
	}
	thread_local std::vector<uint8_t> bytes;
	bytes.resize(copy.size);
	const bool  read = Libs::LibKernel::Memory::TryReadBacking(copy.vaddr, bytes.data(), copy.size);
	const auto* kept = copy.stream->Mapped().data() + copy.offset;
	static std::atomic<uint64_t> checked {0};
	static std::atomic<uint64_t> changed {0};
	const auto count = checked.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((!read || std::memcmp(bytes.data(), kept, copy.size) != 0) &&
	    changed.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::printf("stream-reuse verify: 0x%016" PRIx64 " size 0x%" PRIx64
		            " changed within the epoch\n",
		            copy.vaddr, copy.size);
	}
	if (count % 100000 == 0) {
		std::printf("stream-reuse verify: reuses=%" PRIu64 " changed=%" PRIu64 "\n", count,
		            changed.load(std::memory_order_relaxed));
		std::fflush(stdout);
	}
	return true;
}

namespace {
// KYTY_DEBUG_OBTAIN_STATS=1: every 5 s, how often ObtainBuffer took each path and its mean time
// (steady clock, so the times include the clock reads): where binding a draw's or dispatch's
// buffers goes.
struct ObtainStats {
	enum Path : uint32_t {
		Written,     // A written binding (GPU-dirty bookkeeping).
		Clean,       // Read, every 64 KiB slice CPU-clean: nothing to copy.
		TexelNone,   // ...a texel buffer, and no image starts at its address.
		TexelImage,  // ...a texel buffer an image starts on (SynchronizeBufferFromImage ran on).
		StreamReuse, // Read, CPU-dirty, small: this epoch's stream copy again.
		StreamCopy,  // Read, only CPU-modified, small: copied into the stream ring.
		SyncNone,    // Read, a slice CPU-dirty: the locked upload pass found nothing to copy.
		SyncUpload,  // ...and copied pages (UploadCopies).
		Count
	};
	static constexpr std::array<const char*, Count> Names {
	    "written", "clean", "texel-none", "texel-image", "stream-reuse", "stream-copy", "sync-none",
	    "sync-upload"};
	std::array<uint64_t, Count> calls {};
	std::array<uint64_t, Count> ns {};
	uint64_t                    total = 0;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	static bool Enabled() {
		static const bool enabled = KYTY_GETENV("KYTY_DEBUG_OBTAIN_STATS") != nullptr;
		return enabled;
	}
	void Note(uint32_t path, uint64_t elapsed_ns) {
		calls[path]++;
		ns[path] += elapsed_ns;
		if ((++total & 4095u) != 0 || std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
			return;
		}
		const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
		std::string line = "obtain-stats: calls/s (us each):";
		for (uint32_t i = 0; i < Count; i++) {
			line += fmt::format(" {}={:.0f} ({:.2f})", Names[i], static_cast<double>(calls[i]) / seconds,
			                    calls[i] != 0 ? static_cast<double>(ns[i]) / 1000.0 / static_cast<double>(calls[i])
			                                  : 0.0);
		}
		std::printf("%s\n", line.c_str());
		std::fflush(stdout);
		*this = {};
	}
};
ObtainStats g_obtain_stats;
} // namespace

// KYTY_DEBUG_LABEL_STATS (see bufferCache.h). GPU thread only.
struct LabelStats {
	std::map<uint64_t, uint32_t> labels; // Label address -> size, of the writes so far.
	std::array<uint64_t, 3>      writes {}; // Clean page, GPU-dirty page only, GPU-written bytes.
	uint64_t                     binding_reads     = 0; // Read bindings over label bytes.
	uint64_t                     binding_writes    = 0; // Written bindings over label bytes.
	uint64_t                     copy_reads        = 0; // Copies from label bytes.
	uint64_t                     copy_writes       = 0; // Copies into label bytes...
	uint64_t                     copy_writes_gpu   = 0; // ...from GPU-written bytes.
	uint64_t                     copy_writes_cpu   = 0; // ...done on the CPU (no GPU bytes).
	bool                         in_copy           = false;
	uint64_t                     example_read      = 0;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	[[nodiscard]] bool Overlaps(uint64_t vaddr, uint64_t size) const {
		auto next = labels.lower_bound(vaddr);
		if (next != labels.end() && next->first < vaddr + size) {
			return true;
		}
		if (next != labels.begin()) {
			const auto previous = std::prev(next);
			return previous->first + previous->second > vaddr;
		}
		return false;
	}
};
static LabelStats g_label_stats;

bool BufferCache::LabelStatsEnabled() {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_LABEL_STATS") != nullptr;
	return enabled;
}

void BufferCache::NoteLabelWrite(uint64_t vaddr, uint32_t size) {
	auto&      stats = g_label_stats;
	const auto kind  = !IsRegionGpuModified(vaddr, size) ? 0 : !HasGpuDirtyBytes(vaddr, size) ? 1 : 2;
	stats.writes[kind]++;
	// Labels on clean pages reach the cached buffer through the next upload anyway.
	if (kind != 0 && stats.labels.size() < 65536) {
		stats.labels[vaddr] = std::max(stats.labels[vaddr], size);
	}
	if (std::chrono::steady_clock::now() - stats.start < std::chrono::seconds(5)) {
		return;
	}
	const auto seconds =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - stats.start).count();
	const auto rate = [&](uint64_t count) { return static_cast<double>(count) / seconds; };
	std::printf("label-stats: per s: writes clean=%.0f gpu-page=%.0f gpu-bytes=%.0f | label bytes read by "
	            "bindings=%.0f written by bindings=%.0f copied from=%.0f copied into=%.0f (gpu src=%.0f "
	            "cpu=%.0f) | labels known=%zu example read=0x%016" PRIx64 "\n",
	            rate(stats.writes[0]), rate(stats.writes[1]), rate(stats.writes[2]),
	            rate(stats.binding_reads), rate(stats.binding_writes), rate(stats.copy_reads),
	            rate(stats.copy_writes), rate(stats.copy_writes_gpu), rate(stats.copy_writes_cpu),
	            stats.labels.size(), stats.example_read);
	std::fflush(stdout);
	auto labels = std::move(stats.labels);
	stats       = {};
	stats.labels = std::move(labels);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	m_last_no_image_stable = false;
	if (LabelStatsEnabled() && !g_label_stats.in_copy && g_label_stats.Overlaps(vaddr, size))
	    [[unlikely]] {
		(is_written ? g_label_stats.binding_writes : g_label_stats.binding_reads)++;
		if (!is_written) {
			g_label_stats.example_read = vaddr;
		}
	}
	struct StatsScope {
		bool                                  on;
		std::chrono::steady_clock::time_point start;
		uint32_t                              path = ObtainStats::Clean;
		~StatsScope() {
			if (on) [[unlikely]] {
				g_obtain_stats.Note(path, static_cast<uint64_t>(
				                              std::chrono::duration_cast<std::chrono::nanoseconds>(
				                                  std::chrono::steady_clock::now() - start)
				                                  .count()));
			}
		}
	} stats_scope {ObtainStats::Enabled(), ObtainStats::Enabled() ? std::chrono::steady_clock::now()
	                                                              : std::chrono::steady_clock::time_point {}};
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	// Draws bind most buffers read-only and unchanged since their last upload: the lock-free
	// summary then answers what the locked tracker passes below would find (nothing to copy).
	const bool cpu_clean = !is_written && m_memory_tracker.IsRegionCpuCleanHint(vaddr, size);
	// Draws of one BDA epoch need not see CPU writes made during it (see
	// RenderContext::PrepareBda), so a range this epoch already copied into the stream ring is
	// bound again rather than copied again (Sky Garden binds most such ranges several times an
	// epoch): while the copy's command buffer is current and no buffer or image became
	// GPU-written since. KYTY_DEBUG_STREAM_REUSE=0 copies every time.
	StreamCopy* copy_slot = nullptr;
	uint64_t    epoch     = 0;
	if (!is_written && !cpu_clean && size <= CACHING_PAGESIZE && StreamReuseEnabled()) {
		epoch     = m_scheduler.Context().CurrentBdaEpoch();
		copy_slot = &m_stream_copies[((vaddr >> 4u) ^ (vaddr >> 16u) ^ size) % StreamCopySlots];
		const auto& copy = *copy_slot;
		if (epoch != 0 && copy.epoch == epoch && copy.vaddr == vaddr && copy.size == size &&
		    copy.tick == m_scheduler.CurrentTick() && copy.gpu_writes == m_gpu_dirty_generation &&
		    copy.image_writes == m_texture_cache.GpuModifiedGeneration() &&
		    copy.stream == &ActiveStream()) {
			if (!VerifyStreamReuse(copy)) {
				stats_scope.path = ObtainStats::StreamReuse;
				TouchStreamOwner(id, vaddr, size);
				return {copy.stream, copy.offset};
			}
		}
	}
	if (!is_written && !cpu_clean && size <= CACHING_PAGESIZE &&
	    m_memory_tracker.IsRegionOnlyCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::StreamCopy);
		auto& stream          = ActiveStream();
		auto [mapped, offset] = stream.Map(size, alignment, false);
		if (mapped != nullptr) {
			// No GPU-written bytes in the range (IsRegionOnlyCpuModified), so no read protection
			// either: a plain copy from guest memory, without the backing-store lookup.
			std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
			stream.Commit();
			RecordUpload(UploadSource::Stream, vaddr, size);
			if (copy_slot != nullptr && epoch != 0) {
				*copy_slot = {.vaddr        = vaddr,
				              .size         = size,
				              .epoch        = epoch,
				              .tick         = m_scheduler.CurrentTick(),
				              .gpu_writes   = m_gpu_dirty_generation,
				              .image_writes = m_texture_cache.GpuModifiedGeneration(),
				              .stream       = &stream,
				              .offset       = offset};
			}
			stats_scope.path = ObtainStats::StreamCopy;
			TouchStreamOwner(id, vaddr, size);
			return {&stream, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	m_last_sync_uploaded = false;
	m_last_image_found   = false;
	if (!cpu_clean) {
		(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
		stats_scope.path = m_last_sync_uploaded ? ObtainStats::SyncUpload : ObtainStats::SyncNone;
	} else if (is_texel_buffer) {
		// SynchronizeBuffer's image half: the CPU upload has nothing to copy.
		(void)SynchronizeBufferFromImage(buffer, vaddr, size);
		stats_scope.path = m_last_image_found ? ObtainStats::TexelImage : ObtainStats::TexelNone;
	}
	if (is_written) {
		stats_scope.path = ObtainStats::Written;
		NoteAddrWatchUpload(vaddr, size, cpu_clean ? "gpu-write-mark(cpu-clean)" : "gpu-write-mark");
		WatchGpuWrite(vaddr, size);
		if (m_gpu_modified_ranges.Add(vaddr, size)) {
			m_gpu_dirty_generation++;
		}
		m_texture_cache.OnBufferGpuWrite(vaddr, size);
		if (m_readback_queue != nullptr || ReadbackAgeEnabled()) [[unlikely]] {
			NoteGpuWrite(vaddr, size);
		}
		if (!m_hot_pages.empty()) {
			QueueEagerReadback(vaddr, size);
		}
		if (DrainStats::Enabled()) {
			DrainStats::NoteQueueWrite(vaddr, size);
			DrainStats::RecordGpuWrite(vaddr, size);
			if (m_texture_cache.IsKnownDccMetadata(vaddr, size)) {
				DrainStats::Record(DrainStats::Kind::DccMetaWrite, size);
			}
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

// Streamed textures (Sky Garden: 2048x2048 and 4096x4096 BC textures, 5-22 MB) are written by
// the CPU a part at a time, and each draw that finds one changed re-uploads it whole, several
// times a second, though only ~1-6% of its bytes changed (KYTY_DEBUG_UPLOADS, 2026-09-28). From
// the second staged upload of a range within two seconds on, a buffer holds the range instead:
// the memory tracker then copies only the pages written since, and the image uploads from the
// buffer. KYTY_DEBUG_AB=imagebuf stages every upload in alternate windows.
bool BufferCache::IsRepeatedImageStage(uint64_t vaddr, uint64_t size) {
	static constexpr uint64_t MinSize = 1024 * 1024;
	static constexpr auto     Window  = std::chrono::seconds(2);
	static const bool         ab      = AbSelected("imagebuf");
	if (size < MinSize || (ab && AbFeatureOff())) {
		return false;
	}
	const auto now   = std::chrono::steady_clock::now();
	auto&      stage = m_image_stages[(vaddr >> 16u) % m_image_stages.size()];
	const bool hit   = stage.vaddr == vaddr && stage.size == size && now - stage.time <= Window;
	stage            = {.vaddr = vaddr, .size = size, .time = now};
	return hit;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const UploadSourceScope upload_source(UploadSource::Image);
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size) || IsRepeatedImageStage(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr) {
		EXIT("BufferCache: failed to map staging for guest image backing\n");
	}
	if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		// Part of the range is unmapped (Ghost of Yotei: a texture whose layout reaches past its
		// allocation). Read the mapped 4 KiB pages and leave the rest zero.
		static constexpr uint64_t Page = 0x1000;
		auto* bytes = static_cast<uint8_t*>(staging);
		std::memset(bytes, 0, size);
		uint64_t missing = 0;
		for (uint64_t done = 0; done < size;) {
			const auto count = std::min(size - done, Page - ((vaddr + done) & (Page - 1u)));
			if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr + done, bytes + done, count)) {
				std::memset(bytes + done, 0, count);
				missing += count;
			}
			done += count;
		}
		static std::atomic<uint32_t> reports {0};
		if (reports.fetch_add(1, std::memory_order_relaxed) < 8u) {
			std::printf("Warning: image source 0x%016" PRIx64 " size=0x%" PRIx64
			            " is partly unmapped (0x%" PRIx64 " bytes read as zero)\n",
			            vaddr, size, missing);
		}
	}
	m_staging_buffer.Commit();
	RecordUpload(UploadSource::Image, vaddr, size);
	RecordImageChunks(vaddr, size);
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	NoteAddrWatchUpload(vaddr, size, "fill");
	// KYTY_DEBUG_GDS_LOG=1: GDS fills too (see CopyBuffer).
	static const bool gds_log = KYTY_GETENV("KYTY_DEBUG_GDS_LOG") != nullptr;
	if (gds_log && is_gds) [[unlikely]] {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 200000) {
			std::printf("gds-fill: gds:0x%" PRIx64 " size=0x%" PRIx64 " value=0x%x\n", vaddr, size, value);
			std::fflush(stdout);
		}
	}
	// KYTY_DEBUG_FILL_LOG=<hex value>: every GPU fill with that value (address, size, gds), to match
	// a sentinel later found inside another structure (the title cloth: 0x7fffffff).
	static const int64_t fill_log = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_FILL_LOG");
		return text != nullptr ? static_cast<int64_t>(std::strtoull(text, nullptr, 16)) : int64_t {-1};
	}();
	if (fill_log >= 0 && value == static_cast<uint32_t>(fill_log)) [[unlikely]] {
		static std::atomic<uint32_t> lines {0};
		if (lines.fetch_add(1, std::memory_order_relaxed) < 3000) {
			std::printf("fill-log: 0x%08x [0x%016" PRIx64 "+0x%llx] gds=%d\n", value, vaddr,
			            static_cast<unsigned long long>(size), is_gds ? 1 : 0);
			std::fflush(stdout);
		}
	}
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	// While KYTY_DEBUG_FRAME_PASSES records: every CP fill, whether it cleared a known metadata
	// surface and which path it takes (CPU mapping or GPU buffer), so a clear that the pass list
	// cannot show (no draw or dispatch) is visible next to it (the title flow-state region map).
	// An HTile word with SMEM (bits 9:8) 0 also clears the stencil kept in that HTile (ClearMeta).
	const bool meta_hit = m_texture_cache.ClearMeta(vaddr, ((value >> 8u) & 3u) == 0);
	if (FramePassesRecording()) [[unlikely]] {
		std::printf("frame-pass cp-fill 0x%" PRIx64 "+0x%" PRIx64 " value=%08x meta=%d gpu_mod=%d\n",
		            vaddr, size, value, meta_hit ? 1 : 0, IsRegionGpuModified(vaddr, size) ? 1 : 0);
	}
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images. A page
		// that is already CPU-dirty does not fault: start a BDA epoch either way.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		m_scheduler.Context().AdvanceBdaEpochForGpuWrite();
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, TextureCache::GpuWriteSource::Fill);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	OnCommandRecorded();
}

void BufferCache::MirrorLabel(uint64_t vaddr, const void* data, uint32_t size) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0) {
		EXIT("BufferCache: label mirror must be dword aligned\n");
	}
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, false, true);
	if (m_graphics.buffer_marker_enabled) {
		// As the PS5 writes a label: once all earlier commands are done, without holding later
		// ones. The fills' AllCommands barriers stalled the GPU at every label (~300 a frame in
		// The title, +2.4 ms GPU). Later accesses of these bytes order after a marker through their
		// own AllCommands barriers, as after a transfer write.
		const auto native = m_scheduler.Current().Handle();
		for (uint32_t offset = 0; offset < size; offset += 4) {
			uint32_t word = 0;
			std::memcpy(&word, static_cast<const uint8_t*>(data) + offset, sizeof(word));
			native.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, dst->Handle(),
			                            dst_offset + offset, word);
		}
		OnCommandRecorded();
		return;
	}
	for (uint32_t offset = 0; offset < size; offset += 4) {
		uint32_t word = 0;
		std::memcpy(&word, static_cast<const uint8_t*>(data) + offset, sizeof(word));
		dst->Fill(dst_offset + offset, sizeof(word), word);
	}
	OnCommandRecorded();
}

bool BufferCache::CanWriteBacked(uint64_t vaddr, uint64_t size, bool cpu_images, uint64_t max_size) {
	return size <= max_size && ((vaddr | size) & 3u) == 0 && IsRegionGpuModified(vaddr, size) &&
	       !HasGpuDirtyBytes(vaddr, size) &&
	       (cpu_images ? m_texture_cache.OnlyCpuWrittenImages(vaddr, size)
	                   : !m_texture_cache.IsRegionRegistered(vaddr, size));
}

void BufferCache::InvalidateImagesForWrite(uint64_t vaddr, uint64_t size) {
	m_texture_cache.InvalidateMemory(vaddr, size);
}

void BufferCache::WriteBacked(uint64_t vaddr, const void* data, uint32_t size) {
	NoteAddrWatchUpload(vaddr, size, "write-backed");
	// Upload copies still queued read the backing later: they belong to commands recorded before
	// this write and must not see it.
	JoinAsyncUploads();
	LibKernel::Memory::WriteBacking(vaddr, data, size);
	MirrorLabel(vaddr, data, size);
	// As the fault path's CPU write did: later draws must not reuse reads from before it.
	m_scheduler.Context().AdvanceBdaEpochForGpuWrite();
}

bool BufferCache::CanOverwriteGpuBytes(uint64_t vaddr, uint64_t size, bool cpu_images, bool dword_runs) {
	// Labels are 4 or 8 bytes, naturally aligned: they stay within one tracker page and one
	// caching page. Dword runs check the tracker page themselves.
	const bool shape = dword_runs ? size >= 4 && size <= 256 && ((vaddr | size) & 3u) == 0 &&
	                                    vaddr / TRACKER_PAGE_SIZE == (vaddr + size - 1) / TRACKER_PAGE_SIZE
	                              : (size == 4 || size == 8) && (vaddr & (size - 1)) == 0;
	return shape &&
	       m_gpu_modified_ranges.Intersects(vaddr, size) && !InFlightIntersects(vaddr, size) &&
	       (cpu_images ? m_texture_cache.OnlyCpuWrittenImages(vaddr, size)
	                   : !m_texture_cache.IsRegionRegistered(vaddr, size));
}

uint32_t BufferCache::LabelWriteFlags(uint64_t vaddr, uint64_t size) {
	const bool image = m_texture_cache.IsRegionRegistered(vaddr, size);
	static uint32_t described = 0;
	if (image && described++ < 4) {
		m_texture_cache.DescribeImagesIn(vaddr, size);
	}
	return (IsRegionGpuModified(vaddr, size) ? 1u : 0u) | (HasGpuDirtyBytes(vaddr, size) ? 2u : 0u) |
	       (m_gpu_modified_ranges.Intersects(vaddr, size) ? 4u : 0u) |
	       (InFlightIntersects(vaddr, size) ? 8u : 0u) | (image ? 16u : 0u);
}

void BufferCache::OverwriteGpuBytes(uint64_t vaddr, const void* data, uint32_t size) {
	// The GPU writes recorded before still land in the cached buffer first; the mirror's fill
	// follows them behind a barrier. A later download must not bring their value back: the label
	// bytes leave the GPU-written set (no download of them is in flight, see CanOverwriteGpuBytes).
	// Queued upload copies must not see the write (as in WriteBacked).
	JoinAsyncUploads();
	m_gpu_modified_ranges.Subtract(vaddr, size);
	m_gpu_dirty_generation++;
	LibKernel::Memory::WriteBacking(vaddr, data, size);
	MirrorLabel(vaddr, data, size);
	// A tracker page left without GPU-written bytes is clean: backing and cached buffer agree
	// (the label's write is recorded before any later upload). Kept GPU-dirty, it had no bytes
	// to download, and CPU polls of the label faulted for nothing.
	const auto page = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
	    !InFlightIntersects(page, TRACKER_PAGE_SIZE)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
	}
	m_scheduler.Context().AdvanceBdaEpochForGpuWrite();
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	// KYTY_DEBUG_GDS_LOG=1: every CP copy into or out of GDS (first 200000), with the source's CPU
	// bytes for copies into GDS (the title cs 2ef93a49 takes its item count from GDS).
	static const bool gds_log = KYTY_GETENV("KYTY_DEBUG_GDS_LOG") != nullptr;
	if (gds_log && (dst_gds || src_gds)) [[unlikely]] {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 200000) {
			uint32_t first = 0;
			if (src_memory && src_vaddr != 0) {
				(void)LibKernel::Memory::TryReadBacking(src_vaddr, &first, sizeof(first));
			}
			// KYTY_DEBUG_GDS_LOG=2: for restores (memory -> GDS) also the value the cached host
			// buffer holds right now (a GPU copy and a full wait): differs from the CPU bytes while
			// the tracker calls the slot clean = a lost GPU-dirty mark / stale upload.
			static const bool gpu_read = std::strcmp(KYTY_GETENV("KYTY_DEBUG_GDS_LOG"), "2") == 0;
			int64_t           host     = -1;
			if (gpu_read && dst_gds && src_memory && src_vaddr != 0 && IsRegionRegistered(src_vaddr, 4)) {
				static std::unique_ptr<Buffer> probe;
				if (!probe) {
					probe = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
					                                 vk::BufferUsageFlagBits::eTransferDst, 64);
				}
				auto& source  = m_slot_buffers[FindBuffer(src_vaddr, 4)];
				auto& command = m_scheduler.Current();
				command.EndRendering();
				const auto native = command.Handle();
				vk::MemoryBarrier barrier {};
				barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
				barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
				native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
				                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &barrier, 0,
				                       nullptr, 0, nullptr);
				native.copyBuffer(source.Handle(), probe->Handle(),
				                  vk::BufferCopy {source.Offset(src_vaddr), 0, 4});
				const auto tick = m_scheduler.CurrentTick();
				m_scheduler.Wait(tick);
				probe->Invalidate(0, 4);
				uint32_t value = 0;
				std::memcpy(&value, probe->Mapped().data(), sizeof(value));
				host = value;
			}
			std::printf("gds-copy: dst=%s0x%" PRIx64 " src=%s0x%" PRIx64 " size=0x%" PRIx64
			            " src_cpu_first=0x%x src_gpu_dirty=%d src_cpu_mod=%d src_gpu_mod=%d host=%lld\n",
			            dst_gds ? "gds:" : "", dst_vaddr, src_gds ? "gds:" : "", src_vaddr, size, first,
			            src_memory && src_vaddr != 0 && HasGpuDirtyBytes(src_vaddr, size) ? 1 : 0,
			            src_memory && src_vaddr != 0 && IsRegionCpuModified(src_vaddr, size) ? 1 : 0,
			            src_memory && src_vaddr != 0 && IsRegionGpuModified(src_vaddr, size) ? 1 : 0,
			            static_cast<long long>(host));
			std::fflush(stdout);
		}
	}
	if (FramePassesRecording() && dst_memory) [[unlikely]] {
		std::printf("frame-pass cp-copy dst=0x%" PRIx64 "+0x%" PRIx64 " src=%s0x%" PRIx64 "\n", dst_vaddr,
		            size, src_gds ? "gds:" : "", src_vaddr);
	}
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	struct LabelCopyScope {
		bool active;
		explicit LabelCopyScope(bool on): active(on) {
			if (active) g_label_stats.in_copy = true;
		}
		~LabelCopyScope() {
			if (active) g_label_stats.in_copy = false;
		}
	} label_copy_scope {LabelStatsEnabled()};
	if (LabelStatsEnabled()) [[unlikely]] {
		auto& stats = g_label_stats;
		if (src_memory && stats.Overlaps(src_vaddr, size)) {
			stats.copy_reads++;
		}
		if (dst_memory && stats.Overlaps(dst_vaddr, size)) {
			stats.copy_writes++;
			if (src_memory && IsRegionGpuModified(src_vaddr, size)) {
				stats.copy_writes_gpu++;
			} else if (src_memory && !IsRegionGpuModified(dst_vaddr, size)) {
				stats.copy_writes_cpu++;
			}
		}
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		m_scheduler.Context().AdvanceBdaEpochForGpuWrite(); // As FillBuffer.
		return;
	}
	// KYTY_EOP_GPU_LABELS=1: a small copy whose own source and destination bytes the GPU has not
	// written (only their pages are GPU-dirty) runs here through the backing and is mirrored into
	// the cached buffer in command order, like an end-of-pipe label. The title resets its fences with
	// 4-byte DMA copies: on the GPU they marked the label bytes, and the next label write drained.
	if (static const bool small_copies = [] {
		    const char* text = KYTY_GETENV("KYTY_EOP_GPU_LABELS");
		    return text != nullptr && std::strcmp(text, "0") != 0;
	    }();
	    small_copies && src_memory && dst_memory && CanWriteBacked(dst_vaddr, size) &&
	    !HasGpuDirtyBytes(src_vaddr, size) && !m_texture_cache.IsRegionRegistered(src_vaddr, size)) {
		std::array<uint8_t, 64> bytes {};
		LibKernel::Memory::ReadGuestOnGpuThread(src_vaddr, bytes.data(), size);
		WriteBacked(dst_vaddr, bytes.data(), static_cast<uint32_t>(size));
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size, TextureCache::GpuWriteSource::Copy);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
	OnCommandRecorded();
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	// Downloaded bytes have left the byte set, but their backing is valid only once published.
	// Other bytes of an armed page are valid already: a window download around a CPU fault
	// arms whole pages, and Thread_Gpu reads game tables that share such pages every frame.
	// Strict SRT reads ask this for the same small tables every draw: remember clean pages.
	// KYTY_DEBUG_AB=cleanpages asks the byte sets every time in every other window.
	static const bool ab       = AbSelected("cleanpages");
	const auto        page     = vaddr / TRACKER_PAGE_SIZE;
	const bool        one_page = (vaddr + size - 1) / TRACKER_PAGE_SIZE == page &&
	                      !(ab && AbFeatureOff());
	auto& slot = m_clean_pages[page % m_clean_pages.size()];
	if (one_page && slot.page == page && slot.generation == m_gpu_dirty_generation) {
		return false;
	}
	if (one_page) {
		// Taken before the scan, as TextureCache::IsRegionGpuModified does.
		const auto generation = m_gpu_dirty_generation;
		const auto begin      = page * TRACKER_PAGE_SIZE;
		if (!m_gpu_modified_ranges.Intersects(begin, TRACKER_PAGE_SIZE) &&
		    !InFlightIntersects(begin, TRACKER_PAGE_SIZE)) {
			slot = {page, generation};
			return false;
		}
	}
	return m_gpu_modified_ranges.Intersects(vaddr, size) || InFlightIntersects(vaddr, size);
}

void BufferCache::PruneInFlight() {
	while (!m_inflight_downloads.empty() &&
	       m_scheduler.IsPublished(m_inflight_downloads.front().tick)) {
		m_inflight_downloads.pop_front();
	}
}

bool BufferCache::InFlightIntersects(uint64_t vaddr, uint64_t size) {
	PruneInFlight();
	const auto end = vaddr + size;
	for (const auto& download: m_inflight_downloads) {
		for (const auto& [address, bytes]: download.ranges) {
			if (address < end && address + bytes > vaddr) {
				return true;
			}
		}
	}
	return false;
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::TickFrame() {
	++m_gc_tick;
	ProbePageTable();
}

void BufferCache::SetPageTableProbe(uint64_t entry, uint64_t expected) {
	if (KYTY_GETENV("KYTY_DEBUG_DEVICE_FAULT") == nullptr ||
	    (entry + 1) * sizeof(vk::DeviceAddress) > m_bda_pagetable_buffer.Size()) {
		return;
	}
	m_probe_entry    = entry;
	m_probe_expected = expected;
	std::printf("pagetable-probe: entry %llu expected 0x%016llx, read back once per frame\n",
	            static_cast<unsigned long long>(entry), static_cast<unsigned long long>(expected));
	std::fflush(stdout);
}

void BufferCache::ProbePageTable() {
	if (m_probe_entry == UINT64_MAX) [[likely]] {
		return;
	}
	if (m_probe_buffer == nullptr) {
		m_probe_buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                          AllFlags,
		                                          sizeof(uint64_t) * m_probe_ticks.size());
	}
	for (size_t slot = 0; slot < m_probe_ticks.size(); slot++) {
		auto& tick = m_probe_ticks[slot];
		if (tick == 0 || !m_scheduler.IsFree(tick)) {
			continue;
		}
		m_probe_buffer->Invalidate(slot * sizeof(uint64_t), sizeof(uint64_t));
		uint64_t value = 0;
		std::memcpy(&value, m_probe_buffer->Mapped().data() + slot * sizeof(uint64_t), sizeof(value));
		m_probe_checked++;
		if (value != m_probe_expected) {
			m_probe_mismatches++;
			if (m_probe_mismatches <= 64) {
				std::printf("pagetable-probe: MISMATCH entry %llu = 0x%016llx, expected 0x%016llx "
				            "(written at tick %llu, frame %llu)\n",
				            static_cast<unsigned long long>(m_probe_entry),
				            static_cast<unsigned long long>(value),
				            static_cast<unsigned long long>(m_probe_expected),
				            static_cast<unsigned long long>(tick),
				            static_cast<unsigned long long>(m_probe_frames));
				std::fflush(stdout);
			}
		}
		tick = 0;
	}
	const auto slot = m_probe_frames % m_probe_ticks.size();
	if (m_probe_ticks[slot] == 0) {
		m_probe_buffer->CopyFrom(m_scheduler.Current(), m_bda_pagetable_buffer,
		                         m_probe_entry * sizeof(vk::DeviceAddress), slot * sizeof(uint64_t),
		                         sizeof(uint64_t), vk::AccessFlagBits::eMemoryWrite,
		                         vk::AccessFlagBits::eHostRead,
		                         vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                         vk::AccessFlagBits::eHostRead);
		m_probe_ticks[slot] = m_scheduler.CurrentTick();
	}
	m_probe_frames++;
	if ((m_probe_frames % 600) == 0) {
		std::printf("pagetable-probe: %llu frames, %llu readbacks checked, %llu mismatches\n",
		            static_cast<unsigned long long>(m_probe_frames),
		            static_cast<unsigned long long>(m_probe_checked),
		            static_cast<unsigned long long>(m_probe_mismatches));
		std::fflush(stdout);
	}
}

void BufferCache::RunGarbageCollector() {
	// Ages are counted in presented frames. Collection runs after every submission, often dozens
	// per frame, so counting submissions would retire buffers the game reuses every few frames.
	ReleaseUnboundBuffers();
	const auto tick = m_gc_tick;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	DrainStats::ReasonScope reason(DrainStats::Reason::BufferGc);

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	// Ages count presented frames (TickFrame): the former submission ages divided by 8.
	const uint64_t age        = std::min<uint64_t>(aggressive ? 10 : 20, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			// Pages armed by an in-flight readback publish with the wait below.
			EXIT_IF(!DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size()) &&
			        !m_memory_tracker.HasArmedPages(buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			g_allocation_counters.game_buffers_collected.fetch_add(1, std::memory_order_relaxed);
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		g_allocation_counters.game_buffers_collected.fetch_add(1, std::memory_order_relaxed);
		// As DeleteBuffer: the page-table clear Unregister records runs with the open command
		// buffer, and a draw or dispatch prepared around this collection may already hold the
		// buffer's device address, so the buffer itself goes once that buffer's tick completes.
		// Erasing it here freed memory a recorded shader still read (Ghost of Yotei: device lost
		// with ReadInvalid at a freed buffer's address, the same instruction in every run).
		DeleteBuffer(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

void BufferCache::PublishBdaHints(uint64_t vaddr, uint64_t size) noexcept {
	m_memory_tracker.PublishBdaHints(vaddr, size);
}

// The zero page and its table entry, before the first draw or dispatch that uses the table: the
// first PrepareBda always synchronizes (its epoch starts behind), through this or the selective
// pass. The scheduler is not recording yet when the cache is constructed.
void BufferCache::EnsureBdaZeroPage() {
	if (m_bda_zero_page_written) {
		return;
	}
	m_bda_zero_page_written = true;
	m_bda_zero_page.Fill(0, m_bda_zero_page.Size(), 0);
	const vk::DeviceAddress address = m_bda_zero_page.BufferDeviceAddress();
	EXIT_IF(address == 0);
	std::printf("bda-zero-page: device=0x%016llx size=0x%llx (table entry %u)\n",
	            static_cast<unsigned long long>(address),
	            static_cast<unsigned long long>(m_bda_zero_page.Size()),
	            static_cast<unsigned>(BDA_ZERO_PAGE_ENTRY));
	WriteDataBuffer(m_bda_pagetable_buffer, BDA_ZERO_PAGE_ENTRY * sizeof(vk::DeviceAddress), &address,
	                sizeof(address));
	SetPageTableProbe(BDA_ZERO_PAGE_ENTRY, address);
}

void BufferCache::SynchronizeBdaLegacy(const RangeSet& mapped) {
	const UploadSourceScope upload_source(UploadSource::Bda);
	EnsureBdaZeroPage();
	const UploadBatchScope upload_batch(*this);
	// Clear before the full walk so writes racing with the walk stay pending. Summaries go
	// first: a hint published in between re-sets its summary bit for the next pass.
	for (size_t summary = 0; summary < MemoryTracker::BDA_SUMMARY_WORDS; ++summary) {
		(void)m_memory_tracker.ConsumeBdaSummaryWord(summary);
	}
	for (size_t word = 0; word < MemoryTracker::BDA_HINT_WORDS; ++word) {
		(void)m_memory_tracker.ConsumeBdaHintWord(word);
	}
	mapped.ForEach(
	    [this](uint64_t begin, uint64_t end) { SynchronizeBuffersInRange(begin, end - begin); });
}

bool BufferCache::SynchronizeBdaSelective(const RangeSet& mapped) {
	const UploadSourceScope upload_source(UploadSource::Bda);
	EnsureBdaZeroPage();
	const UploadBatchScope upload_batch(*this);
	for (size_t summary = 0; summary < MemoryTracker::BDA_SUMMARY_WORDS; ++summary) {
		uint64_t words = m_memory_tracker.ConsumeBdaSummaryWord(summary);
		// On failure, words not yet visited keep their hint bits; restore their summary bits.
		struct SummaryClaim {
			MemoryTracker& tracker;
			size_t         summary;
			uint64_t&      remaining;
			~SummaryClaim() { tracker.RestoreBdaSummary(summary, remaining); }
		} summary_claim {m_memory_tracker, summary, words};
		while (words != 0) {
			const auto word = summary * 64 + static_cast<size_t>(std::countr_zero(words));
			if (!SynchronizeBdaWord(word, mapped)) {
				return false;
			}
			words &= words - 1;
		}
	}
	return true;
}

bool BufferCache::SynchronizeBdaWord(size_t word, const RangeSet& mapped) {
	uint64_t bits = m_memory_tracker.ConsumeBdaHintWord(word);
	struct Claim {
		MemoryTracker& tracker;
		size_t         word;
		uint64_t&      remaining;
		~Claim() { tracker.RestoreBdaHints(word, remaining); }
	} claim {m_memory_tracker, word, bits};
	while (bits != 0) {
		const auto region = word * 64 + static_cast<size_t>(std::countr_zero(bits));
		if (!SynchronizeBdaRegion(region, mapped)) {
			return false;
		}
		// Do not touch the shared word again: a concurrent write may have republished it.
		bits &= bits - 1;
	}
	return true;
}

bool BufferCache::SynchronizeBdaRegion(uint64_t region, const RangeSet& mapped) {
	const auto region_begin = region * TRACKER_REGION_SIZE;
	auto*      manager      = m_memory_tracker.FindRegion(region);
	if (manager == nullptr) {
		// A mapping or registration can precede tracker creation. The current consumer must
		// see those bytes now; the full walk creates (or waits for) the initially dirty manager.
		mapped.ForEachInRange(region_begin, TRACKER_REGION_SIZE,
		                      [this](uint64_t begin, uint64_t end) {
			                      SynchronizeBuffersInRange(begin, end - begin);
		                      });
		return true;
	}
	const auto dirty = m_memory_tracker.SnapshotCpuDirty(*manager);
	if (dirty.None()) {
		return true;
	}
	bool consistent = true;
	mapped.ForEachInRange(region_begin, TRACKER_REGION_SIZE, [&](uint64_t begin, uint64_t end) {
		if (consistent) {
			consistent = SynchronizeDirtyOwners(dirty, region_begin, begin, end);
		}
	});
	return consistent;
}

static bool BdaRunSyncEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BDA_RUN_SYNC");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

// KYTY_BDA_SPAN_SYNC=1: consecutive dirty runs of the same owner are synchronized in one call over
// the span from the first to the last run (the upload still copies only the dirty pages inside).
// Each call costs the pass ~1 us; partial fault windows and demoted hot pages split runs.
static bool BdaSpanSyncEnabled() {
	static const bool enabled = KYTY_GETENV("KYTY_BDA_SPAN_SYNC") != nullptr;
	return enabled;
}

bool BufferCache::SynchronizeDirtyOwners(const RegionBits& dirty, uint64_t region_begin,
                                         uint64_t begin, uint64_t end) {
	if (BdaSpanSyncEnabled() && BdaRunSyncEnabled()) {
		return SynchronizeDirtyOwnerSpans(dirty, region_begin, begin, end);
	}
	uint64_t   synced_end = begin;
	auto       page       = static_cast<size_t>((begin - region_begin) / TRACKER_PAGE_SIZE);
	const auto limit =
	    static_cast<size_t>((end - region_begin + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE);
	while (page < limit) {
		const auto [first, last] = dirty.FirstRangeFrom(page);
		if (first >= limit) {
			break;
		}
		const auto run_end = std::min(region_begin + last * TRACKER_PAGE_SIZE, end);
		for (auto cursor = std::max(region_begin + first * TRACKER_PAGE_SIZE, synced_end);
		     cursor < run_end;) {
			const auto* owner = m_page_table.Find(cursor >> CACHING_PAGEBITS);
			if (owner == nullptr || !*owner) {
				cursor = Common::AlignDown(cursor, CACHING_PAGESIZE) + CACHING_PAGESIZE;
				continue;
			}
			auto* buffer = m_slot_buffers.try_get(*owner);
			if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(cursor, 1)) {
				return false;
			}
			// Only the dirty run: the rest of the owner had no dirty page in the snapshot, and a
			// write since publishes the region's hint again. Walking the whole owner (as before;
			// KYTY_BDA_RUN_SYNC=0) cost Sky Garden ~18 us per synchronized owner.
			const auto owner_begin = BdaRunSyncEnabled() ? cursor
			                                             : std::max(buffer->CpuAddress(), begin);
			const auto owner_end   = std::min(buffer->CpuAddress() + buffer->Size(),
                                            BdaRunSyncEnabled() ? run_end : end);
			(void)SynchronizeBuffer(*buffer, owner_begin, owner_end - owner_begin, false, false);
			synced_end = owner_end;
			cursor     = owner_end;
		}
		page = last;
	}
	return true;
}

bool BufferCache::SynchronizeDirtyOwnerSpans(const RegionBits& dirty, uint64_t region_begin,
                                             uint64_t begin, uint64_t end) {
	BufferId   span_owner {};
	uint64_t   span_begin = 0;
	uint64_t   span_end   = 0;
	const auto flush      = [&] {
		if (span_end > span_begin) {
			(void)SynchronizeBuffer(m_slot_buffers[span_owner], span_begin, span_end - span_begin,
			                        false, false);
		}
		span_begin = span_end = 0;
	};
	uint64_t   synced_end = begin;
	auto       page       = static_cast<size_t>((begin - region_begin) / TRACKER_PAGE_SIZE);
	const auto limit =
	    static_cast<size_t>((end - region_begin + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE);
	while (page < limit) {
		const auto [first, last] = dirty.FirstRangeFrom(page);
		if (first >= limit) {
			break;
		}
		const auto run_end = std::min(region_begin + last * TRACKER_PAGE_SIZE, end);
		for (auto cursor = std::max(region_begin + first * TRACKER_PAGE_SIZE, synced_end);
		     cursor < run_end;) {
			const auto* owner = m_page_table.Find(cursor >> CACHING_PAGEBITS);
			if (owner == nullptr || !*owner) {
				cursor = Common::AlignDown(cursor, CACHING_PAGESIZE) + CACHING_PAGESIZE;
				continue;
			}
			auto* buffer = m_slot_buffers.try_get(*owner);
			if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(cursor, 1)) {
				flush();
				return false;
			}
			const auto owner_end = std::min(buffer->CpuAddress() + buffer->Size(), run_end);
			if (span_end > span_begin && span_owner == *owner) {
				span_end = owner_end;
			} else {
				flush();
				span_owner = *owner;
				span_begin = cursor;
				span_end   = owner_end;
			}
			synced_end = owner_end;
			cursor     = owner_end;
		}
		page = last;
	}
	flush();
	return true;
}

bool BufferCache::CheckBdaHintInvariant(const RangeSet& mapped) {
	bool covered = true;
	for (const auto& [address, id]: m_buffers) {
		const auto& buffer = m_slot_buffers[id];
		mapped.ForEachInRange(address, buffer.Size(), [&](uint64_t begin, uint64_t end) {
			covered = covered && m_memory_tracker.BdaHintsCoverCpuDirty(begin, end - begin);
		});
		if (!covered) {
			break;
		}
	}
	return covered;
}

} // namespace Libs::Graphics
