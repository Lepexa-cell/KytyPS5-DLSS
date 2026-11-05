#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class ReadbackQueue;

// Waits until the uploads handed to the helper thread are in the staging buffer (see
// BufferCache::UploadCopies). Called before recorded commands are submitted to the GPU.
void JoinAsyncUploads() noexcept;
// The uploads handed over so far, as a ticket; JoinAsyncUploads(ticket) waits for those only
// (later ones may still be copying). KYTY_LATE_UPLOAD_JOIN: the submit thread waits for a
// command buffer's ticket right before its vkQueueSubmit instead of the GPU thread at Submit.
[[nodiscard]] uint64_t AsyncUploadTicket() noexcept;
void                   JoinAsyncUploads(uint64_t ticket) noexcept;
// Uploads handed to the helper and not yet copied (diagnostics).
[[nodiscard]] uint64_t AsyncUploadsPending() noexcept;

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	// The entry after the last page holds the zero page's device address: shaders read it once and
	// load from the zero page where a page is not mapped, instead of branching around the load.
	// Unmapped pages keep entry 0, so stores and atomics still see them as not mapped.
	static constexpr uint64_t BDA_ZERO_PAGE_ENTRY = CACHING_NUMPAGES;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    (CACHING_NUMPAGES + 1) * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// After a guest write fault at vaddr (see MemoryTracker::ExpandWriteFault).
	// KYTY_FAULT_AHEAD_PAGES=<pages> sets the window (a power of two, default 16; 1: off).
	void                   ExpandWriteFault(uint64_t vaddr);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// Marks a buffer that shaders read through the BDA page table (never released as unbound).
	void PinBda(BufferId id);
	// A shader faulted on [vaddr, vaddr + size) through the BDA page table (release counters).
	void NoteBdaFault(uint64_t vaddr, uint64_t size);
	// FindBuffer, keeping `id` while it is a live buffer covering the range (as ObtainBuffer does):
	// buffers never overlap, so a live buffer over the range is the one FindBuffer would find.
	[[nodiscard]] BufferId RefindBuffer(BufferId id, uint64_t vaddr, uint64_t size) {
		return !IsBufferInvalid(id) && m_slot_buffers[id].IsInBounds(vaddr, size)
		           ? id
		           : FindBuffer(vaddr, size);
	}
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	// Whether ObtainBuffer(vaddr, size, false, false, id) would return buffer `id` at its own
	// offset and copy nothing: `id` is live and covers the range, and no 64 KiB slice of the range
	// holds a CPU-dirty page (ObtainBuffer's cpu_clean path). Then touches the buffer as that path
	// does. KYTY_DRAW_MEMO keeps a recorded read binding while this holds.
	[[nodiscard]] bool KeepReadBinding(BufferId id, uint64_t vaddr, uint64_t size) {
		if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size) ||
		    !m_memory_tracker.IsRegionCpuCleanHint(vaddr, size)) {
			return false;
		}
		TouchBuffer(m_slot_buffers[id]);
		return true;
	}
	// Whether `handle` is live buffer `id`'s (a binding ObtainBuffer served from it, not from a
	// stream copy).
	[[nodiscard]] bool IsBufferHandle(BufferId id, vk::Buffer handle) const {
		return !IsBufferInvalid(id) && m_slot_buffers[id].Handle() == handle;
	}
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return ActiveStream();
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	// The ring that per-draw stream copies go to (see bufferCache.cpp).
	[[nodiscard]] StreamBuffer& ActiveStream() noexcept;
	// The ring an upload of `size` bytes stages in (KYTY_STAGING_DEVICE, see bufferCache.cpp), and
	// the size of the staging ring `source` is (0: not a staging ring).
	[[nodiscard]] StreamBuffer& UploadStaging(uint64_t size);
	[[nodiscard]] uint64_t      StagingCapacity(vk::Buffer source) const;
	// Device-addressable ring for per-draw parameter records that shaders load by address.
	[[nodiscard]] StreamBuffer& GetDrawRecordBuffer() noexcept { return m_draw_record_buffer; }
	// The zero page's device address (written on first use): a table block's null buffers.
	[[nodiscard]] vk::DeviceAddress ZeroPageAddress() {
		EnsureBdaZeroPage();
		return m_bda_zero_page.BufferDeviceAddress();
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	// KYTY_DEBUG_DEVICE_FAULT: once per frame an 8-byte readback of one page-table entry, compared
	// with the value written there (the combined build probes its zero-page entry). For tests,
	// KYTY_DEBUG_PAGETABLE_PROBE=<entry>:<hex value> sets it.
	void SetPageTableProbe(uint64_t entry, uint64_t expected);
	// KYTY_DEBUG_DEVICE_FAULT: prints which buffer (live, or freed within the last 4096 deletes)
	// holds a device address the driver reported faulting on.
	void DescribeDeviceAddress(uint64_t address) const;
	// Debugging aid: prints the cache buffer owning that guest page and its page-table entry.
	void DebugDescribePage(uint64_t vaddr);
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// A label the CPU side already wrote to the backing (dword aligned): the same bytes are filled
	// into the cached buffer in command order, without marking them GPU-written, so GPU readers
	// see it and a later download of the page keeps it (see KYTY_EOP_GPU_LABELS).
	void MirrorLabel(uint64_t vaddr, const void* data, uint32_t size);
	// A small command-processor write (label, tiny DMA copy) whose page is GPU-dirty but whose own
	// bytes are not, and which no image covers: CanWriteBacked says so, WriteBacked writes the
	// backing without a fault (after pending upload copies) and mirrors it (KYTY_EOP_GPU_LABELS).
	// `cpu_images`: images over the range are allowed when only the CPU ever wrote them
	// (TextureCache::OnlyCpuWrittenImages); the caller first invalidates them as for a CPU write
	// (InvalidateImagesForWrite, KYTY_EOP_IMAGE_LABELS). `max_size`: WRITE_DATA packets go up to
	// 256 bytes (KYTY_WRITE_DATA_BACKED); labels keep 64.
	[[nodiscard]] bool CanWriteBacked(uint64_t vaddr, uint64_t size, bool cpu_images = false,
	                                  uint64_t max_size = 64);
	void WriteBacked(uint64_t vaddr, const void* data, uint32_t size);
	// The same for a label whose own bytes the GPU has written (recorded before it, all on one
	// command stream) and no download is still publishing: the label replaces them, so they leave
	// the GPU-written set and only the label reaches backing and cached buffer (KYTY_EOP_GPU_LABELS=2).
	// `dword_runs`: any 4..256 dword-aligned bytes within one tracker page (WRITE_DATA packets, e.g.
	// the title in Pro mode's 3-dword dispatch args over GPU-written ones), not only 4/8-byte labels.
	[[nodiscard]] bool CanOverwriteGpuBytes(uint64_t vaddr, uint64_t size, bool cpu_images = false,
	                                        bool dword_runs = false);
	void InvalidateImagesForWrite(uint64_t vaddr, uint64_t size);
	void OverwriteGpuBytes(uint64_t vaddr, const void* data, uint32_t size);
	// Diagnostic: why neither path above takes a label. Bits: 1 page GPU-modified, 2 GPU-dirty
	// bytes, 4 GPU-written range, 8 download in flight, 16 an image covers it.
	[[nodiscard]] uint32_t LabelWriteFlags(uint64_t vaddr, uint64_t size);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// KYTY_DEBUG_LABEL_STATS=1 (diagnostic): the end-of-pipe labels the command processor writes
	// (into a clean page, a GPU-dirty page beside them, or over GPU-written bytes) and the GPU work
	// that later reads their bytes (shader bindings, copies), every 5 s: whether a label written
	// only to the backing would need a copy in the cached buffer at all.
	[[nodiscard]] static bool LabelStatsEnabled();
	void NoteLabelWrite(uint64_t vaddr, uint32_t size);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsPageGpuDirtyHint(uint64_t vaddr) const noexcept {
		return m_memory_tracker.IsPageGpuDirtyHint(vaddr);
	}
	// Changes whenever bytes become GPU-dirty or a download starts, which is also the only way a
	// GPU-dirty range can become clean again (see m_clean_pages). GPU thread only.
	[[nodiscard]] uint64_t GpuDirtyGeneration() const noexcept { return m_gpu_dirty_generation; }
	// Changes whenever a buffer is registered or unregistered: while it does not, every buffer id
	// still covers what it covered (KYTY_BUFFER_KEEP). GPU thread only.
	[[nodiscard]] uint64_t BufferSetGeneration() const noexcept { return m_buffer_set_generation; }
	// Whether the last ObtainBuffer of a texel buffer found an image over the range.
	[[nodiscard]] bool LastImageFound() const noexcept { return m_last_image_found; }
	[[nodiscard]] bool LastNoImageStable() const noexcept { return m_last_no_image_stable; }
	// Eager readback of hot pages: memory that CPU reads have faulted on. A write recorded to a
	// hot page is downloaded at the next flush point, so its bytes are usually published before
	// the CPU reads them again. OnCommandRecorded() marks writes of the bindings obtained so far
	// as recorded; only recorded writes are downloaded.
	void               OnCommandRecorded();
	void               RecordEagerReadbacks();
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	void               PublishBdaHints(uint64_t vaddr, uint64_t size) noexcept;
	void               SynchronizeBdaLegacy(const RangeSet& mapped);
	[[nodiscard]] bool SynchronizeBdaSelective(const RangeSet& mapped);
	[[nodiscard]] bool CheckBdaHintInvariant(const RangeSet& mapped);
	// KYTY_DEBUG_BDA_STATS: accounts one PrepareBda synchronization (see bufferCache.cpp).
	[[nodiscard]] static bool BdaStatsEnabled();
	void                      NoteBdaPass(uint64_t elapsed_ns);
	void               TickFrame();
	void               RunGarbageCollector();
	// Thread_Gpu at a draw or dispatch entry (a point between commands): dates the GPU writes logged
	// so far with the open command buffer's tick, never earlier than their own command buffer. Without
	// it they were dated only at the next readback, so every writer looked unsubmitted
	// (KYTY_CP_QUEUED_READBACK, run cr2, 2026-10-31).
	void StampGpuWrites() noexcept;

	// KYTY_DEFER_DIRTY_READS=1 (the title in Pro mode, 2026-10-31): an asynchronous compute dispatch whose
	// program lookup faulted on GPU-written bytes before (a read site, learned per shader) has their
	// download recorded and is suspended, instead of draining the GPU inside the fault; Thread_Gpu
	// records the other queues until the bytes are published. The fight's one drain a frame waited
	// ~9 ms while 43 graphics submissions were ready (run qs2). KYTY_DEFER_DIRTY_READS_OFF_FILE=<path>:
	// no deferral while the file exists (in-run A/B; read sites are still learned).
	[[nodiscard]] static bool DeferDirtyReadsEnabled() noexcept;
	// Thread_Gpu, before the program lookup of a deferrable dispatch of `shader_addr`: the tick whose
	// publication its read sites wait for (their download recorded and submitted here), or 0 when it
	// may run now. The tick is kept for TakeDeferredReadTick.
	[[nodiscard]] uint64_t DeferDirtyReads(uint64_t shader_addr);
	// The tick of the last deferral (0: the dispatch was not deferred for reads), cleared.
	[[nodiscard]] uint64_t TakeDeferredReadTick() noexcept {
		return std::exchange(m_deferred_read_tick, 0);
	}
	[[nodiscard]] bool ReadDeferredInSlice() const noexcept { return m_deferred_read_tick != 0; }
	// A guest interrupt is raised (KYTY_EOP_IRQ_AT_CP: at CP time) while a deferred dispatch is not
	// recorded yet: the game may already write the data it reads at recording (counted, must stay 0).
	void NoteInterruptRaised() noexcept {
		m_defer_reads.interrupts += m_deferred_shader != 0 ? 1u : 0u;
	}
	// Thread_Gpu with every queue waiting: waits for a deferred dispatch's bytes (its fault's wait).
	void WaitDeferredRead(uint64_t tick);
	// The shader whose dispatch Thread_Gpu prepares while reads are deferrable (0: none); a read fault
	// meanwhile is learned as its read site (NoteReadSiteFault).
	static inline thread_local uint64_t t_read_site_shader = 0;
	struct ReadSiteScope {
		explicit ReadSiteScope(uint64_t shader_addr) noexcept: previous(t_read_site_shader) {
			t_read_site_shader = shader_addr;
		}
		~ReadSiteScope() { t_read_site_shader = previous; }
		KYTY_CLASS_NO_COPY(ReadSiteScope);
		uint64_t previous;
	};
	// Thread_Gpu read fault at vaddr (renderContext.cpp).
	void NoteReadSiteFault(uint64_t vaddr);

private:
	friend struct BufferCacheTestAccess;
	friend struct PerformanceMemoryTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	void TouchStreamOwner(BufferId id, uint64_t vaddr, uint64_t size);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	// A BDA synchronization pass records nothing but read-only, non-texel uploads while it scans.
	// Inside this scope they are queued and recorded behind one barrier pair instead of one pair
	// each (the title: ~275 uploads a frame); only the outermost scope records them, at its end.
	// Nothing else may be recorded while uploads are queued. KYTY_UPLOAD_BATCH=0 records each
	// upload at once as before.
	class UploadBatchScope {
	public:
		explicit UploadBatchScope(BufferCache& cache);
		~UploadBatchScope();
		KYTY_CLASS_NO_COPY(UploadBatchScope);

	private:
		BufferCache& m_cache;
		bool         m_previous;
	};
	struct PendingUpload {
		vk::Buffer                  source;
		vk::Buffer                  destination;
		std::vector<vk::BufferCopy> copies;
	};
	void QueueUpload(vk::Buffer source, vk::Buffer destination, std::vector<vk::BufferCopy>&& copies);
	// Records the queued uploads; also runs at the start of every submit
	// (CommandScheduler::SetPreSubmitHook), possibly inside an upload callback of the memory
	// tracker: it must never call the tracker or submit.
	void FlushUploadBatch();
	// deferred: the pages' protections are queued (PageManager::DeferProtects).
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size, bool deferred = false);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Records downloads of the range's unarmed GPU-dirty pages, arms them, and queues their
	// publication, which finalizes them. Returns false when there was nothing to arm.
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Thread_Gpu: arms the readback window around [vaddr, vaddr+size).
	void RecordReadback(uint64_t vaddr, uint64_t size, bool is_write);
	// Other threads: publish GPU-dirty pages without draining Thread_Gpu.
	void ReadMemoryAsync(uint64_t vaddr, uint64_t size, bool is_write);
	void MarkReadbackHot(uint64_t vaddr);
	// GPU writes that may still be in flight (KYTY_DEBUG_READBACK_AGE): NoteGpuWrite logs a
	// written range; Thread_Gpu, between commands, InflightWriteTick returns the latest tick of a
	// possibly unfinished write overlapping [begin, end) (0: none). Writes get their tick at the
	// next point between commands, so one is never dated earlier than its command buffer.
	// KYTY_READBACK_QUEUE=1 (ReadbackQueue). Thread_Gpu, for a guest thread's readback: when every
	// GPU write of the readback window has finished, arms its GPU-dirty pages for a download on
	// the readback queue and returns it; null otherwise. The guest thread then submits and
	// publishes it (PublishQueuedReadback); a priority operation at the arming tick does so too,
	// for waiters that only know that tick.
	struct QueuedReadback;
	[[nodiscard]] std::shared_ptr<QueuedReadback> QueueReadback(uint64_t vaddr, uint64_t size,
	                                                            bool is_write);
	void PublishQueuedReadback(QueuedReadback& request);
	void                   NoteGpuWrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t InflightWriteTick(uint64_t begin, uint64_t end, uint64_t completed);
	[[nodiscard]] bool     GpuWriteLogComplete(uint64_t completed) const noexcept {
		return m_gpu_writes_on && completed >= m_gpu_writes_from;
	}
	void QueueEagerReadback(uint64_t vaddr, uint64_t size);
	// Whether a recorded download of these bytes has not finished publishing.
	[[nodiscard]] bool InFlightIntersects(uint64_t vaddr, uint64_t size);
	void               PruneInFlight();
	void               EnsureBdaZeroPage();
	[[nodiscard]] bool SynchronizeBdaWord(size_t word, const RangeSet& mapped);
	[[nodiscard]] bool SynchronizeBdaRegion(uint64_t region, const RangeSet& mapped);
	[[nodiscard]] bool SynchronizeDirtyOwners(const RegionBits& dirty, uint64_t region_begin,
	                                          uint64_t begin, uint64_t end);
	[[nodiscard]] bool SynchronizeDirtyOwnerSpans(const RegionBits& dirty, uint64_t region_begin,
	                                              uint64_t begin, uint64_t end);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	// Never registered, so never evicted, unmapped or overlapped by a game buffer.
	Buffer                                            m_bda_zero_page;
	bool                                              m_bda_zero_page_written = false;
	struct FreedBuffer {
		uint64_t bda  = 0;
		uint64_t size = 0;
		uint64_t cpu  = 0;
		uint64_t tick = 0;
	};
	std::array<FreedBuffer, 4096> m_freed_buffers {};
	size_t                       m_freed_next = 0;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	StreamBuffer                                      m_draw_record_buffer;
	// KYTY_DEBUG_AB=streamhost: a stream ring in device memory for the A/B (see ActiveStream).
	std::unique_ptr<StreamBuffer> m_stream_device;
	// KYTY_STAGING_DEVICE=1: the upload staging ring in device memory (see UploadStaging).
	std::unique_ptr<StreamBuffer> m_staging_device;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	// KYTY_RELEASE_UNBOUND (see ReleaseUnboundBuffers).
	struct ReleasedBuffer {
		uint64_t size = 0;
		uint64_t tick = 0;
		uint64_t age  = 0;
	};
	struct ReleaseStats {
		uint64_t released         = 0;
		uint64_t released_bytes   = 0;
		uint64_t rebound          = 0;
		uint64_t rebound_bytes    = 0;
		uint64_t refaults         = 0;
		uint64_t refaults_printed = 0;
		uint64_t kept             = 0; // old but pinned / GPU-dirty / armed
		std::array<uint64_t, 4> rebound_after {}; // frames from release to rebinding
		std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
	};
	std::map<uint64_t, ReleasedBuffer> m_released;
	ReleaseStats                       m_release_stats;
	uint64_t                           m_release_tick = 0;
	void ReleaseUnboundBuffers();
	void NoteReleasedOverlap(uint64_t vaddr, uint64_t size, bool bda_fault);
	void     ProbePageTable();
	std::unique_ptr<Buffer>  m_probe_buffer;
	uint64_t                 m_probe_entry      = UINT64_MAX;
	uint64_t                 m_probe_expected   = 0;
	std::array<uint64_t, 4>  m_probe_ticks      {};
	uint64_t                 m_probe_frames     = 0;
	uint64_t                 m_probe_checked    = 0;
	uint64_t                 m_probe_mismatches = 0;
	uint64_t m_readback_token     = 0;
	// KYTY_DEFER_DIRTY_READS: [vaddr, vaddr + size) readable without a full drain: 0 now; else the
	// tick of its download (recorded here, page by page, and submitted when it was in the open
	// buffer); UINT64_MAX when this way cannot help (image memory, bytes without an owner).
	uint64_t ArmGpuThreadRead(uint64_t vaddr, uint64_t size);
	// The latest tick of an unpublished download holding bytes of the range (0: none).
	uint64_t InFlightTick(uint64_t vaddr, uint64_t size);
	struct ReadSite {
		uint64_t shader = 0;
		uint64_t vaddr  = 0; // ReadSiteBytes-aligned.
		uint32_t period = 0; // m_read_site_period when it was learned.
	};
	static constexpr uint64_t ReadSiteBytes = 64;
	// A site is forgotten ReadSitePeriods 10-s periods after it was learned: a deferral prevents the
	// fault that would show the shader still reads there, so a site whose shader moved on to other
	// bytes kept deferring its dispatches for bytes it no longer reads (review 2026-11-01). A site
	// still read faults once more and is learned again.
	static constexpr uint32_t ReadSitePeriods = 3;
	// the title in Pro mode learned 62 sites in the fight's first 10 s with 16 slots (run dr2, round robin).
	std::array<ReadSite, 64>  m_read_sites {};
	size_t                    m_read_site_next     = 0;
	uint32_t                  m_read_site_period   = 1;
	uint64_t                  m_deferred_read_tick = 0;
	uint64_t                  m_deferred_shader    = 0; // A deferred dispatch not recorded yet.
	struct DeferReadStats {
		uint32_t checks      = 0;
		uint32_t deferred    = 0;
		uint32_t interrupts  = 0;
		uint32_t flushes     = 0;
		uint32_t unsupported = 0;
		uint32_t gave_up     = 0;
		uint32_t off         = 0;
		uint32_t learned     = 0;
		uint32_t expired     = 0;
		uint32_t fallbacks   = 0;
		uint64_t fallback_ns = 0;
		uint32_t streak      = 0; // Deferrals in a row (re-written bytes); a long streak drains.
		std::chrono::steady_clock::time_point next_print {};
	};
	DeferReadStats m_defer_reads;
	// UploadBatchScope: whether uploads are queued, the queue, and each destination's span of
	// queued copies.
	bool                                                                     m_upload_batch = false;
	std::vector<PendingUpload>                                               m_pending_uploads;
	std::unordered_map<VkBuffer, std::pair<vk::DeviceSize, vk::DeviceSize>> m_pending_spans;
	// Tracker pages hit by CPU read faults, with the tick of their latest fault.
	static constexpr size_t HotReadbackPages = 64;
	std::unordered_map<uint64_t, uint64_t> m_hot_pages;
	std::vector<uint64_t>                  m_eager_pending; // Written by unrecorded commands.
	std::vector<uint64_t>                  m_eager_ready;   // Written by recorded commands.
	// Downloaded byte ranges whose backing is not written yet, oldest first. They left
	// m_gpu_modified_ranges when recorded; the other bytes of their pages are already valid.
	struct InFlightDownload {
		uint64_t                                   tick = 0;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	std::deque<InFlightDownload> m_inflight_downloads;
	struct GpuWrite {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	std::unique_ptr<ReadbackQueue> m_readback_queue;
	std::vector<GpuWrite> m_gpu_writes;
	size_t                m_gpu_writes_head    = 0;
	size_t                m_gpu_writes_stamped = 0;
	bool                  m_gpu_writes_on      = false;
	// Writes before this tick were not logged (the log started or overflowed).
	uint64_t              m_gpu_writes_from    = UINT64_MAX;
	// HasGpuDirtyBytes: tracker pages found clean, valid while m_gpu_dirty_generation is
	// unchanged. It is bumped when bytes become GPU-dirty or a download starts; nothing else can
	// dirty a page. GPU thread only, like HasGpuDirtyBytes.
	struct CleanPage {
		uint64_t page       = UINT64_MAX;
		uint64_t generation = 0;
	};
	std::array<CleanPage, 64> m_clean_pages {};
	uint64_t                  m_gpu_dirty_generation = 1;
	// ObtainBuffer's stream-ring copies of the current BDA epoch, by range (see there).
	struct StreamCopy {
		uint64_t vaddr        = 0;
		uint64_t size         = 0;
		uint64_t epoch        = 0; // 0: none.
		uint64_t tick         = 0;
		uint64_t gpu_writes   = 0; // m_gpu_dirty_generation.
		uint64_t image_writes = 0; // TextureCache::GpuModifiedGeneration.
		Buffer*  stream       = nullptr;
		uint64_t offset       = 0;
	};
	static constexpr size_t StreamCopySlots = 4096;
	std::vector<StreamCopy> m_stream_copies = std::vector<StreamCopy>(StreamCopySlots);
	// KYTY_VERIFY_STREAM_REUSE=1: whether to copy again anyway (after comparing; see there).
	[[nodiscard]] static bool VerifyStreamReuse(const StreamCopy& copy);
	// ObtainBufferForImage's staged uploads, by range, with the time of the latest.
	struct ImageStage {
		uint64_t                              vaddr = 0;
		uint64_t                              size  = 0;
		std::chrono::steady_clock::time_point time {};
	};
	std::array<ImageStage, 256> m_image_stages {};
	// KYTY_TEXEL_IMAGE_MEMO: ranges no image starts on, with the texture cache's generation of
	// the range then (see SynchronizeBufferFromImage).
	struct NoImageRange {
		uint64_t vaddr      = 0;
		uint64_t size       = 0;
		uint64_t generation = 0;
		uint64_t whole      = 0; // The whole image set's generation it was last confirmed under.
	};
	std::array<NoImageRange, 8192> m_no_image_ranges {};
	// KYTY_DEBUG_OBTAIN_STATS: what the last SynchronizeBuffer / SynchronizeBufferFromImage did.
	bool m_last_sync_uploaded = false;
	bool m_last_image_found   = false;
	// The last SynchronizeBufferFromImage found no image, and a lookup gives the same while the
	// image set generation stands (its no-image memo's condition).
	bool     m_last_no_image_stable  = false;
	uint64_t m_buffer_set_generation = 0;
	[[nodiscard]] bool          IsRepeatedImageStage(uint64_t vaddr, uint64_t size);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
