#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/switches.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/drawSpeculator.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/drainStats.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/threadSampler.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fmt/format.h>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <set>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

// KYTY_TS_LOG=<skip> (diagnostic): after that many GPU timestamp writes, print the next 1200
// with their destination and value, and the flips between them, to see how a game pairs them.
static void LogTimestampWrite(const char* kind, uint32_t event, uint64_t dst, uint64_t value,
                              bool counted = true) {
	static const int64_t skip = [] {
		const char* text = KYTY_GETENV("KYTY_TS_LOG");
		return text != nullptr ? std::strtoll(text, nullptr, 10) : int64_t {-1};
	}();
	if (skip < 0) {
		return;
	}
	static std::atomic<int64_t> seen {0};
	const auto index = counted ? seen.fetch_add(1) : seen.load();
	if (index < skip || index >= skip + 1200) {
		return;
	}
	std::printf("ts #%" PRId64 " %s ev=0x%02" PRIx32 " dst=0x%016" PRIx64 " value=%" PRIu64 "\n",
	            index, kind, event, dst, value);
}

// Guest GPU timestamps are the reference clock at the time Thread_Gpu parses the packet, which
// includes the emulator's own stalls. With --gpu-timestamp-scale above 100, time since the last
// flip is stretched by that percentage, so a game that sizes its dynamic resolution from GPU
// timestamps sees more GPU time and leaves headroom. Each flip returns to the real clock, so
// values never drift from it by more than a frame's stretch.
static std::atomic<uint64_t> g_timestamp_anchor {0};

// Debugging aid: KYTY_DEBUG_FRAME_LIMIT_MS=<ms> makes each flip wait until <ms> have passed since
// the previous one, as a slow render thread would, to see how the game's timing copes with low
// frame rates. With KYTY_DEBUG_FRAME_LIMIT_FILE=<path> it applies only while that file exists.
static void DebugFrameLimit() {
	DebugFrameRepeatFlip();
	DrawRepeatFlip();
	DebugWatchTick();
	if (StampTraceEnabled()) [[unlikely]] {
		StampTraceFrame();
	}
	static const double limit_ms = [] {
		const char* value = KYTY_GETENV("KYTY_DEBUG_FRAME_LIMIT_MS");
		return value != nullptr ? std::atof(value) : 0.0;
	}();
	if (limit_ms <= 0.0) {
		return;
	}
	static const char* file   = KYTY_GETENV("KYTY_DEBUG_FRAME_LIMIT_FILE");
	static uint32_t    flips  = 0;
	static bool        active = false;
	if (flips++ % 15 == 0) {
		FILE* probe = file != nullptr ? std::fopen(file, "rb") : nullptr;
		active      = file == nullptr || probe != nullptr;
		if (probe != nullptr) {
			std::fclose(probe);
		}
	}
	static auto last = std::chrono::steady_clock::now();
	const auto  step = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
	    std::chrono::duration<double, std::milli>(limit_ms));
	if (active) {
		// Sleep up to the last millisecond and spin the rest: a late wake-up would otherwise cost
		// a few percent of the frame rate.
		const auto target = last + step;
		std::this_thread::sleep_until(target - std::chrono::milliseconds(1));
		while (std::chrono::steady_clock::now() < target) {
			_mm_pause();
		}
	}
	// A flip that is at most one step late keeps the schedule, so wake-up errors do not add up.
	const auto now = std::chrono::steady_clock::now();
	last           = active && now < last + 2 * step ? last + step : now;
}

// With KYTY_DEBUG_MRT_SHADER set, the GPU thread keeps its last 96 PM4 packets (header and up to 8
// dwords) so that DebugDumpRecentPm4 can print what led up to a traced draw: clears, fills and
// register writes for its targets.
struct RecentPm4 {
	std::array<uint32_t, 9> dwords {};
	uint32_t                count = 0;
};
static thread_local std::array<RecentPm4, 96> t_recent_pm4 {};
static thread_local uint32_t                  t_recent_pm4_next = 0;

static void RecordRecentPm4(const uint32_t* packet, uint32_t remaining_dw) {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_MRT_SHADER") != nullptr;
	if (!enabled) {
		return;
	}
	auto& slot = t_recent_pm4[t_recent_pm4_next++ % t_recent_pm4.size()];
	slot.count = std::min<uint32_t>(remaining_dw, static_cast<uint32_t>(slot.dwords.size()));
	if ((packet[0] >> 30u) == 3u) {
		slot.count = std::min<uint32_t>(slot.count, KYTY_PM4_LEN(packet[0]));
	}
	std::copy_n(packet, slot.count, slot.dwords.begin());
}

void DebugDumpRecentPm4() {
	std::printf("recent-pm4: last %zu packets, oldest first (header op=bits 8-15, then dwords)\n",
	            t_recent_pm4.size());
	for (uint32_t i = 0; i < t_recent_pm4.size(); i++) {
		const auto& slot = t_recent_pm4[(t_recent_pm4_next + i) % t_recent_pm4.size()];
		if (slot.count == 0) {
			continue;
		}
		std::string line = fmt::format("recent-pm4: op=0x{:02x}", (slot.dwords[0] >> 8u) & 0xffu);
		for (uint32_t k = 0; k < slot.count; k++) {
			line += fmt::format(" {:08x}", slot.dwords[k]);
		}
		std::printf("%s\n", line.c_str());
	}
	std::fflush(stdout);
}

static uint64_t GuestGpuTimestamp() {
	const uint64_t percent = Config::GetGpuTimestampScalePercent();
	const auto     now     = Sync::ReadReferenceClock();
	const auto     anchor  = g_timestamp_anchor.load(std::memory_order_relaxed);
	if (percent == 100 || anchor == 0 || now <= anchor) {
		return now;
	}
	return anchor + (now - anchor) * percent / 100u;
}

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		// Commands run between packets on behalf of other threads.
		DrainStats::Pm4OpScope op(DrainStats::NoPm4Op);
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Submission submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

CommandProcessor::CommandProcessor(RenderContext& renderer, int interrupt_event_id)
    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::RestartSpeculation(const Pm4Execution& execution) {
	thread_local std::vector<DrawSpeculator::Cursor> stack;
	stack.clear();
	for (const auto& cursor: execution.m_buffer_stack) {
		stack.push_back({cursor.commands, cursor.offset_dw});
	}
	m_speculator->Restart(*this, stack);
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_predicate_skip                   = false;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::ProcessPssr(const Pssr::Request& request) {
	m_renderer.GetRenderExecutor().ProcessPssr(m_submit_id, CurrentBuffer(), request);
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	m_renderer.GetBufferCache().RecordEagerReadbacks();
	GetScheduler().Flush();
}

void CommandProcessor::BufferFlushIfGpuIdle() {
	m_renderer.GetBufferCache().RecordEagerReadbacks();
	auto& scheduler = GetScheduler();
	// An idle GPU finishes a label's few draws in tens of microseconds, so submitting at every
	// idle label made hundreds of tiny submits a frame whose CPU cost starved the GPU. Batch
	// until the minimum interval since the last submit has passed.
	const bool batched =
	    scheduler.MicrosSinceSubmit() >= Config::GetLabelFlushIntervalUs();
	if (batched && scheduler.IsFree(scheduler.CurrentTick() - 1)) {
		scheduler.Flush();
	}
}

// The open tick an interrupt was batched into (see BufferFlushForInterrupt): a deferred slice end
// still submits it (GuestGpu::FlushSlice). GPU thread only.
static uint64_t g_interrupt_batched_tick = 0;

void CommandProcessor::BufferFlushForInterrupt() {
	// An interrupt fires when its tick completes, so that tick must be submitted. Within the
	// minimum interval it batches instead: the end of the submission slice submits at the latest.
	m_renderer.GetBufferCache().RecordEagerReadbacks();
	auto& scheduler = GetScheduler();
	if (scheduler.MicrosSinceSubmit() >= Config::GetLabelFlushIntervalUs()) {
		scheduler.Flush();
	} else {
		g_interrupt_batched_tick = scheduler.CurrentTick();
	}
}

void CommandProcessor::BufferFlushAndWait() {
	GetScheduler().FlushAndWait();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

void CommandProcessor::AdvanceBdaEpoch() {
	m_renderer.AdvanceBdaEpoch();
}

void CommandProcessor::ReportMipStats(void* dst, uint32_t size, bool reset) {
	m_renderer.ReportMipStats(dst, size, reset);
}

// KYTY_EOP_GPU_LABELS=1: an end-of-pipe label on a page the GPU has written goes to the backing
// without a write fault, and into the cached buffer in command order (BufferCache::MirrorLabel);
// WAIT_REG_MEM reads it byte-exactly (ReadGuestOnGpuThread). The plain write faulted and needed
// that page's GPU bytes first: a full GPU drain per label. The title keeps its fences next to compute
// output in one page (19 labels and ~4.7 ms of drains per frame).
static bool EopGpuLabels() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_EOP_GPU_LABELS");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

static bool EopOverwriteLabels() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_EOP_GPU_LABELS");
		return text != nullptr && std::strcmp(text, "2") == 0;
	}();
	return enabled;
}

static bool EopImageLabels() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_EOP_IMAGE_LABELS");
		return EopGpuLabels() && text != nullptr && std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

// KYTY_DEBUG_EOP_LABEL_DRAIN=1 (diagnostic, slow): an end-of-pipe label is written only after the
// GPU finished all work recorded before it, as the PS5 GPU writes it, instead of when the CP reaches
// the packet. Tests whether the game reusing memory after an early label is a race (the title haze).
static bool EopLabelDrain() {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_EOP_LABEL_DRAIN") != nullptr;
	return enabled;
}

// KYTY_EOP_LABEL_AT_GPU=1 (experiment, the title haze): an end-of-pipe label that comes with an
// interrupt reaches guest memory when the GPU has finished the work before it, as on the PS5, not
// when the CP reaches the packet. The cached buffer still gets the value in command order
// (MirrorLabel); the guest-visible store runs on the priority thread before the interrupt. The CP's
// own WAIT_REG_MEM reads the pending value (the GPU executes in order). Not covered yet: an upload
// of the page between CP time and completion carries the old value; COND_EXEC/predication reads.
static bool EopLabelAtGpu() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_EOP_LABEL_AT_GPU");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

// KYTY_EOP_LABEL_AT_GPU=2: every end-of-pipe label, with or without an interrupt, reaches guest
// memory at GPU completion (the PS5's order for the game's CPU); only labels with an interrupt
// flush at once, the others land when their command buffer is submitted and done (the title writes
// ~300 a frame). Test of whether the haze comes from the game seeing polled fences early.
static bool EopLabelAtGpuAll() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_EOP_LABEL_AT_GPU");
		return text != nullptr && std::strcmp(text, "2") == 0;
	}();
	return enabled;
}

struct PendingEopLabel {
	uint64_t address = 0;
	uint64_t value   = 0;
	uint32_t size    = 0;
	uint64_t tick    = 0;
};
static std::mutex                   g_pending_labels_mutex;
static std::vector<PendingEopLabel> g_pending_labels;
static std::atomic<uint32_t>        g_pending_labels_count {0};

// The value a deferred label store will write over [address, address + size), if one is pending.
static bool PendingEopLabelValue(uint64_t address, uint32_t size, uint64_t& value) {
	if (g_pending_labels_count.load(std::memory_order_acquire) == 0) {
		return false;
	}
	std::lock_guard lock(g_pending_labels_mutex);
	for (auto it = g_pending_labels.rbegin(); it != g_pending_labels.rend(); ++it) {
		if (it->address == address && it->size == size) {
			value = it->value;
			return true;
		}
	}
	return false;
}

// Labels the KYTY_EOP_IMAGE_LABELS paths took (printed with the "eop-label:" diagnostic).
static std::atomic<uint32_t> g_eop_image_labels {0};

static void WriteEopLabel(BufferCache& cache, void* dst, const void* data, uint32_t size) {
	const auto address = reinterpret_cast<uint64_t>(dst);
	if (BufferCache::LabelStatsEnabled()) [[unlikely]] {
		cache.NoteLabelWrite(address, size);
	}
	// KYTY_DEBUG_EOP_LABEL_CASES=1: how often each case occurs (page GPU-dirty, label bytes
	// GPU-dirty). It ran for every label (the title: ~30k/s, two tracker lookups each) until 2026-10-23.
	static const bool label_cases = KYTY_GETENV("KYTY_DEBUG_EOP_LABEL_CASES") != nullptr;
	if (label_cases && EopGpuLabels()) [[unlikely]] {
		static std::array<std::atomic<uint32_t>, 4> cases {};
		static std::atomic<uint32_t>                total {0};
		const int c = (cache.IsRegionGpuModified(address, size) ? 2 : 0) |
		              (cache.HasGpuDirtyBytes(address, size) ? 1 : 0);
		cases[c].fetch_add(1, std::memory_order_relaxed);
		if (total.fetch_add(1, std::memory_order_relaxed) % 2000 == 1999) {
			std::printf("eop-label: clean=%u bytes_only=%u page_gpu=%u page_and_bytes=%u image=%u "
			            "last=0x%016" PRIx64 "\n",
			            cases[0].load(), cases[1].load(), cases[2].load(), cases[3].load(),
			            g_eop_image_labels.load(std::memory_order_relaxed), address);
			std::fflush(stdout);
		}
	}
	if (EopGpuLabels() && cache.CanWriteBacked(address, size)) {
		cache.WriteBacked(address, data, size);
		return;
	}
	// KYTY_EOP_GPU_LABELS=2 also overwrites label bytes the GPU wrote (the title resets its fences with
	// 4-byte DMA copies of GPU-computed values): all GPU work is recorded on one command stream, so
	// the copy lands before the label's fill (TheCruZ GTA fork, WriteAroundGpuWrites).
	if (EopOverwriteLabels() && cache.CanOverwriteGpuBytes(address, size)) {
		cache.OverwriteGpuBytes(address, data, size);
		return;
	}
	// KYTY_EOP_IMAGE_LABELS=1: the same two ways for a label inside images that only ever held CPU
	// data. The title keeps two fences in the last page of old 144x64 BC3 textures' memory, and the
	// first is reset at each frame's start: the fault path drained the GPU for 1.4 ms per frame.
	// The images take the write first, as the fault path's invalidation did.
	if (EopImageLabels()) {
		if (cache.CanWriteBacked(address, size, true)) {
			g_eop_image_labels.fetch_add(1, std::memory_order_relaxed);
			cache.InvalidateImagesForWrite(address, size);
			cache.WriteBacked(address, data, size);
			return;
		}
		if (EopOverwriteLabels() && cache.CanOverwriteGpuBytes(address, size, true)) {
			g_eop_image_labels.fetch_add(1, std::memory_order_relaxed);
			cache.InvalidateImagesForWrite(address, size);
			cache.OverwriteGpuBytes(address, data, size);
			return;
		}
	}
	// KYTY_DEBUG_EOP_FALLTHROUGH=1: the labels that take neither path above with GPU-written state
	// near them (see LabelWriteFlags), and every 500th label that takes neither.
	static const bool fallthrough_log = KYTY_GETENV("KYTY_DEBUG_EOP_FALLTHROUGH") != nullptr;
	if (fallthrough_log && EopGpuLabels()) [[unlikely]] {
		static std::atomic<uint32_t> count {0};
		static std::atomic<uint32_t> flagged {0};
		const auto                   n     = count.fetch_add(1, std::memory_order_relaxed);
		const auto                   flags = cache.LabelWriteFlags(address, size);
		if ((flags != 0 && flagged.fetch_add(1, std::memory_order_relaxed) < 200) || n % 500 == 0) {
			std::printf("eop-fallthrough #%u addr=0x%016" PRIx64 " size=%u flags=0x%x\n", n, address,
			            size, flags);
			std::fflush(stdout);
		}
	}
	// KYTY_DEBUG_UPLOAD_RACE: how often a label becomes visible to the game while upload copies of
	// earlier commands still wait for the helper thread (WriteBacked joins them, this path does not).
	static const bool race_check = KYTY_GETENV("KYTY_DEBUG_UPLOAD_RACE") != nullptr;
	if (race_check) [[unlikely]] {
		static std::atomic<uint64_t> labels {0};
		static std::atomic<uint64_t> early {0};
		const auto n = labels.fetch_add(1, std::memory_order_relaxed) + 1;
		const auto e = early.fetch_add(AsyncUploadsPending() != 0 ? 1u : 0u, std::memory_order_relaxed);
		if (n % 5000 == 0) {
			std::printf("label-fallthrough: %" PRIu64 " labels, %" PRIu64 " with helper copies pending\n", n,
			            e);
			std::fflush(stdout);
		}
	}
	// KYTY_LABEL_JOIN_UPLOADS=1: as WriteBacked does, the helper's copies of earlier commands finish
	// before the game can see the label; otherwise the game may already write the next frame's data
	// into ranges still being copied (KYTY_DEBUG_UPLOAD_RACE: ~1.5 % of copies in the title).
	static const bool join_uploads = [] {
		const char* text = KYTY_GETENV("KYTY_LABEL_JOIN_UPLOADS");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	// KYTY_LABEL_JOIN_UPLOADS_OFF_FILE=<path>: no join while that file exists (in-run A/B of its cost).
	static FileSwitch join_off(KYTY_GETENV("KYTY_LABEL_JOIN_UPLOADS_OFF_FILE"), "label-join-uploads");
	if (join_uploads && AsyncUploadsPending() != 0 && !join_off.Exists()) {
		JoinAsyncUploads();
	}
	DrainStats::t_eop_label_write = true;
	std::memcpy(dst, data, size);
	DrainStats::t_eop_label_write = false;
}

// KYTY_EOP_LABEL_AT_GPU: see EopLabelAtGpu. Labels inside registered images keep the CP-time path.
static void WriteEopLabelAtGpu(BufferCache& cache, CommandScheduler& scheduler, void* dst,
                               const void* data, uint32_t size) {
	const auto address = reinterpret_cast<uint64_t>(dst);
	static std::atomic<uint32_t> deferred {0};
	static std::atomic<uint32_t> kept {0};
	if ((cache.LabelWriteFlags(address, size) & 16u) != 0 || (size != 4 && size != 8)) {
		kept.fetch_add(1, std::memory_order_relaxed);
		WriteEopLabel(cache, dst, data, size);
		return;
	}
	uint64_t value = 0;
	std::memcpy(&value, data, size);
	cache.MirrorLabel(address, data, size);
	const auto tick = scheduler.CurrentTick();
	{
		std::lock_guard lock(g_pending_labels_mutex);
		g_pending_labels.push_back({address, value, size, tick});
		g_pending_labels_count.store(static_cast<uint32_t>(g_pending_labels.size()),
		                             std::memory_order_release);
	}
	scheduler.DeferPriorityOperation([address, value, size, tick] {
		(void)LibKernel::Memory::TryWriteBacking(address, &value, size);
		std::lock_guard lock(g_pending_labels_mutex);
		const auto found = std::ranges::find_if(g_pending_labels, [&](const PendingEopLabel& label) {
			return label.address == address && label.tick == tick && label.value == value;
		});
		if (found != g_pending_labels.end()) {
			g_pending_labels.erase(found);
		}
		g_pending_labels_count.store(static_cast<uint32_t>(g_pending_labels.size()),
		                             std::memory_order_release);
	});
	if (const auto n = deferred.fetch_add(1, std::memory_order_relaxed); n % 200u == 0u) {
		std::printf("eop-label-at-gpu: deferred=%u kept=%u pending=%u\n", n + 1,
		            kept.load(std::memory_order_relaxed),
		            g_pending_labels_count.load(std::memory_order_relaxed));
		std::fflush(stdout);
	}
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	BreakComputeChain();
	// The CPU may have written the data this wait guards just before its flag.
	AdvanceBdaEpoch();
	T value {};
	if (EopGpuLabels()) {
		// Labels next to GPU-written bytes (see WriteEopLabel) read without a fault and drain.
		LibKernel::Memory::ReadGuestOnGpuThread(reinterpret_cast<uint64_t>(addr), &value,
		                                        sizeof(value));
	} else {
		value = *addr;
	}
	// KYTY_EOP_LABEL_AT_GPU: a label whose guest store waits for the GPU holds the value the GPU
	// will have written before any later command runs.
	if (uint64_t pending = 0; PendingEopLabelValue(reinterpret_cast<uint64_t>(addr), sizeof(T), pending)) {
		value = static_cast<T>(pending);
	}
	const bool satisfied = TestWaitRegMemValue(value, ref, mask, func);
	if (DrainStats::QueueSyncActive()) [[unlikely]] {
		// Address, size, compare function, reference, mask, value read, passed, bytes GPU-written.
		const auto address = reinterpret_cast<uint64_t>(addr);
		DrainStats::QueueSyncLine(
		    "r", fmt::format("{:#x} {} {} {:#x} {:#x} {:#x} {} {}", address, sizeof(T), func,
		                     static_cast<uint64_t>(ref), static_cast<uint64_t>(mask),
		                     static_cast<uint64_t>(value), satisfied ? 1 : 0,
		                     m_renderer.GetBufferCache().IsRegionGpuModified(address, sizeof(T)) ? 1 : 0));
	}
	if (!satisfied) {
		SuspendPm4();
	}
}

// KYTY_DEBUG_QUEUE_SYNC (DrainStats::QueueSyncActive): a CP write of `size` bytes at `address`
// whose first bytes are `value`.
static void QueueSyncWrite(const char* kind, const void* address, uint64_t size, uint64_t value) {
	if (DrainStats::QueueSyncActive()) [[unlikely]] {
		DrainStats::QueueSyncLine("w", fmt::format("{:#x} {} {:#x} {}",
		                                           reinterpret_cast<uint64_t>(address), size, value,
		                                           kind));
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	// KYTY_WRITE_DATA_BACKED=1: a small WRITE_DATA on a page the GPU wrote takes the end-of-pipe
	// label's way (WriteEopLabel: backing plus cached buffer in command order) instead of a write
	// fault that downloads the page first. the title in Pro mode: one such write a fight frame waited ~3 ms for
	// the GPU (gpu-thread-write-fault WRITE_DATA, 2026-10-31).
	static const bool backed = [] {
		const char* text = KYTY_GETENV("KYTY_WRITE_DATA_BACKED");
		return text != nullptr && text[0] == '1';
	}();
	// Up to 64 dwords: a run whose bytes the GPU did not write takes the backed way, one over
	// GPU-written bytes within a page replaces them (the title in Pro mode fight, runs wd2/wd9: 64-dword writes
	// on a GPU page and over GPU-written bytes, 3-dword dispatch args over GPU-written args, each
	// frame); the rest as labels.
	auto&      cache        = m_renderer.GetBufferCache();
	const auto write_bytes  = static_cast<uint32_t>(dw_num * sizeof(uint32_t));
	const auto write_target = reinterpret_cast<uint64_t>(dst);
	// KYTY_DEBUG_WRITE_DATA=1: each WRITE_DATA on a GPU-written page that neither the backed nor
	// the overwrite way below takes (the write faults and drains the GPU, unless the label path's
	// image cases take it), first 100 then every 200th: size, one-address mode, selector, label
	// flags (BufferCache::LabelWriteFlags).
	static const bool debug_write_data = KYTY_GETENV("KYTY_DEBUG_WRITE_DATA") != nullptr;
	if (debug_write_data) [[unlikely]] {
		const auto bytes = static_cast<uint64_t>(write_one_address ? 1u : dw_num) * sizeof(uint32_t);
		const bool taken = backed && !write_one_address &&
		                   ((dw_num <= 64u && cache.CanWriteBacked(write_target, bytes, false, 256)) ||
		                    (dw_num <= 64u && cache.CanOverwriteGpuBytes(write_target, bytes, false, true)));
		static std::atomic<uint32_t> seen {0};
		if (!taken && cache.IsRegionGpuModified(write_target, bytes)) {
			const auto n = seen.fetch_add(1);
			if (n < 100 || n % 200 == 0) {
				std::printf("write-data: #%u 0x%016" PRIx64 " dw=%u one=%d sel=%u flags=0x%x\n", n, write_target,
				            dw_num, write_one_address ? 1 : 0, dst_sel, cache.LabelWriteFlags(write_target, bytes));
				std::fflush(stdout);
			}
		}
	}
	if (backed && !write_one_address && dw_num <= 64u &&
	    cache.CanWriteBacked(write_target, write_bytes, false, 256)) {
		cache.WriteBacked(write_target, src, write_bytes);
	} else if (backed && !write_one_address && dw_num <= 64u &&
	           cache.CanOverwriteGpuBytes(write_target, write_bytes, false, true)) {
		cache.OverwriteGpuBytes(write_target, src, write_bytes);
	} else if (backed && !write_one_address && dw_num <= 16u) {
		WriteEopLabel(cache, dst, src, write_bytes);
	} else if (write_one_address) {
		for (uint32_t i = 0; i < dw_num; i++) {
			dst[0] = src[i];
		}
	} else {
		memcpy(dst, src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
	}
	QueueSyncWrite("wd", dst, write_one_address ? 4u : write_bytes,
	               write_one_address ? src[dw_num - 1]
	               : dw_num >= 2     ? src[0] | (static_cast<uint64_t>(src[1]) << 32u)
	                                 : src[0]);
	// Later draws of this submission must see the write, including through addresses.
	m_renderer.AdvanceBdaEpochForGpuWrite();
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = GuestGpuTimestamp();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	m_renderer.AdvanceBdaEpochForGpuWrite(); // As WriteData.
	DrainStats::Record(DrainStats::Kind::GpuTimestamp, 1);
	LogTimestampWrite("clock", num_bytes, dst_address, value);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		if (!dst_gds) {
			QueueSyncWrite("dfill", reinterpret_cast<const void*>(dst_address_or_offset), num_bytes,
			               src_address_or_offset_or_immediate & 0xffffffffu);
		}
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
	if (!dst_gds) {
		// The value is the source address: a GPU copy, its bytes land in command order.
		QueueSyncWrite("dcopy", reinterpret_cast<const void*>(dst_address_or_offset), num_bytes,
		               src_address_or_offset_or_immediate);
	}
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	if (DrainStats::QueueSyncEnabled()) [[unlikely]] {
		submission.enqueue_ns = DrainStats::TimelineNow();
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	submission.sequence = m_next_sequence++;
	m_outstanding.insert(submission.sequence);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::WaitForSubmissionsBefore(uint64_t sequence) {
	Common::LockGuard lock(m_queue_mutex);
	while (!m_outstanding.empty() && *m_outstanding.begin() < sequence && !m_stopping) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	g_gpu_state  = gpu;
	StartThreadSampler("gpu");
	ReserveCoreForCurrentThread(KYTY_GETENV("KYTY_GPU_CORE"), "gpu");
	StartThreadDumper();

	// KYTY_DEBUG_QUEUES=1: every 5 s, for each queue with work, the submissions completed and the
	// blocked passes in the window, the submissions pending, and the packet the last blocked pass
	// stopped at (for WAIT_REG_MEM, its address, reference and the memory's current value).
	static const bool debug_queues = KYTY_GETENV("KYTY_DEBUG_QUEUES") != nullptr;
	struct QueueStats {
		uint64_t                completed = 0;
		uint64_t                blocked   = 0;
		std::array<uint32_t, 8> packet {};
	};
	std::array<QueueStats, QueueCount> queue_stats {};
	auto next_queue_print = std::chrono::steady_clock::now() + std::chrono::seconds(5);

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		bool                         flush_pending  = false;
		uint64_t                     wait_read_tick = 0;
		{
			Common::LockGuard                    lock(gpu->m_queue_mutex);
			std::optional<DrainStats::WaitTimer> idle;
			uint64_t                             timeline_idle = 0;
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				if (gpu->m_slice_flush_pending) {
					// A deferred submission end (KYTY_SUBMIT_MIN_INTERVAL_US): nothing follows it, so
					// submit it before going idle.
					flush_pending = true;
					break;
				}
				if (!idle) {
					idle.emplace(DrainStats::Kind::GpuThreadIdle);
					timeline_idle = DrainStats::TimelineActive() ? DrainStats::TimelineNow() : 0;
				}
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (timeline_idle != 0) {
				std::printf("tl idle %" PRIu64 " %" PRIu64 "\n", timeline_idle, DrainStats::TimelineNow());
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				// KYTY_DEFER_DIRTY_READS: a submission whose dispatch waits for a download runs again
				// once the download's tick is done, not at every other completion.
				uint64_t read_tick = 0;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (gpu->m_queues[id].empty()) {
						continue;
					}
					const auto& front = gpu->m_queues[id].front();
					if (front.read_tick != 0 &&
					    !gpu->m_renderer.GetCommandScheduler().IsFree(front.read_tick)) {
						read_tick = read_tick == 0 ? front.read_tick : std::min(read_tick, front.read_tick);
						continue;
					}
					if (front.read_tick != 0 || !front.blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0 && gpu->m_slice_flush_pending) {
					// Every queue waits: first submit what the blocked slices recorded (the waits
					// may need the GPU to run it), outside the queue lock, then look again.
					flush_pending = true;
				} else if (selected_queue < 0 && read_tick != 0) {
					// Every other queue waits as well: wait for the download, outside the queue lock
					// (what the dispatch's read fault waited for, only later).
					wait_read_tick = read_tick;
				} else if (selected_queue < 0) {
					gpu->m_processing = false;
					DrainStats::WaitTimer poll(DrainStats::Kind::BlockedPoll);
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				} else {
					auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
					submission  = std::move(queue.front());
					queue.pop_front();
					gpu->m_submission_count--;
					gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
					gpu->m_processing = true;
					has_submission    = true;
				}
			}
		}
		if (flush_pending) {
			// Unless a flip, wait, label or readback submitted the tick since: an empty submit
			// would only start a new tick (and make every stage revalidate its images).
			gpu->m_slice_flush_pending = false;
			if (gpu->m_renderer.GetCommandScheduler().CurrentTick() == gpu->m_slice_flush_tick) {
				gpu->m_gfx_cp->BufferFlush();
			}
			continue;
		}
		if (wait_read_tick != 0) {
			gpu->m_renderer.GetBufferCache().WaitDeferredRead(wait_read_tick);
			continue;
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			command();

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		const bool complete = gpu->Process(submission);
		// The slice stopped at a dispatch waiting for a download (KYTY_DEFER_DIRTY_READS), or not (0).
		submission.read_tick = gpu->m_renderer.GetBufferCache().TakeDeferredReadTick();
		if (debug_queues) {
			auto& stats = queue_stats[submission.queue_id];
			if (complete) {
				stats.completed++;
			} else {
				stats.blocked++;
				auto commands = submission.command_execution.RemainingCommands();
				if (commands.empty()) {
					commands = submission.constant_execution.RemainingCommands();
				}
				stats.packet = {};
				std::copy_n(commands.begin(), std::min<size_t>(commands.size(), stats.packet.size()),
				            stats.packet.begin());
			}
		}

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (debug_queues && std::chrono::steady_clock::now() >= next_queue_print) {
			next_queue_print = std::chrono::steady_clock::now() + std::chrono::seconds(5);
			std::string line = "queues:";
			for (uint32_t id = 0; id < QueueCount; id++) {
				auto&      stats   = queue_stats[id];
				const auto pending = gpu->m_queues[id].size();
				if (stats.completed == 0 && stats.blocked == 0 && pending == 0) {
					continue;
				}
				line += fmt::format(" q{}: done={} blocked={} pending={}", id, stats.completed,
				                    stats.blocked, pending);
				const auto& p      = stats.packet;
				const auto  opcode = (p[0] >> 8u) & 0xffu;
				if (stats.blocked != 0 &&
				    (opcode == Pm4::IT_WAIT_REG_MEM || opcode == Pm4::IT_WAIT_REG_MEM_64)) {
					const auto address =
					    (p[2] & ~3u) | (static_cast<uint64_t>(p[3] & 0x3ffffu) << 32u);
					line += fmt::format(" wait@0x{:x} func={} ref=0x{:x} mask=0x{:x} value=0x{:x}",
					                    address, p[1] & 7u, p[4], p[5],
					                    *reinterpret_cast<const volatile uint32_t*>(address));
				} else if (stats.blocked != 0) {
					line += fmt::format(" op=0x{:02x} [{:08x} {:08x} {:08x} {:08x} {:08x}]", opcode,
					                    p[0], p[1], p[2], p[3], p[4]);
				}
				stats = {};
			}
			std::printf("%s\n", line.c_str());
			std::fflush(stdout);
		}
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
			// A suspend point may be waiting for the oldest outstanding submission.
			const bool oldest = !gpu->m_outstanding.empty() &&
			                    *gpu->m_outstanding.begin() == submission.sequence;
			gpu->m_outstanding.erase(submission.sequence);
			if (oldest) {
				gpu->m_idle.SignalAll();
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

// KYTY_SLICE_FLUSH_DEFER=1: a slice that stops at a wait leaves its commands in the open command
// buffer instead of submitting them. Every queue records into the one scheduler, so the next
// queue's commands follow them in order anyway; the submit (and with it the new tick, after which
// every stage revalidates its images) comes when a submission completes or every queue waits (see
// ThreadRun). The title switched queues ~60 times a frame. KYTY_DEBUG_AB=slicedefer alternates.
static bool SliceFlushDeferEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_SLICE_FLUSH_DEFER");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("slicedefer");
	return enabled && !(ab && AbFeatureOff());
}

// KYTY_SUBMIT_MIN_INTERVAL_US=<n>: a completed guest submission submits only if the last submit is
// at least n us old; otherwise the next submission records on into the same command buffer. The
// idle GPU thread (ThreadRun) submits a deferred end at once, as does an interrupt batched into the
// tick. The title submits ~50 command buffers a frame, each new tick costing the CP ~0.1 ms
// (revalidation). Default 0 = off; KYTY_DEBUG_AB=submitdefer alternates.
static bool SubmitDeferAllowed(uint64_t micros_since_submit) {
	static const uint64_t interval = [] {
		const char* text = KYTY_GETENV("KYTY_SUBMIT_MIN_INTERVAL_US");
		return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
	}();
	static const bool ab = AbSelected("submitdefer");
	return interval != 0 && micros_since_submit < interval && !(ab && AbFeatureOff());
}

void GuestGpu::FlushSlice(CommandProcessor& cp, bool complete) {
	// An interrupt batched into the open tick is not deferred further: it fires only once its tick
	// is submitted and done (code review 2026-10-18).
	auto&      scheduler = m_renderer.GetCommandScheduler();
	const auto tick      = scheduler.CurrentTick();
	const bool defer     = complete ? SubmitDeferAllowed(scheduler.MicrosSinceSubmit())
	                                : SliceFlushDeferEnabled();
	if (defer && g_interrupt_batched_tick != tick) {
		m_slice_flush_pending = true;
		m_slice_flush_tick    = tick;
		return;
	}
	m_slice_flush_pending = false;
	cp.BufferFlush();
}

// The submission the GPU thread is processing (diagnostics: KYTY_DEBUG_DISPATCH_INPUTS prints it).
std::atomic<uint64_t> g_debug_submission_seq {0};
std::atomic<uint32_t> g_debug_submission_queue {0};
// Diagnostics: the arguments address of an indirect dispatch the CP turns into a direct one.
uint64_t g_debug_cp_indirect_args = 0;

bool GuestGpu::Process(Submission& submission) {
	g_debug_submission_seq.store(submission.sequence, std::memory_order_relaxed);
	g_debug_submission_queue.store(submission.queue_id, std::memory_order_relaxed);
	const bool first_slice = !submission.started;
	auto&      cp          = GetProcessor(submission.queue_id);

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}
	// KYTY_DEBUG_QUEUE_SYNC: each slice's start (sequence, when the game submitted it, first slice,
	// type) and end ("end" complete, "blk" stopped at a wait).
	const bool queue_sync = DrainStats::QueueSyncActive();
	if (queue_sync) [[unlikely]] {
		DrainStats::QueueSyncLine("sub", fmt::format("{} {} {} {}", submission.sequence,
		                                             submission.enqueue_ns, first_slice ? 1 : 0,
		                                             static_cast<int>(submission.type)));
	}

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					cp.SetConstantStream(true);
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					cp.SetConstantStream(false);
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			if (progressed) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				FlushSlice(cp, complete);
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			// A slice stopped by a read deferral submitted its commands already (BufferCache::
			// ArmGpuThreadRead): marking the new, empty tick for a slice flush would submit it empty.
			if (submission.command_execution.MadeProgress() &&
			    !m_renderer.GetBufferCache().ReadDeferredInSlice()) {
				if (complete) {
					m_renderer.RunGarbageCollector();
				}
				FlushSlice(cp, complete);
			} else if (complete) {
				m_renderer.RunGarbageCollector();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			cp.EmitGlobalBarrier();
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] { ready->release(); });
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	if (queue_sync) [[unlikely]] {
		DrainStats::QueueSyncLine(complete ? "end" : "blk", std::to_string(submission.sequence));
	}
	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;
	// A submission starting or resuming: its draws must see the CPU writes made before it.
	AdvanceBdaEpoch();
	if (DrawSpeculator::Enabled() && !IsAsyncComputeQueue() && !m_constant_stream &&
	    !execution.m_buffer_stack.empty()) {
		if (m_speculator == nullptr) {
			m_speculator = std::make_unique<DrawSpeculator>(m_renderer, m_interrupt_event_id);
		}
		// An execution resuming where this processor last left it continues the stream the walk
		// is already ahead in: restarting it would throw away the draws it has prepared (a
		// submission resumes several times a frame; one draw in ten was prepared by the GPU
		// thread itself because of it). KYTY_SPEC_KEEP_ON_RESUME=0 restarts as before.
		static const bool keep_on_resume = [] {
			const char* value = KYTY_GETENV("KYTY_SPEC_KEEP_ON_RESUME");
			return value == nullptr || std::strcmp(value, "0") != 0;
		}();
		const auto& top = execution.m_buffer_stack.back();
		if (!(keep_on_resume && m_spec_execution == &execution &&
		      m_spec_packet == top.commands.data() + top.offset_dw && !m_speculator->RestartDue())) {
			RestartSpeculation(execution);
		}
	}
	struct SpeculationResumePoint {
		CommandProcessor& processor;
		Pm4Execution&     execution;
		~SpeculationResumePoint() {
			processor.m_spec_execution = &execution;
			processor.m_spec_packet    = execution.m_buffer_stack.empty()
			                                 ? nullptr
			                                 : execution.m_buffer_stack.back().commands.data() +
			                                       execution.m_buffer_stack.back().offset_dw;
		}
	} speculation_resume_point {*this, execution};

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution),
		      previous_async(DrainStats::t_async_compute) {
			g_current_processor          = &processor;
			g_current_execution          = &execution;
			DrainStats::t_async_compute  = processor.IsAsyncComputeQueue();
		}
		~ExecutionScope() {
			g_current_processor         = previous_processor;
			g_current_execution         = previous_execution;
			DrainStats::t_async_compute = previous_async;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
		bool              previous_async;
	} execution_scope(*this, execution);

	// Loading bursts span several submissions. When the previous submission created several
	// pipelines, walk this one from its first packet, so its first new pipeline is prefetched
	// too instead of starting the walk only after compiling it. Single misses during play do not
	// qualify, so ordinary frames never pay for a walk.
	constexpr uint64_t BurstPipelines = 3;
	const auto&        pipelines      = m_renderer.GetPipelineCache();
	const uint64_t     created =
	    pipelines.GraphicsPipelinesCreated() + pipelines.ComputePipelinesCreated();
	if (execution.m_buffer_stack.size() == 1 && execution.m_buffer_stack.back().offset_dw == 0) {
		if (m_last_submission_created >= BurstPipelines && m_lookahead_draws_left == 0) {
			RunPipelineLookahead(execution);
		} else if (Config::DispatchLookaheadEnabled() &&
		           DispatchLookaheadDue(pipelines.ComputePipelinesCreated())) {
			RunDispatchLookahead(execution);
		}
		m_submission_created_start = created;
	}

	ProcessPm4(execution);
	DrainStats::SetPm4Op(DrainStats::NoPm4Op);
	if (execution.m_buffer_stack.empty()) {
		m_last_submission_created = pipelines.GraphicsPipelinesCreated() +
		                            pipelines.ComputePipelinesCreated() -
		                            m_submission_created_start;
		return Pm4ProcessResult::Complete;
	}
	return Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

static bool IsDrawOpcode(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE: return true;
		default: return false;
	}
}

static bool IsDispatchOpcode(uint32_t opcode) {
	return opcode == Pm4::IT_DISPATCH_DIRECT || opcode == Pm4::IT_DISPATCH_INDIRECT;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	// KYTY_DEBUG_DRAW_PHASES: the time in each handler and between them (see Pm4OpTimer).
	static const bool timed       = DrawPhaseTimer::Hash() != 0;
	uint64_t          handler_end = 0;
	uint64_t          cpx_end     = 0;
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);
		RecordRecentPm4(packet, remaining_dw);

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		// KYTY_DEBUG_PREDICATION_STATS=1 prints, every 5 s, how many predicated packets ran and how
		// many were skipped, per opcode, to see whether predication drops whole passes.
		static const bool predication_stats = KYTY_GETENV("KYTY_DEBUG_PREDICATION_STATS") != nullptr;
		if (predication_stats && (packet_header & 1u) != 0) [[unlikely]] {
			static std::array<std::array<uint64_t, 2>, 256> counts {};
			static auto window = std::chrono::steady_clock::now();
			counts[opcode & 0xffu][ShouldSkipPredicatedPackets() ? 1 : 0]++;
			if (std::chrono::steady_clock::now() - window > std::chrono::seconds(5)) {
				window = std::chrono::steady_clock::now();
				std::string line = "predication:";
				for (uint32_t op = 0; op < 256; op++) {
					if (counts[op][0] + counts[op][1] != 0) {
						line += fmt::format(" op{:02x}={}/{}", op, counts[op][0], counts[op][1]);
					}
				}
				std::printf("%s (ran/skipped)\n", line.c_str());
				std::fflush(stdout);
				counts = {};
			}
		}
		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];
		DrainStats::SetPm4Op(DrainStats::Pm4Op(opcode, KYTY_PM4_R(packet_header)));
		DrainStats::t_predicated = (packet_header & 1u) != 0;

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		const bool     draw     = IsDrawOpcode(opcode) || IsDispatchOpcode(opcode);
		const auto&    pipelines = m_renderer.GetPipelineCache();
		const uint64_t pipelines_before =
		    draw ? pipelines.GraphicsPipelinesCreated() + pipelines.ComputePipelinesCreated() : 0;
		// Draws are numbered as the speculation walk numbers them; each takes its speculation.
		const bool speculated =
		    m_speculator != nullptr &&
		    (IsDrawOpcode(opcode) ||
		     (IsDispatchOpcode(opcode) && SpeculateDispatchesNow())) &&
		    DrawSpeculator::Enabled();
		if (speculated) {
			t_speculated_draw = m_speculator->Take(packet);
		}
		uint64_t handler_start = 0;
		if (timed) [[unlikely]] {
			handler_start = DrawPhaseTimer::Now();
			if (handler_end != 0) {
				g_pm4_ops.between += handler_start - handler_end;
			}
			g_pm4_ops.outcome = Pm4OpTimer::Drawn;
		}
		// KYTY_DEBUG_CP_SPLIT_STATS: per-frame handler classes for the CP split (tl cpx).
		static const bool cpx       = DrainStats::CpSplitStatsEnabled();
		uint64_t          cpx_start = 0;
		if (cpx) [[unlikely]] {
			cpx_start = __rdtsc();
			if (cpx_end != 0) {
				DrainStats::g_cpx_between.fetch_add(cpx_start - cpx_end, std::memory_order_relaxed);
			}
		}
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		if (cpx) [[unlikely]] {
			cpx_end = __rdtsc();
			(IsDrawOpcode(opcode)       ? DrainStats::g_cpx_draw
			 : IsDispatchOpcode(opcode) ? DrainStats::g_cpx_dispatch
			                            : DrainStats::g_cpx_other)
			    .fetch_add(cpx_end - cpx_start, std::memory_order_relaxed);
		}
		if (timed) [[unlikely]] {
			handler_end       = DrawPhaseTimer::Now();
			const auto ticks = handler_end - handler_start;
			g_pm4_ops.ticks[opcode] += ticks;
			g_pm4_ops.counts[opcode]++;
			if (opcode == Pm4::IT_NOP) {
				const auto r = KYTY_PM4_R(packet_header) & 63u;
				g_pm4_ops.nop_ticks[r] += ticks;
				g_pm4_ops.nop_counts[r]++;
			}
			if (IsDrawOpcode(opcode)) {
				const auto kind = opcode == Pm4::IT_DRAW_INDEX_AUTO ? 1u : 0u;
				g_pm4_ops.outcome_ticks[kind][g_pm4_ops.outcome] += ticks;
				g_pm4_ops.outcome_counts[kind][g_pm4_ops.outcome]++;
			}
		}
		if (speculated) {
			m_speculator->Release(t_speculated_draw);
			t_speculated_draw = nullptr;
			if (execution.m_suspended) {
				// A deferred dispatch runs again at the resume and would be numbered twice.
				m_speculator->LoseStep();
			}
		}
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return;
		}
		if (draw && DrainStats::QueueSyncEnabled()) [[unlikely]] {
			DrainStats::NoteQueueCall(IsDispatchOpcode(opcode));
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
		if (speculated && m_speculator->RestartDue()) {
			RestartSpeculation(execution);
		}
		if (draw) {
			if (m_lookahead_rewalk && LookaheadWorkFinished()) {
				RunPipelineLookahead(execution);
			} else if (m_lookahead_draws_left > 0) {
				m_lookahead_draws_left--;
			} else if (pipelines.GraphicsPipelinesCreated() + pipelines.ComputePipelinesCreated() !=
			           pipelines_before) {
				// Compute queues are walked too: their command processors run on this thread,
				// and their dispatches are among the largest compiles.
				RunPipelineLookahead(execution);
			}
		}
	}
}

void CommandProcessor::RunPipelineLookahead(const Pm4Execution& execution) {
	KYTY_PROFILER_FUNCTION();
	DrainStats::WaitTimer walk_timer(DrainStats::Kind::Lookahead);
	// With asynchronous pipelines the walk translates shaders on worker threads and does not wait
	// for them: the draws pick the work up later. Otherwise it translates them itself, so it can
	// queue each draw's pipeline parts right away. (Translating in parallel first and queuing the
	// parts in a second walk measured no better: the pipeline compiles are the critical path, and
	// the translations only compete with them for the worker threads.)
	auto&    pipelines = m_renderer.GetPipelineCache();
	uint32_t draws     = 0;
	uint32_t parts     = 0;
	bool     pending   = false;
	const bool async = Config::AsyncPipelinesEnabled();
	LookaheadPass(execution, async ? ProgramWait::Prefetch : ProgramWait::Wait, draws, parts,
	              pending);
	m_lookahead_draws_left = draws;
	// A prediction in Prefetch mode takes one step per walk (translate, then compile the module,
	// then queue the pipeline), so it walks again as that work finishes.
	m_lookahead_rewalk        = async && pending;
	m_lookahead_jobs_finished = pipelines.BackgroundShaderJobsFinished();
	m_lookahead_time          = std::chrono::steady_clock::now();
	pipelines.LogLookahead(draws, parts);
}

// KYTY_DISPATCH_LOOKAHEAD_IDLE_MS=<ms> (default 2000; 0 walks every submission as before): the
// dispatch look-ahead walks a submission only while compute pipelines were created within that
// time. Once every dispatch's pipeline exists a walk finds nothing, and the title's fight paid ~50
// walks a frame, ~1.2 ms of Thread_Gpu. After a quiet spell the first new pipeline
// compiles at its dispatch, and the walks resume with it. Every command processor runs on the GPU
// thread, so they share the clock.
bool CommandProcessor::DispatchLookaheadDue(uint64_t compute_pipelines_created) {
	static const int64_t idle_ms = [] {
		const char* text = KYTY_GETENV("KYTY_DISPATCH_LOOKAHEAD_IDLE_MS");
		return text != nullptr ? std::strtoll(text, nullptr, 10) : int64_t {2000};
	}();
	static uint64_t last_count  = UINT64_MAX;
	static auto     last_change = std::chrono::steady_clock::now();
	const auto      now         = std::chrono::steady_clock::now();
	if (compute_pipelines_created != last_count) {
		last_count  = compute_pipelines_created;
		last_change = now;
	}
	return idle_ms <= 0 || now - last_change < std::chrono::milliseconds(idle_ms);
}

void CommandProcessor::RunDispatchLookahead(const Pm4Execution& execution) {
	KYTY_PROFILER_FUNCTION();
	DrainStats::WaitTimer walk_timer(DrainStats::Kind::Lookahead);
	// Translates on this thread (ProgramWait::Wait), as the dispatch would, so the walk itself
	// queues the pipeline; the compile then runs while the packets before the dispatch execute.
	// The look-ahead state of RunPipelineLookahead is left alone.
	uint32_t dispatches = 0;
	uint32_t parts      = 0;
	bool     pending    = false;
	LookaheadPass(execution, ProgramWait::Wait, dispatches, parts, pending, true);
	if (parts != 0) {
		m_renderer.GetPipelineCache().LogLookahead(dispatches, parts);
	}
}

bool CommandProcessor::LookaheadWorkFinished() const {
	// All of it finished, or some of it and a walk costs little next to the time since the last.
	constexpr auto MinInterval = std::chrono::milliseconds(16);
	const auto&    pipelines   = m_renderer.GetPipelineCache();
	if (pipelines.BackgroundShaderJobs() == 0) {
		return true;
	}
	return pipelines.BackgroundShaderJobsFinished() != m_lookahead_jobs_finished &&
	       std::chrono::steady_clock::now() - m_lookahead_time >= MinInterval;
}

void CommandProcessor::LookaheadPass(const Pm4Execution& execution, ProgramWait wait,
                                     uint32_t& draws, uint32_t& parts, bool& pending,
                                     bool dispatches_only) {
	// Enough for a loading frame's draws; the walk costs a few microseconds per known draw.
	constexpr uint32_t MaxDraws   = 256;
	constexpr uint32_t MaxPackets = 1u << 16u;

	// A second processor holds the copied register state, so the real handlers can replay register
	// writes into it. Its other state is never used: only register packets reach it.
	auto shadow                     = std::make_unique<CommandProcessor>(m_renderer, m_interrupt_event_id);
	shadow->m_ctx                   = m_ctx;
	shadow->m_saved_ctx             = m_saved_ctx;
	shadow->m_context_state_pushed  = m_context_state_pushed;
	shadow->m_ucfg                  = m_ucfg;
	shadow->m_sh_ctx                = m_sh_ctx;
	shadow->m_user_data_marker      = m_user_data_marker;
	shadow->m_index_type_and_size   = m_index_type_and_size;

	auto&    pipelines = m_renderer.GetPipelineCache();
	auto     stack     = execution.m_buffer_stack;
	draws = 0;
	parts = 0;
	for (uint32_t packets = 0; !stack.empty() && draws < MaxDraws && packets < MaxPackets;
	     packets++) {
		auto& cursor = stack.back();
		if (cursor.offset_dw >= cursor.commands.size()) {
			stack.pop_back();
			continue;
		}
		const auto* packet    = cursor.commands.data() + cursor.offset_dw;
		const auto  total_dw  = static_cast<uint32_t>(cursor.commands.size());
		const auto  remaining = total_dw - cursor.offset_dw;
		const auto  header    = packet[0];
		if (header == 0x80000000u) {
			cursor.offset_dw++;
			continue;
		}
		// Only type-3 packets are followed; anything else ends the walk.
		const auto packet_dw = KYTY_PM4_LEN(header);
		if ((header >> 30u) != 3u || remaining < 2 || packet_dw > remaining) {
			break;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		switch (opcode) {
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX: {
				// The packets the real walk will execute next, through the same handlers.
				const auto handled =
				    g_cp_op_func[opcode](*shadow, header & ~1u, packet + 1, remaining, total_dw) + 1;
				if (handled != packet_dw) {
					stack.clear();
					continue;
				}
				break;
			}
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT: {
				// Their (register, value) pairs live in guest memory, which the handlers read
				// directly: the GPU may still own it this early, and that read would fault and
				// wait. The walk reads the pairs without faulting into a copy and hands the
				// handler that, or stops here when they are not readable yet.
				if (packet_dw != 5u) {
					stack.clear();
					continue;
				}
				const uint64_t address =
				    (static_cast<uint64_t>(packet[1]) & 0xfffffffcu) | (static_cast<uint64_t>(packet[2]) << 32u);
				const uint32_t pairs = packet[4] & 0x3fffu;
				thread_local std::vector<uint32_t> values;
				values.resize(std::max(pairs, 1u) * 2u);
				if (pairs != 0 &&
				    (address == 0 || !LibKernel::Memory::TryReadGpuCleanBacking(
				                         address, values.data(), uint64_t {pairs} * 8u))) {
					stack.clear();
					continue;
				}
				const auto copy = reinterpret_cast<uint64_t>(values.data());
				const std::array<uint32_t, 5> local {packet[0], static_cast<uint32_t>(copy),
				                                     static_cast<uint32_t>(copy >> 32u), packet[3],
				                                     packet[4]};
				const auto handled = g_cp_op_func[opcode](*shadow, header & ~1u, local.data() + 1,
				                                          remaining, total_dw) +
				                     1;
				if (handled != packet_dw) {
					stack.clear();
					continue;
				}
				break;
			}
			case Pm4::IT_CLEAR_STATE: shadow->m_ctx.Reset(); break;
			case Pm4::IT_NOP: {
				const auto r = KYTY_PM4_R(header);
				if (r == Pm4::R_ZERO && packet_dw >= 2 && (packet[1] & 0xffff0000u) == 0x68750000u) {
					// User data markers classify the SGPR writes that follow (see CpOpMarker).
					const auto id = packet[1] & 0xfffu;
					if (id == 0x4u) {
						shadow->SetUserDataMarker(HW::UserSgprType::Vsharp);
					} else if (id == 0xdu) {
						shadow->SetUserDataMarker(HW::UserSgprType::Region);
					}
				} else if (r == Pm4::R_CONTEXT_STATE && packet_dw >= 3) {
					const auto operation = packet[1];
					const bool push      = operation == static_cast<uint32_t>(ContextStateOperation::Push) ||
					                  operation == static_cast<uint32_t>(ContextStateOperation::PushClear);
					const bool pop = operation == static_cast<uint32_t>(ContextStateOperation::Pop);
					if (operation > static_cast<uint32_t>(ContextStateOperation::PushClear) ||
					    (push && shadow->m_context_state_pushed) ||
					    (pop && !shadow->m_context_state_pushed)) {
						stack.clear();
						continue;
					}
					shadow->ApplyContextStateOperation(static_cast<ContextStateOperation>(operation));
				}
				break;
			}
			case Pm4::IT_INDIRECT_BUFFER: {
				// Calls and chains are followed; branches (the 14-dword form) end the walk.
				if (packet_dw != 4u) {
					stack.clear();
					continue;
				}
				const auto* address =
				    reinterpret_cast<const uint32_t*>(packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
				const auto  size  = packet[3] & 0xfffffu;
				const bool  chain = (packet[3] & (1u << 20u)) != 0;
				cursor.offset_dw += packet_dw;
				if (size == 0) {
					continue;
				}
				if (address == nullptr) {
					stack.clear();
					continue;
				}
				const std::span<const uint32_t> commands(address, size);
				if (chain) {
					stack.back() = {commands};
				} else {
					stack.push_back({commands});
				}
				continue;
			}
			case Pm4::IT_DISPATCH_DIRECT:
				// Dispatches count as draws for the covered range, as ProcessPm4 counts them.
				draws++;
				// The dispatch path skips zero-sized dispatches before translating (see
				// CpOpDispatchDirect for the layout: groups x, y, z, then the initiator).
				if (packet_dw == 5u && packet[1] != 0 && packet[2] != 0 && packet[3] != 0) {
					const std::array<uint32_t, 3> groups {packet[1], packet[2], packet[3]};
					parts += pipelines.PrefetchComputePipeline(shadow->m_ctx, shadow->m_sh_ctx,
					                                           packet[4], wait, &pending,
					                                           &groups);
				}
				break;
			case Pm4::IT_DISPATCH_INDIRECT:
				draws++;
				// See CpOpDispatchIndirect: the initiator follows the argument address or offset.
				if ((header & ~1u) == 0xc0021600u) {
					parts += pipelines.PrefetchComputePipeline(shadow->m_ctx, shadow->m_sh_ctx,
					                                           packet[3], wait,
					                                           &pending);
				} else if ((header & ~1u) == 0xc0011600u) {
					parts += pipelines.PrefetchComputePipeline(shadow->m_ctx, shadow->m_sh_ctx,
					                                           packet[2], wait,
					                                           &pending);
				}
				break;
			default:
				if (IsDrawOpcode(opcode) && !dispatches_only) {
					draws++;
					parts += pipelines.PrefetchGraphicsPipeline(shadow->m_ctx, shadow->m_sh_ctx,
					                                            shadow->m_ucfg, wait, &pending);
				}
				// Everything else (waits, events, memory writes, dispatches) leaves the
				// register state alone and is skipped.
				break;
		}
		cursor.offset_dw += packet_dw;
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

uint32_t CommandProcessor::NumInstances() {
	if (m_num_instances_address != 0) {
		// Reading faults and waits if the GPU still owns the arguments; direct draws that rely
		// on an indirect draw's instance count are rare.
		std::memcpy(&m_num_instances, reinterpret_cast<const void*>(m_num_instances_address),
		            sizeof(m_num_instances));
		m_num_instances_address = 0;
	}
	return m_num_instances;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances = num_instances;
	m_num_instances_address = 0;
}

void CommandProcessor::SynchronizePredicate(uint64_t address, uint64_t size) {
	DrainStats::ReasonScope reason(DrainStats::Reason::Predicate);
	if (!GuestRange {address, size}.Valid()) {
		// Host-only command buffers are also used by the PM4 test harness. Unknown ownership
		// cannot justify removing the legacy synchronization.
		BufferFlushAndWait();
		GetScheduler().WaitPriorityOperations(GetScheduler().CurrentTick() - 1);
		return;
	}
	auto&      buffers  = m_renderer.GetBufferCache();
	auto&      textures = m_renderer.GetTextureCache();
	const auto sync     = ClassifyPredicateSync(textures.IsRegionGpuModified(address, size),
	                                            textures.HasPendingDownload(address, size),
	                                            buffers.IsRegionGpuModified(address, size));
	if (sync == PredicateSync::Drain) {
		// Preserve the conservative image path. Queue completion alone does not publish the
		// CPU backing: the priority runner may still own a freed image's pending writeback.
		BufferFlushAndWait();
		GetScheduler().WaitPriorityOperations(GetScheduler().CurrentTick() - 1);
	}
	if (sync == PredicateSync::Download ||
	    (sync == PredicateSync::Drain && buffers.IsRegionGpuModified(address, size))) {
		buffers.ReadMemory(address, size, false);
	}
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;
	if (op != 0) {
		AdvanceBdaEpoch();
	}

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			DrainStats::Record(DrainStats::Kind::OcclusionPredicate, 1);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			if (wait_op != 0) {
				SynchronizePredicate(reinterpret_cast<uint64_t>(address), sizeof(uint64_t));
			}
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

// Reading GPU-written arguments on the CPU would wait for the dispatch that wrote them. A
// mesh-emulated indexed draw instead builds its mesh dispatch from them on the GPU, and an indexed
// list draw with 16- or 32-bit indices draws from them directly: the guest layout is
// VkDrawIndexedIndirectCommand's. (Strips could need a primitive restart scan, and 8-bit indices
// are widened on the CPU.) Returns false when the draw has to take the CPU path.
bool CommandProcessor::TryDrawIndexedIndirectOnGpu(uint64_t address) {
	const bool mesh      = (m_ctx.GetShaderStages() & 0x20u) != 0; // As in PrepareProgram.
	const auto prim_type = m_ucfg.GetPrimType();
	const bool list      = prim_type == Prospero::PrimitiveType::kTriList ||
	                  prim_type == Prospero::PrimitiveType::kLineList ||
	                  prim_type == Prospero::PrimitiveType::kPointList;
	if (!(mesh || (list && m_index_type_and_size != 2u)) || m_index_buffer_size == 0 ||
	    !Config::GpuMeshIndirectEnabled()) {
		return false;
	}
	m_num_instances_address = address + offsetof(DrawIndexedIndirectArgs, instance_count);
	DrawIndex({.index_count    = 1,
	           .index_addr     = reinterpret_cast<const void*>(m_index_base_addr),
	           .instance_count = 1,
	           .offset_source  = DrawOffsetSource::IndirectArgs,
	           .indirect_args  = address,
	           .index_limit    = m_index_buffer_size});
	return true;
}

// The non-indexed counterpart: the guest layout is VkDrawIndirectCommand's, so a draw whose
// primitive type Vulkan draws natively (no mesh emulation, no rect or quad lists) draws from the
// GPU-written arguments directly.
bool CommandProcessor::TryDrawIndirectOnGpu(uint64_t address) {
	static const bool enabled = KYTY_GETENV("KYTY_GPU_DRAW_INDIRECT") == nullptr ||
	                            std::strcmp(KYTY_GETENV("KYTY_GPU_DRAW_INDIRECT"), "0") != 0;
	const bool mesh      = (m_ctx.GetShaderStages() & 0x20u) != 0; // As in PrepareProgram.
	const auto prim_type = m_ucfg.GetPrimType();
	const bool native    = prim_type == Prospero::PrimitiveType::kTriList ||
	                    prim_type == Prospero::PrimitiveType::kTriStrip ||
	                    prim_type == Prospero::PrimitiveType::kLineList ||
	                    prim_type == Prospero::PrimitiveType::kLineStrip ||
	                    prim_type == Prospero::PrimitiveType::kPointList;
	if (!enabled || mesh || !native) {
		return false;
	}
	m_num_instances_address = address + offsetof(DrawIndirectArgs, instance_count);
	DrawIndexAuto({.vertex_count   = 1,
	               .instance_count = 1,
	               .offset_source  = DrawOffsetSource::IndirectArgs,
	               .indirect_args  = address});
	return true;
}

// The VS InstanceID input excludes the first instance on the console; a shader that needs the
// base reads it from the user SGPR the packet names (written above). Vulkan's gl_InstanceIndex
// includes firstInstance, so that draw passes 0 when the shader gets the base through an SGPR:
// otherwise it was counted twice (Ghost of Yotei dust particles read the instance records of
// other particles and came out red).
static uint32_t DrawFirstInstance(IndirectDrawRegisters registers, uint32_t first_instance) {
	return registers.instance_offset != Pm4::SH_NOP ? 0u : first_instance;
}

// Upstream 9f83fd5f: an indirect draw writes its vertex offset, first instance and first index
// into the user SGPRs its packet names, as the command processor does on the console.
static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM || g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed,
                                    IndirectDrawRegisters registers) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	const auto* args_addr =
	    reinterpret_cast<const void*>(m_draw_indirect_args_base_addr + data_offset);
	const auto address  = m_draw_indirect_args_base_addr + data_offset;
	const auto size     = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	const bool gpu_args = m_renderer.GetBufferCache().IsRegionGpuModified(address, size);
	const bool mesh     = (m_ctx.GetShaderStages() & 0x20u) != 0; // As in PrepareProgram.
	if (DrainStats::Enabled()) {
		DrainStats::Record(gpu_args ? DrainStats::Kind::IndirectArgsGpu
		                            : DrainStats::Kind::IndirectArgsCpu,
		                   mesh ? 1 : 0);
	}
	const auto prim_type = m_ucfg.GetPrimType();
	if (registers.Any()) {
		// The register values come from the arguments: read GPU-written ones back first.
		if (gpu_args) {
			m_renderer.GetBufferCache().ReadMemory(address, size, false);
		}
	} else if (gpu_args &&
	           (indexed ? TryDrawIndexedIndirectOnGpu(address) : TryDrawIndirectOnGpu(address))) {
		return;
	}

	// KYTY_DEBUG_STALE_INDIRECT=1: GPU-written arguments the CPU reads from guest memory below
	// (stale unless read back).
	static const bool stale_log = KYTY_GETENV("KYTY_DEBUG_STALE_INDIRECT") != nullptr;
	if (stale_log && gpu_args) [[unlikely]] {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 200) {
			uint32_t words[5] {};
			std::memcpy(words, args_addr, indexed ? 20u : 16u);
			std::printf("stale-indirect: addr=0x%016llx indexed=%d mesh=%d prim=%u index_type=%u "
			            "ib_size=%u cpu_args=%u,%u,%u,%u,%u\n",
			            static_cast<unsigned long long>(address), indexed ? 1 : 0, mesh ? 1 : 0,
			            static_cast<uint32_t>(prim_type), m_index_type_and_size, m_index_buffer_size,
			            words[0], words[1], words[2], words[3], words[4]);
			std::fflush(stdout);
		}
	}
	if (!indexed) {
		DrawIndirectArgs args {};
		std::memcpy(&args, args_addr, sizeof(args));
		PatchIndirectDrawOffsets(*this, registers, args.start_vertex_location,
		                         args.start_instance_location, 0);
		m_num_instances = args.instance_count;
		m_num_instances_address = 0;
		DrawIndexAuto({.vertex_count   = args.vertex_count_per_instance,
		               .instance_count = args.instance_count,
		               .first_vertex   = args.start_vertex_location,
		               .first_instance = DrawFirstInstance(registers, args.start_instance_location),
		               .offset_source  = DrawOffsetSource::IndirectArgs});
		return;
	}

	DrawIndexedIndirectArgs args {};
	std::memcpy(&args, args_addr, sizeof(args));

	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(args.start_index_location) * index_size);

	const uint32_t index_count =
	    (m_index_buffer_size != 0 ? std::min(args.index_count_per_instance, m_index_buffer_size)
	                              : args.index_count_per_instance);
	if (GraphicsRunDebugDumpEnabled() && index_count != args.index_count_per_instance) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("\t DrawIndexIndirect: clamped index_count from %" PRIu32 " to %" PRIu32
			     " using INDEX_BUFFER_SIZE\n",
			     args.index_count_per_instance, index_count);
		}
	}

	PatchIndirectDrawOffsets(*this, registers, args.base_vertex_location,
	                         args.start_instance_location, args.start_index_location);
	m_num_instances = args.instance_count;
	m_num_instances_address = 0;
	DrawIndex({.index_count    = index_count,
	           .index_addr     = index_addr,
	           .instance_count = args.instance_count,
	           .base_vertex    = static_cast<int32_t>(args.base_vertex_location),
	           .first_instance = DrawFirstInstance(registers, args.start_instance_location),
	           .offset_source  = DrawOffsetSource::IndirectArgs});
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, uint32_t draw_initiator,
                                         bool indexed, IndirectDrawRegisters registers) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != 2u);
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		draw_count = *count_addr;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	static const bool stale_log = KYTY_GETENV("KYTY_DEBUG_STALE_INDIRECT") != nullptr;
	if (stale_log) [[unlikely]] {
		const auto base      = m_draw_indirect_args_base_addr + data_offset;
		const bool gpu_count = count_addr != nullptr &&
		                       m_renderer.GetBufferCache().IsRegionGpuModified(
		                           reinterpret_cast<uint64_t>(count_addr), sizeof(uint32_t));
		const bool gpu_args  = m_renderer.GetBufferCache().IsRegionGpuModified(
		    base, uint64_t {std::max(max_count_or_count, 1u)} * stride_in_bytes);
		static std::atomic<uint32_t> logged {0};
		if ((gpu_count || gpu_args) && logged.fetch_add(1) < 200) {
			std::printf("stale-indirect-multi: base=0x%016llx indexed=%d gpu_count=%d gpu_args=%d "
			            "max=%u cpu_count=%u stride=%u prim=%u index_type=%u ib_size=%u\n",
			            static_cast<unsigned long long>(base), indexed ? 1 : 0, gpu_count ? 1 : 0,
			            gpu_args ? 1 : 0, max_count_or_count, draw_count, stride_in_bytes,
			            static_cast<uint32_t>(m_ucfg.GetPrimType()), m_index_type_and_size,
			            m_index_buffer_size);
			std::fflush(stdout);
		}
	}
	if (draw_count == 0) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	uint64_t index_size = 0;
	if (indexed) {
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
	}

	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			const bool gpu_before =
			    m_renderer.GetBufferCache().IsRegionGpuModified(args_addr, args_size);
			if (registers.Any()) {
				if (gpu_before) {
					m_renderer.GetBufferCache().ReadMemory(args_addr, args_size, false);
				}
			} else if (gpu_before && TryDrawIndirectOnGpu(args_addr)) {
				continue;
			}
			auto* args = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			PatchIndirectDrawOffsets(*this, registers, args->start_vertex_location,
			                         args->start_instance_location, 0);
			m_num_instances = args->instance_count;
			m_num_instances_address = 0;
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = DrawFirstInstance(registers, args->start_instance_location),
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		// The count is CPU-written, but a culling pass may have written the arguments.
		const bool gpu_before = m_renderer.GetBufferCache().IsRegionGpuModified(args_addr, args_size);
		if (registers.Any()) {
			if (gpu_before) {
				m_renderer.GetBufferCache().ReadMemory(args_addr, args_size, false);
			}
		} else if (gpu_before && TryDrawIndexedIndirectOnGpu(args_addr)) {
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance, m_index_buffer_size)
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		PatchIndirectDrawOffsets(*this, registers, args->base_vertex_location,
		                         args->start_instance_location, args->start_index_location);
		m_num_instances = args->instance_count;
		m_num_instances_address = 0;
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = DrawFirstInstance(registers, args->start_instance_location),
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	// KYTY_DEBUG_DISPATCH_MODES=1 prints each new DISPATCH_INITIATOR value with the shader address.
	static const bool log_modes = KYTY_GETENV("KYTY_DEBUG_DISPATCH_MODES") != nullptr;
	if (log_modes) [[unlikely]] {
		static std::mutex                   modes_mutex;
		static std::set<uint64_t>           seen;
		const std::lock_guard               lock(modes_mutex);
		const auto& cs  = m_sh_ctx.GetCs().cs_regs;
		const auto  key = (cs.data_addr << 20u) ^ mode;
		if (seen.insert(key).second && seen.size() < 4096) {
			std::printf("dispatch-mode: shader=0x%010llx mode=0x%08x wave=%u\n",
			            static_cast<unsigned long long>(cs.data_addr), mode, Pm4::ComputeWaveSize(mode));
		}
	}

	uint32_t frame_num = 0;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		if (!m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(),
		                                                   thread_group_x, thread_group_y,
		                                                   thread_group_z, mode, MayDeferDispatch(),
		                                                   MayDeferReads())) {
			SuspendPm4();
		}
	}
}

bool CommandProcessor::MayDeferDispatch() const {
	// Only inside a PM4 walk: SuspendPm4 stops it before the packet's cursor advances, and the
	// GPU thread runs the queue's other submissions, then this packet again (blocked submissions
	// are retried when another completes, and at least every 100 ms).
	return g_current_execution != nullptr && Config::DeferDispatchesEnabled();
}

bool CommandProcessor::MayDeferReads() const {
	// The graphics queue keeps its order: only asynchronous queues run beside it on the PS5.
	return g_current_execution != nullptr && IsAsyncComputeQueue() &&
	       BufferCache::DeferDirtyReadsEnabled();
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	// Reading arguments in thread dimensions here makes the GPU thread wait for the GPU each time
	// (the title in Pro mode: ~42 dispatches a frame, 17 of 32 ms per fight frame, 2026-10-30).
	// KYTY_INDIRECT_THREADS_GPU=1 sends them to the GPU path: RenderExecutor::DispatchIndirect turns
	// them into group counts and the shader's dispatch thread extent on the GPU.
	static const bool threads_on_cpu = [] {
		const char* text = KYTY_GETENV("KYTY_INDIRECT_THREADS_GPU");
		return text == nullptr || text[0] != '1';
	}();
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0 && threads_on_cpu) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		g_debug_cp_indirect_args = args_addr;
		DispatchDirect(args->x, args->y, args->z, mode);
		g_debug_cp_indirect_args = 0;
		return;
	}
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	static const bool log_modes = KYTY_GETENV("KYTY_DEBUG_DISPATCH_MODES") != nullptr;
	if (log_modes) [[unlikely]] {
		static std::mutex         modes_mutex;
		static std::set<uint64_t> seen;
		const std::lock_guard     lock(modes_mutex);
		const auto& cs  = m_sh_ctx.GetCs().cs_regs;
		const auto  key = (cs.data_addr << 20u) ^ mode;
		if (seen.insert(key).second && seen.size() < 4096) {
			std::printf("dispatch-mode: indirect shader=0x%010llx mode=0x%08x wave=%u\n",
			            static_cast<unsigned long long>(cs.data_addr), mode,
			            Pm4::ComputeWaveSize(mode));
		}
	}
	if (!m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr,
	                                                     mode, MayDeferDispatch(), MayDeferReads())) {
		SuspendPm4();
	}
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	BreakComputeChain();
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				QueueSyncWrite("irq", dst_gpu_addr, 0, interrupt_context_id);
				m_renderer.GetBufferCache().NoteInterruptRaised();
				Sync::TriggerEopEventAtEndOfPipe(command, m_interrupt_event_id,
				                                 interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	auto write32 = [&](bool with_writeback) {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		QueueSyncWrite(with_interrupt ? "eopi" : "eop", dst, sizeof(data), data);
		if (with_interrupt) {
			m_renderer.GetBufferCache().NoteInterruptRaised();
		}
		if (EopLabelDrain()) [[unlikely]] {
			SynchronizeGpu();
		}
		if ((with_interrupt && EopLabelAtGpu()) || EopLabelAtGpuAll()) {
			WriteEopLabelAtGpu(m_renderer.GetBufferCache(), GetScheduler(), dst, &data, sizeof(data));
		} else {
			WriteEopLabel(m_renderer.GetBufferCache(), dst, &data, sizeof(data));
		}

		if (with_interrupt) {
			if (with_writeback) {
				Sync::WriteAtEndOfPipeWithInterruptWriteBack32(m_submit_id, command, dst,
				                                               data, m_interrupt_event_id,
				                                               interrupt_context_id);
			} else {
				Sync::WriteAtEndOfPipeWithInterrupt32(m_submit_id, command, dst, data,
				                                      m_interrupt_event_id, interrupt_context_id);
			}
		} else if (with_writeback) {
			Sync::WriteAtEndOfPipeWithWriteBack32(m_submit_id, command, dst, data);
		} else {
			Sync::WriteAtEndOfPipe32(m_submit_id, command, dst, data);
		}
		// The deferred store lands only when this command buffer completes: submit it now, or a
		// game waiting for the label (while the CP waits for the game) never gets it.
		if (with_interrupt && EopLabelAtGpu()) {
			BufferFlush();
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					QueueSyncWrite("gds", dst, sizeof(uint32_t), value);
					SynchronizeGpu();
					Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, value & 0xffffu,
					              value >> 16u);
					Sync::WriteAtEndOfPipeGds32(m_submit_id, command, dst, value & 0xffffu,
					                            value >> 16u);
					if (with_interrupt) {
						m_renderer.GetBufferCache().NoteInterruptRaised();
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32(false);
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00: write32(false); return;
						case 0x38: write32(true); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = GuestGpuTimestamp();
					DrainStats::Record(DrainStats::Kind::GpuTimestamp, 1);
					LogTimestampWrite("eop", eop_event_type | (event_index << 8u),
					                  reinterpret_cast<uint64_t>(dst_gpu_addr), value);
				}
				auto write64 = [&](bool with_writeback) {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					QueueSyncWrite(with_interrupt ? "eopi" : "eop", dst, sizeof(value), value);
					if (with_interrupt) {
						m_renderer.GetBufferCache().NoteInterruptRaised();
					}
					if (EopLabelDrain()) [[unlikely]] {
						SynchronizeGpu();
					}
					if ((with_interrupt && EopLabelAtGpu()) || EopLabelAtGpuAll()) {
						WriteEopLabelAtGpu(m_renderer.GetBufferCache(), GetScheduler(), dst, &value,
						                   sizeof(value));
					} else {
						WriteEopLabel(m_renderer.GetBufferCache(), dst, &value, sizeof(value));
					}

					if (with_interrupt) {
						if (with_writeback) {
							Sync::WriteAtEndOfPipeWithInterruptWriteBack64(
							    m_submit_id, command, dst, value, m_interrupt_event_id,
							    interrupt_context_id);
						} else {
							Sync::WriteAtEndOfPipeWithInterrupt64(m_submit_id, command, dst,
							                                      value, m_interrupt_event_id,
							                                      interrupt_context_id);
						}
					} else if (with_writeback) {
						Sync::WriteAtEndOfPipeWithWriteBack64(m_submit_id, command, dst,
						                                      value);
					} else {
						Sync::WriteAtEndOfPipe64(m_submit_id, command, dst, value);
					}
					if (with_interrupt && EopLabelAtGpu()) {
						BufferFlush();
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64(false);
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64(false);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64(false);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64(true);
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64(true);
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64(true);
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier(uint32_t source) {
	// KYTY_DEBUG_GLOBAL_BARRIER_SKIP=1: upper-bound measurement for upstream 70668fbd9 (which drops
	// these barriers): none is recorded. Every 5 s the count by source (event type, 0x100 release
	// mem, 0 suspend point) is printed either way when KYTY_DEBUG_GLOBAL_BARRIER_COUNT=1.
	// Measurement only, the picture may break.
	static const bool skip  = KYTY_GETENV("KYTY_DEBUG_GLOBAL_BARRIER_SKIP") != nullptr;
	static const bool count = KYTY_GETENV("KYTY_DEBUG_GLOBAL_BARRIER_COUNT") != nullptr;
	if (count) [[unlikely]] {
		static std::map<uint32_t, uint64_t> n;
		static auto                         start = std::chrono::steady_clock::now();
		n[source]++;
		if (std::chrono::steady_clock::now() - start >= std::chrono::seconds(5)) {
			std::string line = "global-barrier 5 s:";
			for (const auto& [kind, value]: n) {
				line += fmt::format(" {:#x}={}", kind, value);
			}
			std::printf("%s%s\n", line.c_str(), skip ? " (skipped)" : "");
			std::fflush(stdout);
			n.clear();
			start = std::chrono::steady_clock::now();
		}
	}
	if (skip) [[unlikely]] {
		return;
	}
	Common::LockGuard lock(m_renderer.GetMutex());
	// KYTY_GLOBAL_BARRIER_LITE=1 (see upstream 70668fbd9): ~1100 of these a
	// frame in a the title arena cost the GPU ~3 ms. The renderer's resource barriers order GPU work;
	// ending the pass here keeps the game's flush points for the deferred draw-write barriers.
	if (GlobalBarrierLiteEnabled()) {
		// KYTY_GLOBAL_BARRIER_LITE=2: not even the pass end (the title arena: no draw writes memory,
		// arena13wd), as upstream 70668fbd9; BeginRendering then drops a consumed clear (context.cpp).
		if (GlobalBarrierLiteLevel() < 2) {
			GetScheduler().EndRendering();
		}
		// The full barrier also ended a compute chain (HandleForFullBarrier): a flush point is where
		// the game orders dispatches, so the chain's pending access barrier is recorded here (advisor
		// review 2026-10-22; without it dispatches ran unordered across CS partial flushes).
		auto& buffer = CurrentBuffer();
		if (GetScheduler().Active() && !buffer.IsInvalid() && buffer.ComputeChainPending()) {
			(void)buffer.Handle();
		}
		return;
	}

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask =
	    LaterAccessOr(LaterSite::GuestFlush, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	CurrentBuffer().HandleForFullBarrier().pipelineBarrier2(dependency);
}

void CommandProcessor::BreakComputeChain() {
	if (!GetScheduler().Active()) {
		return;
	}
	Common::LockGuard lock(m_renderer.GetMutex());
	auto& buffer = CurrentBuffer();
	if (!buffer.IsInvalid() && buffer.ComputeChainPending()) {
		(void)buffer.Handle();
	}
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	QueueSyncWrite("irq", nullptr, 0, interrupt_context_id);
	m_renderer.GetBufferCache().NoteInterruptRaised();
	BreakComputeChain();
	Sync::TriggerEopEventAtEndOfPipe(CurrentBuffer(), m_interrupt_event_id, interrupt_context_id);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: EmitGlobalBarrier(event_type); break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier(event_type);
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			EmitGlobalBarrier(event_type);
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			BreakComputeChain();
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			DrainStats::Record(DrainStats::Kind::OcclusionQuery, 1);
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			// KYTY_DEBUG_OCCLUSION_HIDDEN_FILE=<path>: while that file exists, the counter stands
			// still, so every query reports no samples and the game drops what it tests: an upper
			// bound for what real occlusion queries could save, and which objects they cover.
			static const char* hidden_file = KYTY_GETENV("KYTY_DEBUG_OCCLUSION_HIDDEN_FILE");
			static bool        hidden      = false;
			static uint32_t    checks      = 0;
			if (hidden_file != nullptr && checks++ % 256 == 0) {
				FILE* probe = std::fopen(hidden_file, "rb");
				hidden      = probe != nullptr;
				if (probe != nullptr) {
					std::fclose(probe);
				}
			}
			if (!hidden) {
				m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			}
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	DebugFrameLimit();
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	const auto flip_time = Sync::ReadReferenceClock();
	g_timestamp_anchor.store(flip_time, std::memory_order_relaxed);
	LogTimestampWrite("flip", 0, 0, flip_time, false);
	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeOnlyFlip(m_submit_id, command, m_flip.handle, m_flip.index,
	                               m_flip.flip_mode, m_flip.flip_arg, request);
	m_renderer.TickFrame();
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	DebugFrameLimit();
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithFlip32(m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr),
	                                 value, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                 m_flip.flip_arg, request);
	m_renderer.TickFrame();
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	DebugFrameLimit();
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	std::memcpy(dst_gpu_addr, &value, sizeof(value));
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	Sync::WriteAtEndOfPipeWithInterruptWriteBackFlip32(
	    m_submit_id, command, static_cast<uint32_t*>(dst_gpu_addr), value, m_flip.handle,
	    m_flip.index, m_flip.flip_mode, m_flip.flip_arg, request, m_interrupt_event_id);
	m_renderer.TickFrame();
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	m_renderer.TickFrame();
	GetScheduler().Flush();
}

void CommandProcessor::SynchronizeGpu() {
	DrainStats::ReasonScope reason(DrainStats::Reason::GdsReadback);
	GetScheduler().Finish();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
