#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAINSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAINSTATS_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

// Opt-in accounting of host waits on the GPU (--drain-stats). Wait sites record their kind and
// duration under the calling thread's innermost reason scope and the PM4 packet it is executing.
// When disabled, a wait site costs one relaxed load.
namespace Libs::Graphics::DrainStats {

enum class Reason : uint8_t {
	Unattributed,
	GuestReadFault,
	GuestWriteFault,
	GpuThreadReadFault,
	GpuThreadWriteFault,
	KernelInvalidate,
	DccClear,
	Predicate,
	GdsReadback,
	Unmap,
	TexturePendingDownload,
	FreeImage,
	TextureGc,
	BufferGc,
	UploadRingWrap,
	StreamRingWrap,
	DownloadRingWrap,
	FaultBuffer,
	PresentFrame,
	EagerReadback,
	Count,
};

enum class Kind : uint8_t {
	FullDrain,     // Submits the open command buffer and waits for it.
	TickWait,      // Blocks on an already submitted tick.
	PriorityWait,  // Blocks until publication callbacks up to a tick have run.
	BlockedPoll,   // Thread_Gpu idles because every queue is suspended.
	Readback,      // A buffer download was recorded; the value is its size in bytes.
	ReadbackClean, // ReadMemory found nothing to download.
	DccMetaWrite,  // A GPU write covered known DCC metadata; the value is its size in bytes.
	DccCheck,      // A DCC lookup read GPU-written metadata; the value is slices it cleared.
	DccGpuCheck,   // A DCC lookup checked GPU-written metadata on the GPU; the value is slices.
	Submit,        // vkQueueSubmit, including the wait for the queue lock; the value is ns.
	QueueLockWait, // Time a submit waited for the queue lock (held by present or another submit).
	IndirectArgsCpu, // An indirect draw read CPU-clean args; the value counts mesh-emulated draws.
	IndirectArgsGpu, // An indirect draw read GPU-written args; the value counts mesh-emulated draws.
	GpuBusy,       // GPU execution time (union of command-buffer intervals), in ns.
	GpuGap,        // GPU time between command buffers with none executing, in ns.
	OcclusionQuery,     // A ZPASS_DONE occlusion-counter dump.
	OcclusionPredicate, // SET_PREDICATION on occlusion results.
	GpuTimestamp,       // A GPU timestamp write (RELEASE_MEM/EOP or COPY_DATA of the clock).
	ShaderCompile,      // Shader lookup that took at least 1 ms (a new permutation), in ns.
	PipelineCreate,     // Pipeline lookup that took at least 1 ms (a new pipeline), in ns.
	Lookahead,          // A pipeline look-ahead walk, including the shaders it translated, in ns.
	GpuThreadIdle,      // Thread_Gpu waiting for a submission or command, in ns.
	StaleRead,          // A guest read answered with previous bytes (relaxed readback).
	Count,
};

// GPU work zones (KYTY_GPU_ZONES=1 with --drain-stats). The render scheduler timestamps each
// change of zone in its recording buffer and charges the interval up to the next timestamp to
// the zone that opened it. Work recorded without a mark joins the zone before it.
enum class Zone : uint8_t {
	Unmarked,           // From the start of a command buffer to its first mark.
	GameDraw,           // A guest draw, keyed by its pixel shader hash (vertex shader without one).
	GameDrawPredicated, // A GameDraw under SET_PREDICATION, e.g. behind an occlusion query.
	GameDispatch,       // A guest compute dispatch, keyed by its shader hash.
	MeshArgs,           // The mesh indirect-arguments pre-pass.
	Tiler,              // Tiling and detiling compute for texture uploads and downloads.
	DccClear,           // DCC fast-clear resolves.
	FaultBuffer,        // The GPU page-fault buffer scan.
	Blit,               // Blit-helper draws (format conversions and scaled copies).
	ImageCopy,          // Image copies and clears recorded by images and the texture cache.
	BufferCopy,         // Buffer cache and stream-buffer copies, uploads, and downloads.
	GameDispatchAsync,  // A GameDispatch from an asynchronous compute queue (ACB).
	Count,
};

struct ZoneSample {
	Zone     zone;
	uint64_t key;
	uint64_t pixels; // Render area of the interval's first draw, or 0.
	uint64_t ns;
};

// PM4 packets are indexed by opcode; IT_NOP packets carrying a Kyty custom code use 256 + code.
constexpr uint32_t Pm4OpCount = 256 + 64;
constexpr uint32_t NoPm4Op    = Pm4OpCount;

inline std::atomic_bool         g_enabled {false};
inline thread_local Reason      t_reason = Reason::Unattributed;
inline thread_local uint32_t    t_pm4_op = NoPm4Op;
// The compute shader whose bindings are being prepared (diagnostics: gpu-write-blame).
inline thread_local uint64_t    t_binding_shader = 0;
// An end-of-pipe label write is faulting (diagnostics: gpu-write-blame names only these).
inline thread_local bool        t_eop_label_write = false;
// The PM4 packet being executed carries the predication bit.
inline thread_local bool        t_predicated = false;
// Set while an asynchronous compute queue's command processor runs (GPU zones tell its
// dispatches apart: they share the graphics queue here, unlike on the PS5).
inline thread_local bool        t_async_compute = false;
// Guest instruction that raised the page fault being handled on this thread.
inline thread_local uint64_t    t_fault_pc = 0;

[[nodiscard]] inline bool Enabled() noexcept {
	return g_enabled.load(std::memory_order_relaxed);
}

// Draw and dispatch packets Thread_Gpu processed since the last new frame (timeline "tl count"
// lines: whether heavy frames have more calls or more expensive ones).
inline std::atomic<uint64_t> g_frame_draws {0};
inline std::atomic<uint64_t> g_frame_dispatches {0};
inline void NoteCallPacket(bool dispatch) noexcept {
	if (Enabled()) {
		(dispatch ? g_frame_dispatches : g_frame_draws).fetch_add(1, std::memory_order_relaxed);
	}
}
// Per new frame as well ("tl frx" lines): ns Submit spent joining the upload helper, ns and count of
// BDA synchronization passes, bytes uploaded into buffers (heavy frames with the same call count).
inline std::atomic<uint64_t> g_frame_join_ns {0};
inline std::atomic<uint64_t> g_frame_bda_ns {0};
inline std::atomic<uint64_t> g_frame_bda_passes {0};
inline std::atomic<uint64_t> g_frame_upload_bytes {0};
// KYTY_DEBUG_CP_SPLIT_STATS=1 (with the timeline): TSC ticks Thread_Gpu spent per frame in draw,
// dispatch and other PM4 handlers and between handlers ("tl cpx" lines; CP split slice S0).
[[nodiscard]] inline bool CpSplitStatsEnabled() noexcept {
	static const bool enabled = std::getenv("KYTY_DEBUG_CP_SPLIT_STATS") != nullptr;
	return enabled;
}
inline std::atomic<uint64_t> g_cpx_draw {0};
inline std::atomic<uint64_t> g_cpx_dispatch {0};
inline std::atomic<uint64_t> g_cpx_other {0};
inline std::atomic<uint64_t> g_cpx_between {0};
inline void NoteFrameUpload(uint64_t bytes) noexcept {
	if (Enabled()) {
		g_frame_upload_bytes.fetch_add(bytes, std::memory_order_relaxed);
	}
}

// KYTY_DEBUG_GPU_TIMELINE=<start s>:<seconds> (with --drain-stats): during that window after the
// first call, "tl" lines in stdout give each command buffer's handover time on the GPU thread and
// its GPU start and end timestamps, and each GPU-thread idle period (waiting for the game), all in
// ns, so a log reader can align the two clocks.
[[nodiscard]] bool     TimelineActive() noexcept;
[[nodiscard]] uint64_t TimelineNow() noexcept;
// A blocking wait for the GPU (full drain, tick or priority wait) on the timeline: "tl wait"
// <begin ns> <end ns> <kind> <reason> <PM4 packet>.
// The packet, drain reason and thread this thread works for (diagnostic lines).
[[nodiscard]] std::string CurrentContext();
void                   TimelineWait(Kind kind, Reason reason, uint32_t pm4_op,
                                    std::chrono::steady_clock::time_point begin,
                                    std::chrono::steady_clock::time_point end) noexcept;

// KYTY_DEBUG_QUEUE_SYNC=1 (in the KYTY_DEBUG_GPU_TIMELINE window): "tl qs" lines name the guest
// queue (0 graphics, 1 + n asynchronous compute queue n) of each CP label or memory write,
// WAIT_REG_MEM, ACQUIRE_MEM, submission slice and CP wait for the GPU, with the queue's draws and
// dispatches so far; a CP read fault names the queue whose command last wrote the bytes.
// A log reader counts the waits between the queues per frame: could the
// asynchronous queues run on a Vulkan queue of their own (the title in Pro mode, 2026-10-31)?
[[nodiscard]] bool     QueueSyncEnabled() noexcept;
[[nodiscard]] bool     QueueSyncActive() noexcept;
// The guest queue Thread_Gpu processes (see above).
[[nodiscard]] uint32_t CurrentGuestQueue() noexcept;
// Prints "tl qs <what> <ns> <queue> <draws> <dispatches> <text>".
void                   QueueSyncLine(const char* what, const std::string& text) noexcept;
// Thread_Gpu finished a draw or dispatch packet of the current queue.
void                   NoteQueueCall(bool dispatch) noexcept;
// A command of the current queue writes [vaddr, vaddr + size) (buffer cache).
void                   NoteQueueWrite(uint64_t vaddr, uint64_t size) noexcept;
// Thread_Gpu faults reading vaddr: prints the command that last wrote it.
void                   NoteQueueReadFault(uint64_t vaddr) noexcept;

// Starts the periodic reporter. Records are ignored until this is called.
void Start(uint32_t interval_seconds);
void Stop();

void Record(Kind kind, Reason reason, uint32_t pm4_op, uint64_t value) noexcept;
void CountFrame(bool new_frame) noexcept;
// A GPU-memory fault stalled the faulting thread for `ns`; keyed by the faulting instruction.
void RecordFaultSite(uint64_t pc, uint64_t address, bool write, uint64_t ns) noexcept;
// A recorded GPU command writes [vaddr, vaddr+size). Read fault sites report the newest such
// writer of their address and how many frames ago it was recorded.
void RecordGpuWrite(uint64_t vaddr, uint64_t size) noexcept;
// GPU time of one command buffer's zone intervals.
void RecordZones(const ZoneSample* samples, size_t count) noexcept;

inline void Record(Kind kind, uint64_t value) noexcept {
	if (Enabled()) {
		Record(kind, t_reason, t_pm4_op, value);
	}
}

[[nodiscard]] inline Reason CurrentReason() noexcept {
	return t_reason;
}

[[nodiscard]] inline uint32_t Pm4Op(uint32_t opcode, uint32_t custom) noexcept {
	return opcode == 0x10u && custom < 64u ? 256u + custom : (opcode & 0xffu);
}

inline void SetPm4Op(uint32_t op) noexcept {
	t_pm4_op = op;
}

class ReasonScope final {
public:
	explicit ReasonScope(Reason reason) noexcept: m_previous(t_reason) { t_reason = reason; }
	~ReasonScope() { t_reason = m_previous; }
	ReasonScope(const ReasonScope&)            = delete;
	ReasonScope& operator=(const ReasonScope&) = delete;

private:
	Reason m_previous;
};

class Pm4OpScope final {
public:
	explicit Pm4OpScope(uint32_t op) noexcept: m_previous(t_pm4_op) { t_pm4_op = op; }
	~Pm4OpScope() { t_pm4_op = m_previous; }
	Pm4OpScope(const Pm4OpScope&)            = delete;
	Pm4OpScope& operator=(const Pm4OpScope&) = delete;

private:
	uint32_t m_previous;
};

// Times a lookup that is usually a cache hit and records it only when it took at least 1 ms,
// i.e. it compiled something. Costs one relaxed load when drain stats are off.
class SlowLookupTimer final {
public:
	explicit SlowLookupTimer(Kind kind) noexcept: m_kind(kind), m_enabled(Enabled()) {
		if (m_enabled) {
			m_start = std::chrono::steady_clock::now();
		}
	}
	~SlowLookupTimer() {
		if (m_enabled) {
			const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
			                    std::chrono::steady_clock::now() - m_start)
			                    .count();
			if (ns >= 1000000) {
				Record(m_kind, t_reason, t_pm4_op, static_cast<uint64_t>(ns));
			}
		}
	}
	SlowLookupTimer(const SlowLookupTimer&)            = delete;
	SlowLookupTimer& operator=(const SlowLookupTimer&) = delete;

private:
	std::chrono::steady_clock::time_point m_start {};
	Kind                                  m_kind;
	bool                                  m_enabled;
};

// Times one blocking wait. Construct it only on the path that actually blocks.
class WaitTimer final {
public:
	explicit WaitTimer(Kind kind) noexcept: m_kind(kind), m_enabled(Enabled()) {
		if (m_enabled) {
			m_start = std::chrono::steady_clock::now();
		}
	}
	~WaitTimer() {
		if (m_enabled) {
			const auto end     = std::chrono::steady_clock::now();
			const auto elapsed = end - m_start;
			Record(m_kind, t_reason, t_pm4_op,
			       static_cast<uint64_t>(
			           std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
			if (m_kind == Kind::FullDrain || m_kind == Kind::TickWait ||
			    m_kind == Kind::PriorityWait) {
				TimelineWait(m_kind, t_reason, t_pm4_op, m_start, end);
			}
		}
	}
	WaitTimer(const WaitTimer&)            = delete;
	WaitTimer& operator=(const WaitTimer&) = delete;

private:
	std::chrono::steady_clock::time_point m_start {};
	Kind                                  m_kind;
	bool                                  m_enabled;
};

} // namespace Libs::Graphics::DrainStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAINSTATS_H_
