#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace Libs::Graphics {

// Counted for KYTY_DEBUG_DRAW_LOG: dynamic rendering instances begun and ended, and image
// barriers recorded.
struct RenderDebugCounters {
	std::atomic<uint64_t> render_begins {0};
	std::atomic<uint64_t> render_ends {0};
	std::atomic<uint64_t> image_barriers {0};
	std::atomic<bool>     counting {false};     // Set while draws are logged or summarized.
	std::atomic<bool>     log_barriers {false}; // Set while a logged draw records.
};
inline RenderDebugCounters g_render_debug_counters;

// With KYTY_DEBUG_DRAW_PHASES set, KYTY_DEBUG_AB=<features> turns the named features off in every
// other 5 s window, so one run compares both under the same scene and machine load; each
// draw-phases line reports its window's mode. Features: reuse (texture and render-target reuse;
// also KYTY_DEBUG_AB=1), pending (rate-limited GPU progress queries at draw entry), backing
// (lock-free cached guest backing reads), bdaepoch (one BDA synchronization per epoch), scratch
// (pooled tiler scratch buffers), cleanpages (remembered clean pages for GPU-write queries),
// cpwrite (BDA epochs started by the GPU thread's own guest memory writes), streamhost (the stream
// ring in cached host memory rather than device memory), colorclear (skipped color clear
// rechecks), prioritywait (draw entry leaving operations queued rather than waiting for the
// priority thread), pushpayload (push constants and descriptors handed to the recording thread
// as one payload call), pipelinememo (the last graphics pipeline reused without a map lookup),
// depthreuse (a clean depth target keeping its view between draws), regiongen (image lookups and
// views checked against the images over their own range rather than the whole image set),
// streamreuse (a range copied into the stream ring earlier in the BDA epoch bound again),
// metamemo (the last depth surface clear-state answer kept without the texture cache lock),
// storagereuse (a clean storage image keeping its view between draws), rangememo (range
// generations kept while the whole image set is unchanged), budgetcache (VMA's memory budget
// fetched from the driver at most every 500 ms), specspin (the draw speculation thread spinning
// for a restart before sleeping), speclead (draws within SpeculationLead of the GPU thread left to
// it), specreads (a speculation's guest reads kept without making them again while its BDA epoch
// lasts and nothing became GPU-written), exactwrite (a buffer write over exactly one image's
// range leaving the images it only partly overlaps as they were), viewmemo (a resolved texture
// binding starting from the view its description's last binding acquired), imagegroups (a stage
// repeating its program and image descriptors keeping its last images and views), nullreuse (a
// null texture binding keeping its image and view while its descriptor repeats), imagebuf (a
// repeatedly re-uploaded CPU-written image range held in a buffer, copying only written pages),
// suballoc (game buffers up to 64 MiB sharing VMA's memory blocks rather than taking dedicated
// memory), bindplan (descriptor payloads filled from a per-program binding plan).
[[nodiscard]] bool AbSelected(const char* feature) noexcept;
// KYTY_DEBUG_FULL_BARRIERS=1: every draw and dispatch waits for all earlier GPU work and sees all
// its memory writes (RecordFullBarrier). Slow; it tells synchronization bugs (a result that changes)
// from bugs that do not depend on timing.
[[nodiscard]] bool DebugFullBarriers() noexcept;
// KYTY_COMPUTE_CHAIN=1: consecutive dispatches without a guest synchronization point between them
// run without host barriers between them (CommandBuffer::ChainHandle). Off with full barriers.
[[nodiscard]] bool ComputeChainEnabled() noexcept;
// KYTY_DEBUG_READBACK_AGE=1: classifies guest waits for GPU-written bytes by whether their last
// GPU write had finished (see BufferCache::InflightWriteTick).
[[nodiscard]] bool ReadbackAgeEnabled() noexcept;
// Counts dispatches and how many continued a chain; KYTY_COMPUTE_CHAIN=1 prints the share
// every 50000 dispatches.
void NoteComputeChain(bool continued) noexcept;
// Counts dispatches that ended an open rendering instance (the compute chain line prints them).
void NoteDispatchPassBreak(bool rendering) noexcept;
// KYTY_DEBUG_SKIP_SHADERS=<shader hashes, hex, comma-separated>: draws with a listed pixel or
// vertex shader and dispatches of a listed compute shader are left out, to find the draws behind a
// visual defect. An entry <value>/<mask> skips every pixel and compute shader whose hash matches
// value under mask (p: or c: before it limits that to one kind), to bisect an unknown shader.
// Masks leave out only calls that pass bisect: draws of more than two triangles or with
// GPU-written counts, so that the frame's full-screen passes stay.
enum class DebugShaderKind { Pixel, Vertex, Compute };
[[nodiscard]] bool DebugSkipShader(uint64_t shader_hash, DebugShaderKind kind,
                                   bool bisect = true) noexcept;
void               RecordFullBarrier(vk::CommandBuffer command) noexcept;
[[nodiscard]] bool AbFeatureOff() noexcept;
// In-run A/B by a file (KYTY_*_OFF_FILE=<path>): whether the file exists, looked at every 100 ms
// whatever the call rate; each change is printed with the steady clock in seconds, as the run
// scripts log theirs. No path: always false. Usage: static FileSwitch off(KYTY_GETENV("..."), "name").
class FileSwitch {
public:
	FileSwitch(const char* path, const char* name) noexcept: m_path(path), m_name(name) {}
	[[nodiscard]] bool Exists() noexcept;

private:
	const char*          m_path;
	const char*          m_name;
	std::atomic<int64_t> m_next_check {0};
	std::atomic<bool>    m_exists {false};
};
// GPU busy time as drain stats measure it (--drain-stats), for the draw-phases line's gpu-ms/s.
inline std::atomic<uint64_t> g_gpu_busy_ns {0};
// GPU buffers and images created and destroyed, for the draw-phases line's allocation churn.
struct AllocationCounters {
	std::atomic<uint64_t> buffers_created {0}; // Every Buffer, temporary staging included.
	std::atomic<uint64_t> buffers_destroyed {0};
	std::atomic<uint64_t> game_buffers_created {0};   // BufferCache::CreateBuffer.
	std::atomic<uint64_t> game_buffers_joined {0};    // Merged into a larger one (JoinOverlap).
	std::atomic<uint64_t> game_buffers_collected {0}; // Deleted by garbage collection.
	std::atomic<uint64_t> images_created {0};
	std::atomic<uint64_t> images_destroyed {0};
	std::atomic<uint64_t> buffer_create_ns {0}; // Time in vmaCreateBuffer.
};
inline AllocationCounters g_allocation_counters;
// How often a stage binds the same program and descriptors as its previous draw, for the
// draw-phases line (see NoteBindingRepeat in descriptors.cpp).
struct BindingRepeats {
	std::atomic<uint64_t> stages {0};
	std::atomic<uint64_t> same_program {0};
	std::atomic<uint64_t> same_images {0};
	std::atomic<uint64_t> same_buffers {0};
	std::atomic<uint64_t> same_all {0};
};
inline BindingRepeats g_binding_repeats;

// The info display (F3, see SystemOverlay): while it is on, draws count themselves and their
// viewport sizes (see DebugResolution in renderDraw.cpp), and the display reads the results.
struct HudStats {
	std::atomic<bool>     active {false};
	// The most drawn viewport of the last second that is no square (shadow maps), as w << 32 | h.
	std::atomic<uint64_t> render_size {0};
	std::atomic<uint64_t> draws_per_second {0};
	// The game's frame rate as the window title shows it, times 10.
	std::atomic<uint32_t> game_fps_x10 {0};
};
inline HudStats g_hud;

// KYTY_DEBUG_BIND_STATS=1 (with KYTY_DEBUG_DRAW_PHASES): why a stage's image group was not kept,
// and which draws only write depth, printed as "image-groups" and "depth-only" lines after each
// draw-phases line. Render thread only.
struct BindStats {
	enum ImageGroupOutcome : uint32_t {
		NoImages,      // The program binds no images.
		Kept,          // PrepareBindings kept the group.
		Disabled,      // Image groups are off.
		NotRecorded,   // The stage's last draw recorded no group (see refused).
		OtherProgram,  // The last group is of another program.
		OtherImages,   // The image descriptors changed.
		ImageSet,      // An image was registered or unregistered.
		Tick,          // The scheduler's tick moved (a submission).
		GcTick,        // Garbage collection ran.
		OtherImage,    // A lookup would find another image (depth association).
		Dirty,         // An image is not clean.
		ColorClear,    // A color clear check is not current.
		GroupOutcomeCount
	};
	// Why RecordImageGroup recorded none.
	enum Refusal : uint32_t {
		RefusedNone,
		RefusedType,       // Not a sampled texture (storage).
		RefusedLookup,     // The lookup cannot be repeated (generation 0).
		RefusedView,       // The view is not kept.
		RefusedDynamicMip, // Dynamic storage mips.
		RefusalCount
	};
	std::array<uint64_t, GroupOutcomeCount> group {};
	std::array<uint64_t, RefusalCount>      refused {};
	uint64_t                                kept_lost = 0; // Kept, then failed in RebindImages.
	// Time in PrepareBindings' image part and in RebindImages, for kept and other stages.
	std::array<uint64_t, 2> prepare_ticks {}; // [0] kept, [1] other.
	std::array<uint64_t, 2> prepare_count {};
	std::array<uint64_t, 2> rebind_ticks {};
	std::array<uint64_t, 2> rebind_count {};
	// Depth-only draws: no color target, a depth target.
	uint64_t draws           = 0;
	uint64_t depth_draws     = 0; // Of draws.
	uint64_t depth_no_ps     = 0; // Of depth_draws: without a pixel shader.
	uint64_t depth_ps_images = 0; // Of depth_draws: a pixel shader binding images.
	uint64_t depth_vs_images = 0; // Of depth_draws: a vertex stage binding images.
	// Runs of consecutive depth-only draws with the same pipeline, vertex program and target,
	// counted by the run's length (1, 2-3, 4-7, 8-15, 16-63, 64+) in draws.
	std::array<uint64_t, 6> run_draws {};
	// What ended a run: [0] another pipeline, [1] another vertex program, [2] another target,
	// [3] a draw that is not depth-only.
	std::array<uint64_t, 4> run_breaks {};
	// Planned payloads by descriptor set: sets with writes, and those whose writes and infos
	// repeat the previous payload's for the same set under the same pipeline layout.
	std::array<uint64_t, 2> set_payloads {};
	std::array<uint64_t, 2> set_repeats {};
	uint32_t                run_length   = 0;
	const void*             run_pipeline = nullptr;
	const void*             run_program  = nullptr;
	uint64_t                run_target   = 0;
};
inline BindStats g_bind_stats;
[[nodiscard]] bool BindStatsEnabled() noexcept;

// KYTY_DEBUG_DRAW_PHASES=<pixel shader hash> times the render thread's CPU phases of that pixel
// shader's draws and prints their average every 5 s; =all times every draw. Each Mark charges the
// time since the previous mark to its phase; marks outside a Begin/End pair (other threads, other
// work) do nothing.
struct DrawPhaseTimer {
	static constexpr uint64_t AllDraws = ~uint64_t {0};
	enum Phase : uint32_t {
		Setup,            // Draw entry up to shader lookup.
		VertexParams,     // Vertex stage registers and code hash.
		PixelParams,      // Pixel stage registers and code hash.
		PixelProgram,     // Pixel program lookup and resource materialization.
		VertexProgram,    // Vertex program lookup and resource materialization.
		Targets,          // Render target resolution.
		StageTextures,    // PrepareBindings: textures.
		StageSamplers,    // PrepareBindings: samplers.
		StageBindings,    // PrepareBindings: user data.
		FindBuffers,      // PrepareGraphicsBindings: buffer discovery.
		RebindImages,     // PrepareGraphicsBindings: image views.
		BufferViews,      // PrepareGraphicsBindings: storage buffer binding.
		GraphicsBindings, // PrepareGraphicsBindings: uploads and the rest.
		RenderTargets,    // AcquireRenderTargets.
		Pipeline,         // Pipeline lookup.
		Records,          // Mesh draw records.
		Commit,           // CommitBindings.
		Record,           // Dynamic state, rendering and draw commands.
		Tail,             // After the draw.
		Count
	};
	// Spans timed on their own inside the phases; a probe overlaps its phase's time.
	enum Probe : uint32_t {
		TargetImage,      // Render-target FindImage.
		TextureImage,     // Texture FindImage.
		TextureDescribe,  // Texture description lookup and copy.
		BufferWritten,    // ObtainBuffer for written storage buffers.
		BufferRead,       // ObtainBuffer for read-only storage buffers.
		BufferInvalidate, // Texture invalidation behind written storage buffers.
		Upload,           // Flattened SRT and shader data uploads.
		FindFinish,       // FindImage and RefindImage after the lookup (DCC clear checks).
		StreamCopy,       // ObtainBuffer's copies of small CPU-written buffers.
		PendingOps,       // The scheduler's completed operations run at draw entry.
		Bda,              // PrepareBda for stages reading memory through addresses.
		SpeculationReads, // Repeating a speculated stage's guest reads before adopting it.
		ResourceRefresh,  // Refreshing a stage's resources that were not adopted.
		ProgramMatch,     // Finding the refreshed stage's permutation.
		InputsCheck,      // Comparing the registers of speculation-prepared stage inputs.
		UploadCopy,       // Copying guest bytes into the staging buffer (UploadCopies).
		CommitScan,       // CommitBindings: counting descriptors, specialization read checks.
		CommitTransit,    // CommitBindings: image layout transitions.
		CommitWrites,     // CommitBindings: descriptor writes and push data.
		CommitPayload,    // CommitBindings: the payload for the recording thread (or the calls).
		ProbeCount
	};
	class ProbeScope {
	public:
		ProbeScope(DrawPhaseTimer& timer, Probe probe)
		    : m_timer(timer.active ? &timer : nullptr), m_probe(probe),
		      m_start(m_timer != nullptr ? Now() : 0) {}
		~ProbeScope() {
			if (m_timer != nullptr) [[unlikely]] {
				m_timer->probes[m_probe] += Now() - m_start;
			}
		}
		ProbeScope(const ProbeScope&)            = delete;
		ProbeScope& operator=(const ProbeScope&) = delete;

	private:
		DrawPhaseTimer* m_timer;
		Probe           m_probe;
		uint64_t        m_start;
	};
	static uint64_t Hash();
	// The time stamp counter the phases are measured in.
	static uint64_t Now();
	// auto_draw: a DRAW_INDEX_AUTO packet's draw, also accounted on its own line.
	void            Begin(bool auto_draw = false) {
		if (Hash() != 0) [[unlikely]] {
			active    = true;
			auto_kind = auto_draw;
			depth_kind = false;
			current.fill(0);
			probes.fill(0);
			last = Now();
		}
	}
	void Mark(Phase phase) {
		if (active) [[unlikely]] {
			const auto now = Now();
			current[phase] += now - last;
			last = now;
		}
	}
	// Accounts the draw if its pixel shader is the one timed.
	void End(uint64_t pixel_hash);
	// Whether a draw is being timed (between Begin and End).
	[[nodiscard]] bool Active() const noexcept { return active; }
	// Consecutive probes without scopes: Lap() starts, Lap(probe, since) charges the time since.
	[[nodiscard]] uint64_t Lap() const { return active ? Now() : 0; }
	void                   Lap(Probe probe, uint64_t& since) {
		if (active) [[unlikely]] {
			const auto now = Now();
			probes[probe] += now - since;
			since = now;
		}
	}
	// The draw only writes depth (see BindStats), also accounted on its own line.
	void SetDepthOnly() { depth_kind = active; }

private:
	bool                             active    = false;
	bool                             auto_kind = false;
	bool                             depth_kind = false;
	uint64_t                         last      = 0;
	std::array<uint64_t, Count>      current {};
	std::array<uint64_t, ProbeCount> probes {};
};
inline thread_local DrawPhaseTimer g_draw_phases;

// With KYTY_DEBUG_DRAW_PHASES: the render thread's time in each PM4 packet's handler, by opcode,
// and in ProcessPm4 between handlers, printed as a "pm4-ops" line after each draw-phases line.
// Draw handlers include their draw phases; the rest is the time no draw phase covers.
struct Pm4OpTimer {
	// How a draw packet's handler ended (set by the handler, reset before each packet).
	enum Outcome : uint8_t {
		Drawn,
		Empty,         // No vertices or instances.
		MetadataOp,    // A color metadata (clear) operation instead of a draw.
		DepthCopy,     // A depth/stencil copy instead of a draw.
		Resolve,       // A color resolve instead of a draw.
		NoShader,      // No valid vertex shader or no topology.
		NotPrepared,   // Programs pending or no target (PrepareDrawRenderState).
		RectListSkip,  // A rect list with nothing to interpolate.
		OutcomeCount
	};
	std::array<uint64_t, 256> ticks {};
	std::array<uint64_t, 256> counts {};
	uint64_t                  between = 0;
	Outcome                   outcome = Drawn;
	// By outcome: [0] indexed draw packets, [1] DRAW_INDEX_AUTO.
	std::array<std::array<uint64_t, OutcomeCount>, 2> outcome_ticks {};
	std::array<std::array<uint64_t, OutcomeCount>, 2> outcome_counts {};
	// NOP packets by custom code (KYTY_PM4_R).
	std::array<uint64_t, 64> nop_ticks {};
	std::array<uint64_t, 64> nop_counts {};
};
inline thread_local Pm4OpTimer g_pm4_ops;

// KYTY_DEBUG_STAMP_TRACE=1: follows ASTRO BOT's interaction stamp surfaces (the targets of pixel
// shader 63722404bb8c7b26) through each frame: C = fast clear written, G = the same on the GPU,
// E = eliminate, D = stamp draw, O = another draw into the surface, S = the simulation's dispatch
// (fb948a435a4e295e). A digit names the surface. A frame's sequence is printed when its shape
// differs from the previous frame's. GPU thread only.
// KYTY_DEBUG_FRAME_REPEAT=1: see NoteBindingRepeat (descriptors.cpp). Called once per flip.
void DebugFrameRepeatFlip();
// KYTY_DEBUG_FRAME_PASSES=<trigger file>: when that file exists at a flip (checked every 15
// flips), the next frames' draws and dispatches are printed in order as "frame-pass" lines (see
// NoteImageUsers in renderDraw.cpp) and the file is removed: each pass's shaders, samplers and
// every image it reads and writes, to follow a post-processing chain such as temporal
// anti-aliasing. KYTY_DEBUG_FRAME_PASSES_COUNT=<n> frames (default 2). GPU thread only.
[[nodiscard]] bool FramePassesRecording() noexcept;
void               FramePassesFlip() noexcept;
// KYTY_DEBUG_FLIGHT_RECORDER=<frames>: keeps the last <frames> frames of cheap events (image churn,
// HTile metadata entries, depth-target clears, the passes of KYTY_DEBUG_FLIGHT_RECORDER_SHADERS=
// <hash,...> in frame-pass form) in memory and writes them to _flight_<flip>_<why>.txt when F10 is
// pressed, when KYTY_DEBUG_FLIGHT_RECORDER_FILE=<trigger> exists (checked every 15 flips) or two
// flips after an anomaly (a depth target cleared from emulator metadata after it was drawn to in
// the same frame). The game keeps its speed: nothing is printed until a dump (flightRecorder.cpp).
[[nodiscard]] bool FlightRecorderEnabled() noexcept;
// A pass of one of the KYTY_DEBUG_FLIGHT_RECORDER_SHADERS hashes.
[[nodiscard]] bool FlightRecorderWatches(uint64_t pixel_hash, uint64_t vertex_hash) noexcept;
// One event line, printf-style; a no-op while the recorder is off.
void FlightRecord(const char* format, ...);
// Every depth-target bind of a draw: the first bind of each image per frame and every clear are kept.
void FlightRecorderDepthBind(uint32_t image, uint64_t address, uint32_t width, uint32_t height,
                             bool game_clear, bool meta_clear, bool late_clear);
void FlightRecorderRequestDump(const char* why);
void FlightRecorderFlip();
// KYTY_DRAW_MEMO: counts frames (RenderContext::TickFrame), which age the memo's entries.
void DrawMemoFlip() noexcept;
// KYTY_DEBUG_DRAW_REPEAT=1: see NoteDrawRepeat (renderDraw.cpp). Called once per flip.
void DrawRepeatFlip();
// KYTY_DEBUG_WATCH (see debugWatch.cpp). Called once per flip.
void DebugWatchTick();

[[nodiscard]] bool StampTraceEnabled() noexcept;
void               StampTraceDraw(uint64_t pixel_hash, uint64_t target_address) noexcept;
void               StampTraceEvent(char event, uint64_t address) noexcept;
void               StampTraceDispatch(uint64_t compute_hash) noexcept;
void               StampTraceFrame() noexcept;

// KYTY_DEBUG_UPLOADS=1 (with KYTY_DEBUG_DRAW_PHASES): guest memory copied for the GPU, counted
// per source and printed after each draw-phases line with the images and 4 MiB regions copied
// most. A copy counts under its thread's innermost UploadSourceScope.
enum class UploadSource : uint8_t {
	Buffer, // SynchronizeBuffer outside the scopes below, mostly for bound buffers.
	Bda,    // PrepareBda's dirty-page synchronization.
	Stream, // ObtainBuffer's stream-buffer copies of small CPU-written ranges.
	Image,  // Image uploads (whole image ranges).
	Fault,  // Not a copy: guest write faults on tracked pages, counted as one page each.
	BdaPass, // Not a copy: PrepareBda calls.
	BdaSync, // Not a copy: PrepareBda calls that synchronized (the first of their epoch).
	Kernel,  // Not a copy: kernel invalidations (file and AMPR reads into guest memory).
	Count
};
inline thread_local UploadSource t_upload_source = UploadSource::Buffer;
[[nodiscard]] bool UploadStatsEnabled() noexcept;
void               RecordUpload(UploadSource source, uint64_t address, uint64_t bytes) noexcept;
void RecordImageUpload(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                       uint32_t guest_format, uint32_t tile_mode, bool buffer_modified) noexcept;
// KYTY_DEBUG_UPLOADS: compares a staged image upload with the previous one at the same address
// in 64 KiB chunks, to report how much of each re-upload changed.
void RecordImageChunks(uint64_t address, uint64_t size) noexcept;
// KYTY_DEBUG_UPLOAD_CHANGES=1: hashes every 4 KiB page a buffer upload copies and compares it with
// the page's previous upload; prints per 5 s how many uploaded pages were unchanged (bytes a
// shadow comparison could skip).
void RecordUploadPages(uint64_t address, uint64_t size) noexcept;
class UploadSourceScope {
public:
	explicit UploadSourceScope(UploadSource source) noexcept: m_previous(t_upload_source) {
		t_upload_source = source;
	}
	~UploadSourceScope() { t_upload_source = m_previous; }
	UploadSourceScope(const UploadSourceScope&)            = delete;
	UploadSourceScope& operator=(const UploadSourceScope&) = delete;

private:
	UploadSource m_previous;
};

class CommandBuffer;

namespace HW {
class Context;
class UserConfig;
struct RenderTarget;
struct ScanModeControl;
struct ScreenViewport;
} // namespace HW

struct ScissorRect {
	int left   = 0;
	int top    = 0;
	int right  = 0;
	int bottom = 0;
};

uint32_t                 render_target_mask_slot(uint32_t mask, uint32_t slot);
uint32_t                 render_target_first_bound_slot(const CommandBuffer& buffer);
bool                     graphics_debug_dump_enabled();
void                     uc_print(const char* func, const HW::UserConfig& uc);
void                     uc_check(const HW::UserConfig& uc);
std::string              rt_print(const char* func, const HW::RenderTarget& rt);
bool                     RenderIsColorTileModeLinear(Prospero::TileMode tile_mode);
void                     hw_print(const CommandBuffer& buffer);
void                     hw_check(const CommandBuffer& buffer);
void                     LogDrawPhase(const char* draw_name, const char* phase);
ScissorRect calc_final_scissor(const HW::ScreenViewport& vp, const HW::ScanModeControl& smc,
                               vk::Extent2D extent, uint32_t viewport_index);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_
