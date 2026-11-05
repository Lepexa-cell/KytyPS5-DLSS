#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_SPECULATOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_SPECULATOR_H

#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/drawSpeculation.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandProcessor;
class RenderContext;

// Refreshes the shader resources of draws ahead of the GPU thread (Config::SpeculativeDrawsEnabled;
// KYTY_SPECULATE_DRAWS=0 or 1 overrides it). A
// worker thread replays the command stream's register writes from a copy of the GPU thread's
// register state, as the pipeline look-ahead does, and speculates each draw it meets
// (PipelineCache::SpeculateGraphicsPrograms). Both threads number draw packets from the copy's
// position; the GPU thread takes a draw's speculation when it is ready and never waits for one.
// A wrong speculation costs nothing but the worker's time: adoption checks every input again.
class DrawSpeculator {
public:
	struct Cursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	DrawSpeculator(RenderContext& renderer, int interrupt_event_id);
	~DrawSpeculator();
	KYTY_CLASS_NO_COPY(DrawSpeculator);

	[[nodiscard]] static bool Enabled();

	// GPU thread: execution continues at `stack` with `processor`'s register state (a submission
	// starting or resuming, or a draw past a packet the walk stopped at).
	void Restart(const CommandProcessor& processor, std::span<const Cursor> stack);
	// GPU thread, before each draw packet's handler, in order: the draw's speculation, if ready.
	[[nodiscard]] SpeculatedDraw* Take(const uint32_t* packet);
	// GPU thread, after the handler (with Take's result).
	void Release(SpeculatedDraw* draw);
	// GPU thread, after Release: whether the walk has stopped behind the GPU thread and needs a
	// Restart at the next packet.
	[[nodiscard]] bool RestartDue() const;
	// GPU thread, after Release: the packet runs again later (its execution was suspended), so
	// the walk's numbering no longer matches; restart it then.
	void LoseStep() { m_mismatch = true; }

private:
	enum SlotState : uint32_t { Free, Writing, Ready, Taken };
	struct Slot {
		std::atomic<uint32_t> state {Free};
		uint64_t              epoch  = 0;
		uint64_t              seq    = 0;
		const uint32_t*       packet = nullptr;
		SpeculatedDraw        draw;
	};
	struct Snapshot;
	static constexpr uint32_t RingSize = 64;
	// Workers share the walk's draws by sequence number (draw seq belongs to worker seq % count):
	// one worker prepares a draw about as fast as the GPU thread executes one, so alone it only
	// slowly gets ahead after each restart and the GPU thread prepared one draw in twelve itself.
	// KYTY_SPECULATE_THREADS=<1..3> sets the count. Default 1: once the walk no longer waited for
	// GPU readbacks, a second worker only cost time (4.6 -> 5.1-5.6 us per draw in Sky Garden).
	static constexpr uint32_t MaxWorkers = 3;
	static constexpr uint64_t NotStopped = UINT64_MAX;
	// Draws within this many of the GPU thread's next draw are not speculated (see Walk).
	static constexpr uint64_t SpeculationLead = 2;

	void Run(uint32_t worker);
	// Walks from the current snapshot until it stops, speculating the worker's draws it meets.
	void Walk(CommandProcessor& shadow, std::vector<Cursor> stack, uint64_t epoch, uint64_t seq,
	          uint32_t worker);
	// A free slot for draw `seq`, or null when a restart or shutdown interrupted the wait.
	Slot* AcquireSlot(uint64_t seq, uint64_t epoch, uint32_t worker);
	// Waits until the GPU thread has finished draw `seq`; false when interrupted.
	bool  WaitForGpu(uint64_t seq, uint64_t epoch, uint32_t worker);
	[[nodiscard]] bool Interrupted(uint64_t epoch) const;
	// GPU thread: whether `draw`'s reads still give what they gave (see Take).
	[[nodiscard]] bool ReadsCurrent(const SpeculatedDraw& draw);

	RenderContext& m_renderer;
	const int      m_interrupt_event_id;

	std::array<Slot, RingSize> m_ring;
	// GPU thread: the next draw's sequence number and the current epoch.
	uint64_t                   m_gpu_seq = 0;
	// GPU thread: the buffer cache's and texture cache's GPU-write generations at the last
	// Release, and the last draw at whose Release either had changed (see Take).
	uint64_t                   m_seen_buffer_writes = 0;
	uint64_t                   m_seen_image_writes  = 0;
	uint64_t                   m_last_write_seq     = 0;
	std::atomic<uint64_t>      m_gpu_next {0};
	std::atomic<uint64_t>      m_epoch {0};
	// The worker's walk ended before this draw (NotStopped while walking or idle at the end).
	std::atomic<uint64_t>      m_stopped_seq {NotStopped};
	bool                       m_mismatch = false;

	std::mutex                 m_mutex;
	std::condition_variable    m_wake;
	std::unique_ptr<Snapshot>  m_snapshot; // The latest restart, under m_mutex; workers copy it.
	uint64_t                   m_restart_count = 0; // Under m_mutex.
	// m_restart_count, read without the lock: the workers spin on it before sleeping.
	std::atomic<uint64_t>      m_restart_hint {0};
	bool                       m_quit    = false;
	uint32_t                   m_workers = 1;
	std::array<std::atomic<bool>, MaxWorkers> m_waiting {};
	// While a worker waits: the GPU thread's draw count that wakes it (a waking per draw would
	// cost the GPU thread a kernel call each).
	std::array<std::atomic<uint64_t>, MaxWorkers> m_wake_at {};
	std::atomic<bool>          m_quitting {false};
	// Bumped when the worker may stop waiting for a slot (a draw finished, a restart, shutdown).
	std::atomic<uint64_t>      m_signal {0};
	std::vector<std::thread>   m_threads;

	// Counters, printed with KYTY_DEBUG_SPEC_STATS=1.
	struct Stats {
		std::atomic<uint64_t> restarts {0};
		std::atomic<uint64_t> walked {0};
		std::atomic<uint64_t> speculated {0};
		std::atomic<uint64_t> behind {0};
		std::atomic<uint64_t> taken {0};
		std::atomic<uint64_t> missed {0};
		std::atomic<uint64_t> mismatched {0};
		std::atomic<uint64_t> deferred {0};
		std::atomic<uint64_t> table_unread {0};
		std::atomic<uint64_t> table_unknown {0};
		std::array<std::atomic<uint64_t>, 256> stops {};
	};
	Stats m_stats;
	void  PrintStats();
	void  Stop(uint64_t seq, uint64_t epoch, uint32_t opcode);
	void  Signal();
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_DRAW_SPECULATOR_H
