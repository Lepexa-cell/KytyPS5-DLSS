#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPUTEPIPELINEQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPUTEPIPELINEQUEUE_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// Worker threads that create compute pipelines (vkCreateComputePipelines), so the GPU thread never
// compiles one itself and several compile at once: the shader journal's replay at startup, the
// look-ahead's prefetches and the dispatches that need one now. Needs no pipeline libraries.
class ComputePipelineQueue {
public:
	// At most `background_threads` workers (at least one) run background jobs at a time, so the
	// other jobs always find a free worker.
	ComputePipelineQueue(uint32_t threads, uint32_t background_threads);
	~ComputePipelineQueue();
	KYTY_CLASS_NO_COPY(ComputePipelineQueue);

	// Queues `compile` under `id` and returns its result, null when it failed or was dropped. An
	// `urgent` job (a dispatch waits for it) goes ahead of the queued others; the rest run in the
	// order they came. A `background` job (the precompile replay's) runs at the lowest priority on
	// a limited number of workers; promoting it lifts both.
	std::shared_future<vk::Pipeline> Post(uint64_t id, Common::UniqueFunction<vk::Pipeline>&& compile,
	                                      bool urgent, bool background = false);
	// Moves the queued job `id`, if any, ahead of the queued others.
	void Promote(uint64_t id);
	// Runs the job `id` on the calling thread if no worker started it yet, for a caller that would
	// otherwise wait for it behind the running compiles. Returns whether it ran it.
	bool RunHere(uint64_t id);
	// Drops the queued jobs (their results become null) and joins the threads; a running compile
	// finishes first. Must run before the driver cache is destroyed.
	void Stop();

	[[nodiscard]] uint32_t ThreadCount() const { return static_cast<uint32_t>(m_threads.size()); }

private:
	struct Job {
		uint64_t                             id         = 0;
		bool                                 urgent     = false;
		bool                                 background = false;
		std::promise<vk::Pipeline>           result;
		Common::UniqueFunction<vk::Pipeline> compile;
	};

	void        Enqueue(Job&& job);
	// The first queued job a worker may take now; m_mutex must be held.
	std::deque<Job>::iterator NextRunnable();
	void        Work(const std::stop_token& stop);
	static void Run(Job& job);

	std::mutex                  m_mutex;
	std::condition_variable_any m_available;
	std::deque<Job>             m_jobs;
	bool                        m_stopped = false;
	uint32_t                    m_background_max     = 1;
	uint32_t                    m_background_running = 0;
	std::vector<std::jthread>   m_threads;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_COMPUTEPIPELINEQUEUE_H_
