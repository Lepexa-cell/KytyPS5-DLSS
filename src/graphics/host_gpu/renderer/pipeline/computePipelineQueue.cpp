#include "graphics/host_gpu/renderer/pipeline/computePipelineQueue.h"

#include "common/profiler.h"

#include <algorithm>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

ComputePipelineQueue::ComputePipelineQueue(uint32_t threads, uint32_t background_threads)
    : m_background_max(std::clamp(background_threads, 1u, std::max(threads, 1u))) {
	m_threads.reserve(std::max(threads, 1u));
	for (uint32_t i = 0; i < std::max(threads, 1u); i++) {
		m_threads.emplace_back([this](const std::stop_token& stop) { Work(stop); });
	}
}

ComputePipelineQueue::~ComputePipelineQueue() {
	Stop();
}

std::shared_future<vk::Pipeline> ComputePipelineQueue::Post(uint64_t id,
                                                            Common::UniqueFunction<vk::Pipeline>&& compile,
                                                            bool urgent, bool background) {
	Job job;
	job.id         = id;
	job.urgent     = urgent;
	job.background = background && !urgent;
	job.compile = std::move(compile);
	auto result = job.result.get_future().share();
	{
		std::lock_guard lock(m_mutex);
		if (m_stopped) {
			job.result.set_value(nullptr);
			return result;
		}
		Enqueue(std::move(job));
	}
	m_available.notify_one();
	return result;
}

void ComputePipelineQueue::Enqueue(Job&& job) {
	if (job.urgent) {
		const auto first_normal = std::ranges::find(m_jobs, false, &Job::urgent);
		m_jobs.insert(first_normal, std::move(job));
	} else {
		m_jobs.push_back(std::move(job));
	}
}

void ComputePipelineQueue::Promote(uint64_t id) {
	std::lock_guard lock(m_mutex);
	const auto      queued = std::ranges::find(m_jobs, id, &Job::id);
	if (queued != m_jobs.end() && !queued->urgent) {
		auto job = std::move(*queued);
		m_jobs.erase(queued);
		job.urgent     = true;
		job.background = false;
		Enqueue(std::move(job));
		m_available.notify_one();
	}
}

std::deque<ComputePipelineQueue::Job>::iterator ComputePipelineQueue::NextRunnable() {
	// Background jobs past their worker limit wait; the jobs behind them may run.
	return std::ranges::find_if(m_jobs, [this](const Job& job) {
		return !job.background || m_background_running < m_background_max;
	});
}

void ComputePipelineQueue::Stop() {
	{
		std::lock_guard lock(m_mutex);
		if (m_stopped) {
			return;
		}
		m_stopped = true;
		for (auto& job: m_jobs) {
			job.result.set_value(nullptr);
		}
		m_jobs.clear();
	}
	for (auto& thread: m_threads) {
		thread.request_stop();
	}
	for (auto& thread: m_threads) {
		if (thread.joinable()) {
			thread.join();
		}
	}
}

void ComputePipelineQueue::Work(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("ComputePipelines");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Below the game's priority, as the pipeline library's compile threads: spare cores only.
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
	for (;;) {
		Job job;
		{
			std::unique_lock lock(m_mutex);
			if (!m_available.wait(lock, stop, [this] { return NextRunnable() != m_jobs.end(); })) {
				return;
			}
			const auto next = NextRunnable();
			job             = std::move(*next);
			m_jobs.erase(next);
			m_background_running += job.background ? 1u : 0u;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		// Background jobs (the replay's) yield even to the other compiles.
		if (job.background) {
			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
		}
#endif
		Run(job);
		if (job.background) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
			{
				std::lock_guard lock(m_mutex);
				m_background_running--;
			}
			// A background job waiting for this worker slot may run now.
			m_available.notify_all();
		}
	}
}

void ComputePipelineQueue::Run(Job& job) {
	vk::Pipeline pipeline = nullptr;
	try {
		pipeline = job.compile();
	} catch (...) {
		// A failed compile (the watchdog thread not starting, for one) leaves the result null,
		// and the dispatch compiles the pipeline itself; a broken promise would end the program.
		pipeline = nullptr;
	}
	job.result.set_value(pipeline);
}

bool ComputePipelineQueue::RunHere(uint64_t id) {
	Job job;
	{
		std::lock_guard lock(m_mutex);
		const auto      queued = std::ranges::find(m_jobs, id, &Job::id);
		if (queued == m_jobs.end()) {
			return false;
		}
		job = std::move(*queued);
		m_jobs.erase(queued);
	}
	Run(job);
	return true;
}

} // namespace Libs::Graphics
