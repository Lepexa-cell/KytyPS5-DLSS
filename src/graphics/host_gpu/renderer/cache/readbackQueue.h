#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKQUEUE_H_

#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <span>

namespace Libs::Graphics {

// KYTY_READBACK_QUEUE=1: guest readbacks of bytes whose GPU writes have all finished copy on a
// transfer-only queue (the copy engine) instead of behind everything queued on the graphics
// queue. Idea from the chenxiao07 Demon's Souls fork (KYTY_READBACK_QUEUE there).
//
// Guest threads record and submit here, never Thread_Gpu: its Vulkan calls go through the
// deferred-recording stream (commandHooks.h), and a direct call there would drain that stream.
// Any thread may call; one mutex orders the command pool and the queue.
class ReadbackQueue {
public:
	static constexpr size_t Buffers = 32;

	explicit ReadbackQueue(GraphicContext& graphics): m_device(graphics.device),
	                                                  m_queue(graphics.readback_queue) {
		EXIT_IF(m_queue == nullptr);
		vk::CommandPoolCreateInfo pool {};
		pool.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
		             vk::CommandPoolCreateFlagBits::eTransient;
		pool.queueFamilyIndex = graphics.readback_family;
		EXIT_NOT_IMPLEMENTED(m_device.createCommandPool(&pool, nullptr, &m_pool) !=
		                     vk::Result::eSuccess);
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = m_pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = static_cast<uint32_t>(Buffers);
		EXIT_NOT_IMPLEMENTED(m_device.allocateCommandBuffers(&allocate, m_commands.data()) !=
		                     vk::Result::eSuccess);
		vk::SemaphoreTypeCreateInfo type {};
		type.semaphoreType = vk::SemaphoreType::eTimeline;
		vk::SemaphoreCreateInfo semaphore {};
		semaphore.pNext = &type;
		EXIT_NOT_IMPLEMENTED(m_device.createSemaphore(&semaphore, nullptr, &m_semaphore) !=
		                     vk::Result::eSuccess);
	}
	~ReadbackQueue() {
		{
			std::scoped_lock lock(m_mutex);
			(void)m_queue.waitIdle();
		}
		m_device.destroySemaphore(m_semaphore, nullptr);
		m_device.destroyCommandPool(m_pool, nullptr);
	}
	KYTY_CLASS_NO_COPY(ReadbackQueue);

	// Copies `regions` from `source` to `destination` once `timeline` reaches `wait_value`.
	// Returns the value of this queue's timeline at which the bytes are visible to the host.
	uint64_t Copy(vk::Buffer source, vk::Buffer destination,
	              std::span<const vk::BufferCopy> regions, vk::Semaphore timeline,
	              uint64_t wait_value) {
		std::scoped_lock lock(m_mutex);
		const uint64_t   value   = ++m_submitted;
		const size_t     index   = value % Buffers;
		const auto       command = m_commands[index];
		if (m_used[index] != 0) {
			Wait(m_used[index]);
		}
		m_used[index] = value;
		EXIT_NOT_IMPLEMENTED(command.reset({}) != vk::Result::eSuccess);
		vk::CommandBufferBeginInfo begin {};
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		EXIT_NOT_IMPLEMENTED(command.begin(&begin) != vk::Result::eSuccess);
		command.copyBuffer(source, destination, static_cast<uint32_t>(regions.size()),
		                   regions.data());
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0, nullptr, 0,
		                        nullptr);
		EXIT_NOT_IMPLEMENTED(command.end() != vk::Result::eSuccess);
		vk::TimelineSemaphoreSubmitInfo values {};
		values.waitSemaphoreValueCount   = 1;
		values.pWaitSemaphoreValues      = &wait_value;
		values.signalSemaphoreValueCount = 1;
		values.pSignalSemaphoreValues    = &value;
		const vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eTransfer;
		vk::SubmitInfo               submit {};
		submit.pNext                = &values;
		submit.waitSemaphoreCount   = 1;
		submit.pWaitSemaphores      = &timeline;
		submit.pWaitDstStageMask    = &stage;
		submit.commandBufferCount   = 1;
		submit.pCommandBuffers      = &command;
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores    = &m_semaphore;
		EXIT_NOT_IMPLEMENTED(m_queue.submit(1, &submit, nullptr) != vk::Result::eSuccess);
		return value;
	}

	void Wait(uint64_t value) {
		if (m_known.load(std::memory_order_acquire) >= value) {
			return;
		}
		vk::SemaphoreWaitInfo wait {};
		wait.semaphoreCount = 1;
		wait.pSemaphores    = &m_semaphore;
		wait.pValues        = &value;
		EXIT_NOT_IMPLEMENTED(m_device.waitSemaphores(&wait, UINT64_MAX) != vk::Result::eSuccess);
		uint64_t known = m_known.load(std::memory_order_relaxed);
		while (known < value &&
		       !m_known.compare_exchange_weak(known, value, std::memory_order_acq_rel)) {
		}
	}

private:
	vk::Device                              m_device;
	vk::Queue                               m_queue;
	vk::CommandPool                         m_pool      = nullptr;
	vk::Semaphore                           m_semaphore = nullptr;
	std::array<vk::CommandBuffer, Buffers>  m_commands {};
	std::array<uint64_t, Buffers>           m_used {};
	uint64_t                                m_submitted = 0;
	std::atomic<uint64_t>                   m_known {0};
	std::mutex                              m_mutex;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKQUEUE_H_
