#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/switches.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/handleRegistry.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/renderer/render.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <iterator>
#include <utility>
#include <vector>

#include <atomic>
#include <chrono>
#include <thread>

namespace Libs::Graphics {

namespace {

// KYTY_DEBUG_GPU_CHECKPOINTS: the last checkpoints each pipeline stage of the queue passed, with
// the zone and key (shader hash for game draws and dispatches) recorded for them.
void ReportCheckpoints(GraphicContext& graphics) {
	static constexpr const char* ZoneNames[] = {"unmarked",  "draw",     "draw-predicated",
	                                            "dispatch",  "mesh-args", "tiler",
	                                            "dcc-clear", "fault-buffer", "blit",
	                                            "image-copy", "buffer-copy", "dispatch-async"};
	for (const auto& [name, queue]: {std::pair {"graphics", graphics.queue},
	                                  std::pair {"readback", graphics.readback_queue}}) {
		if (queue == nullptr) {
			continue;
		}
		uint32_t count = 0;
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, nullptr);
		std::vector<VkCheckpointDataNV> data(count, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, data.data());
		std::printf("gpu-checkpoints: %s queue, %u entries (newest checkpoint %" PRIu64 ")\n", name,
		            count, GpuZones::g_checkpoint_next.load() - 1);
		for (const auto& entry: data) {
			const auto  index = reinterpret_cast<uintptr_t>(entry.pCheckpointMarker);
			const auto& mark  = GpuZones::g_checkpoints[index % GpuZones::CheckpointRing];
			const auto  zone  = static_cast<size_t>(mark.zone);
			std::printf("gpu-checkpoint: stage=%s checkpoint=%" PRIu64 " zone=%s key=%016" PRIx64 "\n",
			            vk::to_string(static_cast<vk::PipelineStageFlagBits>(entry.stage)).c_str(),
			            static_cast<uint64_t>(index),
			            zone < std::size(ZoneNames) ? ZoneNames[zone] : "?", mark.key);
			if (queue == graphics.queue && entry.stage == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) {
				GpuZones::g_fault_top_checkpoint.store(index, std::memory_order_relaxed);
			}
		}
		// The marks between the last finished and the last started checkpoint: the hung command
		// is among them (the driver reports only the newest checkpoint per stage).
		uintptr_t first = UINTPTR_MAX;
		uintptr_t last  = 0;
		for (const auto& entry: data) {
			const auto index = reinterpret_cast<uintptr_t>(entry.pCheckpointMarker);
			first            = std::min(first, index);
			last             = std::max(last, index);
		}
		for (uintptr_t index = first; count != 0 && index <= last + 2u && index - first < 64u;
		     index++) {
			const auto& mark = GpuZones::g_checkpoints[index % GpuZones::CheckpointRing];
			const auto  zone = static_cast<size_t>(mark.zone);
			std::printf("gpu-checkpoint-range: checkpoint=%" PRIu64 " zone=%s key=%016" PRIx64
			            " count=%u instances=%u vs=%016" PRIx64 " indirect=%016" PRIx64 "\n",
			            static_cast<uint64_t>(index),
			            zone < std::size(ZoneNames) ? ZoneNames[zone] : "?", mark.key,
			            static_cast<uint32_t>(mark.detail[0] >> 32u),
			            static_cast<uint32_t>(mark.detail[0]), mark.detail[1], mark.detail[2]);
		}
	}
	std::fflush(stdout);
}

// KYTY_DEBUG_OP_RING: the marks of the submissions the GPU had not completed when the device
// was lost (ticks above `completed`), newest last; the hung command is among them.
void ReportOpRing(uint64_t completed) {
	const auto next  = GpuZones::g_checkpoint_next.load(std::memory_order_relaxed);
	const auto first = next > GpuZones::CheckpointRing ? next - GpuZones::CheckpointRing : 1;
	uint64_t   begin = next;
	for (uint64_t index = next; index > first; index--) {
		if (GpuZones::g_checkpoint_ticks[(index - 1) % GpuZones::CheckpointRing] <= completed) {
			break;
		}
		begin = index - 1;
	}
	std::printf("gpu-op-ring: completed tick %" PRIu64 ", %" PRIu64 " marks in flight\n", completed,
	            next - begin);
	static constexpr const char* ZoneNames[] = {"unmarked",  "draw",     "draw-predicated",
	                                            "dispatch",  "mesh-args", "tiler",
	                                            "dcc-clear", "fault-buffer", "blit",
	                                            "image-copy", "buffer-copy", "dispatch-async"};
	for (uint64_t index = begin; index < next; index++) {
		const auto& mark = GpuZones::g_checkpoints[index % GpuZones::CheckpointRing];
		const auto  zone = static_cast<size_t>(mark.zone);
		std::printf("gpu-op: tick=%" PRIu64 " zone=%s key=%016" PRIx64 " count=%u instances=%u vs=%016" PRIx64
		            " indirect=%016" PRIx64 "\n",
		            GpuZones::g_checkpoint_ticks[index % GpuZones::CheckpointRing],
		            zone < std::size(ZoneNames) ? ZoneNames[zone] : "?", mark.key,
		            static_cast<uint32_t>(mark.detail[0] >> 32u), static_cast<uint32_t>(mark.detail[0]),
		            mark.detail[1], mark.detail[2]);
	}
	std::fflush(stdout);
}

// KYTY_DEBUG_DEVICE_FAULT: the address binding events (VK_EXT_device_address_binding_report)
// covering a faulting address, oldest first, and whether its range is still bound.
void ReportAddressBindings(uint64_t address, uint64_t span = 1) {
	if (!GpuZones::g_binding_report) {
		return;
	}
	std::lock_guard lock(GpuZones::g_binding_mutex);
	const auto      next = GpuZones::g_binding_next.load(std::memory_order_relaxed);
	const auto first = next > GpuZones::AddressBindingRing ? next - GpuZones::AddressBindingRing : 0;
	std::printf("device-fault-bindings: address=0x%016" PRIx64 " span=0x%" PRIx64 ", %" PRIu64
	            " events recorded (%" PRIu64 " kept)\n",
	            address, span, next, next - first);
	uint32_t matches     = 0;
	bool     still_bound = false;
	uint64_t last_base   = 0;
	for (uint64_t index = first; index < next; index++) {
		const auto& e = GpuZones::g_bindings[index % GpuZones::AddressBindingRing];
		if (e.size == 0 || address + span <= e.base || address >= e.base + e.size) {
			continue;
		}
		matches++;
		char name[9] = {};
		std::memcpy(name, &e.object_name, 8);
		std::printf("device-fault-binding: #%" PRIu64 " %s base=0x%016" PRIx64 " size=0x%" PRIx64
		            " offset=0x%" PRIx64 " object=%s handle=0x%016" PRIx64 "%s %s mark=%" PRIu64
		            " tick=%" PRIu64 "\n",
		            index, (e.flags & 1u) != 0 ? "BIND  " : "UNBIND", e.base, e.size,
		            address >= e.base ? address - e.base : 0, vk::to_string(static_cast<vk::ObjectType>(e.object_type)).c_str(),
		            e.handle, (e.flags & 2u) != 0 ? " (driver-internal)" : "", name, e.mark, e.tick);
		if ((e.flags & 1u) != 0) {
			still_bound = true;
			last_base   = e.base;
		} else if (e.base == last_base || last_base == 0) {
			still_bound = false;
		}
	}
	if (matches == 0) {
		std::printf("device-fault-bindings: no recorded binding covers the address\n");
		// The nearest bound ranges below and above the address place it: just past the end of an
		// allocation, in a gap between two, or far from anything the driver ever mapped.
		const GpuZones::AddressBinding* below = nullptr;
		const GpuZones::AddressBinding* above = nullptr;
		for (uint64_t index = first; index < next; index++) {
			const auto& e = GpuZones::g_bindings[index % GpuZones::AddressBindingRing];
			if (e.size == 0 || (e.flags & 1u) == 0) {
				continue;
			}
			if (e.base + e.size <= address && (below == nullptr || e.base + e.size > below->base + below->size)) {
				below = &e;
			}
			if (e.base > address && (above == nullptr || e.base < above->base)) {
				above = &e;
			}
		}
		for (const auto& [where, e]: {std::pair {"below", below}, std::pair {"above", above}}) {
			if (e != nullptr) {
				std::printf("device-fault-bindings: nearest bind %s: base=0x%016" PRIx64 " size=0x%" PRIx64
				            " object=%s handle=0x%016" PRIx64 "%s gap=0x%" PRIx64 " mark=%" PRIu64
				            " tick=%" PRIu64 "\n",
				            where, e->base, e->size,
				            vk::to_string(static_cast<vk::ObjectType>(e->object_type)).c_str(), e->handle,
				            (e->flags & 2u) != 0 ? " (driver-internal)" : "",
				            e->base > address ? e->base - address : address - (e->base + e->size), e->mark,
				            e->tick);
			}
		}
	} else {
		std::printf("device-fault-bindings: %u events, range %s at the fault\n", matches,
		            still_bound ? "STILL BOUND" : "ALREADY UNBOUND");
	}
	// What the handle registry knows of the objects that held the address (origin, destroying stack).
	uint64_t described[8] {};
	uint32_t described_count = 0;
	for (uint64_t index = next; index > first && described_count < 8; index--) {
		const auto& e = GpuZones::g_bindings[(index - 1) % GpuZones::AddressBindingRing];
		if (e.size == 0 || address + span <= e.base || address >= e.base + e.size ||
		    (static_cast<vk::ObjectType>(e.object_type) != vk::ObjectType::eImage &&
		     static_cast<vk::ObjectType>(e.object_type) != vk::ObjectType::eBuffer)) {
			continue;
		}
		if (std::find(described, described + described_count, e.handle) != described + described_count) {
			continue;
		}
		described[described_count++] = e.handle;
		HandleRegistry::Describe(e.handle, "device-fault-binding-owner:");
	}
	std::fflush(stdout);
}

// KYTY_DEBUG_DEVICE_FAULT: what the driver recorded for the lost device.
void ReportDeviceFault(GraphicContext& graphics, vk::Result result, uint64_t completed_tick) {
	// The submit thread and a waiting thread both see the loss: report it once.
	static std::atomic_flag reported = ATOMIC_FLAG_INIT;
	static std::atomic<bool> finished {false};
	if (result != vk::Result::eErrorDeviceLost) {
		return;
	}
	if (reported.test_and_set()) {
		// The other thread reports: wait for it before the caller ends the process (a waiting
		// thread's exit cut the report off, 2026-10-19), but not forever.
		for (int i = 0; i < 1000 && !finished.load(std::memory_order_acquire); i++) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		return;
	}
	struct Finish {
		~Finish() { finished.store(true, std::memory_order_release); }
	} finish;
	if (graphics.checkpoints_enabled && result == vk::Result::eErrorDeviceLost) {
		ReportCheckpoints(graphics);
	}
	if (GpuZones::g_op_ring && result == vk::Result::eErrorDeviceLost) {
		ReportOpRing(completed_tick);
	}
	if (DrawResourcesEnabled()) {
		ReportRecentDrawResources();
	}
	if (GpuZones::g_gds_words != nullptr) {
		// Nonzero words of the first 4 KiB of GDS (host-visible): persistent-thread shaders take
		// their item count and work index from here.
		std::printf("device-fault-gds:");
		for (uint32_t i = 0; i < 1024; i++) {
			const uint32_t value = GpuZones::g_gds_words[i];
			if (value != 0) {
				std::printf(" [0x%x]=0x%x", i * 4u, value);
			}
		}
		std::printf("\n");
		std::fflush(stdout);
	}
	if (!graphics.device_fault_enabled || result != vk::Result::eErrorDeviceLost) {
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	if (graphics.device.getFaultInfoEXT(&counts, nullptr) != vk::Result::eSuccess) {
		std::printf("device-fault: no fault info\n");
		std::fflush(stdout);
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	counts.vendorBinarySize = 0;
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos = addresses.data();
	info.pVendorInfos  = vendors.data();
	const auto fetched = graphics.device.getFaultInfoEXT(&counts, &info);
	std::printf("deferred-call: %llu stale handles seen so far (check %s)\n",
	            static_cast<unsigned long long>(HandleRegistry::StaleCount()),
	            HandleRegistry::CheckEnabled() ? "on" : "off");
	std::printf("device-fault: result=%d addresses=%u vendors=%u description=%s\n",
	            static_cast<int>(fetched), counts.addressInfoCount, counts.vendorInfoCount,
	            info.description.data());
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		std::printf("device-fault-address: type=%s address=0x%016" PRIx64 " precision=0x%" PRIx64
		            "\n",
		            vk::to_string(addresses[i].addressType).c_str(),
		            static_cast<uint64_t>(addresses[i].reportedAddress),
		            static_cast<uint64_t>(addresses[i].addressPrecision));
		if (GpuZones::g_describe_device_address != nullptr &&
		    addresses[i].addressType != vk::DeviceFaultAddressTypeEXT::eExecuteInvalid &&
		    addresses[i].addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerUnknown &&
		    addresses[i].addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerInvalid &&
		    addresses[i].addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerFault) {
			GpuZones::g_describe_device_address(static_cast<uint64_t>(addresses[i].reportedAddress));
		}
		if (addresses[i].addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerUnknown &&
		    addresses[i].addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerInvalid &&
		    addresses[i].addressType != vk::DeviceFaultAddressTypeEXT::eInstructionPointerFault) {
			ReportAddressBindings(static_cast<uint64_t>(addresses[i].reportedAddress));
			// A 32-bit byte offset of 0xFFFFFFFC (the ByteAddress sentinel of an out-of-bounds
			// storage buffer access, which the hardware bounds check lets through when offset + 4
			// wraps to 0) puts the access 4 GiB minus 4 above the buffer's base: name the buffers
			// whose base lies in the window the faulting page allows for.
			const auto reported  = static_cast<uint64_t>(addresses[i].reportedAddress);
			const auto precision = std::max<uint64_t>(static_cast<uint64_t>(addresses[i].addressPrecision), 1);
			if (reported >= 0xFFFFFFFCull) {
				const auto base = reported - 0xFFFFFFFCull;
				std::printf("device-fault-wrap: if the access was base + 0xFFFFFFFC, base is in [0x%016" PRIx64
				            ", 0x%016" PRIx64 ")\n",
				            base, base + precision);
				if (GpuZones::g_describe_device_address != nullptr) {
					GpuZones::g_describe_device_address(base);
				}
				ReportAddressBindings(base, precision);
			}
		}
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		std::printf("device-fault-vendor: code=0x%" PRIx64 " data=0x%" PRIx64 " %s\n",
		            static_cast<uint64_t>(vendors[i].vendorFaultCode),
		            static_cast<uint64_t>(vendors[i].vendorFaultData),
		            vendors[i].description.data());
	}
	std::fflush(stdout);
}

} // namespace

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
	GpuZones::g_binding_tick = &m_current_tick;
}

MasterSemaphore::~MasterSemaphore() {
	GpuZones::g_binding_tick = nullptr;
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	ReportDeviceFault(m_graphics, result, m_gpu_tick.load(std::memory_order_acquire));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::ReportFault(vk::Result result) {
	ReportDeviceFault(m_graphics, result, m_gpu_tick.load(std::memory_order_acquire));
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	// KYTY_DEBUG_WAIT_TIMEOUT=<ms>: a wait longer than that prints the
	// semaphore's value, the tick waited for and the newest tick handed out, then waits on.
	static const uint64_t timeout_ns = [] {
		const char* value = KYTY_GETENV("KYTY_DEBUG_WAIT_TIMEOUT");
		return value != nullptr ? std::strtoull(value, nullptr, 0) * 1000000ull : UINT64_MAX;
	}();
	const auto wait_begin = std::chrono::steady_clock::now();
	auto       result     = m_graphics.device.waitSemaphores(&wait_info, timeout_ns);
	if (result == vk::Result::eTimeout) {
		uint64_t counter = 0;
		(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
		std::printf("gpu-wait: tick %" PRIu64 " not reached after %" PRIu64 " ms: semaphore at %" PRIu64
		            ", newest tick %" PRIu64 "\n",
		            tick, timeout_ns / 1000000, counter, CurrentTick());
		std::fflush(stdout);
		result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	}
	ReportDeviceFault(m_graphics, result, m_gpu_tick.load(std::memory_order_acquire));
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuWaits);
	Common::DebugCounters::Add(
	    Common::DebugCounters::Counter::GpuWaitNs,
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                              std::chrono::steady_clock::now() - wait_begin)
	                              .count()));
	Refresh();
}

} // namespace Libs::Graphics
