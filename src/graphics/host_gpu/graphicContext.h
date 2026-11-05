#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	bool                               compute_subgroup_size_control_enabled = false;
	// VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT (lookups in full driver cache shards).
	bool                               pipeline_creation_cache_control = false;
	bool                               sample_rate_shading_enabled           = false;
	bool                               shader_image_int64_atomics_enabled    = false;
	bool                               maximal_reconvergence_enabled         = false;
	// bool fp64_denorm_preserve = false; // Temporarily disabled.
	// Layout support is sufficient for static pipeline feedback flags. Dynamic state is optional.
	bool                               attachment_feedback_loop_enabled      = false;
	bool                                      attachment_feedback_loop_dynamic_enabled = false;
	bool                               provoking_vertex_last_enabled         = false;
	// VK_EXT_conditional_rendering: lets GPU-written DCC clear keys be applied without readback.
	bool                               conditional_rendering_enabled         = false;
	// VK_EXT_graphics_pipeline_library: compile pipeline stages as libraries and link them.
	bool                               pipeline_library_enabled              = false;
	bool                               pipeline_library_fast_linking         = false;
	// VK_KHR_pipeline_executable_properties, enabled only for KYTY_DEBUG_PIPELINE_STATS.
	bool                               pipeline_executable_info_enabled      = false;
	// KYTY_DEBUG_DEVICE_FAULT=1: VK_EXT_device_fault is on, a lost device reports what faulted.
	bool                               device_fault_enabled                  = false;
	// KYTY_DEBUG_DEVICE_FAULT=1: VK_EXT_device_address_binding_report is on, every bind and
	// unbind of a GPU address range is kept in GpuZones::g_bindings (see vulkanWindow.cpp).
	bool                               address_binding_enabled               = false;
	vk::DebugUtilsMessengerEXT         binding_messenger                     = nullptr;
	// KYTY_DEBUG_GPU_CHECKPOINTS=1: VK_NV_device_diagnostic_checkpoints is on (see GpuZones).
	bool                               checkpoints_enabled                   = false;
	// KYTY_EOP_GPU_LABELS: VK_AMD_buffer_marker is on, labels reach cached buffers as end-of-pipe
	// marker writes (BufferCache::MirrorLabel). KYTY_EOP_LABEL_MARKERS=0 keeps the fills.
	bool                               buffer_marker_enabled                 = false;
	// KYTY_AFTERMATH=1 (build with KYTY_AFTERMATH): VK_NV_device_diagnostics_config is on, the GPU debugger
	// Aftermath collects a GPU crash dump when the device is lost (see aftermath.h).
	bool                               aftermath_enabled                     = false;
	bool                               depth_color_copy_enabled               = false;
	bool                               supports_block_texel_view              = false;
	// Storage buffer word accesses leave their range check to robustBufferAccess2, and storage
	// buffer ranges are rounded down to whole dwords (see
	// ShaderRecompiler::SetHardwareStorageBufferBounds).
	bool                               hardware_storage_buffer_bounds         = false;
	bool                                      mesh_shader_enabled                   = false;
	// Descriptor indexing for bindless images: runtime arrays indexed non-uniformly, partially
	// bound and updated after bind.
	bool                               bindless_enabled                      = false;
	uint32_t                           bindless_max_sampled_images           = 0;
	uint32_t                           bindless_max_samplers                 = 0;
	vk::DescriptorSetLayout            bindless_layout                       = nullptr;
	vk::DescriptorSet                  bindless_set                          = nullptr;
	// Swapchain presents so far; the bindless table resolves requested textures once per frame.
	std::atomic<uint64_t>              presented_frames {0};
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	vk::ShaderStageFlags               required_subgroup_size_stages         = {};
	Common::Mutex                      queue_mutex;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;
	// KYTY_READBACK_QUEUE=1: a transfer queue of another family for guest readbacks (see
	// ReadbackQueue). Buffers are then shared by both families (shared_queue_families).
	uint32_t                           readback_family = static_cast<uint32_t>(-1);
	vk::Queue                          readback_queue  = nullptr;
	std::array<uint32_t, 2>            shared_queue_families {};
	// Makes a buffer usable on both queues when the readback queue exists.
	void ShareBuffer(vk::BufferCreateInfo& info) const noexcept {
		if (readback_queue != nullptr) {
			info.sharingMode           = vk::SharingMode::eConcurrent;
			info.queueFamilyIndexCount = 2;
			info.pQueueFamilyIndices   = shared_queue_families.data();
		}
	}

	[[nodiscard]] const vk::PhysicalDeviceProperties& GetPhysicalDeviceProperties() const {
		return physical_device_properties;
	}

	[[nodiscard]] const vk::PhysicalDeviceMemoryProperties&
	GetPhysicalDeviceMemoryProperties() const {
		return physical_device_memory_properties;
	}

	[[nodiscard]] vk::FormatProperties GetFormatProperties(vk::Format format) const {
		std::scoped_lock lock(m_format_properties_mutex);
		auto [it, inserted] = m_format_properties.try_emplace(format);
		if (inserted) {
			physical_device.getFormatProperties(format, &it->second);
		}
		return it->second;
	}

	[[nodiscard]] vk::Result GetImageFormatProperties(vk::Format format, vk::ImageType type,
	                                                  vk::ImageTiling            tiling,
	                                                  vk::ImageUsageFlags        usage,
	                                                  vk::ImageCreateFlags       flags,
	                                                  vk::ImageFormatProperties* properties) const {
		using Key = std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
		                       vk::ImageCreateFlags>;
		std::scoped_lock lock(m_image_format_properties_mutex);
		auto [it, inserted] =
		    m_image_format_properties.try_emplace(Key {format, type, tiling, usage, flags});
		if (inserted) {
			it->second.first = physical_device.getImageFormatProperties(format, type, tiling, usage,
			                                                            flags, &it->second.second);
		}
		if (properties != nullptr) {
			*properties = it->second.second;
		}
		return it->second.first;
	}

	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
		return subgroup_size == 64u || compute_subgroup_size_control_enabled;
	}

	[[nodiscard]] vk::DeviceSize StorageMinAlignment() const {
		const auto alignment = physical_device_properties.limits.minStorageBufferOffsetAlignment;
		return alignment != 0 ? alignment : 1;
	}

	[[nodiscard]] bool CreateAllocator();
	void               DestroyAllocator();
	void               LogMemoryBudget() const;
	[[nodiscard]] bool CanReportMemoryUsage() const noexcept { return memory_budget_ext_enabled; }
	[[nodiscard]] uint64_t GetDeviceMemoryUsage() const;
	[[nodiscard]] uint64_t GetTotalMemoryBudget() const;

	struct HeapUsage {
		uint64_t usage            = 0; // Whole process, as reported by VK_EXT_memory_budget.
		uint64_t budget           = 0;
		uint64_t size             = 0;
		uint64_t allocation_bytes = 0; // Allocated through VMA.
		uint32_t allocations      = 0;
		bool     device_local     = false;
	};
	[[nodiscard]] std::vector<HeapUsage> GetHeapUsage() const;
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	mutable std::mutex                                 m_format_properties_mutex;
	mutable std::map<vk::Format, vk::FormatProperties> m_format_properties;
	mutable std::mutex                                 m_image_format_properties_mutex;
	mutable std::map<std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
	                            vk::ImageCreateFlags>,
	                 std::pair<vk::Result, vk::ImageFormatProperties>>
	    m_image_format_properties;
};

struct VulkanImageState {
	vk::PipelineStageFlags2 pl_stage    = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        access_mask = vk::AccessFlagBits2::eNone;
	vk::ImageLayout         layout      = vk::ImageLayout::eUndefined;
};

struct VulkanImage {
	VulkanImage() = default;
	KYTY_CLASS_NO_COPY(VulkanImage);

	vk::Format                    format      = vk::Format::eUndefined;
	vk::ImageType                 image_type  = vk::ImageType::e2D;
	vk::Extent3D                  extent      = {1, 1, 1};
	uint32_t                      layers      = 1;
	uint32_t                      mip_levels  = 1;
	uint32_t                      samples     = 1;
	// Created with the plain parameters the image recycler keys on (see GraphicContext::DeleteImage).
	bool                          recyclable  = false;
	vk::ImageUsageFlags           usage       = {};
	vk::ImageCreateFlags          flags       = {};
	vk::Image                     image       = nullptr;
	VulkanImageState              state;
	std::vector<VulkanImageState> subresource_states;
	VmaAllocation                allocation = nullptr;
};



} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
