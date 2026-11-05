#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_

#define VK_NO_PROTOTYPES
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC    1
#define VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL 1
#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_EXCEPTIONS

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"

#include <cstdint>
#include <fmt/format.h>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_hash.hpp>

namespace Libs::Graphics {

vk::Format  VulkanFormat(Prospero::BufferFormat guest_format);
void        RequireVulkanSuccess(vk::Result result, const char* operation);
vk::ShaderModule CompileSPV(std::span<const uint32_t> code, vk::Device device);

[[nodiscard]] constexpr bool IsHdrSwapchainFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32: return true;
		default: return false;
	}
}

// What a barrier to all later commands makes earlier writes visible to: every write and every
// device read except INDIRECT_COMMAND_READ (and the conditional-rendering read, whose only producer
// records its own barrier). MEMORY_READ includes indirect-argument fetches, and waiting for those
// makes some drivers drain much more: on a current GPU a barrier between two dispatches costs 4.1 us
// with MEMORY_READ and 1.2 us with these reads (CarRamRod 8b26c9064). Every command that reads
// indirect arguments on the GPU records its own barrier for them.
inline constexpr vk::AccessFlags LaterAccess =
    vk::AccessFlagBits::eIndexRead | vk::AccessFlagBits::eVertexAttributeRead |
    vk::AccessFlagBits::eUniformRead | vk::AccessFlagBits::eInputAttachmentRead |
    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eColorAttachmentRead |
    vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eTransferRead |
    vk::AccessFlagBits::eMemoryWrite;
inline constexpr vk::AccessFlags2 LaterAccess2 =
    vk::AccessFlagBits2::eIndexRead | vk::AccessFlagBits2::eVertexAttributeRead |
    vk::AccessFlagBits2::eUniformRead | vk::AccessFlagBits2::eInputAttachmentRead |
    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eColorAttachmentRead |
    vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eTransferRead |
    vk::AccessFlagBits2::eMemoryWrite;

// KYTY_LATER_ACCESS=1: barriers to all later commands name LaterAccess instead of their wider
// destination access (off by default until measured on the title in Pro mode). KYTY_LATER_ACCESS_OFF_FILE=<path>:
// the old masks while that file exists (in-run A/B; looked at every 1024 barriers). The indirect
// dispatch's own barrier covers all earlier writes whenever the switch is set, off file or not.
// KYTY_LATER_ACCESS_SITES=<hex>: only the sites whose bit (LaterSite) is set take LaterAccess
// (default all; the title in Pro mode hung on its splash screen with all of them, run la1/la2).
enum class LaterSite : uint32_t {
	Dispatch,    // compute -> all commands after a dispatch (ShaderAccessBarrier)
	BufferCache, // WRITE_DATA copies, buffer uploads, upload batches
	Fill,        // StreamBuffer::Fill (labels, fills)
	CopyFrom,    // StreamBuffer::CopyFrom's default destination access
	Image,       // image upload / download
	Tiler,       // tiler passes
	GuestFlush,  // the guest's global barrier
	Htile,       // HTile synthesis
	DccResolver, // DCC clear predicates
	DccClear,    // the predicated DCC clear
};
[[nodiscard]] bool LaterAccessEnabled() noexcept;
[[nodiscard]] bool LaterAccessOn(LaterSite site) noexcept;
[[nodiscard]] inline vk::AccessFlags LaterAccessOr(LaterSite site, vk::AccessFlags old) noexcept {
	return LaterAccessOn(site) ? LaterAccess : old;
}
[[nodiscard]] inline vk::AccessFlags2 LaterAccessOr(LaterSite site, vk::AccessFlags2 old) noexcept {
	return LaterAccessOn(site) ? LaterAccess2 : old;
}

template <typename Handle, typename... Args>
void SetVulkanObjectNameF(vk::Device device, Handle handle, fmt::format_string<Args...> format,
                          Args&&... args) {
	// Also with --rd: a RenderDoc capture then names images and buffers by their guest address.
	if ((!Config::GraphicsDebugDumpEnabled() && !Config::RenderDocEnabled()) || device == nullptr ||
	    handle == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT == nullptr) {
		return;
	}

	const auto                      name = fmt::format(format, std::forward<Args>(args)...);
	vk::DebugUtilsObjectNameInfoEXT info {};
	info.objectType   = Handle::objectType;
	info.objectHandle = static_cast<uint64_t>(
	    reinterpret_cast<uintptr_t>(static_cast<typename Handle::CType>(handle)));
	info.pObjectName = name.c_str();
	(void)device.setDebugUtilsObjectNameEXT(&info);
}

// With --rd: a marker in the command buffer, so a RenderDoc capture shows what the emulator
// decided at that point (an upload from guest memory, a skipped draw) next to the API calls.
template <typename... Args>
void InsertCaptureLabelF(vk::CommandBuffer command, fmt::format_string<Args...> format,
                         Args&&... args) {
	if (!Config::RenderDocEnabled() || command == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdInsertDebugUtilsLabelEXT == nullptr) [[likely]] {
		return;
	}
	const auto             text = fmt::format(format, std::forward<Args>(args)...);
	vk::DebugUtilsLabelEXT label {};
	label.pLabelName = text.c_str();
	command.insertDebugUtilsLabelEXT(&label);
}

template <typename T, typename Enumerator>
[[nodiscard]] std::vector<T> EnumerateVulkan(const char* operation, Enumerator&& enumerate) {
	for (;;) {
		uint32_t count = 0;
		RequireVulkanSuccess(enumerate(&count, nullptr), operation);
		if (count == 0) {
			return {};
		}

		std::vector<T> values(count);
		const auto     result = enumerate(&count, values.data());
		if (result == vk::Result::eSuccess) {
			values.resize(count);
			return values;
		}
		if (result != vk::Result::eIncomplete) {
			RequireVulkanSuccess(result, operation);
		}
	}
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_
