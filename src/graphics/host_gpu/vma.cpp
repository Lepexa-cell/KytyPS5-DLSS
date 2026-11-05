#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/switches.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/handleRegistry.h"
#include "graphics/host_gpu/renderer/debug.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

// Small images come from a pool of 64 MiB blocks. The driver prefers a dedicated allocation for
// render targets, so without the pool each one is a kernel allocation of its own: Astro Bot
// recreates a 960x540 depth surface as a color image and back every frame (~3% of the GPU
// thread in allocation and free calls). KYTY_SMALL_IMAGE_POOL=0 turns the pool off;
// KYTY_DEBUG_AB=imagepool alternates.
namespace {
constexpr uint64_t SmallImageBytes     = 8u * 1024u * 1024u;
constexpr uint64_t SmallImageBlockSize = 64u * 1024u * 1024u;

struct SmallImagePool {
	std::mutex mutex;
	VmaPool    pool   = VK_NULL_HANDLE;
	bool       failed = false;
};
SmallImagePool g_small_image_pool;

bool SmallImagePoolEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_SMALL_IMAGE_POOL");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("imagepool");
	return enabled && !(ab && AbFeatureOff());
}

// An upper bound of the image's memory: every texel at the format's block size, all levels.
uint64_t EstimateImageBytes(const vk::ImageCreateInfo& info) {
	const uint64_t texels = uint64_t {info.extent.width} * info.extent.height * info.extent.depth *
	                        info.arrayLayers * static_cast<uint32_t>(info.samples);
	const uint64_t bytes  = texels * std::max<uint32_t>(vk::blockSize(info.format), 1u);
	return info.mipLevels > 1 ? bytes + bytes / 2u : bytes;
}

VmaPool GetSmallImagePool(VmaAllocator allocator, const vk::ImageCreateInfo& image_info,
                          const VmaAllocationCreateInfo& alloc_info) {
	std::scoped_lock lock(g_small_image_pool.mutex);
	if (g_small_image_pool.pool != VK_NULL_HANDLE || g_small_image_pool.failed) {
		return g_small_image_pool.pool;
	}
	uint32_t type_index = 0;
	if (vmaFindMemoryTypeIndexForImageInfo(
	        allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
	        &alloc_info, &type_index) != VK_SUCCESS) {
		g_small_image_pool.failed = true;
		return VK_NULL_HANDLE;
	}
	VmaPoolCreateInfo pool_info {};
	pool_info.memoryTypeIndex = type_index;
	pool_info.blockSize       = SmallImageBlockSize;
	if (vmaCreatePool(allocator, &pool_info, &g_small_image_pool.pool) != VK_SUCCESS) {
		g_small_image_pool.pool   = VK_NULL_HANDLE;
		g_small_image_pool.failed = true;
	}
	return g_small_image_pool.pool;
}
} // namespace

namespace {

// VMA fetches the memory budget from the driver again once 30 allocations or frees have passed
// since the last fetch (VmaAllocator_T::GetHeapBudgets), and a fetch took the AMD driver about a
// millisecond: with Sky Garden's buffer and image churn, about 1% of the GPU thread. Budget
// queries are answered from a fetch younger than FetchInterval instead, with each heap's usage
// moved by the device memory VMA allocated and freed since, as VMA itself estimates usage between
// its fetches. Only the other processes' usage is then up to FetchInterval old.
// KYTY_DEBUG_AB=budgetcache fetches every time in every other window.
struct BudgetCache {
	static constexpr auto FetchInterval = std::chrono::milliseconds(500);

	std::mutex                                            mutex;
	PFN_vkGetPhysicalDeviceMemoryProperties2              fetch = nullptr;
	bool                                                  valid = false;
	std::chrono::steady_clock::time_point                 fetched;
	VkPhysicalDeviceMemoryProperties                      memory {};
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>         usage {};
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>         budget {};
	std::array<int64_t, VK_MAX_MEMORY_HEAPS>              allocated_at_fetch {};
	// VMA's device memory per heap, kept by its allocate and free callbacks.
	std::array<std::atomic<int64_t>, VK_MAX_MEMORY_HEAPS> allocated {};
	std::array<uint32_t, VK_MAX_MEMORY_TYPES>             type_heap {};
};

BudgetCache g_budget_cache;

void VKAPI_PTR CountAllocation(VmaAllocator /*allocator*/, uint32_t memory_type,
                               VkDeviceMemory /*memory*/, VkDeviceSize size, void* /*user_data*/) {
	g_budget_cache.allocated[g_budget_cache.type_heap[memory_type]].fetch_add(
	    static_cast<int64_t>(size), std::memory_order_relaxed);
}

void VKAPI_PTR CountFree(VmaAllocator /*allocator*/, uint32_t memory_type, VkDeviceMemory /*memory*/,
                         VkDeviceSize size, void* /*user_data*/) {
	g_budget_cache.allocated[g_budget_cache.type_heap[memory_type]].fetch_sub(
	    static_cast<int64_t>(size), std::memory_order_relaxed);
}

void VKAPI_PTR CachedMemoryProperties2(VkPhysicalDevice                     physical_device,
                                       VkPhysicalDeviceMemoryProperties2* properties) {
	auto& cache  = g_budget_cache;
	auto* budget = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
	static const bool ab = AbSelected("budgetcache");
	if (budget == nullptr ||
	    budget->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT ||
	    budget->pNext != nullptr || (ab && AbFeatureOff())) {
		cache.fetch(physical_device, properties);
		return;
	}
	std::scoped_lock lock(cache.mutex);
	const auto       now = std::chrono::steady_clock::now();
	if (!cache.valid || now - cache.fetched >= BudgetCache::FetchInterval) {
		cache.fetch(physical_device, properties);
		cache.memory = properties->memoryProperties;
		for (uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; heap++) {
			cache.usage[heap]              = budget->heapUsage[heap];
			cache.budget[heap]             = budget->heapBudget[heap];
			cache.allocated_at_fetch[heap] = cache.allocated[heap].load(std::memory_order_relaxed);
		}
		cache.fetched = now;
		cache.valid   = true;
		return;
	}
	properties->memoryProperties = cache.memory;
	for (uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; heap++) {
		const auto delta =
		    cache.allocated[heap].load(std::memory_order_relaxed) - cache.allocated_at_fetch[heap];
		const auto usage = static_cast<int64_t>(cache.usage[heap]) + delta;
		budget->heapUsage[heap]  = static_cast<VkDeviceSize>(std::max<int64_t>(usage, 0));
		budget->heapBudget[heap] = cache.budget[heap];
	}
}

} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	VmaDeviceMemoryCallbacks memory_callbacks {};
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
		auto& cache = g_budget_cache;
		cache.fetch = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(
		        static_cast<VkInstance>(instance), "vkGetPhysicalDeviceMemoryProperties2"));
		EXIT_IF(cache.fetch == nullptr);
		cache.valid = false;
		const auto& properties = physical_device.getMemoryProperties();
		for (uint32_t type = 0; type < properties.memoryTypeCount; type++) {
			cache.type_heap[type] = properties.memoryTypes[type].heapIndex;
		}
		functions.vkGetPhysicalDeviceMemoryProperties2KHR = CachedMemoryProperties2;
		memory_callbacks.pfnAllocate = CountAllocation;
		memory_callbacks.pfnFree     = CountFree;
		info.pDeviceMemoryCallbacks  = &memory_callbacks;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	{
		std::scoped_lock lock(g_small_image_pool.mutex);
		if (g_small_image_pool.pool != VK_NULL_HANDLE) {
			vmaDestroyPool(allocator, g_small_image_pool.pool);
			g_small_image_pool.pool = VK_NULL_HANDLE;
		}
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

std::vector<GraphicContext::HeapUsage> GraphicContext::GetHeapUsage() const {
	std::vector<HeapUsage> heaps;
	if (allocator == nullptr) {
		return heaps;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps.push_back({.usage            = budgets[heap].usage,
		                 .budget           = budgets[heap].budget,
		                 .size             = properties.size,
		                 .allocation_bytes = budgets[heap].statistics.allocationBytes,
		                 .allocations      = budgets[heap].statistics.allocationCount,
		                 .device_local     = static_cast<bool>(properties.flags &
		                                                       vk::MemoryHeapFlagBits::eDeviceLocal)});
	}
	return heaps;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}


// Images freed lately, kept to serve a creation with the same parameters: ASTRO BOT's Sky Garden
// uses one 960x540 surface as depth target and as R32 texture in turn, which replaced its image
// three times a frame (create, destroy and the driver's kernel calls: about 4% of the GPU
// thread). A recycled image has undefined contents, as a new one. KYTY_IMAGE_RECYCLE=0 turns it
// off.
namespace {
struct RecycledImage {
	vk::Image     image;
	VmaAllocation allocation = nullptr;
	vk::Format    format     = vk::Format::eUndefined;
	vk::ImageType image_type = vk::ImageType::e2D;
	vk::Extent3D  extent {};
	uint32_t      layers     = 0;
	uint32_t      mip_levels = 0;
	uint32_t      samples    = 0;
	vk::ImageUsageFlags  usage {};
	vk::ImageCreateFlags flags {};
	uint64_t             serial = 0;
};
constexpr size_t           RecycledImagesMax = 24;
std::mutex                 g_recycled_mutex;
std::vector<RecycledImage> g_recycled;
uint64_t                   g_recycled_serial = 0;

bool ImageRecycleEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_IMAGE_RECYCLE");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

bool ImageRecyclable(const vk::ImageCreateInfo& info) {
	return ImageRecycleEnabled() && info.pNext == nullptr &&
	       info.tiling == vk::ImageTiling::eOptimal &&
	       info.sharingMode == vk::SharingMode::eExclusive &&
	       info.initialLayout == vk::ImageLayout::eUndefined;
}
} // namespace

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	image.recyclable = ImageRecyclable(image_info);
	if (image.recyclable) {
		std::scoped_lock lock {g_recycled_mutex};
		const auto found = std::ranges::find_if(g_recycled, [&](const RecycledImage& r) {
			return r.format == image_info.format && r.image_type == image_info.imageType &&
			       r.extent == image_info.extent && r.layers == image_info.arrayLayers &&
			       r.mip_levels == image_info.mipLevels &&
			       r.samples == static_cast<uint32_t>(image_info.samples) &&
			       r.usage == image_info.usage && r.flags == image_info.flags;
		});
		if (found != g_recycled.end()) {
			image.image      = found->image;
			image.allocation = found->allocation;
			g_recycled.erase(found);
			image.format     = image_info.format;
			image.image_type = image_info.imageType;
			image.extent     = image_info.extent;
			image.layers     = image_info.arrayLayers;
			image.mip_levels = image_info.mipLevels;
			image.samples    = static_cast<uint32_t>(image_info.samples);
			image.usage      = image_info.usage;
			image.flags      = image_info.flags;
			image.state      = {.layout = image_info.initialLayout};
			image.subresource_states.clear();
			// A new generation: a call recorded for the handle's previous life is stale.
			HandleRegistry::Register(reinterpret_cast<uint64_t>(static_cast<VkImage>(image.image)),
			                         HandleRegistry::Kind::Image);
			return true;
		}
	}

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	vk::Result       result       = vk::Result::eErrorUnknown;
	if (SmallImagePoolEnabled() && EstimateImageBytes(image_info) <= SmallImageBytes) {
		if (const auto pool = GetSmallImagePool(allocator, image_info, alloc_info);
		    pool != VK_NULL_HANDLE) {
			auto pooled = alloc_info;
			pooled.pool = pool;
			result      = static_cast<vk::Result>(vmaCreateImage(
			         allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
			         &pooled, &native_image, &image.allocation, nullptr));
			if (result != vk::Result::eSuccess) {
				// Another memory type, or the pool is out of memory: allocate as before.
				native_image     = VK_NULL_HANDLE;
				image.allocation = nullptr;
			}
		}
	}
	if (native_image == VK_NULL_HANDLE) {
		result = static_cast<vk::Result>(
		    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
		                   &alloc_info, &native_image, &image.allocation, nullptr));
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();
	HandleRegistry::Register(reinterpret_cast<uint64_t>(static_cast<VkImage>(image.image)),
	                         HandleRegistry::Kind::Image);

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	vk::Image     destroy_image      = image.image;
	VmaAllocation destroy_allocation = image.allocation;
	HandleRegistry::Unregister(reinterpret_cast<uint64_t>(static_cast<VkImage>(image.image)));
	if (image.recyclable) {
		std::scoped_lock lock {g_recycled_mutex};
		g_recycled.push_back({image.image, image.allocation, image.format, image.image_type,
		                      image.extent, image.layers, image.mip_levels, image.samples,
		                      image.usage, image.flags, ++g_recycled_serial});
		destroy_image      = nullptr;
		destroy_allocation = nullptr;
		if (g_recycled.size() > RecycledImagesMax) {
			const auto oldest = std::ranges::min_element(g_recycled, {}, &RecycledImage::serial);
			destroy_image      = oldest->image;
			destroy_allocation = oldest->allocation;
			g_recycled.erase(oldest);
		}
	}
	if (destroy_image) {
		vmaDestroyImage(allocator, destroy_image, destroy_allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
