#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <deque>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout layout);

private:
	static constexpr uint32_t DescriptorSetBatch = 32;

	// Sets are never returned to the driver: a set is handed out again once the GPU has finished
	// the tick it was last used in. In steady state no set is allocated at all (allocation was
	// about 4% of the GPU thread in ASTRO BOT's Sky Garden).
	struct Used {
		vk::DescriptorSet set;
		uint64_t          tick = 0;
	};
	struct Sets {
		std::vector<vk::DescriptorSet> fresh;
		std::deque<Used>               used; // In tick order.
	};

	void Allocate(vk::DescriptorSetLayout layout, std::vector<vk::DescriptorSet>& sets);
	void CreateDescriptorPool();

	GraphicContext&                                   m_graphics;
	MasterSemaphore&                                  m_master_semaphore;
	std::vector<vk::DescriptorPool>                   m_pools;
	std::unordered_map<vk::DescriptorSetLayout, Sets> m_sets;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
