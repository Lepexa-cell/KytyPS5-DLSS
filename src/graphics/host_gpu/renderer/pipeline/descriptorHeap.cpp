#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
};

} // namespace

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	for (const auto pool: m_pools) {
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(layout == nullptr);

	auto&             sets = m_sets[layout];
	const auto        tick = m_master_semaphore.CurrentTick();
	vk::DescriptorSet set;
	if (!sets.fresh.empty()) {
		set = sets.fresh.back();
		sets.fresh.pop_back();
	} else if (!sets.used.empty() && sets.used.front().tick < tick &&
	           m_master_semaphore.IsFree(sets.used.front().tick)) {
		set = sets.used.front().set;
		sets.used.pop_front();
	} else {
		Allocate(layout, sets.fresh);
		set = sets.fresh.back();
		sets.fresh.pop_back();
	}
	sets.used.push_back({set, tick});
	return set;
}

void DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, std::vector<vk::DescriptorSet>& sets) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);
	std::array<vk::DescriptorSet, DescriptorSetBatch> batch {};

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.pSetLayouts = layouts.data();
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		allocate.descriptorPool = m_pools.back();
		for (uint32_t count = DescriptorSetBatch; count != 0; count /= 2) {
			allocate.descriptorSetCount = count;
			const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.data());
			if (result == vk::Result::eSuccess) {
				sets.insert(sets.end(), batch.begin(), batch.begin() + count);
				return;
			}
			EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
			        result != vk::Result::eErrorFragmentedPool);
		}
		CreateDescriptorPool();
	}
	EXIT("DescriptorHeap: a new descriptor pool cannot hold one set\n");
}

void DescriptorHeap::CreateDescriptorPool() {
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount;
	create.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
	create.pPoolSizes    = DescriptorPoolSizes.data();
	vk::DescriptorPool pool;
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &pool) != vk::Result::eSuccess);
	m_pools.push_back(pool);
}

} // namespace Libs::Graphics
