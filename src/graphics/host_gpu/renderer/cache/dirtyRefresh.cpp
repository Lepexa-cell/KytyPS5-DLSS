#include "graphics/host_gpu/renderer/cache/dirtyRefresh.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/dirty_refresh1_spv.h"
#include "gpu_tiler_shaders/dirty_refresh2_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

namespace {

struct RefreshParams {
	uint32_t source_word;
	uint32_t bitmap_word;
	uint32_t width;
	uint32_t height;
	uint32_t blocks_per_row;
};

// 1 GiB of tracked image memory.
constexpr uint32_t ArenaWords = 8u * 1024u * 1024u;

} // namespace

DirtyRefresh::DirtyRefresh(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_arena(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
              AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
              uint64_t {ArenaWords} * sizeof(uint32_t)) {
	SetVulkanObjectNameF(graphics.device, m_arena.Handle(), "Dirty Track Bitmaps");
	m_free.push_back({0, ArenaWords});

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	    "create dirty refresh descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(RefreshParams)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create dirty refresh pipeline layout");

	for (uint32_t i = 0; i < 2; i++) {
		const auto module = i == 0 ? CompileSPV(DIRTY_REFRESH1_SPV, graphics.device)
		                           : CompileSPV(DIRTY_REFRESH2_SPV, graphics.device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = m_pipeline_layout;
		const auto result = graphics.device.createComputePipelines(nullptr, 1, &pipeline_info,
		                                                           nullptr, &m_pipelines[i]);
		graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create dirty refresh pipeline");
		SetVulkanObjectNameF(graphics.device, m_pipelines[i], "Kyty.DirtyRefresh{}", i + 1);
	}
}

DirtyRefresh::~DirtyRefresh() {
	for (auto pipeline: m_pipelines) {
		if (pipeline != nullptr) {
			m_graphics.device.destroyPipeline(pipeline, nullptr);
		}
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

uint32_t DirtyRefresh::Allocate(uint32_t words) {
	// Slots start on 16-byte boundaries.
	words = Common::AlignUp(words, 4u);
	for (auto it = m_free.begin(); it != m_free.end(); ++it) {
		if (it->words >= words) {
			const auto slot = it->slot;
			it->slot += words;
			it->words -= words;
			if (it->words == 0) {
				m_free.erase(it);
			}
			return slot;
		}
	}
	return NoSlot;
}

void DirtyRefresh::Free(uint32_t slot, uint32_t words) {
	words   = Common::AlignUp(words, 4u);
	auto at = std::lower_bound(m_free.begin(), m_free.end(), slot,
	                           [](const FreeRun& run, uint32_t value) { return run.slot < value; });
	at      = m_free.insert(at, {slot, words});
	if (std::next(at) != m_free.end() && at->slot + at->words == std::next(at)->slot) {
		at->words += std::next(at)->words;
		m_free.erase(std::next(at));
	}
	if (at != m_free.begin() && std::prev(at)->slot + std::prev(at)->words == at->slot) {
		std::prev(at)->words += at->words;
		m_free.erase(at);
	}
}

void DirtyRefresh::RecordClear(vk::CommandBuffer command, uint32_t slot, uint32_t words) {
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eClear;
	before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);
	command.fillBuffer(m_arena.Handle(), uint64_t {slot} * sizeof(uint32_t),
	                   uint64_t {Common::AlignUp(words, 4u)} * sizeof(uint32_t), 0u);
	vk::MemoryBarrier2 after {};
	after.srcStageMask         = vk::PipelineStageFlagBits2::eClear;
	after.srcAccessMask        = vk::AccessFlagBits2::eTransferWrite;
	after.dstStageMask         = vk::PipelineStageFlagBits2::eAllCommands;
	after.dstAccessMask        = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
}

void DirtyRefresh::RecordRefresh(vk::CommandBuffer command, vk::Buffer source,
                                 uint64_t source_offset, uint64_t size, uint32_t slot,
                                 uint32_t words, const Target& target) {
	GpuZones::Mark(command, DrainStats::Zone::ImageCopy);
	EXIT_IF(!Available() || source == nullptr || target.view == nullptr || size == 0 ||
	        source_offset % sizeof(uint32_t) != 0 ||
	        (target.texel_words != 1 && target.texel_words != 2));
	const auto alignment      = m_graphics.StorageMinAlignment();
	const auto binding_offset = Common::AlignDown(source_offset, alignment);
	const auto binding_size   = Common::AlignUp(source_offset - binding_offset + size, 4u);
	const auto max_range = m_graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	EXIT_IF(binding_size > max_range);

	// The guest's marks and writes of the source bytes finish before the copy reads them.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo buffers[] {
	    {source, binding_offset, binding_size},
	    {m_arena.Handle(), uint64_t {slot} * sizeof(uint32_t),
	     uint64_t {Common::AlignUp(words, 4u)} * sizeof(uint32_t)},
	};
	const vk::DescriptorImageInfo image {nullptr, target.view, vk::ImageLayout::eGeneral};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		if (index < 2) {
			writes[index].descriptorType = vk::DescriptorType::eStorageBuffer;
			writes[index].pBufferInfo    = &buffers[index];
		} else {
			writes[index].descriptorType = vk::DescriptorType::eStorageImage;
			writes[index].pImageInfo     = &image;
		}
	}
	const RefreshParams params {
	    .source_word    = static_cast<uint32_t>((source_offset - binding_offset) / sizeof(uint32_t)),
	    .bitmap_word    = 0,
	    .width          = target.width,
	    .height         = target.height,
	    .blocks_per_row = target.blocks_per_row,
	};
	const auto pipeline = m_pipelines[target.texel_words - 1];
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params),
	                      &params);
	command.dispatch((target.width + 7) / 8, (target.height + 7) / 8, 1);

	// The copy has read the marks before they are zeroed (RecordClear waits for all reads).
	RecordClear(command, slot, words);
}

} // namespace Libs::Graphics
