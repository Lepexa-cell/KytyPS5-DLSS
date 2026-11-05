#include "graphics/host_gpu/renderer/cache/htileWriter.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/htile_synth_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>

namespace Libs::Graphics {

namespace {

struct SynthParams {
	uint32_t width;
	uint32_t height;
	uint32_t row_texels;
	uint32_t slice_texels;
	uint32_t first_word;
	uint32_t layer_words;
	uint32_t pitch_blocks;
	uint32_t format;
};

constexpr uint32_t BlockWidth  = 1024;
constexpr uint32_t BlockHeight = 512;
constexpr uint64_t BlockBytes  = 32768;

} // namespace

HtileWriter::HtileWriter(GraphicContext& graphics): m_graphics(graphics) {
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	    "create HTile descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(SynthParams)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create HTile pipeline layout");

	const auto module = CompileSPV(HTILE_SYNTH_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create HTile pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "Kyty.HtileSynth");
}

HtileWriter::~HtileWriter() {
	if (m_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

uint64_t HtileWriter::LayerBytes(uint32_t width, uint32_t height) noexcept {
	return static_cast<uint64_t>(Common::AlignUp(width, BlockWidth) / BlockWidth) *
	       (Common::AlignUp(height, BlockHeight) / BlockHeight) * BlockBytes;
}

void HtileWriter::Record(vk::CommandBuffer command, vk::Buffer depth, uint64_t depth_size,
                         const Buffer& htile, uint64_t htile_offset, const Surface& surface) {
	GpuZones::Mark(command, DrainStats::Zone::ImageCopy);
	const auto layer_bytes = surface.layer_bytes;
	EXIT_IF(!Available() || depth == nullptr || depth_size == 0 || surface.width == 0 ||
	        surface.height == 0 || surface.layers == 0 || htile_offset % sizeof(uint32_t) != 0 ||
	        layer_bytes < LayerBytes(surface.width, surface.height) ||
	        layer_bytes % sizeof(uint32_t) != 0);
	const auto alignment      = m_graphics.StorageMinAlignment();
	const auto binding_offset = Common::AlignDown(htile_offset, alignment);
	const auto binding_size   = htile_offset - binding_offset + layer_bytes * surface.layers;
	const auto max_range = m_graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	EXIT_IF(binding_offset > htile.Size() || binding_size > htile.Size() - binding_offset ||
	        binding_size > max_range || depth_size > max_range);

	// The guest's earlier reads and writes of the HTile range finish before it is rewritten.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo infos[] {
	    {depth, 0, Common::AlignUp(depth_size, sizeof(uint32_t))},
	    {htile.Handle(), binding_offset, binding_size},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const SynthParams params {
	    .width        = surface.width,
	    .height       = surface.height,
	    .row_texels   = surface.row_texels,
	    .slice_texels = surface.slice_texels,
	    .first_word   = static_cast<uint32_t>((htile_offset - binding_offset) / sizeof(uint32_t)),
	    .layer_words  = static_cast<uint32_t>(layer_bytes / sizeof(uint32_t)),
	    .pitch_blocks = Common::AlignUp(surface.width, BlockWidth) / BlockWidth,
	    .format       = static_cast<uint32_t>(surface.format),
	};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params),
	                      &params);
	command.dispatch((surface.width + 7) / 8, (surface.height + 7) / 8, surface.layers);

	vk::MemoryBarrier2 after {};
	after.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	after.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	after.dstAccessMask =
	    LaterAccessOr(LaterSite::Htile, vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
}

} // namespace Libs::Graphics
