#include "graphics/host_gpu/renderer/dispatchThreadGroups.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/dispatch_thread_groups_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace Libs::Graphics {

namespace {

// Mirrors the push-constant block of dispatch_thread_groups.comp.
struct ShaderParams {
	uint32_t arguments_word;
	uint32_t output_word;
	uint32_t group_size_x;
	uint32_t group_size_y;
	uint32_t group_size_z;
	uint32_t extent_word;
	uint32_t max_groups;
	uint32_t shader_low;
};

} // namespace

DispatchThreadGroups::DispatchThreadGroups(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_report(std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, AllFlags, 16)) {
	std::memset(m_report->Mapped().data(), 0, 16);
	m_report->Flush(0, 16);
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	    "create dispatch thread groups descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(ShaderParams)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create dispatch thread groups pipeline layout");

	const auto module = CompileSPV(DISPATCH_THREAD_GROUPS_SPV, graphics.device);
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
	RequireVulkanSuccess(result, "create dispatch thread groups pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "Kyty.DispatchThreadGroups");
}

DispatchThreadGroups::~DispatchThreadGroups() {
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

void DispatchThreadGroups::Record(vk::CommandBuffer command, const Buffer& arguments,
                                  uint64_t arguments_offset, const Buffer& output,
                                  uint64_t output_offset, const std::array<uint32_t, 3>& group_size,
                                  const vk::DescriptorBufferInfo& extent, uint32_t extent_dword,
                                  uint32_t max_groups, uint64_t shader_hash) {
	ReportClipped();
	const auto alignment = m_graphics.StorageMinAlignment();
	EXIT_IF(arguments_offset % sizeof(uint32_t) != 0 || output_offset % sizeof(uint32_t) != 0);
	const auto arguments_binding = Common::AlignDown(arguments_offset, alignment);
	const auto output_binding    = Common::AlignDown(output_offset, alignment);
	const auto arguments_size    = arguments_offset - arguments_binding + ArgumentsSize;
	const auto output_size       = output_offset - output_binding + OutputSize;
	EXIT_IF(arguments_binding > arguments.Size() ||
	        arguments_size > arguments.Size() - arguments_binding ||
	        output_binding > output.Size() || output_size > output.Size() - output_binding);

	// The game's dispatch wrote the arguments earlier in this queue, and an earlier dispatch may
	// still read a slot this ring is reusing.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	// The shader data is uploaded at a storage-aligned offset (NativeUpload). Without an extent the
	// output stands in for the unused binding.
	const bool with_extent = extent.buffer != nullptr;
	EXIT_IF(with_extent && (extent.offset % alignment != 0 ||
	                        (uint64_t {extent_dword} + 3u) * sizeof(uint32_t) > extent.range));
	const vk::DescriptorBufferInfo infos[] {
	    {arguments.Handle(), arguments_binding, arguments_size},
	    {output.Handle(), output_binding, output_size},
	    with_extent ? extent : vk::DescriptorBufferInfo {output.Handle(), output_binding, output_size},
	    {m_report->Handle(), 0, m_report->Size()},
	};
	std::array<vk::WriteDescriptorSet, 4> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const ShaderParams shader_params {
	    .arguments_word =
	        static_cast<uint32_t>((arguments_offset - arguments_binding) / sizeof(uint32_t)),
	    .output_word  = static_cast<uint32_t>((output_offset - output_binding) / sizeof(uint32_t)),
	    .group_size_x = std::max(group_size[0], 1u),
	    .group_size_y = std::max(group_size[1], 1u),
	    .group_size_z = std::max(group_size[2], 1u),
	    .extent_word  = with_extent ? extent_dword : UINT32_MAX,
	    .max_groups   = max_groups,
	    .shader_low   = static_cast<uint32_t>(shader_hash),
	};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(shader_params), &shader_params);
	command.dispatch(1, 1, 1);

	vk::MemoryBarrier2 after {};
	after.srcStageMask         = vk::PipelineStageFlagBits2::eComputeShader;
	after.srcAccessMask        = vk::AccessFlagBits2::eShaderWrite;
	after.dstStageMask         = vk::PipelineStageFlagBits2::eDrawIndirect |
	                     vk::PipelineStageFlagBits2::eComputeShader;
	after.dstAccessMask        = vk::AccessFlagBits2::eIndirectCommandRead |
	                      vk::AccessFlagBits2::eShaderStorageRead;
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
}

// The report is read as the GPU left it by now (an earlier dispatch's pass, maybe a frame late):
// a line at the first cut and at every tenfold count.
void DispatchThreadGroups::ReportClipped() {
	if (!m_report->IsCoherent()) {
		m_report->Invalidate(0, 8);
	}
	uint32_t words[2] {};
	std::memcpy(words, m_report->Mapped().data(), sizeof(words));
	if (words[0] == m_reported || (m_reported != 0 && words[0] < m_reported * 10u)) {
		return;
	}
	m_reported = words[0];
	std::printf("dispatch-groups: %u indirect dispatches asked more groups than their LDS binding holds "
	            "and were cut down (last: shader ...%08x)\n",
	            words[0], words[1]);
	std::fflush(stdout);
}

} // namespace Libs::Graphics
