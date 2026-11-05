#include "graphics/host_gpu/renderer/image/blitHelper.h"

#include "common/assert.h"
#include "gpu_blit_shaders/gpu_blit_color_to_ms_depth_spv.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_blit_scale_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/renderTarget.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

BlitHelper::BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {
	vk::DescriptorSetLayoutBinding texture_binding {};
	texture_binding.binding         = 0;
	texture_binding.descriptorType  = vk::DescriptorType::eSampledImage;
	texture_binding.descriptorCount = 1;
	texture_binding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = 1;
	descriptor_info.pBindings    = &texture_binding;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                                 &m_descriptor_layout),
	                     "create BlitHelper descriptor layout");

	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts    = &m_descriptor_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_pipeline_layout),
	    "create BlitHelper pipeline layout");

	m_vertex_shader   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, m_graphics.device);
	m_fragment_shader = CompileSPV(GPU_BLIT_COLOR_TO_MS_DEPTH_SPV, m_graphics.device);

	vk::DescriptorSetLayoutBinding scale_binding {};
	scale_binding.binding         = 0;
	scale_binding.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	scale_binding.descriptorCount = 1;
	scale_binding.stageFlags      = vk::ShaderStageFlagBits::eFragment;
	vk::DescriptorSetLayoutCreateInfo scale_descriptor_info {};
	scale_descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	scale_descriptor_info.bindingCount = 1;
	scale_descriptor_info.pBindings    = &scale_binding;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(
	                         &scale_descriptor_info, nullptr, &m_scale_descriptor_layout),
	                     "create BlitHelper scale descriptor layout");

	vk::PushConstantRange scale_push {};
	scale_push.stageFlags = vk::ShaderStageFlagBits::eFragment;
	scale_push.size       = 2 * sizeof(float);
	vk::PipelineLayoutCreateInfo scale_layout_info {};
	scale_layout_info.setLayoutCount         = 1;
	scale_layout_info.pSetLayouts            = &m_scale_descriptor_layout;
	scale_layout_info.pushConstantRangeCount = 1;
	scale_layout_info.pPushConstantRanges    = &scale_push;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&scale_layout_info, nullptr,
	                                                            &m_scale_pipeline_layout),
	                     "create BlitHelper scale pipeline layout");

	vk::SamplerCreateInfo sampler_info {};
	sampler_info.magFilter    = vk::Filter::eLinear;
	sampler_info.minFilter    = vk::Filter::eLinear;
	sampler_info.mipmapMode   = vk::SamplerMipmapMode::eNearest;
	sampler_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
	sampler_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
	RequireVulkanSuccess(m_graphics.device.createSampler(&sampler_info, nullptr, &m_scale_sampler),
	                     "create BlitHelper scale sampler");

	m_scale_shader = CompileSPV(GPU_BLIT_SCALE_SPV, m_graphics.device);
}

BlitHelper::~BlitHelper() {
	for (const auto& pipeline: m_pipelines) {
		m_graphics.device.destroyPipeline(pipeline.handle, nullptr);
	}
	for (const auto& pipeline: m_scale_pipelines) {
		m_graphics.device.destroyPipeline(pipeline.handle, nullptr);
	}
	if (m_scale_sampler != nullptr) {
		m_graphics.device.destroySampler(m_scale_sampler, nullptr);
	}
	if (m_scale_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_scale_shader, nullptr);
	}
	if (m_scale_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_scale_pipeline_layout, nullptr);
	}
	if (m_scale_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_scale_descriptor_layout, nullptr);
	}
	if (m_fragment_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_fragment_shader, nullptr);
	}
	if (m_vertex_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_vertex_shader, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

vk::Pipeline BlitHelper::GetPipeline(PipelineKey key) {
	const auto cached = std::ranges::find(m_pipelines, key, &Pipeline::key);
	if (cached != m_pipelines.end()) {
		return cached->handle;
	}

	const auto samples = vulkan_sample_count(key.samples);
	EXIT_IF(samples == vk::SampleCountFlagBits {} || key.format == vk::Format::eUndefined);

	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = m_vertex_shader;
	stages[0].pName  = "main";
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = m_fragment_shader;
	stages[1].pName  = "main";

	vk::PipelineVertexInputStateCreateInfo vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = samples;
	vk::PipelineDepthStencilStateCreateInfo depth {};
	depth.depthTestEnable  = VK_TRUE;
	depth.depthWriteEnable = VK_TRUE;
	depth.depthCompareOp   = vk::CompareOp::eAlways;
	vk::PipelineColorBlendStateCreateInfo color_blend {};
	const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic.pDynamicStates    = dynamic_states.data();

	vk::PipelineRenderingCreateInfo rendering {};
	rendering.depthAttachmentFormat = key.format;

	vk::GraphicsPipelineCreateInfo create {};
	create.pNext               = &rendering;
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &input_assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pDepthStencilState  = &depth;
	create.pColorBlendState    = &color_blend;
	create.pDynamicState       = &dynamic;
	create.layout              = m_pipeline_layout;

	vk::Pipeline pipeline = nullptr;
	RequireVulkanSuccess(
	    m_graphics.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline),
	    "create color-to-MS-depth pipeline");
	m_pipelines.push_back({key, pipeline});
	return pipeline;
}

void BlitHelper::ReinterpretColorAsMsDepth(Image& source, Image& destination) {
	const auto& source_info      = source.info;
	const auto& destination_info = destination.info;
	EXIT_IF(DepthAspectTransferFormat(source_info.pixel_format) != vk::Format::eUndefined ||
	        DepthAspectTransferFormat(destination_info.pixel_format) == vk::Format::eUndefined ||
	        source_info.samples != 1 || destination_info.samples <= 1 ||
	        destination_info.samples > 4 || source.backing.image_type != vk::ImageType::e2D ||
	        destination.backing.image_type != vk::ImageType::e2D ||
	        source_info.extent.width != destination_info.extent.width ||
	        source_info.extent.height != destination_info.extent.height ||
	        source_info.extent.depth != 1 || destination_info.extent.depth != 1 ||
	        source.backing.image == nullptr || destination.backing.image == nullptr);
	m_scheduler.EndRendering();

	ImageViewInfo source_view_info {};
	source_view_info.format = source_info.pixel_format;
	source_view_info.type   = vk::ImageViewType::e2D;
	source_view_info.aspect = vk::ImageAspectFlagBits::eColor;
	source_view_info.usage  = vk::ImageUsageFlagBits::eSampled;
	const auto source_view  = source.FindView(source_view_info);

	ImageViewInfo destination_view_info {};
	destination_view_info.format = destination_info.pixel_format;
	destination_view_info.type   = vk::ImageViewType::e2D;
	destination_view_info.aspect = vk::ImageAspectFlagBits::eDepth;
	destination_view_info.usage  = vk::ImageUsageFlagBits::eDepthStencilAttachment;
	const auto destination_view  = destination.FindView(destination_view_info);

	auto& command_buffer = m_scheduler.Current();
	auto  command        = command_buffer.Handle();
	GpuZones::Mark(command, DrainStats::Zone::Blit);
	source.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {},
	               command);
	destination.Transit(ColorToMsDepthLayout, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {},
	                    command);

	vk::RenderingAttachmentInfo depth_attachment {};
	depth_attachment.imageView               = destination_view;
	depth_attachment.imageLayout             = ColorToMsDepthLayout;
	depth_attachment.loadOp                  = vk::AttachmentLoadOp::eClear;
	depth_attachment.storeOp                 = vk::AttachmentStoreOp::eStore;
	depth_attachment.clearValue.depthStencil = {0.0f, 0};

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent = {destination_info.extent.width, destination_info.extent.height};
	rendering.layerCount        = 1;
	rendering.pDepthAttachment  = &depth_attachment;
	command.beginRendering(&rendering);

	vk::DescriptorImageInfo descriptor_image {};
	descriptor_image.imageView   = source_view;
	descriptor_image.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::WriteDescriptorSet descriptor_write {};
	descriptor_write.dstBinding      = 0;
	descriptor_write.descriptorCount = 1;
	descriptor_write.descriptorType  = vk::DescriptorType::eSampledImage;
	descriptor_write.pImageInfo      = &descriptor_image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_pipeline_layout, 0, 1,
	                             &descriptor_write);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics,
	                     GetPipeline({destination_info.samples, destination_info.pixel_format}));

	const vk::Viewport viewport {0.0f,
	                             0.0f,
	                             static_cast<float>(destination_info.extent.width),
	                             static_cast<float>(destination_info.extent.height),
	                             0.0f,
	                             1.0f};
	const vk::Rect2D   scissor {{0, 0},
	                            {destination_info.extent.width, destination_info.extent.height}};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
	// The blit replaced the pipeline and dynamic state the game's draws recorded.
	command_buffer.InvalidateGraphicsState();
}

vk::Pipeline BlitHelper::GetScalePipeline(vk::Format format) {
	const PipelineKey key {1, format};
	const auto        cached = std::ranges::find(m_scale_pipelines, key, &Pipeline::key);
	if (cached != m_scale_pipelines.end()) {
		return cached->handle;
	}
	EXIT_IF(format == vk::Format::eUndefined);

	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = m_vertex_shader;
	stages[0].pName  = "main";
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = m_scale_shader;
	stages[1].pName  = "main";

	vk::PipelineVertexInputStateCreateInfo vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
	vk::PipelineDepthStencilStateCreateInfo depth {};
	vk::PipelineColorBlendAttachmentState blend_attachment {};
	blend_attachment.colorWriteMask = vk::ColorComponentFlagBits::eR |
	                                  vk::ColorComponentFlagBits::eG |
	                                  vk::ColorComponentFlagBits::eB |
	                                  vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo color_blend {};
	color_blend.attachmentCount = 1;
	color_blend.pAttachments    = &blend_attachment;
	const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic.pDynamicStates    = dynamic_states.data();

	vk::PipelineRenderingCreateInfo rendering {};
	rendering.colorAttachmentCount    = 1;
	rendering.pColorAttachmentFormats = &format;

	vk::GraphicsPipelineCreateInfo create {};
	create.pNext               = &rendering;
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &input_assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pDepthStencilState  = &depth;
	create.pColorBlendState    = &color_blend;
	create.pDynamicState       = &dynamic;
	create.layout              = m_scale_pipeline_layout;

	vk::Pipeline pipeline = nullptr;
	RequireVulkanSuccess(
	    m_graphics.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline),
	    "create colour scale pipeline");
	m_scale_pipelines.push_back({key, pipeline});
	return pipeline;
}

void BlitHelper::ScaleColor(vk::CommandBuffer command, Image& source, vk::ImageView source_view,
                            vk::Extent2D source_size, Image& destination,
                            vk::Format destination_format, vk::Extent2D destination_size) {
	EXIT_IF(source_view == nullptr || destination.backing.image == nullptr ||
	        source.info.extent.width == 0 || source.info.extent.height == 0 ||
	        destination_size.width == 0 || destination_size.height == 0);

	ImageViewInfo destination_view_info {};
	destination_view_info.format = destination_format;
	destination_view_info.type   = vk::ImageViewType::e2D;
	destination_view_info.aspect = vk::ImageAspectFlagBits::eColor;
	destination_view_info.usage  = vk::ImageUsageFlagBits::eColorAttachment;
	const auto destination_view  = destination.FindView(destination_view_info);

	GpuZones::Mark(command, DrainStats::Zone::Blit);
	source.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {},
	               command);
	destination.Transit(vk::ImageLayout::eColorAttachmentOptimal,
	                    vk::AccessFlagBits2::eColorAttachmentWrite, {}, command);

	vk::RenderingAttachmentInfo color_attachment {};
	color_attachment.imageView   = destination_view;
	color_attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	color_attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	color_attachment.storeOp     = vk::AttachmentStoreOp::eStore;

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent   = destination_size;
	rendering.layerCount          = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments   = &color_attachment;
	command.beginRendering(&rendering);

	vk::DescriptorImageInfo descriptor_image {};
	descriptor_image.sampler     = m_scale_sampler;
	descriptor_image.imageView   = source_view;
	descriptor_image.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::WriteDescriptorSet descriptor_write {};
	descriptor_write.dstBinding      = 0;
	descriptor_write.descriptorCount = 1;
	descriptor_write.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
	descriptor_write.pImageInfo      = &descriptor_image;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, m_scale_pipeline_layout, 0, 1,
	                             &descriptor_write);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, GetScalePipeline(destination_format));
	// Target pixel centre p maps to source texel position p * source / destination, as uv.
	const std::array<float, 2> uv_scale {
	    static_cast<float>(source_size.width) /
	        (static_cast<float>(destination_size.width) * static_cast<float>(source.info.extent.width)),
	    static_cast<float>(source_size.height) / (static_cast<float>(destination_size.height) *
	                                              static_cast<float>(source.info.extent.height))};
	command.pushConstants(m_scale_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0,
	                      sizeof(uv_scale), uv_scale.data());

	const vk::Viewport viewport {0.0f,
	                             0.0f,
	                             static_cast<float>(destination_size.width),
	                             static_cast<float>(destination_size.height),
	                             0.0f,
	                             1.0f};
	const vk::Rect2D   scissor {{0, 0}, destination_size};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
}

} // namespace Libs::Graphics
