#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <compare>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
class Image;
struct GraphicContext;

class BlitHelper final {
public:
	inline static constexpr auto ColorToMsDepthLayout =
	    vk::ImageLayout::eDepthStencilAttachmentOptimal;

	BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler);
	~BlitHelper();
	KYTY_CLASS_NO_COPY(BlitHelper);

	void ReinterpretColorAsMsDepth(Image& source, Image& destination);
	// Draws `source_view` (its top-left `source_size` texels) scaled into the top-left
	// `destination_size` pixels of `destination`, viewed as `destination_format`, with a linear
	// filter. Outside a render pass, into `command`.
	void ScaleColor(vk::CommandBuffer command, Image& source, vk::ImageView source_view,
	                vk::Extent2D source_size, Image& destination, vk::Format destination_format,
	                vk::Extent2D destination_size);

private:
	struct PipelineKey {
		uint32_t   samples = 1;
		vk::Format format  = vk::Format::eUndefined;

		auto operator<=>(const PipelineKey&) const = default;
	};

	struct Pipeline {
		PipelineKey  key;
		vk::Pipeline handle = nullptr;
	};

	[[nodiscard]] vk::Pipeline GetPipeline(PipelineKey key);
	[[nodiscard]] vk::Pipeline GetScalePipeline(vk::Format format);

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::ShaderModule        m_vertex_shader     = nullptr;
	vk::ShaderModule        m_fragment_shader   = nullptr;
	std::vector<Pipeline>   m_pipelines;
	// ScaleColor: combined image sampler + push constants, pipelines by colour format.
	vk::DescriptorSetLayout m_scale_descriptor_layout = nullptr;
	vk::PipelineLayout      m_scale_pipeline_layout   = nullptr;
	vk::ShaderModule        m_scale_shader            = nullptr;
	vk::Sampler             m_scale_sampler           = nullptr;
	std::vector<Pipeline>   m_scale_pipelines;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BLITHELPER_H_
