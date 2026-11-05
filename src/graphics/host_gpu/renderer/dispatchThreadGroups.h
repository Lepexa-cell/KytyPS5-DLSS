#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHTHREADGROUPS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHTHREADGROUPS_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

// Turns a GPU-written guest DISPATCH_INDIRECT argument block in thread dimensions into a
// VkDispatchIndirectCommand of group counts on the GPU: a one-thread compute pass, so the CPU
// never reads the arguments (the title in Pro mode: ~42 such dispatches a frame, each a full GPU drain of
// ~0.4 ms while the GPU thread read the counts, 2026-10-30).
class DispatchThreadGroups {
public:
	static constexpr uint64_t ArgumentsSize = 12;
	static constexpr uint64_t OutputSize    = 12;

	DispatchThreadGroups(GraphicContext& graphics, CommandScheduler& scheduler);
	~DispatchThreadGroups();
	KYTY_CLASS_NO_COPY(DispatchThreadGroups);

	// Records the conversion and the barriers that order it after earlier GPU writes of the
	// arguments and before the indirect dispatch. Binds its own compute pipeline and pushed set:
	// record it before the dispatch's own bindings. Outside a rendering instance. `extent`: the
	// dispatch's uploaded shader data and the DWORD of its dispatch thread extent, which gets the
	// thread counts (no buffer: the program reads no extent). `max_groups`: the groups the
	// dispatch's LDS binding holds (UINT32_MAX: no LDS); more are cut down to fit, and the next
	// Record prints how many dispatches were cut so far (`shader_hash`: whose, in that line).
	void Record(vk::CommandBuffer command, const Buffer& arguments, uint64_t arguments_offset,
	            const Buffer& output, uint64_t output_offset, const std::array<uint32_t, 3>& group_size,
	            const vk::DescriptorBufferInfo& extent, uint32_t extent_dword, uint32_t max_groups,
	            uint64_t shader_hash);

private:
	void ReportClipped();

	GraphicContext&         m_graphics;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipeline          = nullptr;
	std::unique_ptr<Buffer> m_report; // dispatches cut down, last shader (dispatch_thread_groups.comp)
	uint32_t                m_reported = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHTHREADGROUPS_H_
