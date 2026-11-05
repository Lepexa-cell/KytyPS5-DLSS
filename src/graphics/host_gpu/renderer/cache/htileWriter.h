#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HTILEWRITER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HTILEWRITER_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Writes a depth target's HTile (min/max depth per 8x8 tile) into guest memory from a linear copy
// of its depth, so guest shaders that read HTile see what the PS5's DB would have left there (see
// htile_synth.comp and TextureCache::SyncHtile).
class HtileWriter {
public:
	enum class DepthFormat : uint32_t { Float32 = 0, Unorm16 = 1, Unorm24 = 2 };

	struct Surface {
		uint32_t    width        = 0;
		uint32_t    height       = 0;
		uint32_t    layers       = 0;
		uint32_t    row_texels   = 0; // Depth row stride of the linear copy.
		uint32_t    slice_texels = 0; // Depth layer stride of the linear copy.
		uint64_t    layer_bytes  = 0; // HTile bytes per layer.
		DepthFormat format       = DepthFormat::Float32;
	};

	explicit HtileWriter(GraphicContext& graphics);
	~HtileWriter();
	KYTY_CLASS_NO_COPY(HtileWriter);

	[[nodiscard]] bool Available() const noexcept { return m_pipeline != nullptr; }

	// HTile bytes one layer of a width x height surface needs: 32 KiB per 1024x512 pixels.
	[[nodiscard]] static uint64_t LayerBytes(uint32_t width, uint32_t height) noexcept;

	// Records the rebuild of every layer: `depth` holds the linear depth (written by a copy that
	// already made its writes available), `htile` at `htile_offset` the guest HTile. Must be
	// recorded outside a rendering instance.
	void Record(vk::CommandBuffer command, vk::Buffer depth, uint64_t depth_size,
	            const Buffer& htile, uint64_t htile_offset, const Surface& surface);

private:
	GraphicContext&         m_graphics;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipeline          = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HTILEWRITER_H_
