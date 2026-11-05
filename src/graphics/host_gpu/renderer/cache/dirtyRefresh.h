#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DIRTYREFRESH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DIRTYREFRESH_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
struct GraphicContext;

// KYTY_GPU_DIRTY_TRACK: one bit per 4-byte word of a tracked image's memory, set on the GPU by the
// recompiled buffer stores of compute shaders (see TrackDword in BindingLayout), and the pass that
// copies the marked words from the tiled guest bytes into the image (dirty_refresh.comp). The bitmaps
// live in one arena; a slot is a run of 32-bit words.
// The F9 menu's switch for KYTY_GPU_DIRTY_TRACK (as KYTY_GPU_DIRTY_TRACK_OFF_FILE: writes mark images
// whole again while paused); available only with the switch on.
[[nodiscard]] bool DirtyTrackAvailable() noexcept;
[[nodiscard]] bool DirtyTrackPaused() noexcept;
void               SetDirtyTrackPaused(bool paused) noexcept;

class DirtyRefresh {
public:
	static constexpr uint32_t NoSlot = UINT32_MAX;

	struct Target {
		vk::ImageView view            = nullptr; // R32Uint or R32G32Uint storage view, layout General.
		uint32_t      texel_words     = 0;       // 1 or 2.
		uint32_t      width           = 0;
		uint32_t      height          = 0;
		uint32_t      blocks_per_row  = 0;
	};

	DirtyRefresh(GraphicContext& graphics, CommandScheduler& scheduler);
	~DirtyRefresh();
	KYTY_CLASS_NO_COPY(DirtyRefresh);

	[[nodiscard]] bool Available() const noexcept { return m_pipelines[0] != nullptr; }

	// Bitmap words for `bytes` of image memory.
	[[nodiscard]] static uint32_t SlotWords(uint64_t bytes) noexcept {
		return static_cast<uint32_t>((bytes + 127u) / 128u);
	}
	// NoSlot when the arena is full. A new slot's words are undefined: clear them before use.
	[[nodiscard]] uint32_t Allocate(uint32_t words);
	void                   Free(uint32_t slot, uint32_t words);
	[[nodiscard]] vk::DeviceAddress SlotAddress(uint32_t slot) const noexcept {
		return m_arena.BufferDeviceAddress() + uint64_t {slot} * sizeof(uint32_t);
	}

	// Zeroes a slot: the guest's earlier marks are made available first, later marks wait for it.
	void RecordClear(vk::CommandBuffer command, uint32_t slot, uint32_t words);
	// Copies the marked words of the slot from `source` (the image's guest bytes at
	// `source_offset`, `size` of them) into the target's texels, then zeroes the slot.
	void RecordRefresh(vk::CommandBuffer command, vk::Buffer source, uint64_t source_offset,
	                   uint64_t size, uint32_t slot, uint32_t words, const Target& target);

private:
	struct FreeRun {
		uint32_t slot;
		uint32_t words;
	};

	GraphicContext&         m_graphics;
	Buffer                  m_arena;
	std::vector<FreeRun>    m_free;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipelines[2] {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DIRTYREFRESH_H_
