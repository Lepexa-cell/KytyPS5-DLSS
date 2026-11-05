#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFORMANCEOVERLAY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFORMANCEOVERLAY_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Called for every guest flip that reaches the screen, with the guest's video-out size.
void PerformanceOverlayRecordFlip(uint32_t guest_width, uint32_t guest_height) noexcept;

// Called for every PS5 Pro upscaling request (renderPssr.cpp): the upscaler that ran (a string
// literal) and the render and output sizes.
void PerformanceOverlayRecordPssr(const char* upscaler, uint32_t render_width, uint32_t render_height,
                                  uint32_t output_width, uint32_t output_height) noexcept;

[[nodiscard]] bool PerformanceOverlayEnabled() noexcept;
void               TogglePerformanceOverlay() noexcept;
// Switches between the compact and the detailed panel, showing the panel if it was hidden.
void TogglePerformanceOverlayDetails() noexcept;

// F9 menu: exact image refresh, the panel, diagnostic captures.
// While it is open the keys go to it (systemOverlay.cpp): Up/Down choose a row, Left/Right change
// it, Enter changes it or runs it, F9 or Esc close the menu.
enum class DebugMenuKey { Up, Down, Left, Right, Enter, Close };
[[nodiscard]] bool DebugMenuOpen() noexcept;
void               ToggleDebugMenu() noexcept;
void               PressDebugMenuKey(DebugMenuKey key) noexcept;

// Changes when the panel needs a repaint without a guest flip (toggled, or the guest stalled).
[[nodiscard]] uint64_t PerformanceOverlayRevision() noexcept;

// Adds the panel's fonts (Segoe UI from the Windows font folder, when present) to the current
// ImGui context's font atlas after its default font. Once, right after creating the context.
void PerformanceOverlayLoadFonts();

// Must run between ImGui::NewFrame() and ImGui::Render(), on the present thread while the
// graphics subsystem is alive. Draws in the top-left corner and returns the panel's bottom
// edge, or 0 when hidden.
float DrawPerformanceOverlay(const GraphicContext& graphics, vk::Extent2D extent);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFORMANCEOVERLAY_H_
