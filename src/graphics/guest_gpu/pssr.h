#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSSR_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSSR_H_

#include <cstdint>

namespace Libs::Graphics::Pssr {

// PS5 Pro upscaling (PSSR) requests, routed from libScePsml to the GPU command processor (after
// upstream PR 1553, 339ff29db, which handles PSSR1 for Astro Bot).
//
// On a PS5 Pro the game reserves space in its command buffer and calls the Psml dispatch
// function, which writes the upscaler's GPU work there. Kyty instead fills that space with one
// IT_NOP/R_PSSR packet carrying a copy of the request, so the request runs in GPU order.
// PSSR2 (Psml_mfsr2) request of the title (2026-10-29, read from the dispatch lrJwpLjKXRc; run pro32,
// KYTY_PSSR_DUMP):
//   +0x00..+0x27  pointers to image descriptors (T#), 5 slots: 0 output (3840x2160 11_11_10 float),
//                 1 colour (render size, 11_11_10 float, before post-processing), 2 motion vectors
//                 (R16G16 float), 3 depth (R32 float), 4 a 1x1 R32 float (exposure)
//   +0x34         float 8.1e-4 (constant)
//   +0x3c, +0x40  render width, height (this frame: dynamic resolution)
//   +0x44, +0x48  jitter in render pixels (Halton, -0.5..0.5)
//   +0x4c, +0x50  motion vector scale to render pixels (-width/2, height/2)
//   +0x5c, +0x60  near 0.1, far 1000
//   +0x64         1 = reset (first frame, a cut), +0x68 1
// The renderer still identifies the slots by size and format (RenderExecutor::ProcessPssr).

constexpr uint32_t Magic        = 0x52535350u; // "PSSR"
constexpr uint32_t Version      = 2;
constexpr uint32_t ParamsDwords = 0x50; // 320 bytes copied from the request
constexpr uint32_t TextureSlots = 7;

struct Request {
	uint32_t magic;
	uint32_t version;
	uint64_t params_address;            // guest address of the request (for logs)
	uint32_t params[ParamsDwords];      // the request itself
	uint32_t textures[TextureSlots][8]; // T# behind each pointer slot, copied at dispatch
	uint32_t texture_mask;              // bit i: slot i had a descriptor
};

constexpr uint32_t RequestDwords = sizeof(Request) / sizeof(uint32_t);
static_assert(sizeof(Request) % sizeof(uint32_t) == 0);

} // namespace Libs::Graphics::Pssr

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_PSSR_H_
