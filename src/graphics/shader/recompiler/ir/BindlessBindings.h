#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_

#include <array>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Descriptor set 2 of a pipeline that samples bindless images (the host side is BindlessTable).
// Sets 0 and 1 are the pipeline's own descriptors (vertex-side and pixel stages).
inline constexpr uint32_t BindlessDescriptorSet  = 2;
inline constexpr uint32_t BindlessImages2D       = 0;
inline constexpr uint32_t BindlessImages2DArray  = 1;
inline constexpr uint32_t BindlessImagesCube     = 2;
inline constexpr uint32_t BindlessImages3D       = 3;
inline constexpr uint32_t BindlessTranslation    = 4;
inline constexpr uint32_t BindlessFeedback       = 5;
// Samplers mirrored from guest sampler heaps (16-byte S# records); slot 0 is the default sampler.
inline constexpr uint32_t BindlessSamplers       = 6;
// 2D and 2D-array images of an integer format (Ghost of Yotei: R8_UINT terrain maps): a shader
// whose heap holds them reads them through these arrays with an integer result type.
inline constexpr uint32_t BindlessImages2DUint      = 7;
inline constexpr uint32_t BindlessImages2DArrayUint = 8;
inline constexpr uint32_t BindlessImages2DSint      = 9;
inline constexpr uint32_t BindlessImages2DArraySint = 10;
// The image arrays, in placeholder order: the four float arrays, then the four integer arrays.
inline constexpr uint32_t BindlessImageArrayCount   = 8;
inline constexpr std::array<uint32_t, BindlessImageArrayCount> BindlessImageArrayBindings {
    BindlessImages2D,         BindlessImages2DArray,     BindlessImagesCube,
    BindlessImages3D,         BindlessImages2DUint,      BindlessImages2DArrayUint,
    BindlessImages2DSint,     BindlessImages2DArraySint};
// The index of an image array's binding in BindlessImageArrayBindings (UINT32_MAX: none).
constexpr uint32_t BindlessImageArrayIndex(uint32_t binding) {
	if (binding <= BindlessImages3D) {
		return binding;
	}
	if (binding >= BindlessImages2DUint && binding <= BindlessImages2DArraySint) {
		return binding - BindlessImages2DUint + 4u;
	}
	return UINT32_MAX;
}
// A translation entry for a key whose texture is not resident yet; the shader samples slot 0.
inline constexpr uint32_t BindlessPending        = 0xffffffffu;
// The S# a sampler without a usable heap entry gets: wrap on every axis, trilinear, full LOD range.
inline constexpr std::array<uint32_t, 4> BindlessDefaultSampler {
    0x00000000u,                             // wrap on every axis
    0xfffu << 12u,                           // max_lod 255.9
    (1u << 20u) | (1u << 22u) | (2u << 26u), // bilinear mag/min, linear mip
    0x00000000u};

// Research: the tail of the feedback buffer holds loop-watchdog trip reports. Word
// WatchdogReportBase counts reports; report n is WatchdogReportWords words starting at
// WatchdogReportBase + WatchdogReportWords * (n + 1). No heap region reaches this far.
inline constexpr uint32_t WatchdogReportBase     = 0xf0000u;
inline constexpr uint32_t WatchdogReportWords    = 64;
inline constexpr uint32_t WatchdogReportSlots    = 1023;
inline constexpr uint32_t WatchdogReportPhis     = 24; // header phi values per lane half

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
