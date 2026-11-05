#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	bool                        bindless_images = false;
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        early_dump                 = false;
	const char*                 dump_label                 = nullptr;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	ShaderStageInputInfo        input_info;
};

struct TranslateResult {
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
	// The shader could not be translated (an unsupported encoding or resource); its draws and
	// dispatches are skipped.
	bool        skip_dispatch = false;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

// Plain word loads and stores of storage buffers leave their range check to the device: with
// robustBufferAccess2, a load outside the bound range reads zero and a store there is dropped, as
// the shader's comparison with the range would do. Set before any compile, only when the device
// enables robustBufferAccess2 and nullDescriptor, checks storage buffer ranges in units of at most
// four bytes, and the renderer rounds storage buffer ranges down to whole dwords.
void SetHardwareStorageBufferBounds(bool enabled);
[[nodiscard]] bool HardwareStorageBufferBounds();

// KYTY_TABLE_MODE=1 (research, off by default): vertex and pixel shaders that only read their
// buffers read them and the flattened SRT through a per-draw block (IR::BindingLayout::table)
// instead of storage buffer descriptors. Set before any compile.
void SetTableMode(bool enabled);
[[nodiscard]] bool TableMode();

// Packed FP16 arithmetic runs as native 16-bit float vectors (shaderFloat16 with FP16 denormals
// and signed zero/Inf/NaN preserved) instead of through F32. Set before any compile, only when the
// device enables those; KYTY_NATIVE_FP16=0 turns it off.
void SetNativeFloat16(bool enabled);
[[nodiscard]] bool NativeFloat16();

// A compute shader whose FLOAT_MODE flushes 32-bit denormals in and out (FP_DENORM_32 = 0, as the
// default 0xc0) declares DenormFlushToZero for 32-bit floats instead of flushing each float input
// explicitly (usa-ratt's GT7 fork, round 98). Set before any compile, only when the device can
// flush 32-bit denormals apart from the other widths; KYTY_DENORM_FLUSH=0 turns it off.
void SetDenormFlushF32(bool enabled);
[[nodiscard]] bool DenormFlushF32();

// A float bound for a 16-bit float format (half-float render targets, written storage images of
// 16-bit float formats) is clamped to the half range first, as the guest GPU's conversion (D3D's,
// rounding toward zero) stores the largest half where the host's gives infinity (usa-ratt's GT7
// fork, cfa8a479e). KYTY_FLOAT_STORE_CLAMP=0 turns it off.
[[nodiscard]] bool FloatStoreClampEnabled();

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
