#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <stdexcept>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// A planned program the emitter cannot express (a translator bug for that shader). The caller
// skips the shader's draws and dispatches instead of ending the session.
class ValidationError: public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};

std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info);

// Set once at device creation: the device enabled VK_KHR_shader_maximal_reconvergence
// (KYTY_MAX_RECONVERGENCE). Guest waves leave scalar loops (V_READFIRSTLANE waterfalls,
// S_FF1/V_READLANE walks) once a mask built from ballots is empty; every lane of the host
// subgroup must have reconverged at the loop for that, which only maximal reconvergence
// guarantees (CarRamRod a828533fd: the driver left lanes behind otherwise).
void SetMaximalReconvergence(bool enabled);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
