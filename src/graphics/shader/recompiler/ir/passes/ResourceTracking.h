#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <stdexcept>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Thrown when a descriptor source has a shape the tracker does not support; the shader is
// then skipped instead of ending the emulator (KYTY_STRICT_RESOURCE_TRACKING=1 restores the exit).
struct ResourceTrackingError: std::runtime_error {
	using std::runtime_error::runtime_error;
};

// Resolves native descriptor sources, plans their scalar reads, and assigns dense resource bindings.
void TrackResources(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_ */
