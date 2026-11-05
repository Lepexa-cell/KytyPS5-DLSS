#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_AFTERMATH_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_AFTERMATH_H_

#include <cstdint>
#include <span>

// Aftermath GPU crash dumps (debug builds with KYTY_AFTERMATH):
// KYTY_AFTERMATH=1 enables them before the Vulkan device is created. A lost device then writes
// _crashdumps/kyty-<time>-<n>.nv-gpudmp (open with the vendor's GPU debugger) and the same decoded as
// .json, plus the shader debug information the driver hands over (.nvdbg) and every SPIR-V
// module the emulator created, as _crashdumps/shaders/<aftermath hash>.spv, so the GPU debugger can name
// the faulting shader and instruction.
namespace Libs::Graphics::Aftermath {

// KYTY_AFTERMATH=1 was set (and the build has the library).
[[nodiscard]] bool Requested();
// Crash dumps are on for the devices created after EnableCrashDumps.
[[nodiscard]] bool Enabled();
// Before vkCreateDevice. False when not requested or the library refused.
bool EnableCrashDumps();
// Keeps a copy of the SPIR-V under its Aftermath shader hash for the dump's shader lookup.
void NoteShaderModule(std::span<const uint32_t> spirv);
// Appends "<aftermath hash> <guest shader hash> <stage>" to _crashdumps/shaders/guest.txt, so a
// dump's "Shader hash" names the game shader (e.g. for KYTY_DEBUG_DUMP_SPIRV).
void NoteGuestShader(std::span<const uint32_t> spirv, uint64_t guest_hash, const char* stage);
void Disable();

} // namespace Libs::Graphics::Aftermath

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_AFTERMATH_H_ */
