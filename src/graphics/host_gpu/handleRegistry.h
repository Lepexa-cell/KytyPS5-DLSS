#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_HANDLE_REGISTRY_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_HANDLE_REGISTRY_H_

#include <cstdint>

// The VkBuffer and VkImage handles the emulator created, with a generation per creation, so a
// Vulkan call recorded on the GPU thread and replayed later on the record thread
// (commandHooks.cpp) can tell whether a handle it carries was destroyed or recreated in between:
// the driver would copy into freed or reused memory; RenderDoc's layer crashes on it.
namespace Libs::Graphics::HandleRegistry {

enum class Kind : uint8_t { Buffer, Image, ImageView };

// `origin` describes the object (kind of buffer, guest address, size; extent and format), kept
// for the stale report.
void Register(uint64_t handle, Kind kind, const char* origin = nullptr);
// Remembers the destroying call stack (module offsets) for the stale report.
void Unregister(uint64_t handle);

// The generation the handle was registered under: 0 for a handle never registered (not checked
// later), otherwise the stamp a recorded call keeps.
[[nodiscard]] uint64_t Stamp(uint64_t handle);
// False (and a logged "deferred-call: stale ..." line) when the handle has been destroyed or
// recreated since `stamp`; true for an untracked handle (stamp 0) or an unchanged one.
[[nodiscard]] bool Same(uint64_t handle, uint64_t stamp, const char* call, const char* role);
// KYTY_DEFERRED_HANDLE_CHECK=1/0; the default is on with --rd or KYTY_DEBUG_DEVICE_FAULT.
[[nodiscard]] bool CheckEnabled();
// How many stale handles the checks found so far (printed with the device-fault report).
[[nodiscard]] uint64_t StaleCount();
// Whether the handle was registered and has been destroyed since (a command recorded with it now
// would use freed memory).
[[nodiscard]] bool IsDead(uint64_t handle);
// Prints what the registry knows of the handle (kind, origin, alive or destroyed, destroying stack).
void Describe(uint64_t handle, const char* prefix);

} // namespace Libs::Graphics::HandleRegistry

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_HANDLE_REGISTRY_H_ */
