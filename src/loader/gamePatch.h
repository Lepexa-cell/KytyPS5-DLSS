#ifndef KYTY_LOADER_GAME_PATCH_H_
#define KYTY_LOADER_GAME_PATCH_H_

#include "common/common.h"

#include <filesystem>
#include <vector>

namespace Loader {

struct Program;

namespace GamePatch {

bool Apply(const std::filesystem::path& plan_path, Program* main_program,
           const std::vector<Program*>& programs);
bool ApplyPending(Program* program);
// Patches the emulator applies by itself for a known game and version (after a cheat file, which
// wins where both touch the same code).
void ApplyBuiltIn(Program* main_program);
void Clear();
// The GPU timestamp scale an enabled mod of the cheat file asks for (its optional
// "gpu_timestamp_scale" field, 25-1000), or 0: e.g. a fixed-resolution patch that needs the game's
// dynamic resolution on its top step. Read before the game starts.
[[nodiscard]] uint32_t TimestampScale(const std::filesystem::path& plan_path);

} // namespace GamePatch

} // namespace Loader

#endif // KYTY_LOADER_GAME_PATCH_H_
