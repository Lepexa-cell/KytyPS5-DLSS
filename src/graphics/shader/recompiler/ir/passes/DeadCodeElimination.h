#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Program;

void RemoveIdentities(const BlockList& blocks);
// `roots`: values the program refers to outside the IR (resource plan), kept with their inputs.
void EliminateDeadCode(const BlockList& blocks, std::span<const Value> roots = {});
// The values the resource plan of `program` refers to directly (descriptor dwords, indirect
// descriptor selectors, scalar reads): roots for EliminateDeadCode while the plan is still
// extracted from the IR. Off by default: KYTY_DCE_PLAN_ROOTS=1 returns them.
std::vector<Value> PlanRoots(const Program& program);
// Replaces selects of a lane's new or old value by the new value where the old one is never
// needed: every consumer of the result is masked by the select's condition again.
// Returns how many it replaced.
uint32_t EliminateMaskedSelects(const BlockList& blocks);

} // namespace Libs::Graphics::ShaderRecompiler::IR
