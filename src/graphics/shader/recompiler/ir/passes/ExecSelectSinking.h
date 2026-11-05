#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ExecSelectSinkingStats {
	uint32_t selects = 0; // selects examined
	uint32_t sunk    = 0; // selects replaced by their chosen value
};

// Every guest VGPR write is Select(exec, value, old) so that inactive lanes keep their register.
// Within a stretch where EXEC stays the same SSA value e, a chain of writes
//   v1 = Select(e, a, old); v2 = Select(e, g(v1), v1)
// needs the select only where the value leaves the stretch: v1 can be `a`, provided
//   - v1's FALSE-operand uses in selects on the same e take `old` instead, and
//   - every other use of v1, and transitively of what is computed from it, either is the TRUE
//     operand of a select on e, or a memory operation whose EXEC operand is e, or a lane-local
//     instruction (no phi, no cross-lane read, no implicit-derivative image op) whose own uses
//     qualify the same way.
// Active lanes see the same values as before; inactive lanes compute values that every consumer
// either discards (select on e, EXEC-masked memory op) or that are discarded further on, and the
// last select of the chain restores their old register. Run after SSA construction and read-lane
// elimination; leaves replaced selects as identities for RemoveIdentities.
ExecSelectSinkingStats SinkExecSelects(const BlockList& blocks);

} // namespace Libs::Graphics::ShaderRecompiler::IR
