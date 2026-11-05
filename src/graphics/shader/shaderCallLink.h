#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCALLLINK_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCALLLINK_H_

#include "graphics/shader/shaderCompiler.h"

#include <cstdint>
#include <span>
#include <vector>

// KYTY_PRO_SWAPPC_LINK=1 (the title in Pro mode, 2026-11-03): real S_SWAPPC_B64 calls whose target comes
// from a table the shader loads from user data (d0f76103's shadow hit functions) are linked: the
// call jumps to a stub after the code that compares the target with every function the tables
// have named so far and runs a copy of it; an unknown target returns at once, as
// KYTY_PRO_SWAPPC_SKIP does. Needs KYTY_PRO_SWAPPC_SKIP=1 for the calls it cannot follow
// (d0f76103's pc 0x3e8), which the decoder rejects otherwise.
namespace Libs::Graphics::ShaderCallLink {

struct Site {
	uint32_t call_word  = 0; // word index of the S_SWAPPC_B64
	uint32_t target_reg = 0; // SGPR pair holding the target
	uint32_t return_reg = 0; // SGPR pair receiving the return address
	uint32_t table_sgpr = 0; // user-data SGPR pair holding the pointer to the table descriptor
	int32_t  table_offset = 0;
};

struct Function {
	uint64_t              target = 0;
	std::vector<uint32_t> body; // without its return
};

// Call sites of the program in `code` the linker can follow; `code_words` is where decoding ends.
std::vector<Site> FindSites(std::span<const uint32_t> code, uint32_t user_data_count,
                            uint32_t& code_words);
// The body of a leaf function at `code` returning through `return_reg`, or false when it is not
// a straight leaf (a branch, a nested call, a write to the return pair, too long).
bool ReadLeaf(std::span<const uint32_t> code, uint32_t return_reg, std::vector<uint32_t>& body);
// The program with each site's call replaced by a jump into its stub; `functions[i]` belongs to
// `sites[i]`.
std::vector<uint32_t> Build(std::span<const uint32_t> code, uint32_t code_words,
                            std::span<const Site> sites,
                            std::span<const std::vector<Function>> functions);

// Points `linked` at the linked program of `params` when the shader has linkable calls and the
// switch is on; the linked code lives as long as the process.
bool Link(const ShaderParams& params, ShaderParams& linked);

} // namespace Libs::Graphics::ShaderCallLink

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCALLLINK_H_ */
