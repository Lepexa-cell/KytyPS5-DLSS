#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// `table`: lay out a program that qualifies (TableEligible) in table mode (BindingLayout::table).
void AllocateBindings(Program& program, uint32_t push_data_start_dword = 0,
                      bool lds_storage = false, bool table = false);

// A vertex or pixel shader that only reads its buffers, without the BDA page table or GDS, and
// has buffers or a flattened SRT to read.
bool TableEligible(const Program& program);

struct SharedMemoryResources {
	bool lds = false;
	bool gds = false;
};

// The shader-data DWORDs before the memory offsets (user data registers, dispatch thread
// counts) the layout of this translated program will have at most; resource specialization can
// only drop user data reads.
uint32_t ShaderDataBaseDwords(const Program& program);

SharedMemoryResources CollectMemoryResources(const Program& program, std::vector<uint32_t>& buffers);
bool UsesFlattenedSrt(const Program& program);

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_ */
