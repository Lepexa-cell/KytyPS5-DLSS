#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include "common/assert.h"
#include "common/switches.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <array>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

[[noreturn]] void BindingFail(const char* message) {
	EXIT("shader binding layout failed: %s", message);
	std::abort();
}

void CollectShaderData(const Program& program, BindingLayout& layout) {
	std::array<bool, NumScalarRegs> registers {};
	bool uses_dispatch_threads = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (!inst.HasUses()) {
				continue;
			}
			uses_dispatch_threads |= inst.GetOpcode() == ValueOpcode::GetDispatchThreadExtent;
			if (inst.GetOpcode() != ValueOpcode::GetUserData) {
				continue;
			}
			if (inst.Arg(0).GetType() != Type::ScalarReg) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			const auto index = RegIndex(inst.Arg(0).ScalarRegister());
			if (index >= NumScalarRegs) {
				BindingFail("typed shader contains an invalid user-data register");
			}
			registers[index] = true;
		}
	}
	for (uint32_t index = 0; index < registers.size(); index++) {
		if (registers[index]) {
			layout.user_data_registers.push_back(index);
		}
	}
	layout.memory_offset_dword = static_cast<uint32_t>(layout.user_data_registers.size());
	if (uses_dispatch_threads) {
		layout.dispatch_thread_dword = layout.memory_offset_dword;
		layout.memory_offset_dword += 3u;
	}
}

void AddBinding(BindingLayout& layout, DescriptorBindingKind kind,
                std::vector<uint32_t> resources = {}) {
	layout.descriptors.push_back({kind, std::move(resources)});
}

} // namespace

uint32_t ShaderDataBaseDwords(const Program& program) {
	BindingLayout layout;
	CollectShaderData(program, layout);
	return layout.memory_offset_dword;
}

SharedMemoryResources CollectMemoryResources(const Program& program, std::vector<uint32_t>& buffers) {
	std::array<bool, ShaderInfo::MaxBuffers> live_buffers {};
	SharedMemoryResources shared;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (BufferAccessOf(op) == BufferAccess::None &&
			    SharedAccessOf(op) == SharedAccess::None &&
			    AddressOpcodeInfoOf(op).access == AddressAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size()) {
				BindingFail("typed shader contains invalid memory metadata");
			}
			const auto& memory = program.memory_info[index];
			if (memory.planning_only) {
				continue;
			}
			shared.lds |= memory.kind == ResourceKind::FlatLocal;
			if (SharedAccessOf(op) != SharedAccess::None) {
				if (memory.kind != ResourceKind::Lds && memory.kind != ResourceKind::Gds) {
					BindingFail("typed shader contains invalid shared-memory metadata");
				}
				shared.gds |= memory.kind == ResourceKind::Gds;
				shared.lds |= memory.kind == ResourceKind::Lds;
			} else if (memory.kind == ResourceKind::Buffer || memory.kind == ResourceKind::ScalarBuffer) {
				EXIT_IF(memory.resource >= program.info.buffers.size());
				live_buffers.at(memory.resource) = true;
				for (const auto child: program.info.buffers[memory.resource].indirect_resources) {
					live_buffers.at(child) = true;
				}
			}
		}
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (live_buffers.at(i)) {
			buffers.push_back(i);
		}
	}
	return shared;
}

bool UsesFlattenedSrt(const Program& program) {
	const auto uses_mapping = [](const auto& resource) {
		return resource.indirect_search_iterations != 0u;
	};
	return std::ranges::any_of(program.blocks, [](const Block* block) {
		return std::ranges::any_of(*block, [](const Inst& inst) {
			return inst.GetOpcode() == ValueOpcode::ReadConst;
		});
	}) || std::ranges::any_of(program.info.buffers, uses_mapping) ||
	       std::ranges::any_of(program.info.images, [&](const ImageResource& image) {
		       // A bindless image reads its heap region from the flattened SRT.
		       return uses_mapping(image) || image.bindless;
	       }) ||
	       std::ranges::any_of(program.info.samplers, &SamplerResource::bindless);
}

bool GpuDirtyTrackEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_GPU_DIRTY_TRACK");
		return text != nullptr && text[0] == '1';
	}();
	return enabled;
}

bool TableEligible(const Program& program) {
	if ((program.stage != ShaderType::Vertex && program.stage != ShaderType::Pixel) ||
	    program.info.uses_dma) {
		return false;
	}
	if (std::ranges::any_of(program.info.buffers, [](const BufferResource& buffer) {
		    return buffer.written || buffer.atomic;
	    })) {
		return false;
	}
	std::vector<uint32_t> live_buffers;
	if (CollectMemoryResources(program, live_buffers).gds) {
		return false;
	}
	return !program.info.buffers.empty() || UsesFlattenedSrt(program);
}

void AllocateBindings(Program& program, uint32_t push_data_start_dword, bool lds_storage,
                      bool table) {
	if (!program.shader_info_complete || program.binding_layout_complete) {
		EXIT("shader binding layout failed: %s", !program.shader_info_complete
		                                             ? "shader info is not ready"
		                                             : "binding layout already allocated");
	}
	BindingLayout next;
	std::vector<uint32_t> live_buffers;
	const auto shared = CollectMemoryResources(program, live_buffers);
	CollectShaderData(program, next);
	next.table = table && TableEligible(program);
	// Every buffer stays bound, at its own index: the renderer prepares and offsets buffers by
	// their index in program.info.buffers. A table program reads its offsets from the block.
	next.memory_offset_count =
	    next.table ? 0u : static_cast<uint32_t>(program.info.buffers.size());
	next.table_srt_dword =
	    next.table ? static_cast<uint32_t>(program.info.buffers.size()) * TableBufferEntryDwords +
	                     TableSrtHeaderDwords
	               : 0u;
	// Strides and formats read at run time, only for the buffers that need them: a generic
	// format's word holds its stride too.
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (program.info.buffers[i].descriptor_format == GenericBufferFormat) {
			next.buffer_formats.push_back(i);
		} else if ((program.info.buffers[i].packed_stride & DynamicBufferStride) != 0u) {
			next.buffer_strides.push_back(i);
		}
	}
	// Stores through GPU-selected descriptors pass no bound-buffer store site: such a program's
	// writes would not all be marked.
	next.track = GpuDirtyTrackEnabled() && program.stage == ShaderType::Compute && !next.table &&
	             !program.has_address_writes &&
	             std::ranges::any_of(program.info.buffers, [](const BufferResource& buffer) {
		             return buffer.written || buffer.atomic;
	             });
	next.reserve_object_id     = program.stage != ShaderType::Compute;
	next.push_data_start_dword = next.PushStartFor(push_data_start_dword);

	if (!program.info.buffers.empty() && !next.table) {
		// Draw binding accesses this first group directly when memory_offset_count is nonzero.
		std::vector<uint32_t> resources(program.info.buffers.size());
		for (uint32_t i = 0; i < resources.size(); i++) {
			resources[i] = i;
		}
		AddBinding(next, DescriptorBindingKind::Buffers, std::move(resources));
	}

	std::array<std::vector<uint32_t>, ImageBindingCount> image_groups;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			EXIT("shader binding layout failed: image %u has an invalid binding class", i);
		}
		const auto group = ImageBindingIndex(*kind);
		if (group >= image_groups.size()) {
			EXIT("shader binding layout failed: image %u has an unmapped binding class", i);
		}
		auto&      resources = image_groups[group];
		const auto dynamic   = program.info.images[i].mip_mode == ImageMipMode::Dynamic;
		const auto count     = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			EXIT("shader binding layout failed: image %u has invalid specialized mip count %u", i,
			     program.info.images[i].mip_count);
		}
		resources.insert(resources.end(), count, i);
	}
	for (uint32_t i = 0; i < image_groups.size(); i++) {
		if (!image_groups[i].empty()) {
			AddBinding(next, static_cast<DescriptorBindingKind>(FirstImageBinding + i),
			           std::move(image_groups[i]));
		}
	}

	if (!program.info.samplers.empty()) {
		std::vector<uint32_t> resources(program.info.samplers.size());
		for (uint32_t i = 0; i < resources.size(); i++) {
			resources[i] = i;
		}
		AddBinding(next, DescriptorBindingKind::Samplers, std::move(resources));
	}
	if (shared.gds) {
		AddBinding(next, DescriptorBindingKind::Gds);
	}
	if (shared.lds && lds_storage) {
		AddBinding(next, DescriptorBindingKind::SharedMemory);
	}
	if (program.info.uses_dma) {
		AddBinding(next, DescriptorBindingKind::BdaPagetable);
		AddBinding(next, DescriptorBindingKind::FaultBuffer);
	}
	if (UsesFlattenedSrt(program) && !next.table) {
		AddBinding(next, DescriptorBindingKind::FlattenedSrt);
	}

	if (next.ShaderDataDwords() != 0 && !next.UsesPushData()) {
		AddBinding(next, DescriptorBindingKind::ShaderData);
	}

	program.bindings                = std::move(next);
	program.binding_layout_complete = true;
}

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind) {
	for (const auto& binding: layout.descriptors) {
		if (binding.kind == kind) {
			return &binding;
		}
	}
	return nullptr;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
