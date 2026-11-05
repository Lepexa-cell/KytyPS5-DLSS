#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "common/switches.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

#include <fmt/format.h>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

// KYTY_WAVE64_WARPS=1 runs eligible wave64 workgroups as two host subgroups (split waves).
bool SplitWavesEnabled() {
	static const bool enabled = [] {
		const char* value = KYTY_GETENV("KYTY_WAVE64_WARPS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_WAVE64_LOCAL_BRANCHES=0 keeps every branch of a split wave whole-wave (A/B runs).
bool SplitWaveLocalBranchesEnabled() {
	static const bool enabled = [] {
		const char* value = KYTY_GETENV("KYTY_WAVE64_LOCAL_BRANCHES");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// KYTY_WAVE64_LOCAL_WATERFALL=1 lets each half of a split wave read its own waterfall key. Off by
// default: scalar state built from the key would then differ between the halves, and Kyty
// Astro's whole-wave branches take their emitted condition (ConditionRef argument 1) from scalar
// state, not from a vote over both halves.
std::unordered_set<const IR::Inst*> SplitWaveLocal(std::unordered_set<const IR::Inst*> insts) {
	static const bool waterfall = [] {
		const char* value = KYTY_GETENV("KYTY_WAVE64_LOCAL_WATERFALL");
		return value != nullptr && value[0] == '1';
	}();
	if (!waterfall) {
		std::erase_if(insts, [](const IR::Inst* inst) {
			return inst->GetOpcode() == IR::ValueOpcode::ReadFirstLane;
		});
	}
	return insts;
}

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	throw ValidationError(fmt::format("SPIR-V validation failed: hash=0x{:016x} stage={} reason={}",
	                                  program.shader_hash, static_cast<unsigned>(program.stage),
	                                  reason));
}

void ValidateNativeProgram(const IR::Program& program, bool lds_storage) {
	using Kind                                             = IR::DescriptorBindingKind;
	constexpr auto                               KindCount = static_cast<size_t>(Kind::Count);
	std::array<std::vector<uint32_t>, KindCount> expected;
	std::array<bool, KindCount>                  present {};
	const auto                                   Dense = [](size_t size) {
		std::vector<uint32_t> values(size);
		for (uint32_t i = 0; i < values.size(); i++) {
			values[i] = i;
		}
		return values;
	};
	auto Expect = [&](Kind kind, std::vector<uint32_t> resources = {}) {
		const auto index = static_cast<size_t>(kind);
		present[index]   = true;
		expected[index]  = std::move(resources);
	};
	const bool table = program.bindings.table;
	if (!program.info.buffers.empty() && !table) {
		Expect(Kind::Buffers, Dense(program.info.buffers.size()));
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto kind = IR::DescriptorBindingForImage(program.info.images[i]);
		if (!kind.has_value()) {
			Fail(program, "native shader plan has an invalid image class");
		}
		present[static_cast<size_t>(*kind)] = true;
		const auto dynamic = program.info.images[i].mip_mode == IR::ImageMipMode::Dynamic;
		const auto count   = dynamic ? program.info.images[i].mip_count : 1u;
		if (count == 0u || (!dynamic && program.info.images[i].mip_count != 1u)) {
			Fail(program, "native shader plan has an invalid image mip descriptor count");
		}
		expected[static_cast<size_t>(*kind)].insert(expected[static_cast<size_t>(*kind)].end(),
		                                            count, i);
	}
	if (!program.info.samplers.empty()) {
		Expect(Kind::Samplers, Dense(program.info.samplers.size()));
	}
	// Every buffer is bound (checked above); the shared-memory kinds come from the accesses.
	std::vector<uint32_t> live_buffers;
	const auto            shared = IR::CollectMemoryResources(program, live_buffers);
	if (shared.gds) {
		Expect(Kind::Gds);
	}
	if (shared.lds && lds_storage) {
		Expect(Kind::SharedMemory);
	}
	if (program.info.uses_dma) {
		Expect(Kind::BdaPagetable);
		Expect(Kind::FaultBuffer);
	}
	if (IR::UsesFlattenedSrt(program) && !table) {
		Expect(Kind::FlattenedSrt);
	}
	if (program.bindings.ShaderDataDwords() != 0 && !program.bindings.UsesPushData()) {
		Expect(Kind::ShaderData);
	}

	std::array<bool, KindCount> seen {};
	for (const auto& binding: program.bindings.descriptors) {
		const auto kind = static_cast<size_t>(binding.kind);
		if (kind >= KindCount || seen[kind] || !present[kind] ||
		    binding.resources != expected[kind]) {
			Fail(program, "native descriptor groups do not match shader topology");
		}
		seen[kind] = true;
	}
	for (size_t i = 0; i < KindCount; i++) {
		if (present[i] != seen[i]) {
			Fail(program, "native shader plan is missing a required descriptor group");
		}
	}
	const auto has_shader_data_storage = present[static_cast<size_t>(Kind::ShaderData)];
	const auto shader_data_dwords = program.bindings.ShaderDataDwords();
	const auto user_data_dwords = program.bindings.user_data_registers.size();
	const bool has_dispatch_threads = program.bindings.dispatch_thread_dword != IR::PushData::NoStart;
	if ((program.bindings.UsesPushData() &&
	     !IR::PushData::CanFit(program.bindings.push_data_start_dword, shader_data_dwords)) ||
	    (has_dispatch_threads && (program.stage != ShaderType::Compute ||
	                              program.bindings.dispatch_thread_dword != user_data_dwords)) ||
	    program.bindings.memory_offset_dword != user_data_dwords + (has_dispatch_threads ? 3u : 0u) ||
	    program.bindings.memory_offset_count != (table ? 0u : program.info.buffers.size()) ||
	    (table && !IR::TableEligible(program)) ||
	    has_shader_data_storage != (shader_data_dwords != 0 && !program.bindings.UsesPushData()) ||
	    !std::is_sorted(program.bindings.user_data_registers.begin(),
	                    program.bindings.user_data_registers.end()) ||
	    std::adjacent_find(program.bindings.user_data_registers.begin(),
	                       program.bindings.user_data_registers.end()) !=
	        program.bindings.user_data_registers.end()) {
		Fail(program, "native shader-data layout is inconsistent");
	}

	const auto planning_only_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       const auto op = use.user->GetOpcode();
			       if (op != IR::ValueOpcode::LoadAddressU32 &&
			           op != IR::ValueOpcode::ReadConstBuffer) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].planning_only;
		       });
	};
	// A GPU-selected descriptor serves loads and, since raw stores go through BDA too, stores and
	// atomics (MemoryInfo::SupportsIndirectBufferStore/Atomic).
	const auto indirect_buffer_handle = [&](const IR::Inst& handle) {
		return program.info.uses_dma && handle.NumArgs() == 4u && !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::BufferAccessOf(use.user->GetOpcode()) == IR::BufferAccess::None) {
				       return false;
			       }
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::IndirectBuffer;
		       });
	};
	const auto local_flat_handle = [&](const IR::Inst& handle) {
		return !handle.Uses().empty() &&
		       std::ranges::all_of(handle.Uses(), [&](const IR::Use& use) {
			       if (IR::AddressOpcodeInfoOf(use.user->GetOpcode()).access == IR::AddressAccess::None)
				       return false;
			       const auto index = use.user->Flags<IR::MemoryFlags>().index;
			       return index < program.memory_info.size() &&
			              program.memory_info[index].kind == IR::ResourceKind::FlatLocal;
		       });
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto dense = inst.Flags<uint32_t>();
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::GetBufferResource:
					if (planning_only_handle(inst) || indirect_buffer_handle(inst)) {
						break;
					}
					if (dense >= program.info.buffers.size()) {
						Fail(program, "typed buffer handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetAddressResource:
					if (planning_only_handle(inst)) {
						break;
					}
					if (inst.NumArgs() != 2 || (!program.info.uses_dma && !local_flat_handle(inst))) {
						Fail(program, "typed address handle has invalid DMA metadata");
					}
					break;
				case IR::ValueOpcode::GetScratchResource:
					if (inst.NumArgs() != 0 || program.scratch_dwords == 0) {
						Fail(program, "typed scratch handle has invalid shader metadata");
					}
					break;
				case IR::ValueOpcode::GetImageResource:
					if (dense >= program.info.images.size()) {
						Fail(program, "typed image handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::GetSamplerResource:
					if (dense >= program.info.samplers.size()) {
						Fail(program, "typed sampler handle has an invalid dense resource");
					}
					break;
				case IR::ValueOpcode::ReadConst: {
					const auto slot = inst.Arg(1).Resolve();
					if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
					    slot.U32() >= program.srt_reads.size()) {
						Fail(program, "flattened SRT read has an invalid dense slot");
					}
					break;
				}
				default: break;
			}
		}
	}
}

// An upper bound of a U32 value built from constants, the lane id and simple integer arithmetic,
// or nothing. Phi cycles (a loop-carried address) are unbounded.
std::optional<uint64_t> UpperBoundU32(IR::Value value, uint32_t wave_size,
                                      std::vector<const IR::Inst*>& visiting) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == IR::Type::U32 ? std::optional<uint64_t>(value.U32())
		                                        : std::nullopt;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || visiting.size() > 64u ||
	    std::ranges::find(visiting, inst) != visiting.end()) {
		return std::nullopt;
	}
	visiting.push_back(inst);
	const auto arg = [&](size_t index) {
		return UpperBoundU32(inst->Arg(index), wave_size, visiting);
	};
	const auto immediate = [&](size_t index) -> std::optional<uint32_t> {
		const auto operand = inst->Arg(index).Resolve();
		return operand.IsImmediate() && operand.GetType() == IR::Type::U32
		           ? std::optional<uint32_t>(operand.U32())
		           : std::nullopt;
	};
	std::optional<uint64_t> result;
	switch (inst->GetOpcode()) {
		case IR::ValueOpcode::LaneId: result = wave_size - 1u; break;
		case IR::ValueOpcode::IAdd32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = *a + *b;
			}
			break;
		case IR::ValueOpcode::IMul32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = *a * *b;
			}
			break;
		case IR::ValueOpcode::ShiftLeftLogical32:
		case IR::ValueOpcode::ShiftRightLogical32: {
			const auto a = arg(0);
			const auto b = immediate(1);
			if (a && b && *b < 32u) {
				result = inst->GetOpcode() == IR::ValueOpcode::ShiftLeftLogical32 ? *a << *b
				                                                                  : *a >> *b;
			}
			break;
		}
		case IR::ValueOpcode::BitwiseAnd32: {
			const auto a = arg(0), b = arg(1);
			if (a || b) {
				result = std::min(a.value_or(UINT32_MAX), b.value_or(UINT32_MAX));
			}
			break;
		}
		case IR::ValueOpcode::BitwiseOr32:
		case IR::ValueOpcode::BitwiseXor32:
			if (const auto a = arg(0), b = arg(1); a && b) {
				result = std::bit_ceil(std::max(*a, *b) + 1u) - 1u;
			}
			break;
		case IR::ValueOpcode::SelectU32:
			if (const auto a = arg(1), b = arg(2); a && b) {
				result = std::max(*a, *b);
			}
			break;
		case IR::ValueOpcode::Phi: {
			uint64_t bound = 0;
			bool     known = inst->NumArgs() != 0;
			for (size_t index = 0; known && index < inst->NumArgs(); index++) {
				const auto incoming = arg(index);
				known               = incoming.has_value();
				bound               = known ? std::max(bound, *incoming) : bound;
			}
			if (known) {
				result = bound;
			}
			break;
		}
		default: break;
	}
	visiting.pop_back();
	return result && *result <= UINT32_MAX ? result : std::nullopt;
}

} // namespace

Emitter::SpirvRequirements Emitter::AnalyzeProgramRequirements(const IR::Program& program) {
	SpirvRequirements requirements {};
	// A non-compute stage keeps LDS in a function-scope array of its own, which the driver places in
	// local memory: at the 8192-dword default that is 32 KB per invocation, enough to lose the
	// device. Size it from the addresses the shader can form instead.
	uint64_t function_lds_bytes     = 0;
	bool     function_lds_unbounded = false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			requirements.float64 |= inst.GetType() == IR::Type::F64;
			requirements.float16 |= inst.GetOpcode() == IR::ValueOpcode::FPAddF16x2 ||
			                        inst.GetOpcode() == IR::ValueOpcode::FPMulF16x2 ||
			                        inst.GetOpcode() == IR::ValueOpcode::FPFmaF16x2 ||
			                        inst.GetOpcode() == IR::ValueOpcode::FPSaturateF16x2 ||
			                        inst.GetOpcode() == IR::ValueOpcode::FPMinF16x2 ||
			                        inst.GetOpcode() == IR::ValueOpcode::FPMaxF16x2 ||
			                        inst.GetOpcode() == IR::ValueOpcode::FPMinTriF16x2 ||
			                        inst.GetOpcode() == IR::ValueOpcode::FPMaxTriF16x2 ||
			                        inst.GetOpcode() == IR::ValueOpcode::FPMedTriF16x2;
			if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::Atomic &&
			    inst.GetType() == IR::Type::U64) {
				requirements.buffer_int64_atomics = true;
			}
			const auto address_access = IR::AddressOpcodeInfoOf(inst.GetOpcode()).access;
			if (address_access != IR::AddressAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					Fail(program, "address operation has invalid memory metadata");
				}
				const auto& memory = program.memory_info[memory_index];
				const auto  kind   = memory.kind;
				if (kind == IR::ResourceKind::Scratch || kind == IR::ResourceKind::FlatLocal) {
					if (program.scratch_dwords == 0) {
						Fail(program, "scratch operation has no per-thread storage");
					}
					requirements.function_scratch = true;
					if (kind == IR::ResourceKind::FlatLocal && program.stage != ShaderType::Compute &&
					    program.stage != ShaderType::Mesh) {
						requirements.function_lds = true;
					}
				} else if (address_access == IR::AddressAccess::Write) {
					Fail(program, "writable FLAT/GLOBAL addresses require GPU ownership tracking");
				} else {
					// Descriptor stores can alias coherent physical-address loads.
					requirements.coherent_buffers |= memory.coherent;
				}
			}
			if (IR::BufferAccessOf(inst.GetOpcode()) != IR::BufferAccess::None) {
				const auto memory_index = inst.Flags<IR::MemoryFlags>().index;
				if (memory_index >= program.memory_info.size()) {
					Fail(program, "buffer operation has invalid memory metadata");
				}
				const auto& memory = program.memory_info[memory_index];
				if (memory.kind == IR::ResourceKind::IndirectBuffer &&
				    inst.GetOpcode() != IR::ValueOpcode::ReadConstBuffer) {
					requirements.subgroup_local_invocation_id = true;
				}
				if (memory.kind == IR::ResourceKind::Buffer) {
					requirements.coherent_buffers |= memory.coherent;
					if (memory.resource >= program.info.buffers.size()) {
						Fail(program, "buffer operation has invalid resource metadata");
					}
					const auto bits = StorageBufferElementBits(program, memory);
					requirements.buffer_u8 |= bits == 8u;
					requirements.buffer_u16 |= bits == 16u;
					if ((program.info.buffers[memory.resource].packed_stride & (1u << 20u)) != 0u) {
						if (program.stage != ShaderType::Compute) {
							Fail(program, "buffer ADD_TID is only valid for compute shaders");
						}
						requirements.subgroup_local_invocation_id = true;
					}
				}
			}
			const auto shared_access = IR::SharedAccessOf(inst.GetOpcode());
			if (shared_access != IR::SharedAccess::None) {
				const auto index = inst.Flags<IR::MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					Fail(program, "shared operation has invalid memory metadata");
				}
				const auto kind = program.memory_info[index].kind;
				if (kind != IR::ResourceKind::Lds && kind != IR::ResourceKind::Gds) {
					Fail(program, "shared operation has invalid resource kind");
				}
				if (shared_access == IR::SharedAccess::Atomic &&
				    IR::SharedComponentCount(inst.GetOpcode()) == 2u) {
					if (kind != IR::ResourceKind::Lds || program.stage != ShaderType::Compute) {
						Fail(program, "64-bit shared atomics require compute LDS");
					}
					requirements.shared_int64_atomics = true;
				}
				if (program.stage != ShaderType::Compute && program.stage != ShaderType::Mesh &&
				    kind == IR::ResourceKind::Lds) {
					requirements.function_lds = true;
					std::vector<const IR::Inst*> visiting;
					const auto address = UpperBoundU32(inst.Arg(0), program.wave_size, visiting);
					if (address.has_value()) {
						const auto& memory = program.memory_info[index];
						function_lds_bytes = std::max(
						    function_lds_bytes,
						    *address + std::max(memory.offset, memory.secondary_offset) +
						        4u * IR::SharedComponentCount(inst.GetOpcode()));
					} else {
						function_lds_unbounded = true;
					}
				}
				if (shared_access == IR::SharedAccess::Append ||
				    shared_access == IR::SharedAccess::Consume) {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
				}
			}
			switch (inst.GetOpcode()) {
				case IR::ValueOpcode::StoreCompletion: requirements.subgroup_barrier = true; break;
				// Off by default: the module then stays as before (EmitLdsWaitBarrier).
				case IR::ValueOpcode::LdsWaitBarrier:
					requirements.subgroup_barrier |= LdsWaitBarrierEnabled() || LdsAccessBarrierEnabled();
					break;
				case IR::ValueOpcode::BvhIntersect: requirements.bvh = true; break;
				case IR::ValueOpcode::Bvh8Intersect: requirements.bvh8 = true; break;
				case IR::ValueOpcode::BvhStack: requirements.bvh_stack = true; break;
				case IR::ValueOpcode::Ballot: requirements.subgroup_ballot = true; break;
				case IR::ValueOpcode::DppMoveU32:
				case IR::ValueOpcode::ReadFirstLane:
				case IR::ValueOpcode::ReadLane: {
					requirements.subgroup_ballot  = true;
					requirements.subgroup_shuffle = true;
					if (inst.GetOpcode() == IR::ValueOpcode::DppMoveU32) {
						requirements.subgroup_local_invocation_id = true;
					}
					break;
				}
				case IR::ValueOpcode::DppUpdateU32:
				case IR::ValueOpcode::WriteLane: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::Permlane16U32: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::SwizzleU32:
				case IR::ValueOpcode::PermuteU32:
				case IR::ValueOpcode::BpermuteU32: {
					requirements.subgroup_ballot              = true;
					requirements.subgroup_shuffle             = true;
					requirements.subgroup_local_invocation_id = true;
					break;
				}
				case IR::ValueOpcode::IsHelperInvocation:
					// gl_HelperInvocation is declared with the ballot support of pixel shaders.
					requirements.subgroup_ballot |= program.stage == ShaderType::Pixel;
					break;
				case IR::ValueOpcode::LaneId:
					requirements.subgroup_local_invocation_id |=
					    program.stage != ShaderType::TessellationControl;
					break;
				case IR::ValueOpcode::ImageQueryLod: requirements.compute_derivatives = true; break;
				case IR::ValueOpcode::ImageGatherRaw:
					requirements.image_gather_extended = true;
					break;
				case IR::ValueOpcode::SetAttribute: {
					const auto index = inst.Flags<IR::ExportFlags>().index;
					if (index >= program.export_info.size()) {
						Fail(program, "attribute export has invalid metadata");
					}
					if (program.stage == ShaderType::Pixel &&
					    program.export_info[index].vm) {
						requirements.pixel_valid_mask = true;
					}
					break;
				}
				default: break;
			}
		}
	}
	if (requirements.function_lds && !function_lds_unbounded &&
	    function_lds_bytes < 8192u * sizeof(uint32_t)) {
		requirements.function_lds_dwords =
		    static_cast<uint32_t>((function_lds_bytes + sizeof(uint32_t) - 1u) / sizeof(uint32_t));
	}
	return requirements;
}

std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info) {
	using namespace Emitter;

	if (program.stage != ShaderType::Compute && program.stage != ShaderType::Vertex &&
	    program.stage != ShaderType::Pixel && program.stage != ShaderType::Mesh &&
	    program.stage != ShaderType::Local && program.stage != ShaderType::TessellationControl &&
	    program.stage != ShaderType::TessellationEvaluation) {
		Fail(program, "binary SPIR-V emitter received an unsupported shader stage");
	}
	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete || !program.binding_layout_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	ValidateNativeProgram(program, program.stage == ShaderType::Compute &&
	                                   input_info.compute != nullptr && input_info.compute->lds_storage);
	IR::ValidateProgram(program, true);
	EmitterState state(program, input_info);
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	const bool narrow_host =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u;
	// Split waves need the workgroup to be one whole wave: their barriers synchronize the
	// workgroup, and every lane exists. Explicitly laid out LDS cannot hold a separate exchange.
	state.split_wave =
	    narrow_host && SplitWavesEnabled() && program.stage == ShaderType::Compute &&
	    std::max(workgroup->threads_num[0], 1u) * std::max(workgroup->threads_num[1], 1u) *
	            std::max(workgroup->threads_num[2], 1u) == 64u &&
	    !state.requirements.compute_derivatives && !state.requirements.shared_int64_atomics &&
	    state.lds_storage_class == spv::StorageClassWorkgroup;
	state.lane_count = narrow_host && !state.split_wave ? 2u : 1u;
	// KYTY_DEBUG_WAVE_DECISION=<hex hash>: how this shader's guest wave maps to host invocations.
	static const uint64_t wave_decision = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_WAVE_DECISION");
		return text != nullptr ? std::strtoull(text, nullptr, 16) : uint64_t {0};
	}();
	if (wave_decision != 0 && program.shader_hash == wave_decision) [[unlikely]] {
		std::printf("wave-decision: hash=%016llx stage=%u wave=%u host_subgroup=%u threads=%ux%ux%u "
		            "lane_count=%u split=%d\n",
		            static_cast<unsigned long long>(program.shader_hash),
		            static_cast<unsigned>(program.stage), program.wave_size,
		            workgroup != nullptr ? workgroup->host_subgroup_size : 0u,
		            workgroup != nullptr ? workgroup->threads_num[0] : 0u,
		            workgroup != nullptr ? workgroup->threads_num[1] : 0u,
		            workgroup != nullptr ? workgroup->threads_num[2] : 0u, state.lane_count,
		            state.split_wave ? 1 : 0);
		std::fflush(stdout);
	}
	if (state.split_wave && SplitWaveLocalBranchesEnabled()) {
		// A first emission, with every candidate branch local to a half, finds the blocks that
		// synchronize the halves regardless of which branches stay whole-wave.
		std::unordered_set<const IR::Block*> sync_blocks;
		{
			EmitterState probe(program, input_info);
			probe.split_wave       = true;
			probe.lane_count       = 1;
			// The probe places every barrier the final emission will: waterfall reads follow the
			// same policy in both.
			probe.wave_local_insts = SplitWaveLocal(SplitWaveLocalInstructions(program, {}, true));
			probe.wave_sync_blocks = &sync_blocks;
			DefineModule(probe);
			EmitProgram(probe);
		}
		state.wave_local_insts = SplitWaveLocal(SplitWaveLocalInstructions(program, sync_blocks, false));
	}
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

namespace {
std::atomic<bool> g_maximal_reconvergence {false};
} // namespace

void SetMaximalReconvergence(bool enabled) {
	g_maximal_reconvergence.store(enabled, std::memory_order_relaxed);
}

// KYTY_MAX_RECONVERGENCE=1 marks every shader MaximallyReconvergesKHR; =<hash>[,<hash>...] only
// those (the execution mode changes the SPIR-V, so every marked shader compiles again).
bool Emitter::MaximalReconvergenceFor(uint64_t shader_hash) {
	if (!g_maximal_reconvergence.load(std::memory_order_relaxed)) {
		return false;
	}
	static const Common::Switches::HashList hashes = [] {
		const char* text = KYTY_GETENV("KYTY_MAX_RECONVERGENCE");
		return Common::Switches::HashList(text != nullptr && std::strcmp(text, "1") == 0 ? nullptr
		                                                                                 : text);
	}();
	return hashes.hashes.empty() || hashes.Contains(shader_hash);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
