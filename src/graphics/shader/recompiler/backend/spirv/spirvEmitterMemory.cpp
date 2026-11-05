#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/switches.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t AndCondition(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, spv::OpLogicalAnd, TypeBool(state), lhs, rhs);
}

bool IsStorageBufferKind(IR::ResourceKind kind) {
	return kind == IR::ResourceKind::Buffer || kind == IR::ResourceKind::ScalarBuffer ||
	       kind == IR::ResourceKind::Gds;
}

bool IsStorageBufferAccess(const MemoryResourceAccess& resource) {
	return IsStorageBufferKind(resource.kind);
}

// Loads an element that may be out of bounds; out-of-bounds lanes get zero. A storage-buffer load
// is issued at its own index without a branch and its value is replaced by zero: the device
// always enables robustBufferAccess, which keeps an out-of-bounds load harmless. Leaving the
// address alone keeps constant offsets foldable into the load (clamping the index instead made
// Astro Bot's hottest pixel shaders up to 18% slower). AMD's compiler spends far longer on a
// branch around every load than on a select: pipeline creation for Astro Bot's largest compute
// shader is about 20% faster. LDS and scratch have no such guarantee and keep the branch.
template <typename Fn>
uint32_t LoadElementOrZero(EmitterState& state, const MemoryResourceAccess& resource,
                           uint32_t in_bounds, uint32_t index, Fn&& load) {
	if (!IsStorageBufferAccess(resource)) {
		return EmitValueOrZeroIfCondition(state, in_bounds, [&]() { return load(index); });
	}
	if (in_bounds == ConstantBool(state, true)) {
		return load(index);
	}
	return Select(state, TypeU32(state), in_bounds, load(index), ConstantU32(state, 0));
}

uint32_t EmitDsMaskedLaneRead(EmitterState& state, uint32_t source, uint32_t target,
                              uint32_t exec) {
	// These reads stay within 32-lane halves of the wave: a split wave's host subgroups.
	if (state.lane_count == 2 || state.split_wave) {
		target = Binary(state, spv::OpBitwiseAnd, TypeU32(state), target, ConstantU32(state, 31));
	}
	const auto shuffled = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), shuffled,
	                          ConstantU32(state, spv::ScopeSubgroup), source, target);
	const auto source_exec = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeBool(state), source_exec,
	                          ConstantU32(state, spv::ScopeSubgroup), exec, target);
	const auto source_active =
	    AndCondition(state, source_exec, EmitSubgroupLaneActiveBool(state, target));
	return Select(state, TypeU32(state), source_active, shuffled, ConstantU32(state, 0));
}

struct BufferAddress {
	uint32_t offset;
	uint32_t byte;
};

BufferAddress CalculateBufferAddress(EmitterState& state, uint32_t index, uint32_t offset,
                                     uint32_t soffset, uint32_t immediate, uint32_t stride,
                                     uint32_t swizzle, uint32_t index_stride) {
	const auto zero = ConstantU32(state, 0);
	const auto one  = ConstantU32(state, 1);
	const auto add = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero ? rhs : rhs == zero ? lhs
		                                      : Binary(state, spv::OpIAdd, TypeU32(state), lhs, rhs);
	};
	const auto mul = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero || rhs == zero ? zero
		       : lhs == one              ? rhs
		       : rhs == one              ? lhs
		                                 : Binary(state, spv::OpIMul, TypeU32(state), lhs, rhs);
	};
	if (immediate != 0u) {
		offset = add(offset, ConstantU32(state, immediate));
	}
	auto address = add(mul(index, stride), offset);
	if (swizzle != 0u) {
		const auto index_shift =
		    Binary(state, spv::OpIAdd, TypeU32(state), index_stride, ConstantU32(state, 3));
		const auto indices = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                            ConstantU32(state, 1), index_shift);
		const auto index_msb =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), index, index_shift);
		const auto index_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), index,
		           Binary(state, spv::OpISub, TypeU32(state), indices, ConstantU32(state, 1)));
		const auto offset_msb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u));
		const auto offset_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, 3u));
		const auto msb = mul(add(mul(index_msb, stride), offset_msb), indices);
		const auto lsb = add(Binary(state, spv::OpShiftLeftLogical, TypeU32(state), index_lsb,
		                            ConstantU32(state, 2u)), offset_lsb);
		address = Select(state, TypeU32(state), swizzle, add(msb, lsb), address);
	}
	return {offset, add(address, soffset)};
}

uint32_t BufferLane(EmitterState& state) {
	// ADD_TID indexes within the guest wave, even when a host subgroup holds two wave32s.
	return Binary(state, spv::OpBitwiseAnd, TypeU32(state), EmitSubgroupLocalInvocationId(state),
	              ConstantU32(state, state.program.wave_size == 32u ? 31u : 63u));
}

uint32_t BufferByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto&      state  = ctx.state;
	const auto packed = StorageBufferPackedStride(state, mem);
	const auto stride = packed & 0x3fffu;
	auto       index  = ctx.Arg(inst, 1);
	if ((packed & (1u << 20u)) != 0u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, BufferLane(state));
	}
	// A DynamicBufferStride buffer: this dispatch's stride from shader data, nonzero.
	const bool dynamic = (packed & IR::DynamicBufferStride) != 0u;
	const bool swizzle = (dynamic || stride != 0u) && (packed & (1u << 14u)) != 0u;
	return CalculateBufferAddress(state, index, ctx.Arg(inst, 2), ctx.Arg(inst, 3), mem.offset,
	                              dynamic ? state.buffer_strides[mem.resource]
	                                      : ConstantU32(state, stride),
	                              swizzle ? ConstantBool(state, true) : 0u,
	                              ConstantU32(state, (packed >> 16u) & 3u))
	    .byte;
}

uint32_t AddU64Low(EmitterState& state, uint32_t low, uint32_t high, uint32_t add_low,
                   uint32_t add_high, uint32_t& out_high) {
	const auto result = Binary(state, spv::OpIAdd, TypeU32(state), low, add_low);
	const auto carry  = Binary(state, spv::OpULessThan, TypeBool(state), result, low);
	out_high =
	    Binary(state, spv::OpIAdd, TypeU32(state),
	           Binary(state, spv::OpIAdd, TypeU32(state), high, add_high),
	           Select(state, TypeU32(state), carry, ConstantU32(state, 1), ConstantU32(state, 0)));
	return result;
}

uint32_t AddAddressOffset(EmitterState& state, const IR::MemoryInfo& mem, uint32_t low,
                          uint32_t& high) {
	const auto immediate = static_cast<int32_t>(mem.offset);
	if (immediate == 0) return low;
	const auto immediate_low  = ConstantU32(state, static_cast<uint32_t>(immediate));
	const auto immediate_high = ConstantU32(state, immediate < 0 ? UINT32_MAX : 0u);
	return AddU64Low(state, low, high, immediate_low, immediate_high, high);
}

uint32_t ScratchByteAddress(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t low,
                            uint32_t high) {
	auto& state = ctx.state;
	low = AddAddressOffset(state, mem, low, high);
	const auto valid = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, 0));
	return Select(state, TypeU32(state), valid, low, ConstantU32(state, UINT32_MAX));
}

uint32_t GuestAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	auto  low   = ctx.Arg(inst, 1);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, ~3u));
	}
	uint32_t address = 0;
	if (mem.address_is_full) {
		address = DeviceAddressFromWords(state, low, ctx.Arg(inst, 2));
	} else {
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) {
			ctx.Fail(inst, "has no address base pair");
			return ConstantDeviceAddress(state, 0);
		}
		auto base_low = ctx.Arg(*handle, 0);
		if (mem.kind == IR::ResourceKind::ScalarAddress) {
			base_low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), base_low,
			                  ConstantU32(state, ~3u));
		}
		const auto base = DeviceAddressFromWords(state, base_low, ctx.Arg(*handle, 1));
		address         = Binary(state, spv::OpIAdd, TypeScalarU64(state), base,
		                         Unary(state, spv::OpUConvert, TypeScalarU64(state), low));
	}
	auto immediate = static_cast<int32_t>(mem.offset);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		immediate = static_cast<int32_t>(static_cast<uint32_t>(immediate) & ~3u);
	}
	return immediate == 0
	           ? address
	           : Binary(state, spv::OpIAdd, TypeScalarU64(state), address,
	                    ConstantDeviceAddress(
	                        state, static_cast<uint64_t>(static_cast<int64_t>(immediate))));
}

uint32_t FaultElementPointer(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.fault_buffer_variable, ConstantU32(state, 0), index);
	return pointer;
}

void RecordBdaFault(EmitterState& state, uint32_t page) {
	const auto word =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5));
	const auto bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	// One atomic OR: lanes faulting on pages of the same bitmap word keep each other's bits.
	state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(),
	                          FaultElementPointer(state, word), ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), bit);
}

// A DWORD load at a physical address for FLAT/GLOBAL loads with DLC or GLC (IR::MemoryInfo::coherent):
// through a Coherent block member and Volatile, so it is not served from a stale cache.
uint32_t LoadCoherentU32(EmitterState& state, uint32_t address) {
	const auto block = state.builder.DecoratedType(
	    spv::OpTypeStruct,
	    {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	     {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
	     {spv::OpDecorate, {spv::DecorationBlock}}},
	    TypeU32(state));
	const auto base = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr,
	                          TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block), base,
	                          address);
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU32Pointer(state), pointer, base,
	                          ConstantU32(state, 0));
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
	                          spv::MemoryAccessAlignedMask | spv::MemoryAccessVolatileMask,
	                          uint32_t {4});
	return value;
}

// A dword load by guest address; zero where `active` (default true) is false or the page is not
// mapped. With the zero page there is no branch: an unmapped or inactive lane reads a valid
// address and its value is replaced by zero.
uint32_t LoadBdaDword(ValueEmitContext& ctx, uint32_t address, uint32_t active = 0,
                      bool coherent = false) {
	auto& state = ctx.state;
	if (state.bda_load_pointer_function != 0) {
		const auto always = active == 0 || active == ConstantBool(state, true);
		uint32_t   value  = 0;
		if (coherent) {
			value = LoadCoherentU32(
			    state, GetBdaLoadPointer(state, address, always ? ConstantBool(state, true) : active));
		} else {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
			    GetBdaLoadPointer(state, address, always ? ConstantBool(state, true) : active));
			value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
			                          spv::MemoryAccessAlignedMask, uint32_t {4});
		}
		return always ? value : Select(state, TypeU32(state), active, value, ConstantU32(state, 0));
	}
	if (active != 0 && active != ConstantBool(state, true)) {
		return EmitValueOrZeroIfCondition(state, active,
		                                  [&]() { return LoadBdaDword(ctx, address, 0, coherent); });
	}
	const auto bda     = GetBdaPointer(state, address);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantDeviceAddress(state, 0));
	return EmitValueOrZeroIfCondition(state, present, [&]() {
		if (coherent) {
			return LoadCoherentU32(state, bda);
		}
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                          bda);
		const auto         value     = state.builder.AllocateId();
		constexpr uint32_t alignment = sizeof(uint32_t);
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
		                          spv::MemoryAccessAlignedMask, alignment);
		return value;
	});
}

std::array<uint32_t, 2> LoadElementWordPair(EmitterState& state, uint32_t address, uint32_t bytes,
                                            uint32_t active, bool coherent);

// usable[i]: the DWORD belongs to the element, is mapped, and `active` holds. With the zero page
// (`zero_page`) every pointer is readable and usable[i] does not test the mapping: an unmapped
// page reads zeros.
struct ElementWords {
	uint32_t                misalign;
	uint32_t                first_page;
	uint32_t                second_page;
	uint32_t                count;
	bool                    zero_page;
	std::array<uint32_t, 5> pointer;
	std::array<uint32_t, 5> usable;
};

ElementWords MapElementWords(EmitterState& state, uint32_t address, uint32_t bytes, uint32_t active,
                             uint32_t count, bool load, bool dword_aligned, bool masks);
std::array<uint32_t, 5> LoadElementWords(EmitterState& state, const ElementWords& window,
                                         const std::array<uint32_t, 5>& usable, bool coherent);

// `coherent`: a FLAT/GLOBAL load with DLC or GLC (see LoadCoherentU32).
uint32_t LoadBda(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t bits,
                 bool coherent = false) {
	auto& state = ctx.state;
	if (state.bda_load_pointer_function != 0 && BdaNoWordMasksEnabled()) {
		// KYTY_BDA_NO_WORD_MASKS: the two DWORDs unmasked (both readable with the zero page), shifted
		// as one 64-bit value so the second only contributes above the first's remaining bytes,
		// and one select for `active` on the result.
		const auto u32    = TypeU32(state);
		const auto u64    = TypeScalarU64(state);
		const auto window = MapElementWords(state, address, ConstantU32(state, bits / 8u), active, 2u,
		                                    true, false, false);
		const auto words  = LoadElementWords(state, window, {}, coherent);
		const auto pair   = Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, words[0]),
		                           Binary(state, spv::OpShiftLeftLogical, u64,
		                                  Unary(state, spv::OpUConvert, u64, words[1]),
		                                  ConstantDeviceAddress(state, 32)));
		const auto shift  = Binary(state, spv::OpShiftLeftLogical, u64,
		                           Unary(state, spv::OpUConvert, u64, window.misalign),
		                           ConstantDeviceAddress(state, 3));
		auto value = Unary(state, spv::OpUConvert, u32, Binary(state, spv::OpShiftRightLogical, u64, pair, shift));
		if (bits != 32u) {
			value = Binary(state, spv::OpBitwiseAnd, u32, value, ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
		}
		return active == ConstantBool(state, true) ? value
		                                           : Select(state, u32, active, value, ConstantU32(state, 0));
	}
	// The one or two DWORDs the bytes cover, through one page lookup (two when they cross a page)
	// under one branch. Zero when `active` is false.
	const auto [first, second] =
	    LoadElementWordPair(state, address, ConstantU32(state, bits / 8u), active, coherent);
	const auto byte =
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	           Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3));
	const auto shift =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3));
	const auto upper_shift =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                  Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 4), byte),
	                  ConstantU32(state, 3)),
	           ConstantU32(state, 3));
	const auto merged =
	    Binary(state, spv::OpBitwiseOr, TypeU32(state),
	           Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, shift),
	           Binary(state, spv::OpShiftLeftLogical, TypeU32(state), second, upper_shift));
	return bits == 32u ? merged
	                   : Binary(state, spv::OpBitwiseAnd, TypeU32(state), merged,
	                            ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
}

// With hardware storage buffer bounds, byte offsets at or past this are out of bounds whatever
// the access size (no binding is 2 GiB long): offset + size never wraps the 32-bit check, and as
// an access chain index (signed) even a byte view's index stays positive, so it cannot address
// below the binding.
constexpr uint32_t HardwareBoundsByteCap = 0x7fffff00u;

uint32_t ByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	if (mem.kind == IR::ResourceKind::Buffer) {
		const auto resource = ctx.state.program.bindings.table
		                          ? mem.resource
		                          : ResourceForDescriptor(ctx.state, IR::DescriptorBindingKind::Buffers,
		                                                  mem.resource);
		const auto prefix = ctx.state.memory_byte_offsets[resource];
		const auto address = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state),
		                            BufferByteAddress(ctx, inst, mem), prefix);
		// The host binding prefix must not wrap an out-of-range guest address into the buffer.
		const auto wrapped =
		    Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), address, prefix);
		if (!HardwareStorageBufferBounds() || ctx.state.program.bindings.table) {
			return Select(ctx.state, TypeU32(ctx.state), wrapped, address,
			              ConstantU32(ctx.state, UINT32_MAX));
		}
		// Hardware bounds check offset + size in 32 bits: an access in the last bytes below 4 GiB
		// wraps to a small end and passes, and the GPU reads 4 GiB past the binding. Capped, the
		// largest access (an element window of at most 256 bytes) stays below 4 GiB.
		const auto cap = ConstantU32(ctx.state, HardwareBoundsByteCap);
		return EmitGlsl<GLSLstd450UMin, IR::Type::U32>(
		    ctx.state, Select(ctx.state, TypeU32(ctx.state), wrapped, address, cap), cap);
	}
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		if (mem.offset == 0u) {
			return ctx.Arg(inst, 0);
		}
		return Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), ctx.Arg(inst, 0),
		              ConstantU32(ctx.state, mem.offset));
	}
	if (mem.kind != IR::ResourceKind::Scratch) {
		EXIT("physical address memory must use the BDA emitter\n");
	}
	return ScratchByteAddress(ctx, mem, ctx.Arg(inst, 1), ctx.Arg(inst, 2));
}

uint32_t DwordIndex(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto address = ByteAddress(ctx, inst, mem);
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		// RDNA2 DS region addresses retain bits [15:2] after adding the byte offset.
		address = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                 ConstantU32(ctx.state, 0xffffu));
	}
	return Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state),
	              address, ConstantU32(ctx.state, 2));
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index);

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits, bool sign_extend);

uint32_t LoadWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource) {
	const auto index = DwordIndex(ctx, inst, mem);
	if (resource.element_bits == 32u) {
		const auto in_bounds = EmitWordAccessInBounds(ctx.state, resource, index);
		return LoadElementOrZero(ctx.state, resource, in_bounds, index, [&](uint32_t element) {
			return LoadWordInBounds(ctx, resource, element);
		});
	}
	return LoadElementOrZero(ctx.state, resource,
	                         EmitMemoryElementInBounds(ctx.state, resource, index), index,
	                         [&](uint32_t element) { return LoadWordInBounds(ctx, resource, element); });
}

// LDS and scratch always hold element 0 (they are arrays of a known nonzero size).
bool HasElementZero(const EmitterState& state, IR::ResourceKind kind) {
	return (kind == IR::ResourceKind::Lds && LdsDwordCount(state) != 0u) ||
	       (kind == IR::ResourceKind::Scratch && state.program.scratch_dwords != 0u);
}

// A DWORD of LDS or scratch without a branch: where `valid` is false the read goes to element 0
// and gives zero. Two nested branches (EXEC, then bounds) around every DS read gave Yotei's
// LDS-heavy compute shaders thousands of blocks for the driver's compiler to work through.
uint32_t LoadLocalWordOrZero(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t index, uint32_t valid) {
	auto&      state = ctx.state;
	const auto safe  = Select(state, TypeU32(state), valid, index, ConstantU32(state, 0));
	return Select(state, TypeU32(state), valid, LoadWordInBounds(ctx, resource, safe),
	              ConstantU32(state, 0));
}

uint32_t LoadWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	if (HasElementZero(ctx.state, mem.kind)) {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		if (resource.element_bits == 32u) {
			const auto index = DwordIndex(ctx, inst, mem);
			return LoadLocalWordOrZero(
			    ctx, resource, index,
			    AndCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1),
			                 EmitMemoryElementInBounds(ctx.state, resource, index)));
		}
	}
	if (IsStorageBufferKind(mem.kind)) {
		// As in LoadElementOrZero: robustBufferAccess keeps the load of a lane outside EXEC
		// harmless, so EXEC selects the value instead of branching around the load.
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return Select(ctx.state, TypeU32(ctx.state), ctx.Arg(inst, inst.NumArgs() - 1),
		              LoadWordPrepared(ctx, inst, mem, resource), ConstantU32(ctx.state, 0));
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadWordPrepared(ctx, inst, mem, resource);
	});
}

uint32_t LoadSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const MemoryResourceAccess& resource, uint32_t bits,
                             bool sign_extend) {
	const auto address = ByteAddress(ctx, inst, mem);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	return LoadElementOrZero(
	    ctx.state, resource, EmitMemoryElementInBounds(ctx.state, resource, index), index,
	    [&](uint32_t element) {
		    return LoadSubwordInBounds(ctx, resource, address, element, bits, sign_extend);
	    });
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index) {
	const auto value   = ctx.state.builder.AllocateId();
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	if (resource.physical) {
		EmitBufferElementLoad(ctx.state, TypeU32(ctx.state), value, pointer, resource);
		return value;
	}
	ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), value, pointer,
	                              resource.memory_access);
	return value;
}

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits, bool sign_extend) {
	uint32_t value;
	if (resource.element_bits < 32u) {
		const auto loaded = ctx.state.builder.AllocateId();
		if (resource.physical) {
			EmitBufferElementLoad(ctx.state, TypeStorageBufferElement(ctx.state, bits), loaded,
			                      EmitMemoryElementPointer(ctx.state, resource, index), resource);
		} else {
			ctx.state.builder.AddFunction(spv::OpLoad, TypeStorageBufferElement(ctx.state, bits),
			                              loaded, EmitMemoryElementPointer(ctx.state, resource, index),
			                              resource.memory_access);
		}
		value = Unary(ctx.state, spv::OpUConvert, TypeU32(ctx.state), loaded);
	} else {
		const auto word = LoadWordInBounds(ctx, resource, index);
		const auto byte = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                         ConstantU32(ctx.state, 3));
		const auto shift = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), byte,
		                          ConstantU32(ctx.state, 3));
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
		               Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), word, shift),
		               ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu));
	}
	if (!sign_extend) return value;
	const auto left = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), value,
	                         ConstantU32(ctx.state, 32u - bits));
	return Binary(ctx.state, spv::OpShiftRightArithmetic, TypeU32(ctx.state), left,
	              ConstantU32(ctx.state, 32u - bits));
}

uint32_t LoadSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits,
                     bool sign_extend) {
	if (IsStorageBufferKind(mem.kind)) {
		// As in LoadWord.
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return Select(ctx.state, TypeU32(ctx.state), ctx.Arg(inst, inst.NumArgs() - 1),
		              LoadSubwordPrepared(ctx, inst, mem, resource, bits, sign_extend),
		              ConstantU32(ctx.state, 0));
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadSubwordPrepared(ctx, inst, mem, resource, bits, sign_extend);
	});
}

Prospero::BufferFormat BufferFormat(const ValueEmitContext& ctx, const IR::MemoryInfo& mem) {
	return mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
	                 : StorageBufferFormat(ctx.state, mem);
}

IR::MemoryInfo RebaseRawComponent(IR::MemoryInfo mem, uint32_t component) {
	mem.offset += component * 4u;
	mem.data_dwords     = 1u;
	mem.component_index = component;
	return mem;
}

using Format::FormattedSource;
using Format::FormattedSourceKind;

FormattedSource ResolveFormattedSource(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t                        output_component) {
	if (mem.typed) {
		return output_component < info.component_count
		           ? FormattedSource {FormattedSourceKind::Memory, output_component}
		           : FormattedSource {};
	}
	const auto selector = GetDstSel(ctx.state.program.info.buffers[mem.resource].descriptor_swizzle,
	                                output_component);
	const auto source = Format::ResolveFormattedSource(info, selector);
	if (source.kind == FormattedSourceKind::Invalid) {
		ExitDescriptorBindingFailure(ctx.state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer descriptor has reserved dst_sel");
	}
	return source;
}

uint32_t FormattedConstant(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                           FormattedSourceKind kind) {
	return ConstantU32(ctx.state, Format::FormattedConstantBits(info, kind));
}

template <typename LoadWordFn, typename LoadSubwordFn>
uint32_t LoadFormattedComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                                FormattedSource source, LoadWordFn&& load_word,
                                LoadSubwordFn&& load_subword) {
	if (source.kind != FormattedSourceKind::Memory) {
		return FormattedConstant(ctx, info, source.kind);
	}
	const auto component = source.component;
	const auto bits      = info.component_bits[component];
	auto raw = info.packed_bitfield || bits == 32u ? load_word(component)
	                                                : load_subword(component, bits);
	if (info.packed_bitfield || (bits < 32u && IsSignedFormatComponent(info.type))) {
		const auto type =
		    IsSignedFormatComponent(info.type) ? TypeI32(ctx.state) : TypeU32(ctx.state);
		const auto source_value =
		    type == TypeI32(ctx.state) ? Unary(ctx.state, spv::OpBitcast, type, raw) : raw;
		const auto extracted = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(IsSignedFormatComponent(info.type) ? spv::OpBitFieldSExtract
		                                                                 : spv::OpBitFieldUExtract,
		                              type, extracted, source_value,
		                              ConstantU32(ctx.state, info.packed_bitfield
		                                                         ? info.component_bit_offset[component]
		                                                         : 0u),
		                              ConstantU32(ctx.state, bits));
		raw = type == TypeI32(ctx.state)
		          ? Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), extracted)
		          : extracted;
	}
	return NormalizeFormatComponent(ctx.state, info, component, raw);
}

void StoreSubwordInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t address, uint32_t index,
                          uint32_t bits, uint32_t data) {
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	if (resource.element_bits < 32u) {
		ctx.state.builder.AddFunction(
		    spv::OpStore, pointer,
		    Unary(ctx.state, spv::OpUConvert, TypeStorageBufferElement(ctx.state, bits), data),
		    resource.memory_access);
		MarkTrackedWrite(ctx.state, resource, index, resource.element_bits);
		return;
	}
	const auto shift   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
	                                   ConstantU32(ctx.state, 3)),
	                            ConstantU32(ctx.state, 3));
	const auto mask    = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu), shift);
	const auto value   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), data,
	                                   ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu)),
	                            shift);
	const auto merge   = [&](uint32_t old) {
		return Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), old,
		                     Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), mask)),
		              value);
	};
	if (mem.kind == IR::ResourceKind::Scratch) {
		const auto old = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), old, pointer);
		ctx.state.builder.AddFunction(spv::OpStore, pointer, merge(old));
	} else {
		AtomicUpdate(ctx.state, pointer, mem.kind, merge);
	}
	MarkTrackedWrite(ctx.state, resource, index, resource.element_bits);
}

void StoreSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t bits, uint32_t data) {
	const auto address = ByteAddress(ctx, inst, mem);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	EmitIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		    StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data);
	    });
}

void StoreSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		StoreSubwordPrepared(ctx, inst, mem, resource, bits, ctx.Arg(inst, inst.NumArgs() - 2));
	});
}

void StoreWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t data) {
	ctx.state.builder.AddFunction(spv::OpStore,
	                              EmitMemoryElementPointer(ctx.state, resource, index), data,
	                              resource.memory_access);
	MarkTrackedWrite(ctx.state, resource, index, resource.element_bits);
}

template <typename Fn>
void ForEachLocalFlatAccess(ValueEmitContext& ctx, const IR::Inst& inst, Fn&& emit) {
	auto& state = ctx.state;
	auto mem = ctx.Memory(inst);
	auto high = ctx.Arg(inst, 2);
	const auto low = AddAddressOffset(state, mem, ctx.Arg(inst, 1), high);
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), low, ConstantU32(state, 2));
	const auto aligned = Binary(state, spv::OpIEqual, TypeBool(state), ConstantU32(state, 0),
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 3)));
	for (const auto kind: {IR::ResourceKind::Scratch, IR::ResourceKind::Lds}) {
		mem.kind = kind;
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto aperture = kind == IR::ResourceKind::Scratch ? Decoder::PrivateApertureHigh
		                                                       : Decoder::SharedApertureHigh;
		const auto selected = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, aperture));
		const auto valid = AndCondition(state, aligned,
		    AndCondition(state, selected, EmitMemoryElementInBounds(state, resource, index)));
		emit(resource, index, valid);
	}
}

uint32_t LoadLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		uint32_t result = 0;
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			const auto value = EmitValueOrZeroIfCondition(ctx.state, valid, [&]() {
				return LoadWordInBounds(ctx, resource, index);
			});
			result = result == 0 ? value : Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), result, value);
		});
		return result;
	});
}

void StoreLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			EmitIfCondition(ctx.state, valid, [&]() {
				StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, inst.NumArgs() - 2));
			});
		});
	});
}

void StoreWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t data) {
	const auto index     = DwordIndex(ctx, inst, mem);
	const auto in_bounds = resource.element_bits == 32u
	                           ? EmitWordAccessInBounds(ctx.state, resource, index)
	                           : EmitMemoryElementInBounds(ctx.state, resource, index);
	EmitIfCondition(ctx.state, in_bounds, [&]() { StoreWordInBounds(ctx, resource, index, data); });
}

void StoreWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	auto& state = ctx.state;
	if (HasElementZero(state, mem.kind)) {
		// One branch for EXEC and the bounds together.
		const auto access = PrepareMemoryResourceAccess(state, mem);
		if (access.element_bits == 32u) {
			const auto index = DwordIndex(ctx, inst, mem);
			const auto data  = ctx.Arg(inst, inst.NumArgs() - 2);
			EmitIfCondition(state,
			                AndCondition(state, ctx.Arg(inst, inst.NumArgs() - 1),
			                             EmitMemoryElementInBounds(state, access, index)),
			                [&]() { StoreWordInBounds(ctx, access, index, data); });
			return;
		}
	}
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto store = [&](uint32_t resource) {
			if (mem.kind == IR::ResourceKind::Buffer) {
				const auto& buffer = state.program.info.buffers[resource];
				if (buffer.packed_stride == 0u &&
				    buffer.source < state.program.descriptor_sources.size() &&
				    state.program.descriptor_sources[buffer.source].indirect_descriptor) return;
			}
			mem.resource = resource;
			const auto access = PrepareMemoryResourceAccess(state, mem);
			StoreWordPrepared(ctx, inst, mem, access, data);
		};
		if (mem.kind != IR::ResourceKind::Buffer ||
		    state.program.info.buffers[mem.resource].indirect_root != mem.resource) {
			store(mem.resource);
			return;
		}
		const auto& buffer = state.program.info.buffers[mem.resource];
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr || buffer.indirect_resources.empty() ||
		    !HasFlattenedSrt(state) || buffer.indirect_search_iterations == 0u) {
			ctx.Fail(inst, "has no indirect buffer runtime mapping");
			return;
		}
		const auto selected = EmitIndirectResourceIndex(
		    state, ctx.Arg(*handle, 0), buffer.indirect_mapping_offset,
		    buffer.indirect_search_iterations, UINT32_MAX);
		const auto merge = state.builder.AllocateId();
		std::vector<uint32_t> labels(buffer.indirect_resources.size());
		std::vector<uint32_t> branches {spv::OpSwitch, selected, merge};
		for (uint32_t i = 0; i < labels.size(); ++i) {
			labels[i] = state.builder.AllocateId();
			branches.push_back(i);
			branches.push_back(labels[i]);
		}
		state.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
		state.builder.AddFunction(branches);
		for (uint32_t i = 0; i < labels.size(); ++i) {
			EmitLabel(state, labels[i]);
			store(buffer.indirect_resources[i]);
			state.builder.AddFunction(spv::OpBranch, merge);
		}
		EmitLabel(state, merge);
	});
}

template <typename Fn>
uint32_t EmitAtomicAccess(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& operation) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto index = DwordIndex(ctx, inst, mem);
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return EmitValueOrZeroIfCondition(
		    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
			    const auto result = operation(EmitMemoryElementPointer(ctx.state, resource, index));
			    MarkTrackedWrite(ctx.state, resource, index, resource.element_bits);
			    return result;
		    });
	});
}

template <typename Fn>
uint32_t EmitAtomicUpdate(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& replacement) {
	const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		return AtomicUpdate(ctx.state, pointer, mem.kind, [&](uint32_t old) {
			return replacement(ctx.state, old, value);
		});
	});
}

uint32_t AtomicIncrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old >= limit ? 0 : old + 1 (unsigned).
	const auto wrap = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), old, limit);
	const auto next = Binary(state, spv::OpIAdd, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, ConstantU32(state, 0), next);
}

uint32_t AtomicDecrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old == 0 || old > limit ? limit : old - 1 (unsigned).
	const auto zero  = Binary(state, spv::OpIEqual, TypeBool(state), old, ConstantU32(state, 0));
	const auto above = Binary(state, spv::OpUGreaterThan, TypeBool(state), old, limit);
	const auto wrap  = Binary(state, spv::OpLogicalOr, TypeBool(state), zero, above);
	const auto next  = Binary(state, spv::OpISub, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, limit, next);
}

struct PreparedFormattedMemory {
	Format::BufferFormatInfo info;
	MemoryResourceAccess     resource;
	uint32_t                 base = 0;
	uint32_t                 in_bounds = 0;
};

PreparedFormattedMemory PrepareFormattedMemory(ValueEmitContext& ctx, const IR::Inst& inst,
                                               const IR::MemoryInfo& mem,
                                               const MemoryResourceAccess& resource,
                                               const Format::BufferFormatInfo& info) {
	PreparedFormattedMemory plan;
	plan.info      = info;
	plan.resource  = resource;
	const auto start = ByteAddress(ctx, inst, mem);
	// Check the whole, unaligned format span before aligning the actual memory address.
	const auto end = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), start,
	                        ConstantU32(ctx.state, info.byte_size - 1u));
	const auto last_index = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), end,
	                               ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	plan.in_bounds = AndCondition(
	    ctx.state, Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), end, start),
	    EmitMemoryElementInBounds(ctx.state, plan.resource, last_index));
	plan.base = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), start,
	                   ConstantU32(ctx.state, ~(std::min(4u, info.byte_size) - 1u)));
	return plan;
}

std::pair<uint32_t, uint32_t> FormattedComponentAddress(EmitterState& state,
                                                       const PreparedFormattedMemory& plan,
                                                       uint32_t component) {
	// Components advance sequentially from one address; do not reapply buffer swizzling.
	const auto offset = Format::GetFormatComponentByteOffset(plan.info, component);
	const auto address = offset == 0u ? plan.base : Binary(
	    state, spv::OpIAdd, TypeU32(state), plan.base, ConstantU32(state, offset));
	return {address, Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                        ConstantU32(state, std::countr_zero(plan.resource.element_bits / 8u)))};
}

uint32_t LoadFormattedInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                               const PreparedFormattedMemory& plan, uint32_t output_component) {
	return LoadFormattedComponent(
	    ctx, plan.info, ResolveFormattedSource(ctx, mem, plan.info, output_component),
	    [&](uint32_t component) {
		    return LoadWordInBounds(ctx, plan.resource,
		                            FormattedComponentAddress(ctx.state, plan, component).second);
	    },
	    [&](uint32_t component, uint32_t bits) {
		    const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
		    return LoadSubwordInBounds(ctx, plan.resource, address, index, bits, false);
	    });
}

uint32_t ConstructU32Composite(EmitterState& state, uint32_t components,
                               const std::array<uint32_t, 4>& values) {
	if (components == 1u) return values[0];
	const auto            result = state.builder.AllocateId();
	std::vector<uint32_t> words {spv::OpCompositeConstruct, TypeU32Composite(state, components),
	                             result};
	words.insert(words.end(), values.begin(), values.begin() + components);
	state.builder.AddFunction(words);
	return result;
}

uint32_t FormattedOutOfBoundsValue(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                   const PreparedFormattedMemory& plan, uint32_t components) {
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
		values[component] = FormattedConstant(ctx, plan.info, source.kind);
	}
	return ConstructU32Composite(ctx.state, components, values);
}

uint32_t D16LoadComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                          uint32_t value);

// A formatted load through the descriptor's format of a buffer whose format and dst_sel the
// host passes in shader data (IR::GenericBufferFormat).
bool IsGenericFormattedLoad(const ValueEmitContext& ctx, const IR::MemoryInfo& mem) {
	return mem.formatted && !mem.typed && mem.kind == IR::ResourceKind::Buffer &&
	       mem.resource < ctx.state.program.info.buffers.size() &&
	       ctx.state.program.info.buffers[mem.resource].descriptor_format == IR::GenericBufferFormat;
}

uint32_t LoadGenericFormatted(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                              uint32_t components);

uint32_t FormattedLoad(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	if (IsGenericFormattedLoad(ctx, mem)) {
		return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1),
		                                  [&]() { return LoadGenericFormatted(ctx, inst, mem, 1u); });
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto info = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type == Format::ComponentType::Unknown) {
			return LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, 0u), resource);
		}
		const auto plan  = PrepareFormattedMemory(ctx, inst, mem, resource, info);
		const auto value = EmitValueOrDefaultIfCondition(
		    ctx.state, plan.in_bounds, TypeU32(ctx.state),
		    FormattedOutOfBoundsValue(ctx, mem, plan, 1u),
		    [&]() { return LoadFormattedInBounds(ctx, mem, plan, 0u); });
		return mem.d16 ? D16LoadComponent(ctx, info, value) : value;
	});
}

bool IsD16FloatLike(Format::ComponentType type) {
	return type == Format::ComponentType::Float || type == Format::ComponentType::Unorm ||
	       type == Format::ComponentType::Snorm;
}

// A D16 load returns each formatted component as 16 bits: a half for float-like formats, the low
// bits otherwise.
uint32_t D16LoadComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                          uint32_t value) {
	if (IsD16FloatLike(info.type)) {
		const auto pair = EmitCompositeConstructF32x2(
		    ctx.state, EmitBitCastF32U32(ctx.state, value), ConstantF32Value(ctx.state, 0.0f));
		return EmitPackHalf2x16(ctx.state, pair);
	}
	return Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), value,
	              ConstantU32(ctx.state, 0xffffu));
}

// A D16 store takes each component as 16 bits (see D16LoadComponent).
uint32_t D16StoreComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                           uint32_t value) {
	auto& state = ctx.state;
	if (IsD16FloatLike(info.type)) {
		const auto unpacked = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
		                          GlslStd450(state), GLSLstd450UnpackHalf2x16, value);
		const auto low = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), low, unpacked, 0u);
		return Unary(state, spv::OpBitcast, TypeU32(state), low);
	}
	if (IsSignedFormatComponent(info.type)) {
		const auto extended = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), extended,
		                          Unary(state, spv::OpBitcast, TypeI32(state), value),
		                          ConstantU32(state, 0u), ConstantU32(state, 16u));
		return Unary(state, spv::OpBitcast, TypeU32(state), extended);
	}
	return Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, ConstantU32(state, 0xffffu));
}

// A 32-bit float (its bits) as an unsigned 11-bit or 10-bit float (5-bit exponent), rounded
// toward zero as the guest GPU's conversion: beyond the range the largest value, negative values
// zero, infinity and NaN kept.
static uint32_t EmitF32ToUFloatBits(EmitterState& state, uint32_t f32_bits, uint32_t bits) {
	const auto     u32           = TypeU32(state);
	const auto     boolean       = TypeBool(state);
	const uint32_t mantissa_bits = bits == 11u ? 6u : 5u;
	const uint32_t infinity      = 0x1fu << mantissa_bits;
	const uint32_t largest       = infinity - 1u;
	const auto     compare       = [&](spv::Op op, uint32_t value, uint32_t constant) {
        return Binary(state, op, boolean, value, ConstantU32(state, constant));
	};
	const auto exponent = Binary(state, spv::OpBitwiseAnd, u32,
	                             Binary(state, spv::OpShiftRightLogical, u32, f32_bits,
	                                    ConstantU32(state, 23u)),
	                             ConstantU32(state, 0xffu));
	const auto mantissa = Binary(state, spv::OpBitwiseAnd, u32, f32_bits, ConstantU32(state, 0x007fffffu));
	const auto negative = compare(spv::OpUGreaterThanEqual, f32_bits, 0x80000000u);
	const auto special  = compare(spv::OpIEqual, exponent, 0xffu);
	const auto nan      = compare(spv::OpINotEqual, mantissa, 0u);
	// NaN stores zero (upstream 59d8256b3): a NaN in a packed float image feeds every later pass.
	const auto special_bits =
	    Select(state, u32, nan, ConstantU32(state, 0u), ConstantU32(state, infinity));
	// Normal: the exponent rebiased from 127 to 15, the mantissa truncated.
	const auto small_exponent = Binary(state, spv::OpISub, u32, exponent, ConstantU32(state, 127u - 15u));
	const auto normal = Binary(
	    state, spv::OpBitwiseOr, u32,
	    Binary(state, spv::OpShiftLeftLogical, u32, small_exponent, ConstantU32(state, mantissa_bits)),
	    Binary(state, spv::OpShiftRightLogical, u32, mantissa, ConstantU32(state, 23u - mantissa_bits)));
	// Denormal: the mantissa with its hidden bit, shifted down by 136 - mantissa_bits - exponent
	// (under 32 where it is used).
	const auto with_hidden = Binary(state, spv::OpBitwiseOr, u32, mantissa, ConstantU32(state, 0x00800000u));
	const auto shift = Binary(state, spv::OpISub, u32, ConstantU32(state, 136u - mantissa_bits), exponent);
	const auto denormal =
	    Select(state, u32, compare(spv::OpULessThan, exponent, 105u - mantissa_bits), ConstantU32(state, 0u),
	           Binary(state, spv::OpShiftRightLogical, u32, with_hidden, shift));
	const auto finite =
	    Select(state, u32, compare(spv::OpULessThanEqual, exponent, 127u - 15u), denormal,
	           Select(state, u32, compare(spv::OpUGreaterThanEqual, exponent, 127u + 16u),
	                  ConstantU32(state, largest), normal));
	const auto magnitude = Select(state, u32, special, special_bits, finite);
	// Negative values, -infinity included, become zero.
	return Select(state, u32, negative, ConstantU32(state, 0u), magnitude);
}

uint32_t EncodeFormattedStoreComponent(ValueEmitContext& ctx,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t component, uint32_t data) {
	auto& state = ctx.state;
	const auto bits = info.component_bits[component];
	const bool normalized = info.type == Format::ComponentType::Unorm ||
	                        info.type == Format::ComponentType::Snorm;
	const bool scaled = info.type == Format::ComponentType::Uscaled ||
	                    info.type == Format::ComponentType::Sscaled;
	if (normalized || scaled) {
		// Two-bit packed channels are unsigned even in signed formats.
		const bool is_signed = bits != 2u && (info.type == Format::ComponentType::Snorm ||
		                                      info.type == Format::ComponentType::Sscaled);
		const auto max_value = static_cast<float>((1u << (bits - (is_signed ? 1u : 0u))) - 1u);
		const auto lower = normalized ? (is_signed ? -1.0f : 0.0f)
		                              : (is_signed ? -max_value - 1.0f : 0.0f);
		const auto upper = normalized ? 1.0f : max_value;
		auto value = Select(state, TypeF32(state), EmitClassifyF32Bits(state, data).nan,
		                    ConstantF32Value(state, 0.0f), EmitBitCastF32U32(state, data));
		value = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(
		    state, value, ConstantF32Value(state, lower), ConstantF32Value(state, upper));
		if (normalized) {
			value = EmitFPRoundEven32(
			    state, EmitFPMul32(state, value, ConstantF32Value(state, max_value)));
		}
		// Float-to-integer conversion truncates scaled values toward zero.
		value = Unary(state, is_signed ? spv::OpConvertFToS : spv::OpConvertFToU,
		              is_signed ? TypeI32(state) : TypeU32(state), value);
		return is_signed ? Unary(state, spv::OpBitcast, TypeU32(state), value) : value;
	}
	if (bits == 16u && info.type == Format::ComponentType::Float) {
		// Beyond the half range the guest GPU stores the largest half (D3D's conversion), where
		// packHalf2x16 overflows to infinity.
		auto value = EmitBitCastF32U32(state, data);
		if (FloatStoreClampEnabled()) {
			value = EmitClampToHalfRange(state, value);
		}
		const auto pair = EmitCompositeConstructF32x2(state, value, ConstantF32Value(state, 0.0f));
		return EmitPackHalf2x16(state, pair);
	}
	if ((bits == 11u || bits == 10u) && info.type == Format::ComponentType::Float &&
	    FloatStoreClampEnabled()) {
		// Packed unsigned floats (11_11_10, 10_11_11): encoded, not the float's low bits
		// (usa-ratt's GT7 fork, cfa8a479e).
		return EmitF32ToUFloatBits(state, data, bits);
	}
	return data;
}

void StoreFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                             const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                             const Format::BufferFormatInfo& info, uint32_t data,
                             uint32_t components) {
	// RDNA2 formatted stores transfer the entire format, zero-filling missing VGPRs.
	const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
	EmitIfCondition(ctx.state, plan.in_bounds, [&]() {
		auto packed = ConstantU32(ctx.state, 0);
		for (uint32_t component = 0; component < info.component_count; component++) {
			auto value = ConstantU32(ctx.state, 0);
			if (component < components) {
				value = data;
				if (components != 1u) {
					value = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
					                              value, data, component);
				}
				if (mem.d16) {
					value = D16StoreComponent(ctx, info, value);
				}
			}
			value = EncodeFormattedStoreComponent(ctx, info, component, value);
			const auto bits = info.component_bits[component];
			if (info.packed_bitfield) {
				packed = EmitBitFieldInsert(
				    ctx.state, packed, value,
				    ConstantU32(ctx.state, info.component_bit_offset[component]),
				    ConstantU32(ctx.state, bits));
			} else {
				const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
				if (bits == 8u || bits == 16u) {
					StoreSubwordInBounds(ctx, mem, plan.resource, address, index, bits, value);
				} else {
					StoreWordInBounds(ctx, plan.resource, index, value);
				}
			}
		}
		if (info.packed_bitfield) {
			StoreWordInBounds(ctx, plan.resource,
			                  FormattedComponentAddress(ctx.state, plan, 0u).second, packed);
		}
	});
}

void FormattedStore(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto data     = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info     = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type == Format::ComponentType::Unknown) {
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, 0u), resource, data);
			return;
		}
		StoreFormattedPrepared(ctx, inst, mem, resource, info, data, 1u);
	});
}

uint32_t LoadIndirectScalarBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  word1  = ctx.Arg(handle, 1);
	const auto  stride = EmitBitFieldUExtract(state, word1, ConstantU32(state, 16),
	                                         ConstantU32(state, 14));
	const auto size = Binary(
	    state, spv::OpIMul, TypeScalarU64(state),
	    Unary(state, spv::OpUConvert, TypeScalarU64(state),
	          EmitUMax32(state, stride, ConstantU32(state, 1))),
	    Unary(state, spv::OpUConvert, TypeScalarU64(state), ctx.Arg(handle, 2)));
	const auto offset = Binary(
	    state, spv::OpIAdd, TypeScalarU64(state),
	    Unary(state, spv::OpUConvert, TypeScalarU64(state),
	          Binary(state, spv::OpBitwiseAnd, TypeU32(state), ctx.Arg(inst, 1),
	                 ConstantU32(state, ~3u))),
	    ConstantDeviceAddress(state, ctx.Memory(inst).offset & ~3u));
	const auto end = Binary(state, spv::OpIAdd, TypeScalarU64(state), offset,
	                        ConstantDeviceAddress(state, sizeof(uint32_t)));
	const auto in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), end, size);
	const auto address   = [&]() {
		const auto base = DeviceAddressFromWords(
		    state, Binary(state, spv::OpBitwiseAnd, TypeU32(state), ctx.Arg(handle, 0),
		                  ConstantU32(state, ~3u)),
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1, ConstantU32(state, 0xffffu)));
		return Binary(state, spv::OpIAdd, TypeScalarU64(state), base, offset);
	};
	if (state.bda_load_pointer_function != 0) {
		return LoadBdaDword(ctx, address(), in_bounds);
	}
	return EmitValueOrZeroIfCondition(state, in_bounds,
	                                  [&]() { return LoadBdaDword(ctx, address()); });
}

// A formatted load through a GPU-selected V#: the format comes from the descriptor at run time
// and is decoded from tables. The element's bytes are read once, up to five DWORDs from the
// aligned start, with one address translation (two when the element crosses a page).
// The DWORDs an element of `bytes` bytes at a guest address covers: up to five from the aligned
// start, through one address translation (two when the element crosses a buffer-cache page).

// `load`: the words are only read, so the lookups may go through the zero page. `dword_aligned`:
// the address is known to be a multiple of 4; a single aligned DWORD never crosses a page and
// needs no second lookup. `masks` false (zero page only): the caller masks the final value itself
// and usable[] is left unset.
ElementWords MapElementWords(EmitterState& state, uint32_t address, uint32_t bytes, uint32_t active,
                             uint32_t count = 5, bool load = false, bool dword_aligned = false,
                             bool masks = true) {
	const auto u32 = TypeU32(state);
	const auto u64 = TypeScalarU64(state);
	ElementWords result {};
	result.zero_page = load && state.bda_load_pointer_function != 0;
	if (result.zero_page) {
		const auto single = dword_aligned && bytes == ConstantU32(state, 4u);
		result.misalign   = dword_aligned ? ConstantU32(state, 0)
		                                  : Binary(state, spv::OpBitwiseAnd, u32,
		                                           Unary(state, spv::OpUConvert, u32, address),
		                                           ConstantU32(state, 3));
		const auto aligned = dword_aligned ? address
		                                   : Binary(state, spv::OpBitwiseAnd, u64, address,
		                                            ConstantDeviceAddress(state, ~uint64_t {3}));
		result.count = std::min<uint32_t>(single ? 1u : count, static_cast<uint32_t>(result.pointer.size()));
		result.first_page  = GetBdaLoadPointer(state, aligned, active);
		result.second_page = result.first_page;
		if (single) {
			result.pointer[0] = result.first_page;
			result.usable[0]  = masks ? active : 0u;
			return result;
		}
		const auto need = Binary(state, spv::OpIAdd, u32, result.misalign, bytes);
		constexpr auto PageSize = static_cast<uint32_t>(BufferCache::CACHING_PAGESIZE);
		const auto in_page    = Binary(state, spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, aligned),
		                               ConstantU32(state, PageSize - 1u));
		const auto first_room = Binary(state, spv::OpISub, u32, ConstantU32(state, PageSize), in_page);
		const auto crosses    = Binary(state, spv::OpUGreaterThan, TypeBool(state), need, first_room);
		// KYTY_BDA_FAST_DWORD: the first DWORD is always on the first page (at least 4 bytes are
		// left on it), and every DWORD of an aligned access of whole DWORDs is wanted.
		const bool fast       = BdaFastDwordEnabled();
		const bool all_wanted = fast && dword_aligned && bytes == ConstantU32(state, result.count * 4u);
		if (BdaSingleCrossBranchEnabled(state.program.shader_hash)) {
			// KYTY_BDA_SINGLE_CROSS_BRANCH: one branch per access on whether it crosses into the next
			// page. Not crossing (nearly always), every DWORD is on the first page; only the crossing
			// block looks the second page up and picks each DWORD's page. A second lookup and a page
			// choice per DWORD on every access cost the driver's compiler most of its time (bc8e).
			std::array<uint32_t, 5> same_page {};
			for (uint32_t word = 0; word < result.count; ++word) {
				same_page[word] = word == 0u ? result.first_page
				                             : Binary(state, spv::OpIAdd, u64, result.first_page,
				                                      ConstantDeviceAddress(state, word * 4u));
				if (masks) {
					result.usable[word] = word == 0u || all_wanted
					                          ? active
					                          : AndCondition(state, active,
					                                         Binary(state, spv::OpULessThan, TypeBool(state),
					                                                ConstantU32(state, word * 4u), need));
				}
			}
			result.pointer[0] = result.first_page;
			if (result.count == 1u) {
				return result;
			}
			const auto header      = state.current_label;
			const auto cross_label = state.builder.AllocateId();
			const auto merge_label = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
			state.builder.AddFunction(spv::OpBranchConditional, crosses, cross_label, merge_label);
			EmitLabel(state, cross_label);
			// Only lanes that cross get here: `active` alone gates the second page's fault record.
			result.second_page = GetBdaLoadPointer(
			    state, Binary(state, spv::OpIAdd, u64, aligned, Unary(state, spv::OpUConvert, u64, first_room)),
			    active);
			std::array<uint32_t, 5> crossing {};
			const bool              mux = BdaCrossMuxEnabled();
			for (uint32_t word = 1; word < result.count; ++word) {
				const auto offset    = ConstantU32(state, word * 4u);
				const auto on_second = [&]() {
					return Binary(state, spv::OpIAdd, u64, result.second_page,
					              Unary(state, spv::OpUConvert, u64,
					                    Binary(state, spv::OpISub, u32, offset, first_room)));
				};
				if (mux) {
					// KYTY_BDA_SINGLE_CROSS_BRANCH=2: the page choice as a bit mask instead of a select:
					// offset - first_room is negative (first_room is 4..16384) exactly when the DWORD is
					// on the first page, so its sign spread over 64 bits picks the first page's pointer.
					const auto first_mask = Unary(
					    state, spv::OpSConvert, u64,
					    Binary(state, spv::OpShiftRightArithmetic, u32,
					           Binary(state, spv::OpISub, u32, offset, first_room), ConstantU32(state, 31)));
					crossing[word] = Binary(
					    state, spv::OpBitwiseOr, u64, Binary(state, spv::OpBitwiseAnd, u64, same_page[word], first_mask),
					    Binary(state, spv::OpBitwiseAnd, u64, on_second(), Unary(state, spv::OpNot, u64, first_mask)));
				} else {
					const auto in_first = Binary(state, spv::OpULessThan, TypeBool(state), offset, first_room);
					crossing[word]      = Select(state, u64, in_first, same_page[word], on_second());
				}
			}
			const auto cross_exit = state.current_label;
			state.builder.AddFunction(spv::OpBranch, merge_label);
			EmitLabel(state, merge_label);
			for (uint32_t word = 1; word < result.count; ++word) {
				result.pointer[word] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpPhi, u64, result.pointer[word], crossing[word], cross_exit,
				                          same_page[word], header);
			}
			// The V# values cached in the header dominate the merge block (KYTY_INDIRECT_BOUNDS_ONCE).
			if (state.indirect_vsharp_label == header) {
				state.indirect_vsharp_label = merge_label;
			}
			return result;
		}
		result.second_page = GetBdaLoadPointer(
		    state, Binary(state, spv::OpIAdd, u64, aligned, Unary(state, spv::OpUConvert, u64, first_room)),
		    AndCondition(state, active, crosses));
		for (uint32_t word = 0; word < result.count; ++word) {
			const auto offset = ConstantU32(state, word * 4u);
			if (fast && word == 0u) {
				result.pointer[word] = result.first_page;
			} else {
				const auto in_first = Binary(state, spv::OpULessThan, TypeBool(state), offset, first_room);
				result.pointer[word] = Select(
				    state, u64, in_first,
				    Binary(state, spv::OpIAdd, u64, result.first_page, Unary(state, spv::OpUConvert, u64, offset)),
				    Binary(state, spv::OpIAdd, u64, result.second_page,
				           Unary(state, spv::OpUConvert, u64, Binary(state, spv::OpISub, u32, offset, first_room))));
			}
			if (masks) {
				result.usable[word] = word == 0u || all_wanted
				                          ? active
				                          : AndCondition(state, active,
				                                         Binary(state, spv::OpULessThan, TypeBool(state),
				                                                offset, need));
			}
		}
		return result;
	}
	result.misalign = Binary(state, spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, address),
	                         ConstantU32(state, 3));
	const auto aligned = Binary(state, spv::OpBitwiseAnd, u64, address,
	                            ConstantDeviceAddress(state, ~uint64_t {3}));
	const auto need    = Binary(state, spv::OpIAdd, u32, result.misalign, bytes);
	constexpr auto PageSize = static_cast<uint32_t>(BufferCache::CACHING_PAGESIZE);
	const auto in_page    = Binary(state, spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, aligned),
	                               ConstantU32(state, PageSize - 1u));
	const auto first_room = Binary(state, spv::OpISub, u32, ConstantU32(state, PageSize), in_page);
	const auto crosses    = Binary(state, spv::OpUGreaterThan, TypeBool(state), need, first_room);
	const auto first_page = EmitValueOrDefaultIfCondition(
	    state, active, u64, ConstantDeviceAddress(state, 0), [&]() { return GetBdaPointer(state, aligned); });
	const auto second_page = EmitValueOrDefaultIfCondition(
	    state, AndCondition(state, active, crosses), u64, ConstantDeviceAddress(state, 0), [&]() {
		    return GetBdaPointer(state, Binary(state, spv::OpIAdd, u64, aligned,
		                                       Unary(state, spv::OpUConvert, u64, first_room)));
	    });
	const auto nonzero64 = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantDeviceAddress(state, 0));
	};
	result.first_page  = first_page;
	result.second_page = second_page;
	result.count       = std::min<uint32_t>(count, static_cast<uint32_t>(result.pointer.size()));
	for (uint32_t word = 0; word < result.count; ++word) {
		const auto offset   = ConstantU32(state, word * 4u);
		const auto wanted   = Binary(state, spv::OpULessThan, TypeBool(state), offset, need);
		const auto in_first = Binary(state, spv::OpULessThan, TypeBool(state), offset, first_room);
		result.pointer[word] = Select(
		    state, u64, in_first,
		    Binary(state, spv::OpIAdd, u64, first_page, Unary(state, spv::OpUConvert, u64, offset)),
		    Binary(state, spv::OpIAdd, u64, second_page,
		           Unary(state, spv::OpUConvert, u64, Binary(state, spv::OpISub, u32, offset, first_room))));
		const auto present = Select(state, TypeBool(state), in_first, nonzero64(first_page),
		                            nonzero64(second_page));
		result.usable[word] = AndCondition(state, active, AndCondition(state, wanted, present));
	}
	return result;
}

// Loads the first window.count words of an element under one branch (some page is mapped)
// instead of one branch per word: a word that is not usable reads the element's first mapped
// DWORD, a valid address, and gives zero. A branch and a page lookup per DWORD made Yotei's
// buffer-heavy compute shaders take minutes in the driver's compiler (cs bc8eaa9cea77f9c3).
// With the zero page every pointer is readable and there is no branch at all; words past
// window.count are zero then, and a word whose usable[] is 0 is returned unmasked.
std::array<uint32_t, 5> LoadElementWords(EmitterState& state, const ElementWords& window,
                                         const std::array<uint32_t, 5>& usable,
                                         bool coherent = false) {
	const auto u32    = TypeU32(state);
	const auto u64    = TypeScalarU64(state);
	std::array<uint32_t, 5> result {};
	if (window.zero_page) {
		result.fill(ConstantU32(state, 0));
		for (uint32_t word = 0; word < window.count; ++word) {
			uint32_t value = 0;
			if (coherent) {
				value = LoadCoherentU32(state, window.pointer[word]);
			} else {
				const auto physical = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), physical,
				                          window.pointer[word]);
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLoad, u32, value, physical, spv::MemoryAccessAlignedMask,
				                          uint32_t {4});
			}
			result[word] = usable[word] == 0u || usable[word] == ConstantBool(state, true)
			                   ? value
			                   : Select(state, u32, usable[word], value, ConstantU32(state, 0));
		}
		return result;
	}
	const auto zero64 = ConstantDeviceAddress(state, 0);
	const auto safe   = Select(state, u64,
	                           Binary(state, spv::OpINotEqual, TypeBool(state), window.first_page, zero64),
	                           window.first_page, window.second_page);
	const auto mapped = Binary(state, spv::OpINotEqual, TypeBool(state), safe, zero64);
	const auto header = state.current_label;
	const auto then_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, mapped, then_label, merge_label);
	EmitLabel(state, then_label);
	std::array<uint32_t, 5> loaded {};
	for (uint32_t word = 0; word < window.count; ++word) {
		uint32_t value = 0;
		if (coherent) {
			value = LoadCoherentU32(state, Select(state, u64, usable[word], window.pointer[word], safe));
		} else {
			const auto physical = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), physical,
			                          Select(state, u64, usable[word], window.pointer[word], safe));
			value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, u32, value, physical, spv::MemoryAccessAlignedMask,
			                          uint32_t {4});
		}
		loaded[word] = Select(state, u32, usable[word], value, ConstantU32(state, 0));
	}
	const auto then_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	for (uint32_t word = 0; word < window.count; ++word) {
		result[word] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpPhi, u32, result[word], loaded[word], then_exit,
		                          ConstantU32(state, 0), header);
	}
	return result;
}

// The two DWORDs from the aligned start of `bytes` (at most 4) bytes at a guest address; a DWORD
// the bytes do not cover, or that is not mapped, is zero.
std::array<uint32_t, 2> LoadElementWordPair(EmitterState& state, uint32_t address, uint32_t bytes,
                                            uint32_t active, bool coherent) {
	const auto window = MapElementWords(state, address, bytes, active, 2u, true);
	const auto words  = LoadElementWords(state, window, window.usable, coherent);
	return {words[0], words[1]};
}

struct IndirectBufferAccess {
	uint32_t address;
	uint32_t offset;
	uint32_t stride;
	uint32_t swizzle;
	uint32_t format;
	uint32_t selector;
	uint32_t mode;
	uint32_t records;
	uint32_t index_in_bounds;
	uint32_t scalar_in_bounds;
	uint32_t raw_records;
	uint32_t raw_index_in_bounds;
	// KYTY_INDIRECT_BOUNDS_ONCE: the V#'s index-independent values (vsharp.key[0] != 0).
	EmitterState::IndirectVsharpValues vsharp;
};

// The V# words, index, offset, SOFFSET and lane of an access through a GPU-selected V#.
struct IndirectBufferOperands {
	std::array<uint32_t, 4> words;
	uint32_t                index;
	uint32_t                offset;
	uint32_t                soffset;
	uint32_t                lane;
};

IndirectBufferOperands IndirectOperands(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	return {.words   = {ctx.Arg(handle, 0), ctx.Arg(handle, 1), ctx.Arg(handle, 2), ctx.Arg(handle, 3)},
	        .index   = ctx.Arg(inst, 1),
	        .offset  = ctx.Arg(inst, 2),
	        .soffset = ctx.Arg(inst, 3),
	        .lane    = BufferLane(ctx.state)};
}

bool IndirectBoundsOnceEnabled() {
	// KYTY_INDIRECT_BOUNDS_ONCE=0: every access through a GPU-selected V# decodes the V# and
	// selects its bounds per OOB_SELECT mode on its own.
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_INDIRECT_BOUNDS_ONCE");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

// The index-independent values of a V# and SOFFSET, from an earlier access in the same block when
// there is one (the same SSA words are the same V#, and the block's earlier values dominate).
EmitterState::IndirectVsharpValues IndirectVsharp(EmitterState& state,
                                                  const IndirectBufferOperands& operands) {
	if (state.indirect_vsharp_label != state.current_label) {
		state.indirect_vsharp_cache.clear();
		state.indirect_vsharp_label = state.current_label;
	}
	const std::array<uint32_t, 5> key {operands.words[0], operands.words[1], operands.words[2],
	                                   operands.words[3], operands.soffset};
	for (const auto& cached: state.indirect_vsharp_cache) {
		if (cached.key == key) {
			return cached;
		}
	}
	const auto u32     = TypeU32(state);
	const auto boolean = TypeBool(state);
	const auto field   = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, boolean, value, ConstantU32(state, 0));
	};
	const auto mode_is = [&](uint32_t mode, uint32_t value) {
		return Binary(state, spv::OpIEqual, boolean, mode, ConstantU32(state, value));
	};
	EmitterState::IndirectVsharpValues v;
	v.key            = key;
	const auto word1 = operands.words[1];
	const auto word3 = operands.words[3];
	v.records        = operands.words[2];
	v.stride         = field(word1, 16, 14);
	v.swizzle        = AndCondition(state, nonzero(v.stride), nonzero(field(word1, 31, 1)));
	v.not_swizzle    = Unary(state, spv::OpLogicalNot, boolean, v.swizzle);
	v.lane_add       = Select(state, u32, nonzero(field(word3, 23, 1)), operands.lane, ConstantU32(state, 0));
	v.index_stride   = field(word3, 21, 2);
	v.base           = DeviceAddressFromWords(state, operands.words[0], field(word1, 0, 16));
	v.format         = field(word3, 12, 7);
	v.mode           = field(word3, 28, 2);
	v.raw_records    = Binary(state, spv::OpISub, u32, v.records, operands.soffset);
	v.scalar_in_bounds = Binary(state, spv::OpULessThanEqual, boolean, operands.soffset, v.records);
	v.mode0          = mode_is(v.mode, 0);
	v.mode1          = mode_is(v.mode, 1);
	v.mode2_nonzero  = AndCondition(state, mode_is(v.mode, 2), nonzero(v.records));
	v.size           = Select(state, u32, v.swizzle, v.stride, v.raw_records);
	// OOB_SELECT=3 reduces NUM_RECORDS by SOFFSET before its offset/index checks.
	v.raw_static     = AndCondition(state, mode_is(v.mode, 3), v.scalar_in_bounds);
	// The DWORD limit (see IndirectDwordBounds): mode 0 the stride; mode 3 size - 3 when size >= 4,
	// else 0; modes 1 and 2 do not depend on the offset (0).
	const auto size_ok = AndCondition(state, v.raw_static,
	                                  Binary(state, spv::OpUGreaterThanEqual, boolean, v.size, ConstantU32(state, 4)));
	v.base_limit = Select(state, u32, v.mode0, v.stride,
	                      Select(state, u32, size_ok, Binary(state, spv::OpISub, u32, v.size, ConstantU32(state, 3)),
	                             ConstantU32(state, 0)));
	state.indirect_vsharp_cache.push_back(v);
	return v;
}

IndirectBufferAccess PrepareIndirectBuffer(EmitterState& state, const IndirectBufferOperands& operands,
                                           uint32_t memory_offset, bool formatted) {
	if (IndirectBoundsOnceEnabled()) {
		const auto v       = IndirectVsharp(state, operands);
		const auto index   = Binary(state, spv::OpIAdd, TypeU32(state), operands.index, v.lane_add);
		const auto address = CalculateBufferAddress(state, index, operands.offset, operands.soffset,
		                                            memory_offset, v.stride, v.swizzle, v.index_stride);
		return {
		    .address  = Binary(state, spv::OpIAdd, TypeScalarU64(state), v.base,
		                       Unary(state, spv::OpUConvert, TypeScalarU64(state), address.byte)),
		    .offset   = address.offset,
		    .stride   = v.stride,
		    .swizzle  = v.swizzle,
		    .format   = v.format,
		    .selector = formatted ? EmitBitFieldUExtract(state, operands.words[3], ConstantU32(state, 0),
		                                                 ConstantU32(state, 3))
		                          : 0u,
		    .mode     = v.mode,
		    .records  = v.records,
		    .index_in_bounds     = Binary(state, spv::OpULessThan, TypeBool(state), index, v.records),
		    .scalar_in_bounds    = v.scalar_in_bounds,
		    .raw_records         = v.raw_records,
		    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, v.raw_records),
		    .vsharp              = v,
		};
	}
	const auto word1   = operands.words[1];
	const auto records = operands.words[2];
	const auto word3   = operands.words[3];
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto stride  = field(word1, 16, 14);
	const auto swizzle = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto index = Binary(
	    state, spv::OpIAdd, TypeU32(state), operands.index,
	    Select(state, TypeU32(state), nonzero(field(word3, 23, 1)), operands.lane,
	           ConstantU32(state, 0)));
	const auto soffset = operands.soffset;
	const auto address = CalculateBufferAddress(state, index, operands.offset, soffset,
	                                            memory_offset, stride, swizzle,
	                                            field(word3, 21, 2));
	const auto base = DeviceAddressFromWords(state, operands.words[0], field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	return {
	    .address = Binary(state, spv::OpIAdd, TypeScalarU64(state), base,
	                      Unary(state, spv::OpUConvert, TypeScalarU64(state), address.byte)),
	    .offset = address.offset,
	    .stride = stride,
	    .swizzle = swizzle,
	    .format = field(word3, 12, 7),
	    .selector = formatted ? field(word3, 0, 3) : 0u,
	    .mode = field(word3, 28, 2),
	    .records = records,
	    .index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	    .vsharp = {},
	};
}

IndirectBufferAccess PrepareIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	return PrepareIndirectBuffer(ctx.state, IndirectOperands(ctx, inst), ctx.Memory(inst).offset,
	                             ctx.Memory(inst).formatted);
}

// As IndirectBufferInBounds, with the access size as a value (a format decoded at run time).
uint32_t IndirectBufferInBoundsBytes(EmitterState& state, const IndirectBufferAccess& buffer,
                                     uint32_t displacement, uint32_t bytes, bool formatted) {
	const auto offset = displacement == 0u ? buffer.offset : Binary(
	    state, spv::OpIAdd, TypeU32(state), buffer.offset, ConstantU32(state, displacement));
	const auto has_payload = [&](uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size, bytes),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, bytes)));
	};
	const auto structured = AndCondition(
	    state, buffer.index_in_bounds,
	    formatted ? has_payload(buffer.stride)
	              : Binary(state, spv::OpULessThan, TypeBool(state), offset, buffer.stride));
	if (const auto& v = buffer.vsharp; v.key[0] != 0) {
		// The same checks without selects: exactly one OOB_SELECT mode holds, and the size of mode 3
		// (the stride when swizzled, else NUM_RECORDS - SOFFSET) comes with the V#.
		const auto boolean = TypeBool(state);
		const auto raw     = AndCondition(
		    state,
		    AndCondition(state, v.raw_static,
		                 Binary(state, spv::OpLogicalOr, boolean, v.not_swizzle, buffer.raw_index_in_bounds)),
		    has_payload(v.size));
		return Binary(state, spv::OpLogicalOr, boolean,
		              Binary(state, spv::OpLogicalOr, boolean, AndCondition(state, v.mode0, structured),
		                     AndCondition(state, v.mode1, buffer.index_in_bounds)),
		              Binary(state, spv::OpLogicalOr, boolean, v.mode2_nonzero, raw));
	}
	// OOB_SELECT=3 reduces NUM_RECORDS by SOFFSET before its offset/index checks.
	const auto raw = AndCondition(
	    state, buffer.scalar_in_bounds,
	    Select(state, TypeBool(state), buffer.swizzle,
	           AndCondition(state, buffer.raw_index_in_bounds, has_payload(buffer.stride)),
	           has_payload(buffer.raw_records)));
	auto in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 0)),
	    structured, buffer.index_in_bounds);
	in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	    in_bounds,
	    Select(state, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	           Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records, ConstantU32(state, 0)),
	           raw));
	return in_bounds;
}

uint32_t IndirectBufferInBounds(EmitterState& state, const IndirectBufferAccess& buffer,
                                uint32_t displacement, uint32_t bytes, bool formatted) {
	return IndirectBufferInBoundsBytes(state, buffer, displacement, ConstantU32(state, bytes),
	                                   formatted);
}

// IndirectBufferInBounds for the DWORDs of one raw access, sharing the work between them: the
// DWORD at byte displacement d is in bounds when offset + d < limit, or whatever d when `always`.
// Per OOB_SELECT mode: 0 checks offset + d < stride (and the index), 3 checks offset + d <= size
// - 4 (size the stride when swizzled, else NUM_RECORDS - SOFFSET), that is offset + d < size - 3
// for a size of at least 4, and 1 and 2 do not depend on the offset.
struct IndirectDwordBounds {
	uint32_t offset;
	uint32_t limit;
	uint32_t always;
};

IndirectDwordBounds PrepareIndirectDwordBounds(EmitterState& state, const IndirectBufferAccess& buffer,
                                               uint32_t enabled) {
	const auto u32     = TypeU32(state);
	const auto boolean = TypeBool(state);
	const auto zero    = ConstantU32(state, 0);
	if (const auto& v = buffer.vsharp; v.key[0] != 0) {
		// The V# gives its mode's limit; per access only whether the index passes: in mode 0 the
		// index check, in mode 3 the swizzled index check (other modes have limit 0 anyway).
		const auto passes = Binary(
		    state, spv::OpLogicalOr, boolean, AndCondition(state, v.mode0, buffer.index_in_bounds),
		    AndCondition(state, Unary(state, spv::OpLogicalNot, boolean, v.mode0),
		                 Binary(state, spv::OpLogicalOr, boolean, v.not_swizzle, buffer.raw_index_in_bounds)));
		const auto limit  = Select(state, u32, AndCondition(state, enabled, passes), v.base_limit, zero);
		const auto always = AndCondition(
		    state, enabled,
		    Binary(state, spv::OpLogicalOr, boolean, AndCondition(state, v.mode1, buffer.index_in_bounds),
		           v.mode2_nonzero));
		return {buffer.offset, limit, always};
	}
	const auto mode_is = [&](uint32_t mode) {
		return Binary(state, spv::OpIEqual, boolean, buffer.mode, ConstantU32(state, mode));
	};
	const auto size = Select(state, u32, buffer.swizzle, buffer.stride, buffer.raw_records);
	const auto raw  = AndCondition(
	    state, AndCondition(state, buffer.scalar_in_bounds,
	                        Binary(state, spv::OpUGreaterThanEqual, boolean, size, ConstantU32(state, 4))),
	    Select(state, boolean, buffer.swizzle, buffer.raw_index_in_bounds, ConstantBool(state, true)));
	const auto raw_limit =
	    Select(state, u32, raw, Binary(state, spv::OpISub, u32, size, ConstantU32(state, 3)), zero);
	const auto structured_limit = Select(state, u32, buffer.index_in_bounds, buffer.stride, zero);
	auto limit = Select(state, u32, mode_is(0), structured_limit,
	                    Select(state, u32, mode_is(3), raw_limit, zero));
	limit      = Select(state, u32, enabled, limit, zero);
	const auto always = AndCondition(
	    state, enabled,
	    Binary(state, spv::OpLogicalOr, boolean, AndCondition(state, mode_is(1), buffer.index_in_bounds),
	           AndCondition(state, mode_is(2),
	                        Binary(state, spv::OpINotEqual, boolean, buffer.records, zero))));
	return {buffer.offset, limit, always};
}

uint32_t IndirectDwordInBounds(EmitterState& state, const IndirectDwordBounds& bounds,
                               uint32_t displacement) {
	const auto offset = displacement == 0u ? bounds.offset
	                                       : Binary(state, spv::OpIAdd, TypeU32(state), bounds.offset,
	                                                ConstantU32(state, displacement));
	return Binary(state, spv::OpLogicalOr, TypeBool(state),
	              Binary(state, spv::OpULessThan, TypeBool(state), offset, bounds.limit), bounds.always);
}

// The DWORDs of a raw load through a GPU-selected V#: one page lookup for the access (two when it
// crosses a page) and one branch, instead of a lookup and two branches per DWORD.
uint32_t LoadIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components,
                            uint32_t active) {
	auto& state = ctx.state;
	const auto buffer = PrepareIndirectBuffer(ctx, inst);
	const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
	                                  ConstantU32(state, 0));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), buffer.address,
	                         ConstantDeviceAddress(state, ~uint64_t {3}));
	const auto bounds =
	    PrepareIndirectDwordBounds(state, buffer, AndCondition(state, active, valid_format));
	std::array<uint32_t, 5> in_bounds {};
	uint32_t                any = 0;
	for (uint32_t component = 0; component < components; ++component) {
		in_bounds[component] = IndirectDwordInBounds(state, bounds, component * 4u);
		any = any == 0 ? in_bounds[component]
		               : Binary(state, spv::OpLogicalOr, TypeBool(state), any, in_bounds[component]);
	}
	const auto window =
	    MapElementWords(state, base, ConstantU32(state, components * 4u), any, components, true, true);
	std::array<uint32_t, 5> usable {};
	for (uint32_t component = 0; component < components; ++component) {
		// KYTY_BDA_NO_WORD_MASKS: with the zero page every DWORD of the aligned window is wanted
		// and in_bounds implies `any`, so the DWORD's bounds alone select its value.
		usable[component] = window.zero_page && BdaNoWordMasksEnabled()
		                        ? in_bounds[component]
		                        : AndCondition(state, window.usable[component], in_bounds[component]);
	}
	// GLC/DLC loads bypass the caches on the GPU; a spin on another workgroup's flag (decoupled
	// look-back in a prefix scan) would otherwise never see the store.
	const auto words = LoadElementWords(state, window, usable, ctx.Memory(inst).coherent);
	std::array<uint32_t, 4> values {};
	std::copy_n(words.begin(), components, values.begin());
	return ConstructU32Composite(state, components, values);
}

// A raw byte or short load through a GPU-selected descriptor: the bytes at the computed address,
// zero when the access is out of the descriptor's bounds.
uint32_t LoadIndirectSubword(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t bits) {
	auto&      state  = ctx.state;
	const auto buffer = PrepareIndirectBuffer(ctx, inst);
	const auto valid_format =
	    Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format, ConstantU32(state, 0));
	const auto in_bounds = AndCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1),
	    AndCondition(state, valid_format,
	                 IndirectBufferInBounds(state, buffer, 0u, bits / 8u, false)));
	return LoadBda(ctx, buffer.address, in_bounds, bits);
}

// A dword store by guest address: skipped where the BDA page table has no mapping.
void StoreBdaDword(ValueEmitContext& ctx, uint32_t address, uint32_t value) {
	auto&      state   = ctx.state;
	const auto bda     = GetBdaPointer(state, address);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantDeviceAddress(state, 0));
	EmitIfCondition(state, present, [&]() {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda);
		state.builder.AddFunction(spv::OpStore, pointer, value, spv::MemoryAccessAlignedMask,
		                          static_cast<uint32_t>(sizeof(uint32_t)));
	});
}

// A byte or short store by guest address: an atomic AND clears its bits in the dword and an
// atomic OR sets them, so lanes writing other bytes of the same dword do not lose theirs.
void StoreBdaSubword(ValueEmitContext& ctx, uint32_t address, uint32_t value, uint32_t bits) {
	auto&      state   = ctx.state;
	const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address,
	                            ConstantDeviceAddress(state, ~uint64_t {3}));
	const auto bda     = GetBdaPointer(state, aligned);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantDeviceAddress(state, 0));
	EmitIfCondition(state, present, [&]() {
		const auto shift = Binary(
		    state, spv::OpShiftLeftLogical, TypeU32(state),
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		           Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3)),
		    ConstantU32(state, 3));
		const auto field = ConstantU32(state, bits == 8u ? 0xffu : 0xffffu);
		const auto mask  = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), field, shift);
		const auto bitsv = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                          Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, field), shift);
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda);
		const auto scope     = ConstantU32(state, spv::ScopeDevice);
		const auto semantics = ConstantU32(state, spv::MemorySemanticsMaskNone);
		const auto cleared   = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicAnd, TypeU32(state), cleared, pointer, scope, semantics,
		                          Unary(state, spv::OpNot, TypeU32(state), mask));
		const auto set = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), set, pointer, scope, semantics,
		                          bitsv);
	});
}

// A raw store through a GPU-selected descriptor (see MemoryInfo::SupportsIndirectBufferStore).
void StoreIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components,
                         uint32_t bits) {
	auto&      state  = ctx.state;
	const auto buffer = PrepareIndirectBuffer(ctx, inst);
	const auto valid_format =
	    Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format, ConstantU32(state, 0));
	const auto active = AndCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), valid_format);
	const auto data   = ctx.Arg(inst, inst.NumArgs() - 2);
	if (bits != 32u) {
		EmitIfCondition(state,
		                AndCondition(state, active,
		                             IndirectBufferInBounds(state, buffer, 0u, bits / 8u, false)),
		                [&]() { StoreBdaSubword(ctx, buffer.address, data, bits); });
		return;
	}
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), buffer.address,
	                         ConstantDeviceAddress(state, ~uint64_t {3}));
	// One page lookup for the access (two when it crosses a page), as in LoadIndirectBuffer.
	const auto bounds = PrepareIndirectDwordBounds(state, buffer, active);
	std::array<uint32_t, 5> in_bounds {};
	uint32_t                any = 0;
	for (uint32_t component = 0; component < components; ++component) {
		in_bounds[component] = IndirectDwordInBounds(state, bounds, component * 4u);
		any = any == 0 ? in_bounds[component]
		               : Binary(state, spv::OpLogicalOr, TypeBool(state), any, in_bounds[component]);
	}
	const auto window =
	    MapElementWords(state, base, ConstantU32(state, components * 4u), any, components);
	for (uint32_t component = 0; component < components; ++component) {
		uint32_t value = data;
		if (components > 1u) {
			value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data, component);
		}
		EmitIfCondition(state, AndCondition(state, window.usable[component], in_bounds[component]),
		                [&]() {
			                const auto pointer = state.builder.AllocateId();
			                state.builder.AddFunction(spv::OpConvertUToPtr,
			                                          TypePhysicalU32Pointer(state), pointer,
			                                          window.pointer[component]);
			                state.builder.AddFunction(spv::OpStore, pointer, value,
			                                          spv::MemoryAccessAlignedMask,
			                                          static_cast<uint32_t>(sizeof(uint32_t)));
		                });
	}
}

struct DynamicBufferFormat {
	uint32_t layout;  // component count | byte size << 3 | packed << 8 | valid << 9
	uint32_t widths;  // 6 bits per component
	uint32_t offsets; // bit offset in the element, 7 bits per component
	uint32_t type;    // Format::ComponentType
	uint32_t valid;
};

// The layout of a V# format known only at run time: four Private arrays indexed by the 7-bit
// format, declared once per module. A select chain or switch over every format at each of a
// hundred loads made Yotei's tile-lighting shaders too large for the driver to compile.
DynamicBufferFormat LookupDynamicBufferFormat(EmitterState& state, uint32_t format) {
	const auto array = state.builder.Type(spv::OpTypeArray, TypeU32(state), ConstantU32(state, 128));
	if (state.format_tables[0] == 0) {
		constexpr auto last = static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float);
		std::array<std::array<uint32_t, 128>, 4> words {};
		for (uint32_t code = 0; code < 128u; ++code) {
			const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(code));
			if (code == 0u || code > last || info.type == Format::ComponentType::Unknown) {
				words[0][code] = 1u;
				words[1][code] = 32u;
				continue;
			}
			words[0][code] = info.component_count | (info.byte_size << 3u) |
			                 (uint32_t {info.packed_bitfield} << 8u) | (1u << 9u);
			for (uint32_t component = 0; component < 4; ++component) {
				words[1][code] |= info.component_bits[component] << (component * 6u);
				words[2][code] |= info.component_bit_offset[component] << (component * 7u);
			}
			words[3][code] = static_cast<uint32_t>(info.type);
		}
		for (uint32_t table = 0; table < 4u; ++table) {
			std::vector<uint32_t> ids;
			ids.reserve(128);
			for (const auto word: words[table]) {
				ids.push_back(ConstantU32(state, word));
			}
			const auto initializer = state.builder.Constant(spv::OpConstantComposite, array,
			                                                std::span<const uint32_t>(ids));
			state.format_tables[table] = state.builder.DefineGlobalVariable(
			    TypePointer(state, spv::StorageClassPrivate, array), spv::StorageClassPrivate,
			    initializer);
			// Not an interface variable: SPIR-V 1.3 lists only Input/Output ones, and from 1.4 on
			// the builder adds every global variable itself (Builder::AddEntryPoint).
		}
	}
	const auto index = Binary(state, spv::OpBitwiseAnd, TypeU32(state), format, ConstantU32(state, 127));
	const auto load  = [&](uint32_t table) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassPrivate, TypeU32(state)), pointer,
		                          state.format_tables[table], index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	DynamicBufferFormat result {};
	result.layout  = load(0);
	result.widths  = load(1);
	result.offsets = load(2);
	result.type    = load(3);
	result.valid   = Binary(state, spv::OpINotEqual, TypeBool(state),
	                        Binary(state, spv::OpBitwiseAnd, TypeU32(state), result.layout,
	                               ConstantU32(state, 1u << 9u)),
	                        ConstantU32(state, 0));
	return result;
}

// NormalizeFormatComponent with the width and component type as values.
uint32_t NormalizeDynamicBufferComponent(EmitterState& state, uint32_t raw, uint32_t bits,
                                         uint32_t type) {
	const auto is_type = [&](Format::ComponentType value) {
		return Binary(state, spv::OpIEqual, TypeBool(state), type,
		              ConstantU32(state, static_cast<uint32_t>(value)));
	};
	const auto signed_raw = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), signed_raw,
	                          Unary(state, spv::OpBitcast, TypeI32(state), raw),
	                          ConstantU32(state, 0), bits);
	const auto unsigned_float = Unary(state, spv::OpConvertUToF, TypeF32(state), raw);
	const auto signed_float   = Unary(state, spv::OpConvertSToF, TypeF32(state), signed_raw);
	const auto mask =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), ConstantU32(state, UINT32_MAX),
	           Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 32), bits));
	const auto unsigned_max = Unary(state, spv::OpConvertUToF, TypeF32(state), mask);
	const auto signed_max =
	    Unary(state, spv::OpConvertUToF, TypeF32(state),
	          Binary(state, spv::OpShiftRightLogical, TypeU32(state), mask, ConstantU32(state, 1)));
	const auto unorm = Binary(state, spv::OpFDiv, TypeF32(state), unsigned_float, unsigned_max);
	const auto snorm = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), snorm, GlslStd450(state),
	                          GLSLstd450FMax,
	                          Binary(state, spv::OpFDiv, TypeF32(state), signed_float, signed_max),
	                          ConstantF32Value(state, -1.0f));
	auto result = Select(state, TypeU32(state), is_type(Format::ComponentType::Sint),
	                     Unary(state, spv::OpBitcast, TypeU32(state), signed_raw), raw);
	for (const auto [kind, value]: {std::pair {Format::ComponentType::Uscaled, unsigned_float},
	                                std::pair {Format::ComponentType::Sscaled, signed_float},
	                                std::pair {Format::ComponentType::Unorm, unorm},
	                                std::pair {Format::ComponentType::Snorm, snorm}}) {
		result = Select(state, TypeU32(state), is_type(kind),
		                Unary(state, spv::OpBitcast, TypeU32(state), value), result);
	}
	auto floating = raw;
	for (const auto width: {10u, 11u, 16u}) {
		const auto converted = width == 16u ? EmitBitcastF32ToU32(state, EmitF16BitsToF32(state, raw))
		                                    : EmitUFloatToF32Bits(state, raw, width);
		floating =
		    Select(state, TypeU32(state),
		           Binary(state, spv::OpIEqual, TypeBool(state), bits, ConstantU32(state, width)),
		           converted, floating);
	}
	return Select(state, TypeU32(state), is_type(Format::ComponentType::Float), floating, result);
}

// The formatted load of a GenericBufferFormat buffer, with the format values decoded at entry
// (EmitGenericBufferFormats). The whole unaligned element must be in bounds (as
// PrepareFormattedMemory); out of bounds, each output is its dst_sel's constant. Each output
// loads the one or two DWORDs its bits lie in: storage-buffer loads past the end are harmless
// (robust buffer access), so they need no branch.
uint32_t LoadGenericFormatted(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                              uint32_t components) {
	auto&       state    = ctx.state;
	const auto  u32      = TypeU32(state);
	const auto  boolean  = TypeBool(state);
	const auto& format   = state.generic_buffer_formats[mem.resource];
	const auto  resource = PrepareMemoryResourceAccess(state, mem);
	const auto  start    = ByteAddress(ctx, inst, mem);
	const auto  end      = Binary(state, spv::OpIAdd, u32, start, format.end_offset);
	const auto  in_bounds = AndCondition(
	    state, Binary(state, spv::OpUGreaterThanEqual, boolean, end, start),
	    EmitMemoryElementInBounds(state, resource,
	                              Binary(state, spv::OpShiftRightLogical, u32, end, ConstantU32(state, 2))));
	// The hardware aligns the element to min(4, size).
	const auto base       = Binary(state, spv::OpBitwiseAnd, u32, start, format.align_mask);
	const auto first      = Binary(state, spv::OpShiftRightLogical, u32, base, ConstantU32(state, 2));
	const auto misaligned = Binary(state, spv::OpShiftLeftLogical, u32,
	                               Binary(state, spv::OpBitwiseAnd, u32, base, ConstantU32(state, 3)),
	                               ConstantU32(state, 3));
	// The element's DWORDs (16 bytes at most, DWORD-aligned past 4 bytes), loaded once.
	std::array<uint32_t, 5> words {};
	for (uint32_t index = 0; index < 4u; ++index) {
		words[index] = LoadWordInBounds(
		    ctx, resource,
		    index == 0u ? first : Binary(state, spv::OpIAdd, u32, first, ConstantU32(state, index)));
	}
	words[4] = ConstantU32(state, 0);
	const auto pick = [&](uint32_t index, uint32_t offset) {
		auto result = words[3u + offset];
		for (uint32_t i = 3u; i-- > 0;) {
			result = Select(state, u32, Binary(state, spv::OpIEqual, boolean, index, ConstantU32(state, i)),
			                words[i + offset], result);
		}
		return result;
	};
	std::array<uint32_t, 4> values {};
	for (uint32_t output = 0; output < components; ++output) {
		const auto bits     = format.bits[output];
		const auto position = Binary(state, spv::OpIAdd, u32, misaligned, format.bit_offset[output]);
		const auto index    = Binary(state, spv::OpShiftRightLogical, u32, position, ConstantU32(state, 5));
		const auto within   = Binary(state, spv::OpBitwiseAnd, u32, position, ConstantU32(state, 31));
		const auto low      = Binary(state, spv::OpShiftRightLogical, u32, pick(index, 0u), within);
		const auto high     = Select(
		    state, u32, Binary(state, spv::OpIEqual, boolean, within, ConstantU32(state, 0)),
		    ConstantU32(state, 0),
		    Binary(state, spv::OpShiftLeftLogical, u32, pick(index, 1u),
		           Binary(state, spv::OpISub, u32, ConstantU32(state, 32), within)));
		const auto raw = EmitBitFieldUExtract(state, Binary(state, spv::OpBitwiseOr, u32, low, high),
		                                      ConstantU32(state, 0), bits);
		// As NormalizeFormatComponent for 8/16/32-bit components; signed types sign-extend.
		const auto signed_raw = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), signed_raw,
		                          Unary(state, spv::OpBitcast, TypeI32(state), raw), ConstantU32(state, 0),
		                          bits);
		const auto unsigned_float = Unary(state, spv::OpConvertUToF, TypeF32(state), raw);
		const auto signed_float   = Unary(state, spv::OpConvertSToF, TypeF32(state), signed_raw);
		const auto unorm = Binary(state, spv::OpFDiv, TypeF32(state), unsigned_float, format.unsigned_max[output]);
		const auto snorm = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeF32(state), snorm, GlslStd450(state), GLSLstd450FMax,
		                          Binary(state, spv::OpFDiv, TypeF32(state), signed_float,
		                                 format.signed_max[output]),
		                          ConstantF32Value(state, -1.0f));
		auto converted = Select(state, u32, format.half[output],
		                        EmitBitcastF32ToU32(state, EmitF16BitsToF32(state, raw)), raw);
		for (const auto [is_type, value]: {std::pair {format.is_sint, Unary(state, spv::OpBitcast, u32, signed_raw)},
		                                   std::pair {format.is_uscaled, Unary(state, spv::OpBitcast, u32, unsigned_float)},
		                                   std::pair {format.is_sscaled, Unary(state, spv::OpBitcast, u32, signed_float)},
		                                   std::pair {format.is_unorm, Unary(state, spv::OpBitcast, u32, unorm)},
		                                   std::pair {format.is_snorm, Unary(state, spv::OpBitcast, u32, snorm)}}) {
			converted = Select(state, u32, is_type, value, converted);
		}
		values[output] = Select(state, u32, format.constant_one[output], format.one,
		                        Select(state, u32, AndCondition(state, format.from_memory[output], in_bounds),
		                               converted, ConstantU32(state, 0)));
		if (mem.d16) {
			// As D16LoadComponent: a half for float-like formats, the low 16 bits otherwise.
			const auto half = EmitPackHalf2x16(
			    state, EmitCompositeConstructF32x2(state, EmitBitCastF32U32(state, values[output]),
			                                       ConstantF32Value(state, 0.0f)));
			values[output] = Select(state, u32, format.float_like, half,
			                        Binary(state, spv::OpBitwiseAnd, u32, values[output],
			                               ConstantU32(state, 0xffffu)));
		}
	}
	return ConstructU32Composite(state, components, values);
}

uint32_t LoadIndirectFormattedBody(EmitterState& state, const IndirectBufferOperands& operands,
                                   uint32_t memory_offset, uint32_t components, bool d16) {
	const auto u32    = TypeU32(state);
	const auto buffer = PrepareIndirectBuffer(state, operands, memory_offset, true);
	const auto word3  = operands.words[3];
	const auto field  = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto info      = LookupDynamicBufferFormat(state, buffer.format);
	const auto count     = field(info.layout, 0, 3);
	const auto byte_size = field(info.layout, 3, 5);
	const auto integer   = AndCondition(
	    state, info.valid,
	    Binary(state, spv::OpLogicalOr, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), info.type,
	                  ConstantU32(state, static_cast<uint32_t>(Format::ComponentType::Uint))),
	           Binary(state, spv::OpIEqual, TypeBool(state), info.type,
	                  ConstantU32(state, static_cast<uint32_t>(Format::ComponentType::Sint)))));
	const auto one = Select(state, u32, integer, ConstantU32(state, 1), ConstantU32(state, 0x3f800000));
	auto any_read  = ConstantBool(state, false);
	std::array<uint32_t, 4> selectors {};
	for (uint32_t output = 0; output < components; ++output) {
		selectors[output] = field(word3, output * 3u, 3);
		any_read = Binary(state, spv::OpLogicalOr, TypeBool(state), any_read,
		                  Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), selectors[output],
		                         ConstantU32(state, 4)));
	}
	const auto reading = AndCondition(
	    state, AndCondition(state, info.valid, any_read),
	    IndirectBufferInBoundsBytes(state, buffer, 0u, byte_size, true));
	// KYTY_BDA_NO_WORD_MASKS: with the zero page the words need no mask: every value below is
	// selected by `reading`, and the bits a component takes from a DWORD past the element lie
	// above its width, which the extraction drops.
	const bool unmasked = state.bda_load_pointer_function != 0 && BdaNoWordMasksEnabled();
	const auto window   = MapElementWords(state, buffer.address, byte_size, reading, 5u, true, false,
	                                      !unmasked);
	const auto misalign = window.misalign;
	const auto element  = LoadElementWords(state, window, unmasked ? std::array<uint32_t, 5> {} : window.usable);
	const auto pick = [&](uint32_t index) {
		auto result = element[element.size() - 1u];
		for (uint32_t word = element.size() - 1u; word-- > 0;) {
			result = Select(state, u32,
			                Binary(state, spv::OpIEqual, TypeBool(state), index, ConstantU32(state, word)),
			                element[word], result);
		}
		return result;
	};
	std::array<uint32_t, 4> values {};
	for (uint32_t output = 0; output < components; ++output) {
		const auto selector    = selectors[output];
		const auto from_memory = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), selector,
		                                ConstantU32(state, 4));
		// As Format::ResolveFormattedSource: channels repeat past the format's component count.
		const auto component = Binary(
		    state, spv::OpUMod, u32, Binary(state, spv::OpBitwiseAnd, u32, selector, ConstantU32(state, 3)),
		    EmitUMax32(state, count, ConstantU32(state, 1)));
		const auto bits = EmitBitFieldUExtract(
		    state, info.widths, Binary(state, spv::OpIMul, u32, component, ConstantU32(state, 6)),
		    ConstantU32(state, 6));
		const auto bit_offset = EmitBitFieldUExtract(
		    state, info.offsets, Binary(state, spv::OpIMul, u32, component, ConstantU32(state, 7)),
		    ConstantU32(state, 7));
		const auto position = Binary(state, spv::OpIAdd, u32,
		                             Binary(state, spv::OpShiftLeftLogical, u32, misalign, ConstantU32(state, 3)),
		                             bit_offset);
		const auto index  = Binary(state, spv::OpShiftRightLogical, u32, position, ConstantU32(state, 5));
		const auto within = Binary(state, spv::OpBitwiseAnd, u32, position, ConstantU32(state, 31));
		const auto low    = Binary(state, spv::OpShiftRightLogical, u32, pick(index), within);
		const auto high   = Select(
		    state, u32, Binary(state, spv::OpIEqual, TypeBool(state), within, ConstantU32(state, 0)),
		    ConstantU32(state, 0),
		    Binary(state, spv::OpShiftLeftLogical, u32,
		           pick(Binary(state, spv::OpIAdd, u32, index, ConstantU32(state, 1))),
		           Binary(state, spv::OpISub, u32, ConstantU32(state, 32), within)));
		const auto data      = Binary(state, spv::OpBitwiseOr, u32, low, high);
		const auto extracted = EmitBitFieldUExtract(state, data, ConstantU32(state, 0), bits);
		const auto converted = NormalizeDynamicBufferComponent(state, extracted, bits, info.type);
		const auto constant_one =
		    AndCondition(state, info.valid,
		                 Binary(state, spv::OpIEqual, TypeBool(state), selector, ConstantU32(state, 1)));
		values[output] = Select(state, u32, constant_one, one,
		                        Select(state, u32, AndCondition(state, from_memory, reading), converted,
		                               ConstantU32(state, 0)));
		if (d16) {
			// As D16LoadComponent: a half for float-like formats, the low 16 bits otherwise.
			const auto float_like = Binary(
			    state, spv::OpLogicalOr, TypeBool(state),
			    Binary(state, spv::OpIEqual, TypeBool(state), info.type,
			           ConstantU32(state, static_cast<uint32_t>(Format::ComponentType::Float))),
			    Binary(state, spv::OpLogicalOr, TypeBool(state),
			           Binary(state, spv::OpIEqual, TypeBool(state), info.type,
			                  ConstantU32(state, static_cast<uint32_t>(Format::ComponentType::Unorm))),
			           Binary(state, spv::OpIEqual, TypeBool(state), info.type,
			                  ConstantU32(state, static_cast<uint32_t>(Format::ComponentType::Snorm)))));
			const auto half = EmitPackHalf2x16(
			    state, EmitCompositeConstructF32x2(state, EmitBitCastF32U32(state, values[output]),
			                                       ConstantF32Value(state, 0.0f)));
			values[output] = Select(state, u32, float_like, half,
			                        Binary(state, spv::OpBitwiseAnd, u32, values[output],
			                               ConstantU32(state, 0xffffu)));
		}
	}
	return components == 1u ? values[0] : ConstructU32Composite(state, components, values);
}

// EncodeFormattedStoreComponent with the width and component type as values.
uint32_t EncodeDynamicBufferComponent(EmitterState& state, uint32_t data, uint32_t bits, uint32_t type) {
	const auto u32     = TypeU32(state);
	const auto f32     = TypeF32(state);
	const auto boolean = TypeBool(state);
	const auto is_type = [&](Format::ComponentType value) {
		return Binary(state, spv::OpIEqual, boolean, type, ConstantU32(state, static_cast<uint32_t>(value)));
	};
	const auto either = [&](Format::ComponentType a, Format::ComponentType b) {
		return Binary(state, spv::OpLogicalOr, boolean, is_type(a), is_type(b));
	};
	const auto normalized = either(Format::ComponentType::Unorm, Format::ComponentType::Snorm);
	const auto scaled     = either(Format::ComponentType::Uscaled, Format::ComponentType::Sscaled);
	// Two-bit packed channels are unsigned even in signed formats.
	const auto is_signed = AndCondition(
	    state, Binary(state, spv::OpINotEqual, boolean, bits, ConstantU32(state, 2)),
	    either(Format::ComponentType::Snorm, Format::ComponentType::Sscaled));
	auto shift = Binary(state, spv::OpISub, u32, bits,
	                    Select(state, u32, is_signed, ConstantU32(state, 1), ConstantU32(state, 0)));
	shift = Select(state, u32, Binary(state, spv::OpUGreaterThan, boolean, shift, ConstantU32(state, 31)),
	               ConstantU32(state, 31), shift);
	const auto max_value = Unary(
	    state, spv::OpConvertUToF, f32,
	    Binary(state, spv::OpISub, u32,
	           Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1), shift), ConstantU32(state, 1)));
	const auto lower = Select(
	    state, f32, normalized,
	    Select(state, f32, is_signed, ConstantF32Value(state, -1.0f), ConstantF32Value(state, 0.0f)),
	    Select(state, f32, is_signed,
	           Binary(state, spv::OpFSub, f32, Unary(state, spv::OpFNegate, f32, max_value),
	                  ConstantF32Value(state, 1.0f)),
	           ConstantF32Value(state, 0.0f)));
	const auto upper = Select(state, f32, normalized, ConstantF32Value(state, 1.0f), max_value);
	auto value = Select(state, f32, EmitClassifyF32Bits(state, data).nan, ConstantF32Value(state, 0.0f),
	                    EmitBitCastF32U32(state, data));
	value = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(state, value, lower, upper);
	value = Select(state, f32, normalized, EmitFPRoundEven32(state, EmitFPMul32(state, value, max_value)),
	               value);
	// Float-to-integer conversion truncates scaled values toward zero.
	const auto as_signed   = Unary(state, spv::OpBitcast, u32, Unary(state, spv::OpConvertFToS, TypeI32(state), value));
	const auto as_unsigned = Unary(state, spv::OpConvertFToU, u32, value);
	const auto converted   = Select(state, u32, is_signed, as_signed, as_unsigned);
	auto half_input = EmitBitCastF32U32(state, data);
	if (FloatStoreClampEnabled()) {
		// The guest GPU's float-to-half bound (see EncodeFormattedStoreComponent).
		half_input = EmitClampToHalfRange(state, half_input);
	}
	const auto half = EmitPackHalf2x16(
	    state, EmitCompositeConstructF32x2(state, half_input, ConstantF32Value(state, 0.0f)));
	const auto float16 = AndCondition(state, is_type(Format::ComponentType::Float),
	                                  Binary(state, spv::OpIEqual, boolean, bits, ConstantU32(state, 16)));
	return Select(state, u32, Binary(state, spv::OpLogicalOr, boolean, normalized, scaled), converted,
	              Select(state, u32, float16, half, data));
}

// A formatted store through a GPU-selected V#: encoded by the descriptor's format from the
// tables. RDNA2 writes the entire format, zero-filling missing components; DWORDs the element
// covers only in part are updated atomically so other lanes' bytes survive.
void StoreIndirectFormatted(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto&       state   = ctx.state;
	const auto  u32     = TypeU32(state);
	const auto  boolean = TypeBool(state);
	const auto& memory  = ctx.Memory(inst);
	const auto  buffer  = PrepareIndirectBuffer(ctx, inst);
	const auto  field   = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto info      = LookupDynamicBufferFormat(state, buffer.format);
	const auto count     = field(info.layout, 0, 3);
	const auto byte_size = field(info.layout, 3, 5);
	const auto is_type   = [&](Format::ComponentType value) {
		return Binary(state, spv::OpIEqual, boolean, info.type,
		              ConstantU32(state, static_cast<uint32_t>(value)));
	};
	const auto any_of = [&](std::initializer_list<Format::ComponentType> types) {
		auto result = ConstantBool(state, false);
		for (const auto type: types) {
			result = Binary(state, spv::OpLogicalOr, boolean, result, is_type(type));
		}
		return result;
	};
	const auto float_like = any_of({Format::ComponentType::Float, Format::ComponentType::Unorm,
	                                Format::ComponentType::Snorm});
	const auto signed_type = any_of({Format::ComponentType::Sint, Format::ComponentType::Snorm,
	                                 Format::ComponentType::Sscaled});
	const auto writing = AndCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1),
	    AndCondition(state, info.valid, IndirectBufferInBoundsBytes(state, buffer, 0u, byte_size, true)));
	const auto window = MapElementWords(state, buffer.address, byte_size, writing);
	const auto data   = ctx.Arg(inst, inst.NumArgs() - 2);
	std::array<uint32_t, 5> word_data {};
	std::array<uint32_t, 5> word_mask {};
	word_data.fill(ConstantU32(state, 0));
	word_mask.fill(ConstantU32(state, 0));
	for (uint32_t component = 0; component < 4u; ++component) {
		auto value = ConstantU32(state, 0);
		if (component < components) {
			value = data;
			if (components > 1u) {
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, u32, value, data, component);
			}
			if (memory.d16) {
				// As D16StoreComponent: a half for float-like formats, 16 bits otherwise.
				const auto unpacked = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
				                          GlslStd450(state), GLSLstd450UnpackHalf2x16, value);
				const auto low = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), low, unpacked, 0u);
				const auto extended = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), extended,
				                          Unary(state, spv::OpBitcast, TypeI32(state), value),
				                          ConstantU32(state, 0u), ConstantU32(state, 16u));
				value = Select(state, u32, float_like, Unary(state, spv::OpBitcast, u32, low),
				               Select(state, u32, signed_type,
				                      Unary(state, spv::OpBitcast, u32, extended),
				                      Binary(state, spv::OpBitwiseAnd, u32, value,
				                             ConstantU32(state, 0xffffu))));
			}
		}
		const auto bits       = field(info.widths, component * 6u, 6);
		const auto bit_offset = field(info.offsets, component * 7u, 7);
		const auto encoded    = EncodeDynamicBufferComponent(state, value, bits, info.type);
		const auto full       = Binary(state, spv::OpUGreaterThanEqual, boolean, bits, ConstantU32(state, 32));
		const auto field_mask = Select(
		    state, u32, full, ConstantU32(state, UINT32_MAX),
		    Binary(state, spv::OpISub, u32,
		           Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1),
		                  Select(state, u32, full, ConstantU32(state, 0), bits)),
		           ConstantU32(state, 1)));
		const auto exists = Binary(state, spv::OpULessThan, boolean, ConstantU32(state, component), count);
		const auto mask   = Select(state, u32, exists, field_mask, ConstantU32(state, 0));
		const auto bits_v = Binary(state, spv::OpBitwiseAnd, u32, encoded, mask);
		const auto position = Binary(state, spv::OpIAdd, u32,
		                             Binary(state, spv::OpShiftLeftLogical, u32, window.misalign,
		                                    ConstantU32(state, 3)),
		                             bit_offset);
		const auto index  = Binary(state, spv::OpShiftRightLogical, u32, position, ConstantU32(state, 5));
		const auto within = Binary(state, spv::OpBitwiseAnd, u32, position, ConstantU32(state, 31));
		const auto split  = Binary(state, spv::OpINotEqual, boolean, within, ConstantU32(state, 0));
		const auto back   = Binary(state, spv::OpISub, u32, ConstantU32(state, 32), within);
		const auto low_v  = Binary(state, spv::OpShiftLeftLogical, u32, bits_v, within);
		const auto low_m  = Binary(state, spv::OpShiftLeftLogical, u32, mask, within);
		const auto high_v = Select(state, u32, split,
		                           Binary(state, spv::OpShiftRightLogical, u32, bits_v, back),
		                           ConstantU32(state, 0));
		const auto high_m = Select(state, u32, split,
		                           Binary(state, spv::OpShiftRightLogical, u32, mask, back),
		                           ConstantU32(state, 0));
		const auto next = Binary(state, spv::OpIAdd, u32, index, ConstantU32(state, 1));
		for (uint32_t word = 0; word < word_data.size(); ++word) {
			const auto here  = Binary(state, spv::OpIEqual, boolean, index, ConstantU32(state, word));
			const auto after = Binary(state, spv::OpIEqual, boolean, next, ConstantU32(state, word));
			const auto add   = [&](uint32_t current, uint32_t low, uint32_t high) {
				return Binary(state, spv::OpBitwiseOr, u32, current,
				              Binary(state, spv::OpBitwiseOr, u32,
				                     Select(state, u32, here, low, ConstantU32(state, 0)),
				                     Select(state, u32, after, high, ConstantU32(state, 0))));
			};
			word_data[word] = add(word_data[word], low_v, high_v);
			word_mask[word] = add(word_mask[word], low_m, high_m);
		}
	}
	for (uint32_t word = 0; word < word_data.size(); ++word) {
		const auto touched = Binary(state, spv::OpINotEqual, boolean, word_mask[word], ConstantU32(state, 0));
		EmitIfCondition(state, AndCondition(state, window.usable[word], touched), [&]() {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
			                          window.pointer[word]);
			const auto whole = Binary(state, spv::OpIEqual, boolean, word_mask[word],
			                          ConstantU32(state, UINT32_MAX));
			EmitIfCondition(state, whole, [&]() {
				state.builder.AddFunction(spv::OpStore, pointer, word_data[word],
				                          spv::MemoryAccessAlignedMask, uint32_t {4});
			});
			EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, whole), [&]() {
				const auto scope     = ConstantU32(state, spv::ScopeDevice);
				const auto semantics = ConstantU32(state, spv::MemorySemanticsMaskNone);
				state.builder.AddFunction(spv::OpAtomicAnd, u32, state.builder.AllocateId(), pointer,
				                          scope, semantics,
				                          Unary(state, spv::OpNot, u32, word_mask[word]));
				state.builder.AddFunction(spv::OpAtomicOr, u32, state.builder.AllocateId(), pointer,
				                          scope, semantics, word_data[word]);
			});
		});
	}
}

// A call of the helper function holding LoadIndirectFormattedBody for this load's variant,
// created on first use and emitted after the entry function (EmitIndirectFormattedHelpers):
// the decoder inline at each of a hundred loads was megabytes of SPIR-V for the driver.
uint32_t LoadIndirectFormatted(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto&       state  = ctx.state;
	const auto& memory = ctx.Memory(inst);
	auto found = std::ranges::find_if(state.indirect_formatted_helpers, [&](const auto& helper) {
		return helper.components == components && helper.offset == memory.offset &&
		       helper.d16 == memory.d16;
	});
	if (found == state.indirect_formatted_helpers.end()) {
		state.indirect_formatted_helpers.push_back({.components = components,
		                                            .offset     = memory.offset,
		                                            .d16        = memory.d16,
		                                            .function   = state.builder.AllocateId()});
		found = std::prev(state.indirect_formatted_helpers.end());
	}
	const auto operands = IndirectOperands(ctx, inst);
	const auto result   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall,
	                          components == 1u ? TypeU32(state) : TypeU32Composite(state, components),
	                          result, found->function, operands.words[0], operands.words[1],
	                          operands.words[2], operands.words[3], operands.index, operands.offset,
	                          operands.soffset, operands.lane);
	return result;
}

} // namespace

void EmitIndirectFormattedHelpers(EmitterState& state) {
	const auto u32 = TypeU32(state);
	for (size_t i = 0; i < state.indirect_formatted_helpers.size(); i++) {
		const auto helper = state.indirect_formatted_helpers[i];
		const auto ret    = helper.components == 1u ? u32 : TypeU32Composite(state, helper.components);
		const auto type   = state.builder.Type(spv::OpTypeFunction, ret, u32, u32, u32, u32, u32, u32,
		                                       u32, u32);
		state.builder.AddFunction(spv::OpFunction, ret, helper.function, spv::FunctionControlMaskNone,
		                          type);
		std::array<uint32_t, 8> params {};
		for (auto& param: params) {
			param = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionParameter, u32, param);
		}
		EmitLabel(state, state.builder.AllocateId());
		const IndirectBufferOperands operands {.words   = {params[0], params[1], params[2], params[3]},
		                                       .index   = params[4],
		                                       .offset  = params[5],
		                                       .soffset = params[6],
		                                       .lane    = params[7]};
		const auto value =
		    LoadIndirectFormattedBody(state, operands, helper.offset, helper.components, helper.d16);
		state.builder.AddFunction(spv::OpReturnValue, value);
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
}

namespace {

uint32_t LoadBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	if (IsGenericFormattedLoad(ctx, ctx.Memory(inst))) {
		return EmitValueOrDefaultIfCondition(
		    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
		    ConstantU32CompositeZero(state, components),
		    [&]() { return LoadGenericFormatted(ctx, inst, ctx.Memory(inst), components); });
	}
	if (ctx.Memory(inst).kind == IR::ResourceKind::IndirectBuffer && !ctx.Memory(inst).formatted) {
		// The lane's EXEC bit gates the access with its bounds, without a branch of its own.
		return LoadIndirectBuffer(ctx, inst, components, ctx.Arg(inst, inst.NumArgs() - 1));
	}
	if (const auto& mem = ctx.Memory(inst);
	    IsStorageBufferKind(mem.kind) &&
	    Format::GetFormatInfo(mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid)
	            .type == Format::ComponentType::Unknown) {
		// Raw DWORDs of a storage buffer: EXEC selects the values, as in LoadWord.
		const auto              active   = ctx.Arg(inst, inst.NumArgs() - 1);
		const auto              resource = PrepareMemoryResourceAccess(state, mem);
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < components; component++) {
			values[component] =
			    Select(state, TypeU32(state), active,
			           LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource),
			           ConstantU32(state, 0));
		}
		return ConstructU32Composite(state, components, values);
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto mem      = ctx.Memory(inst);
		    if (mem.kind == IR::ResourceKind::IndirectBuffer) {
			    return LoadIndirectFormatted(ctx, inst, components);
		    }
		    const auto resource = PrepareMemoryResourceAccess(state, mem);
		    const auto info = Format::GetFormatInfo(
		        mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		    if (info.type != Format::ComponentType::Unknown) {
			    const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
			    if (!IsStorageBufferAccess(resource)) {
				    return EmitValueOrDefaultIfCondition(
				        state, plan.in_bounds, TypeU32Composite(state, components),
				        FormattedOutOfBoundsValue(ctx, mem, plan, components), [&]() {
					        std::array<uint32_t, 4> values {};
					        for (uint32_t component = 0; component < components; component++) {
						        values[component] = LoadFormattedInBounds(ctx, mem, plan, component);
						        if (mem.d16) {
							        values[component] = D16LoadComponent(ctx, info, values[component]);
						        }
					        }
					        return ConstructU32Composite(state, components, values);
				        });
			    }
			    // As in LoadElementOrZero: the elements are loaded at their own indices, and when
			    // any component is out of bounds each output takes the format's out-of-bounds value.
			    std::array<uint32_t, 4> values {};
			    for (uint32_t component = 0; component < components; component++) {
				    const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
				    values[component] =
				        Select(state, TypeU32(state), plan.in_bounds,
				               LoadFormattedInBounds(ctx, mem, plan, component),
				               FormattedConstant(ctx, plan.info, source.kind));
				    if (mem.d16) {
					    values[component] = D16LoadComponent(ctx, info, values[component]);
				    }
			    }
			    return ConstructU32Composite(state, components, values);
		    }
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    values[component] =
			        LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem       = ctx.Memory(inst);
		const auto resource  = PrepareMemoryResourceAccess(state, mem);
		const auto composite = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info = Format::GetFormatInfo(
		    mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		if (info.type != Format::ComponentType::Unknown) {
			StoreFormattedPrepared(ctx, inst, mem, resource, info, composite, components);
			return;
		}
		for (uint32_t component = 0; component < components; component++) {
			const auto data = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), data, composite,
			                          component);
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource, data);
		}
	});
}

uint32_t LoadWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	if (const auto& mem = ctx.Memory(inst); HasElementZero(state, mem.kind)) {
		const auto              resource = PrepareMemoryResourceAccess(state, mem);
		const auto              base     = ByteAddress(ctx, inst, mem);
		const auto              active   = ctx.Arg(inst, inst.NumArgs() - 1);
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                          ConstantU32(state, 2));
			values[component] = LoadLocalWordOrZero(
			    ctx, resource, index,
			    AndCondition(state, active, EmitMemoryElementInBounds(state, resource, index)));
		}
		return ConstructU32Composite(state, components, values);
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto              mem      = ctx.Memory(inst);
		    const auto              resource = PrepareMemoryResourceAccess(state, mem);
		    const auto              base     = ByteAddress(ctx, inst, mem);
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    const auto address   = component == 0u
			                               ? base
			                               : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                        ConstantU32(state, component * 4u));
			    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
			                                  address, ConstantU32(state, 2));
			    values[component]    = EmitValueOrZeroIfCondition(
			        state, EmitMemoryElementInBounds(state, resource, index),
			        [&]() { return LoadWordInBounds(ctx, resource, index); });
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	if (const auto& mem = ctx.Memory(inst); HasElementZero(state, mem.kind)) {
		// One branch per DWORD for EXEC and its bounds together.
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto base     = ByteAddress(ctx, inst, mem);
		const auto active   = ctx.Arg(inst, inst.NumArgs() - 1);
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                          ConstantU32(state, 2));
			const auto data  = ctx.Arg(inst, component + 1u);
			EmitIfCondition(state,
			                AndCondition(state, active, EmitMemoryElementInBounds(state, resource, index)),
			                [&]() { StoreWordInBounds(ctx, resource, index, data); });
		}
		return;
	}
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem      = ctx.Memory(inst);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto base     = ByteAddress(ctx, inst, mem);
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                              ConstantU32(state, 2));
			EmitIfCondition(state, EmitMemoryElementInBounds(state, resource, index),
			                [&]() {
				                StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, component + 1u));
			                });
		}
	});
}

} // namespace

uint32_t ConstantDeviceAddress(EmitterState& state, uint64_t value) {
	return state.builder.Constant(spv::OpConstant, TypeScalarU64(state),
	                              static_cast<uint32_t>(value),
	                              static_cast<uint32_t>(value >> 32u));
}

uint32_t DeviceAddressFromWords(EmitterState& state, uint32_t low, uint32_t high) {
	const auto low64  = Unary(state, spv::OpUConvert, TypeScalarU64(state), low);
	const auto high64 = Binary(state, spv::OpShiftLeftLogical, TypeScalarU64(state),
	                           Unary(state, spv::OpUConvert, TypeScalarU64(state), high),
	                           ConstantDeviceAddress(state, 32));
	return Binary(state, spv::OpBitwiseOr, TypeScalarU64(state), low64, high64);
}

uint32_t GetBdaPointer(EmitterState& state, uint32_t address) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), result,
	                          state.bda_pointer_function, address);
	return result;
}

uint32_t GetBdaLoadPointer(EmitterState& state, uint32_t address, uint32_t active) {
	if (state.bda_load_pointer_function == 0) {
		const auto pointer = GetBdaPointer(state, address);
		(void)active;
		return pointer;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), result,
	                          state.bda_load_pointer_function, address, active);
	return result;
}

bool BdaFastDwordEnabled() {
	// KYTY_BDA_FAST_DWORD=0: element windows test every DWORD's page and size at run time.
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BDA_FAST_DWORD");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

bool BdaNoWordMasksEnabled() {
	// KYTY_BDA_NO_WORD_MASKS=0: with the zero page, element words are still masked one by one.
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BDA_NO_WORD_MASKS");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

bool BdaSingleCrossBranchEnabled(uint64_t shader_hash) {
	// KYTY_BDA_SINGLE_CROSS_BRANCH=1 (opt-in): one branch per element window on crossing a page;
	// =2: the crossing block chooses pages with bit masks. Off by default: the driver's compiler took
	// longer with either (3e82 35.0 / 27.9 s, c089 88.3 / 87.4 s, 15f4 113.8 / 111.6 s) than with
	// the second lookup and the per-DWORD page selects at every access (21.7, 71.7, 100.8 s).
	// KYTY_BDA_SINGLE_CROSS_SHADERS=<hash>[,...]: =1 for those shaders only.
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BDA_SINGLE_CROSS_BRANCH");
		return text != nullptr && (std::strcmp(text, "1") == 0 || std::strcmp(text, "2") == 0);
	}();
	static const Common::Switches::HashList hashes(KYTY_GETENV("KYTY_BDA_SINGLE_CROSS_SHADERS"));
	return enabled || hashes.Contains(shader_hash);
}

bool BdaCrossMuxEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BDA_SINGLE_CROSS_BRANCH");
		return text != nullptr && std::strcmp(text, "2") == 0;
	}();
	return enabled;
}

bool BdaZeroPageEnabled() {
	// KYTY_BDA_ZERO_PAGE=0: loads by guest address branch on the page being mapped, as stores do.
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BDA_ZERO_PAGE");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	return enabled;
}

namespace {

// get_bda_pointer (load == false): the device address of a guest address, 0 where its page is not
// mapped. get_bda_load_pointer (load == true): a readable device address even there, inside the
// zero page whose address the table holds after its last entry (BufferCache::BDA_ZERO_PAGE_ENTRY),
// so loads need no branch around the access and no phi. Its `active` parameter only gates the
// fault record: lanes that do not access must not fault their (arbitrary) pages in.
uint32_t DefineBdaLookup(EmitterState& state, bool load) {
	const auto type          = TypeScalarU64(state);
	const auto function_type = load ? state.builder.Type(spv::OpTypeFunction, type, type, TypeBool(state))
	                                : state.builder.Type(spv::OpTypeFunction, type, type);
	const auto function    = state.builder.AllocateId();
	const auto address     = state.builder.AllocateId();
	const auto active      = load ? state.builder.AllocateId() : 0u;
	const auto entry_label = state.builder.AllocateId();
	state.builder.AddName(function, load ? "get_bda_load_pointer" : "get_bda_pointer");
	state.builder.AddFunction(spv::OpFunction, type, function, spv::FunctionControlMaskNone,
	                          function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, type, address);
	if (load) {
		state.builder.AddFunction(spv::OpFunctionParameter, TypeBool(state), active);
	}
	EmitLabel(state, entry_label);

	const auto extended = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address,
	                             ConstantDeviceAddress(state, LibKernel::Memory::kExtendedMemoryBase));
	const auto packed = Select(state, type, extended,
	                           Binary(state, spv::OpISub, type, address,
	                                  ConstantDeviceAddress(state,
	                                      LibKernel::Memory::kExtendedMemoryBase - LOWER_ADDRESS_SIZE)),
	                           address);
	const auto page64        = Binary(state, spv::OpShiftRightLogical, type, packed,
	                                  ConstantDeviceAddress(state, BufferCache::CACHING_PAGEBITS));
	const auto page          = Unary(state, spv::OpUConvert, TypeU32(state), page64);
	// An address outside guest memory (a load whose address the GPU computes can produce one)
	// must not index past the page table or the fault bitmap: that faults the device. One between
	// the lower range and the extended base must not take an extended page's entry either.
	const auto in_range = Binary(
	    state, spv::OpLogicalAnd, TypeBool(state),
	    Binary(state, spv::OpLogicalOr, TypeBool(state), extended,
	           Binary(state, spv::OpULessThan, TypeBool(state), address,
	                  ConstantDeviceAddress(state, LOWER_ADDRESS_SIZE))),
	    Binary(state, spv::OpULessThan, TypeBool(state), page64,
	           ConstantDeviceAddress(state, BufferCache::CACHING_NUMPAGES)));
	// The driver inlines this at every access, so it branches only to record a fault: a page out
	// of range reads the table's first entry instead and gives zero (the zero page's entry for a
	// load).
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpAccessChain, TypeStorageBufferElementPointer(state, 64), entry_pointer,
	    state.bda_pagetable_variable, ConstantU32(state, 0),
	    Select(state, TypeU32(state), in_range, page,
	           ConstantU32(state, load ? static_cast<uint32_t>(BufferCache::BDA_ZERO_PAGE_ENTRY) : 0u)));
	const auto base = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, base, entry_pointer);
	const auto mapped =
	    Binary(state, spv::OpINotEqual, TypeBool(state), base, ConstantDeviceAddress(state, 0));
	auto missing = Binary(state, spv::OpLogicalAnd, TypeBool(state), in_range,
	                      Unary(state, spv::OpLogicalNot, TypeBool(state), mapped));
	if (load) {
		missing = Binary(state, spv::OpLogicalAnd, TypeBool(state), active, missing);
	}
	EmitIfCondition(state, missing, [&]() {
		RecordBdaFault(state, page);
		if (state.bda_fault_seen_variable != 0) {
			state.builder.AddFunction(spv::OpStore, state.bda_fault_seen_variable,
			                          ConstantU32(state, 1));
		}
	});
	const auto offset = Binary(state, spv::OpBitwiseAnd, type, address,
	                           ConstantDeviceAddress(state, BufferCache::CACHING_PAGESIZE - 1));
	uint32_t result = 0;
	if (load) {
		// Out of range the entry is the zero page's already; the zero page's address was read once
		// at shader entry.
		const auto zero_page = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, type, zero_page, state.bda_zero_page_variable);
		result = Binary(state, spv::OpIAdd, type, Select(state, type, mapped, base, zero_page), offset);
	} else {
		result = Select(state, type, Binary(state, spv::OpLogicalAnd, TypeBool(state), in_range, mapped),
		                Binary(state, spv::OpIAdd, type, base, offset), ConstantDeviceAddress(state, 0));
	}
	state.builder.AddFunction(spv::OpReturnValue, result);
	state.builder.AddFunction(spv::OpFunctionEnd);
	return function;
}

} // namespace

void DefineGetBdaPointer(EmitterState& state) {
	if (!state.program.info.uses_dma) {
		return;
	}
	state.bda_pointer_function = DefineBdaLookup(state, false);
	if (BdaZeroPageEnabled()) {
		state.bda_zero_page_variable = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state)), spv::StorageClassPrivate);
		state.builder.AddName(state.bda_zero_page_variable, "bda_zero_page");
		state.bda_load_pointer_function = DefineBdaLookup(state, true);
	}
}

void EmitBdaZeroPageInit(EmitterState& state) {
	if (state.bda_zero_page_variable == 0) {
		return;
	}
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state, 64),
	                          entry_pointer, state.bda_pagetable_variable, ConstantU32(state, 0),
	                          ConstantU32(state, static_cast<uint32_t>(BufferCache::BDA_ZERO_PAGE_ENTRY)));
	const auto zero_page = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeScalarU64(state), zero_page, entry_pointer);
	state.builder.AddFunction(spv::OpStore, state.bda_zero_page_variable, zero_page);
}

// A 32-bit atomic through a GPU-selected descriptor: on the guest address, zero where the
// access is out of bounds or the page is not mapped.
uint32_t IndirectBufferAtomic32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto buffer = PrepareIndirectBuffer(ctx, inst);
		const auto valid_format =
		    Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format, ConstantU32(state, 0));
		const auto in_bounds =
		    AndCondition(state, valid_format, IndirectBufferInBounds(state, buffer, 0u, 4u, false));
		return EmitValueOrZeroIfCondition(state, in_bounds, [&]() {
			const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), buffer.address,
			                            ConstantDeviceAddress(state, ~uint64_t {3}));
			const auto bda     = GetBdaPointer(state, aligned);
			const auto present =
			    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantDeviceAddress(state, 0));
			return EmitValueOrZeroIfCondition(state, present, [&]() {
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
				                          bda);
				const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
				EmitAtomicMemoryBarrier(state, IR::ResourceKind::Buffer);
				return old;
			});
		});
	});
}

uint32_t EmitAtomic32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem = ctx.Memory(inst);
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		return IndirectBufferAtomic32(ctx, inst);
	}
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		const auto scope =
		    mem.kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
		const auto old = EmitAtomicOperation(ctx, inst, pointer, scope);
		EmitAtomicMemoryBarrier(ctx.state, mem.kind);
		return old;
	});
}

uint32_t EmitBufferAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem   = ctx.Memory(inst);
	auto&       state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU64(state), ConstantU64(state, 0), [&]() {
		    const auto resource = PrepareStorageBufferResourceAccess(
		        state, mem, state.storage_buffer_u64_variable, TypeStorageBufferPointer(state, 64));
		    const auto byte_address = ByteAddress(ctx, inst, mem);
		    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte_address,
		                              ConstantU32(state, 3u));
		    return EmitValueOrDefaultIfCondition(
		        state, EmitMemoryElementInBounds(state, resource, index), TypeU64(state),
		        ConstantU64(state, 0), [&]() {
			        const auto pointer = EmitStorageBufferElementPointer(
			            state, resource, index, TypeStorageBufferElementPointer(state, 64));
			        const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
			        EmitAtomicMemoryBarrier(state, mem.kind);
			        MarkTrackedWrite(state, resource, index, 64u);
			        return old;
		        });
	    });
}

void EmitSharedAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	EnsureLdsStorage(state);
	EmitIfCondition(state, ctx.Arg(inst, 2), [&]() {
		const auto address = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                            ByteAddress(ctx, inst, mem), ConstantU32(state, 0xfff8u));
		const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
		                          ConstantU32(state, 3u));
		const auto in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index,
		                              ConstantU32(state, LdsDwordCount(state) / 2u));
		EmitIfCondition(state, in_bounds, [&]() {
			auto native_index = index;
			auto semantics = spv::MemorySemanticsWorkgroupMemoryMask;
			if (state.lds_storage_class == spv::StorageClassStorageBuffer) {
				native_index = EmitAddU32(state, index, EmitBinaryU32(
				    state, spv::OpShiftRightLogical, state.lds_base_dwords, ConstantU32(state, 1)));
				semantics = spv::MemorySemanticsUniformMemoryMask;
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, state.lds_storage_class, TypeScalarU64(state)),
			                          pointer, state.lds_u64_variable, ConstantU32(state, 0), native_index);
			const auto value = Unary(state, spv::OpBitcast, TypeScalarU64(state), ctx.Arg(inst, 1));
			state.builder.AddFunction(
			    SpirvAtomicOpcode(inst.GetOpcode()), TypeScalarU64(state), state.builder.AllocateId(),
			    pointer, ConstantU32(state, spv::ScopeWorkgroup),
			    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | semantics), value);
		});
	});
}

uint32_t EmitBufferFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::BufferAtomicFMax32;
	return EmitAtomicUpdate(ctx, inst, mem,
	                        [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                        return EmitFloatAtomicReplacement(state, old, value, max_value);
	                        });
}

void EmitSharedFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::SharedAtomicFMax32;
	EmitAtomicUpdate(ctx, inst, mem,
	                 [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                 return EmitDsFloatAtomicReplacement(state, old, value, max_value);
	                 });
}

uint32_t EmitAppendConsume(ValueEmitContext& ctx, const IR::Inst& inst) {
	const bool append = inst.GetOpcode() == IR::ValueOpcode::DataAppend;
	auto&      state  = ctx.state;
	if (ctx.half == 1) {
		return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
	}
	const auto mem           = ctx.Memory(inst);
	const auto offset        = mem.offset & 0xfffcu;
	auto       address       = ConstantU32(state, offset);
	uint32_t   region_bounds = 0;
	if (mem.kind == IR::ResourceKind::Gds) {
		const auto m0 = ctx.Arg(inst, 0);
		const auto base =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), m0, ConstantU32(state, 16));
		const auto size =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), m0, ConstantU32(state, 0xffffu));
		address = Binary(state, spv::OpIAdd, TypeU32(state), base, address);
		// The entire M0 region must fit the PS5's 48 KiB GDS partition.
		region_bounds = AndCondition(
		    state, Binary(state, spv::OpULessThan, TypeBool(state),
		                  ConstantU32(state, offset + 3u), size),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state),
		           Binary(state, spv::OpIAdd, TypeU32(state), base, size),
		           ConstantU32(state, 0xc000u)));
	}
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
	const auto access = PrepareMemoryResourceAccess(state, mem);
	const auto exec   = ctx.Arg(inst, 1);
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto low    = state.builder.AllocateId();
	const auto high   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	const auto count =
	    WaveHalvesInHostSubgroup(state)
	        ? Unary(state, spv::OpBitCount, TypeU32(state), EmitOwnWaveHalfWord(state, ballot))
	        : Binary(state, spv::OpIAdd, TypeU32(state),
	                 Unary(state, spv::OpBitCount, TypeU32(state), low),
	                 Unary(state, spv::OpBitCount, TypeU32(state), high));
	const auto first = ctx.FirstLane(ballot);
	const auto source_lane =
	    state.lane_count == 2
	        ? Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 31))
	        : first;
	const auto is_first       = Binary(state, spv::OpIEqual, TypeBool(state),
	                                   EmitSubgroupLocalInvocationId(state), source_lane);
	auto bounds = EmitMemoryElementInBounds(state, access, index);
	if (mem.kind == IR::ResourceKind::Gds) {
		bounds = AndCondition(state, bounds, region_bounds);
	}
	const auto condition = AndCondition(
	    state, is_first,
	    AndCondition(state,
	                 state.lane_count == 2 || state.split_wave
	                     ? Binary(state, spv::OpINotEqual, TypeBool(state), count,
	                              ConstantU32(state, 0))
	                     : exec,
	                 bounds));
	const auto atomic = EmitValueOrZeroIfCondition(state, condition, [&]() {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(append ? spv::OpAtomicIAdd : spv::OpAtomicISub, TypeU32(state),
		                          value, EmitMemoryElementPointer(state, access, index),
		                          ConstantU32(state, mem.kind == IR::ResourceKind::Gds
		                                                 ? spv::ScopeDevice
		                                                 : spv::ScopeWorkgroup),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), count);
		return value;
	});
	// A split wave's first active lane may be in the other host subgroup.
	if (state.split_wave) {
		return ExchangeWaveValue(state, atomic, source_lane);
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), atomic, source_lane);
	return result;
}

uint32_t EmitReadConst(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.program.bindings.table) {
		return LoadFlattenedSrtWord(state, ctx.Arg(inst, 1));
	}
	if (state.flattened_srt_variable == 0) {
		ctx.Fail(inst, "requires the flattened SRT descriptor");
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.flattened_srt_variable, ConstantU32(state, 0),
	                          ctx.Arg(inst, 1));
	return EmitNative<spv::OpLoad, IR::Type::U32>(state, pointer);
}

void EmitReadConstBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto mem = ctx.Memory(inst);
	if (mem.planning_only) return;
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		ctx.Define(inst, LoadIndirectScalarBuffer(ctx, inst));
		return;
	}
	auto& state        = ctx.state;
	mem.kind           = IR::ResourceKind::ScalarBuffer;
	auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), ctx.Arg(inst, 1),
	                    ConstantU32(state, 2));
	if (mem.offset >= 4u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, ConstantU32(state, mem.offset >> 2u));
	}
	const auto access    = PrepareMemoryResourceAccess(state, mem);
	// Scalar buffer addressing aligns the base and each offset independently.
	auto element = Binary(
	    state, spv::OpIAdd, TypeU32(state), index,
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.byte_offset, ConstantU32(state, 2)));
	auto condition = EmitWordAccessInBounds(state, access, element);
	if (HardwareStorageBufferBounds() && !access.physical) {
		// SOFFSET and the immediate are aligned and added without wrapping, so an offset carry
		// past 4 GiB gives an element of 2^30 or more. Hardware bounds check the element's byte
		// offset plus size in 32 bits, which would wrap back into the buffer: reject those here,
		// and the last elements below 4 GiB, whose end wraps.
		const auto below = Binary(state, spv::OpULessThan, TypeBool(state), element,
		                          ConstantU32(state, HardwareBoundsByteCap / 4u));
		condition = condition == ConstantBool(state, true) ? below
		                                                   : AndCondition(state, condition, below);
		// A storage buffer load runs unconditionally and only its value is selected (see
		// LoadElementOrZero): the rejected element must not reach the access either.
		element = EmitGlsl<GLSLstd450UMin, IR::Type::U32>(
		    state, element, ConstantU32(state, HardwareBoundsByteCap / 4u));
	}
	ctx.Define(inst, LoadElementOrZero(state, access, condition, element, [&](uint32_t target) {
		           if (access.physical) {
			           const auto value = state.builder.AllocateId();
			           EmitBufferElementLoad(state, TypeU32(state), value,
			                                 EmitMemoryElementPointer(state, access, target), access);
			           return value;
		           }
		           return EmitNative<spv::OpLoad, IR::Type::U32>(
		               state, EmitMemoryElementPointer(state, access, target));
	           }));
}

void EmitLoadMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op  = inst.GetOpcode();
	const auto& mem = ctx.Memory(inst);
	if (op == IR::ValueOpcode::LoadAddressU32 && mem.planning_only) return;
	const auto buffer_components = IR::BufferComponentCount(op);
	const auto shared_components = IR::SharedComponentCount(op);
	const auto address_info      = IR::AddressOpcodeInfoOf(op);
	uint32_t   value;
	if (mem.kind == IR::ResourceKind::FlatLocal)
		value = LoadLocalFlat(ctx, inst);
	else if (buffer_components > 1u ||
	         (buffer_components == 1u && mem.kind == IR::ResourceKind::IndirectBuffer && !mem.formatted &&
	          inst.GetType() != IR::Type::U8 && inst.GetType() != IR::Type::U16))
		// U8/U16 indirect loads take LoadIndirectSubword below: the dword path read whole words
		// for byte/short loads (blocky Yotei roofs).
		value = LoadBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		value = LoadWideShared(ctx, inst, shared_components);
	else if (mem.kind == IR::ResourceKind::ScalarAddress)
		value = LoadBdaDword(ctx, GuestAddress(ctx, inst, mem));
	else if (address_info.access == IR::AddressAccess::Read &&
	         mem.kind != IR::ResourceKind::Scratch)
		value = LoadBda(ctx, GuestAddress(ctx, inst, mem), ctx.Arg(inst, inst.NumArgs() - 1),
		                address_info.data_bits, mem.coherent);
	else if (op == IR::ValueOpcode::LoadBufferU32 && mem.formatted)
		// A per-format switch at each load made shaders with many such loads (Yotei's tile
		// lighting: 16 X loads) grow to megabytes of SPIR-V; the table decoder is shared.
		value = mem.kind != IR::ResourceKind::IndirectBuffer
		            ? FormattedLoad(ctx, inst, mem)
		            : EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1),
		                                         [&]() { return LoadIndirectFormatted(ctx, inst, 1u); });
	else if (mem.kind == IR::ResourceKind::IndirectBuffer &&
	         (inst.GetType() == IR::Type::U8 || inst.GetType() == IR::Type::U16))
		value = LoadIndirectSubword(ctx, inst, inst.GetType() == IR::Type::U8 ? 8u : 16u);
	else if (inst.GetType() == IR::Type::U8)
		value = LoadSubword(ctx, inst, mem, 8, false);
	else if (inst.GetType() == IR::Type::U16)
		value = LoadSubword(ctx, inst, mem, 16, false);
	else
		value = LoadWord(ctx, inst, mem);
	ctx.Define(inst, value);
}

void EmitStoreMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op                = inst.GetOpcode();
	const auto& mem               = ctx.Memory(inst);
	const auto  buffer_components = IR::BufferComponentCount(op);
	const auto  shared_components = IR::SharedComponentCount(op);
	const auto  type              = inst.Arg(inst.NumArgs() - 2).GetType();
	if (mem.kind == IR::ResourceKind::IndirectBuffer && mem.formatted)
		StoreIndirectFormatted(ctx, inst, std::max(buffer_components, 1u));
	else if (mem.kind == IR::ResourceKind::IndirectBuffer)
		StoreIndirectBuffer(ctx, inst, std::max(buffer_components, 1u),
		                    type == IR::Type::U8 ? 8u : type == IR::Type::U16 ? 16u : 32u);
	else if (mem.kind == IR::ResourceKind::FlatLocal)
		StoreLocalFlat(ctx, inst);
	else if (buffer_components > 1u)
		StoreWideBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		StoreWideShared(ctx, inst, shared_components);
	else if (op == IR::ValueOpcode::StoreBufferU32 && mem.formatted)
		FormattedStore(ctx, inst, mem);
	else if (type == IR::Type::U8)
		StoreSubword(ctx, inst, mem, 8);
	else if (type == IR::Type::U16)
		StoreSubword(ctx, inst, mem, 16);
	else
		StoreWord(ctx, inst, mem);
}

uint32_t EmitSharedIncDec(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto replacement =
	    inst.GetOpcode() == IR::ValueOpcode::SharedAtomicInc32 ? AtomicIncrement : AtomicDecrement;
	return EmitAtomicUpdate(ctx, inst, ctx.Memory(inst), replacement);
}

void EmitSharedAtomicMaskedOr32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto keep = Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), ctx.Arg(inst, 1));
	EmitAtomicUpdate(ctx, inst, ctx.Memory(inst),
	                 [keep](EmitterState& state, uint32_t old, uint32_t value) {
		                 return Binary(state, spv::OpBitwiseOr, TypeU32(state),
		                               Binary(state, spv::OpBitwiseAnd, TypeU32(state), old, keep), value);
	                 });
}

uint32_t EmitSwizzleU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	state.builder.AddFunction(spv::OpStore, ctx.scratch_u32_variable, ctx.Arg(inst, 0));
	const auto source = EmitNative<spv::OpLoad, IR::Type::U32>(state, ctx.scratch_u32_variable);
	const auto target = EmitDsSwizzleTargetLane(state, EmitSubgroupLocalInvocationId(state),
	                                            inst.Arg(1).IsImmediate() ? inst.Arg(1).U32() : 0);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

uint32_t EmitPermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state   = ctx.state;
	auto       lane    = EmitSubgroupLocalInvocationId(state);
	if (state.lane_count == 2 || state.split_wave) {
		lane = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane, ConstantU32(state, 31));
	}
	const auto address = ctx.Arg(inst, 1);
	const auto word    = Binary(state, spv::OpShiftRightLogical, TypeU32(state), lane,
	                            ConstantU32(state, 5));
	const auto ballot_word = [&](uint32_t predicate) {
		const auto ballot = state.builder.AllocateId();
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), predicate);
		state.builder.AddFunction(spv::OpVectorExtractDynamic, TypeU32(state), result, ballot, word);
		return result;
	};
	// RDNA2 permutes independently within each 32-lane half. Intersect the source
	// address bit ballots to find this destination's enabled writers without LDS.
	auto writers = ballot_word(ctx.Arg(inst, 2));
	for (uint32_t bit = 0; bit < 5; ++bit) {
		const auto address_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), address,
		                                ConstantU32(state, 1u << (bit + 2)));
		const auto mask = ballot_word(Binary(state, spv::OpINotEqual, TypeBool(state),
		                                      address_bit, ConstantU32(state, 0)));
		const auto lane_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
		                             ConstantU32(state, 1u << bit));
		const auto selected = Select(
		    state, TypeU32(state),
		    Binary(state, spv::OpINotEqual, TypeBool(state), lane_bit, ConstantU32(state, 0)),
		    mask, Unary(state, spv::OpNot, TypeU32(state), mask));
		writers = Binary(state, spv::OpBitwiseAnd, TypeU32(state), writers, selected);
	}
	const auto active = Binary(state, spv::OpINotEqual, TypeBool(state), writers,
	                           ConstantU32(state, 0));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
	                         ConstantU32(state, ~31u));
	const auto source = Select(state, TypeU32(state), active,
	                           Binary(state, spv::OpBitwiseOr, TypeU32(state), base,
	                                  EmitFindUMsb32(state, writers)), lane);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), ctx.Arg(inst, 0), source);
	return Select(state, TypeU32(state), active, result, ConstantU32(state, 0));
}

uint32_t EmitBpermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto source = ctx.Arg(inst, 0);
	const auto index  = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                                  ctx.Arg(inst, 1), ConstantU32(state, 2)),
	                           ConstantU32(state, 31));
	const auto base   = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           EmitSubgroupLocalInvocationId(state), ConstantU32(state, ~31u));
	const auto target = Binary(state, spv::OpBitwiseOr, TypeU32(state), base, index);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

// The format words of IR::GenericBufferFormat buffers (the V#'s format and dst_sel), decoded once
// at function entry for every formatted load of the buffer: the layout from the shared format
// tables (LookupDynamicBufferFormat), and per output component its source, width, bit offset,
// conversion maxima and constant.
void EmitGenericBufferFormats(EmitterState& state) {
	if (state.program.bindings.buffer_formats.empty()) {
		return;
	}
	const auto u32     = TypeU32(state);
	const auto f32     = TypeF32(state);
	const auto boolean = TypeBool(state);
	for (uint32_t k = 0; k < state.program.bindings.buffer_formats.size(); k++) {
		const auto i     = state.program.bindings.buffer_formats[k];
		const auto slot  = EmitShaderDataDwordLoad(state, state.program.bindings.BufferFormatDword() + k);
		const auto field = [&](uint32_t value, uint32_t first, uint32_t count) {
			return EmitBitFieldUExtract(state, value, ConstantU32(state, first), ConstantU32(state, count));
		};
		if ((state.program.info.buffers[i].packed_stride & IR::DynamicBufferStride) != 0u) {
			state.buffer_strides[i] = field(slot, 0, 14);
		}
		// RuntimeBufferSlotWord's base-6 selectors back to 3-bit ones: format | dst_sel << 7.
		auto       selectors = field(slot, 21, 11);
		auto       word      = field(slot, 14, 7);
		for (uint32_t component = 0; component < 4u; ++component) {
			const auto digit = Binary(state, spv::OpUMod, u32, selectors, ConstantU32(state, 6));
			selectors        = Binary(state, spv::OpUDiv, u32, selectors, ConstantU32(state, 6));
			const auto selector = Select(
			    state, u32, Binary(state, spv::OpUGreaterThanEqual, boolean, digit, ConstantU32(state, 2)),
			    Binary(state, spv::OpIAdd, u32, digit, ConstantU32(state, 2)), digit);
			word = Binary(state, spv::OpBitwiseOr, u32, word,
			              Binary(state, spv::OpShiftLeftLogical, u32, selector,
			                     ConstantU32(state, 7u + component * 3u)));
		}
		const auto info      = LookupDynamicBufferFormat(state, field(word, 0, 7));
		const auto count     = EmitUMax32(state, field(info.layout, 0, 3), ConstantU32(state, 1));
		const auto byte_size = EmitUMax32(state, field(info.layout, 3, 5), ConstantU32(state, 1));
		const auto is_type   = [&](Format::ComponentType type) {
			return Binary(state, spv::OpIEqual, boolean, info.type,
			              ConstantU32(state, static_cast<uint32_t>(type)));
		};
		auto& format      = state.generic_buffer_formats[i];
		format.end_offset = Binary(state, spv::OpISub, u32, byte_size, ConstantU32(state, 1));
		format.align_mask = Unary(state, spv::OpNot, u32,
		                          Binary(state, spv::OpISub, u32, EmitUMin32(state, byte_size, ConstantU32(state, 4)),
		                                 ConstantU32(state, 1)));
		format.valid      = info.valid;
		format.one        = Select(state, u32,
		                           Binary(state, spv::OpLogicalOr, boolean, is_type(Format::ComponentType::Uint),
		                                  is_type(Format::ComponentType::Sint)),
		                           ConstantU32(state, 1), ConstantU32(state, 0x3f800000));
		const auto is_float = is_type(Format::ComponentType::Float);
		format.is_sint      = is_type(Format::ComponentType::Sint);
		format.is_uscaled   = is_type(Format::ComponentType::Uscaled);
		format.is_sscaled   = is_type(Format::ComponentType::Sscaled);
		format.is_unorm     = is_type(Format::ComponentType::Unorm);
		format.is_snorm     = is_type(Format::ComponentType::Snorm);
		format.float_like   = Binary(state, spv::OpLogicalOr, boolean, is_float,
		                             Binary(state, spv::OpLogicalOr, boolean, format.is_unorm, format.is_snorm));
		for (uint32_t output = 0; output < 4u; ++output) {
			const auto selector = field(word, 7u + output * 3u, 3);
			format.from_memory[output] = AndCondition(
			    state, info.valid, Binary(state, spv::OpUGreaterThanEqual, boolean, selector, ConstantU32(state, 4)));
			format.constant_one[output] = AndCondition(
			    state, info.valid, Binary(state, spv::OpIEqual, boolean, selector, ConstantU32(state, 1)));
			// As Format::ResolveFormattedSource: channels repeat past the format's component count.
			const auto component = Binary(
			    state, spv::OpUMod, u32, Binary(state, spv::OpBitwiseAnd, u32, selector, ConstantU32(state, 3)),
			    count);
			const auto bits = EmitUMax32(
			    state,
			    EmitBitFieldUExtract(state, info.widths,
			                         Binary(state, spv::OpIMul, u32, component, ConstantU32(state, 6)),
			                         ConstantU32(state, 6)),
			    ConstantU32(state, 1));
			format.bits[output]       = bits;
			format.bit_offset[output] = EmitBitFieldUExtract(
			    state, info.offsets, Binary(state, spv::OpIMul, u32, component, ConstantU32(state, 7)),
			    ConstantU32(state, 7));
			const auto mask = Binary(state, spv::OpShiftRightLogical, u32, ConstantU32(state, UINT32_MAX),
			                         Binary(state, spv::OpISub, u32, ConstantU32(state, 32), bits));
			format.unsigned_max[output] = Unary(state, spv::OpConvertUToF, f32, mask);
			format.signed_max[output]   = Unary(state, spv::OpConvertUToF, f32,
			                                    Binary(state, spv::OpShiftRightLogical, u32, mask, ConstantU32(state, 1)));
			format.half[output]         = AndCondition(
			    state, is_float, Binary(state, spv::OpIEqual, boolean, bits, ConstantU32(state, 16)));
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
