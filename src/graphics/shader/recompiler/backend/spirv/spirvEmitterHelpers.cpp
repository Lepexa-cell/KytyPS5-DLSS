#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "common/switches.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

DppTargetLane EmitDppPermTargetLane(EmitterState& state, uint32_t subid, uint32_t control,
                                    uint32_t lane_bits) {
	const auto lane_mask  = (1u << lane_bits) - 1u;
	const auto group_base = state.builder.AllocateId();
	const auto lane       = state.builder.AllocateId();
	const auto shift      = state.builder.AllocateId();
	const auto selected0  = state.builder.AllocateId();
	const auto selected   = state.builder.AllocateId();
	const auto target     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), group_base, subid,
	                          ConstantU32(state, ~lane_mask));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, lane_mask));
	state.builder.AddFunction(spv::OpIMul, TypeU32(state), shift, lane,
	                          ConstantU32(state, lane_bits));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), selected0,
	                          ConstantU32(state, control), shift);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), selected, selected0,
	                          ConstantU32(state, lane_mask));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, group_base, selected);
	return {target, ConstantBool(state, true)};
}

DppTargetLane EmitDppRowShiftTargetLane(EmitterState& state, uint32_t subid, uint32_t amount,
                                        bool left) {
	const auto row          = state.builder.AllocateId();
	const auto lane         = state.builder.AllocateId();
	const auto lane_shifted = state.builder.AllocateId();
	const auto target       = state.builder.AllocateId();
	const auto valid        = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	if (left) {
		state.builder.AddFunction(spv::OpIAdd, TypeU32(state), lane_shifted, lane,
		                          ConstantU32(state, amount));
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), valid, lane,
		                          ConstantU32(state, 16u - amount));
	} else {
		state.builder.AddFunction(spv::OpISub, TypeU32(state), lane_shifted, lane,
		                          ConstantU32(state, amount));
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), valid, lane,
		                          ConstantU32(state, amount));
	}
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row, lane_shifted);
	return {target, valid};
}

DppTargetLane EmitDppRowRotateRightTargetLane(EmitterState& state, uint32_t subid,
                                              uint32_t amount) {
	const auto row      = state.builder.AllocateId();
	const auto lane     = state.builder.AllocateId();
	const auto in_high  = state.builder.AllocateId();
	const auto minus    = state.builder.AllocateId();
	const auto plus     = state.builder.AllocateId();
	const auto selected = state.builder.AllocateId();
	const auto target   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), in_high, lane,
	                          ConstantU32(state, amount));
	state.builder.AddFunction(spv::OpISub, TypeU32(state), minus, lane, ConstantU32(state, amount));
	state.builder.AddFunction(spv::OpIAdd, TypeU32(state), plus, lane,
	                          ConstantU32(state, 16u - amount));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, in_high, minus, plus);
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row, selected);
	return {target, ConstantBool(state, true)};
}

DppTargetLane EmitDppMirrorTargetLane(EmitterState& state, uint32_t subid, bool half_row) {
	const auto base_mask = half_row ? 0xfffffff8u : 0xfffffff0u;
	const auto lane_mask = half_row ? 7u : 15u;
	const auto base      = state.builder.AllocateId();
	const auto lane      = state.builder.AllocateId();
	const auto mirrored  = state.builder.AllocateId();
	const auto target    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), base, subid,
	                          ConstantU32(state, base_mask));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, lane_mask));
	state.builder.AddFunction(spv::OpISub, TypeU32(state), mirrored, ConstantU32(state, lane_mask),
	                          lane);
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, base, mirrored);
	return {target, ConstantBool(state, true)};
}

DppTargetLane EmitDppTargetLane(EmitterState& state, const IR::DppMoveFlags& flags) {
	const auto subid   = EmitSubgroupLocalInvocationId(state);
	const auto control = flags.control;
	if (flags.dpp8) {
		return EmitDppPermTargetLane(state, subid, control, 3u);
	}
	if (control <= 0xffu) {
		return EmitDppPermTargetLane(state, subid, control, 2u);
	}
	if (control >= 0x101u && control <= 0x10fu) {
		return EmitDppRowShiftTargetLane(state, subid, control & 0xfu, true);
	}
	if (control >= 0x111u && control <= 0x11fu) {
		return EmitDppRowShiftTargetLane(state, subid, control & 0xfu, false);
	}
	if (control >= 0x121u && control <= 0x12fu) {
		return EmitDppRowRotateRightTargetLane(state, subid, control & 0xfu);
	}
	if (control == 0x140u) {
		return EmitDppMirrorTargetLane(state, subid, false);
	}
	if (control == 0x141u) {
		return EmitDppMirrorTargetLane(state, subid, true);
	}
	// row_share:N reads lane N of the caller's row of 16.
	if (control >= 0x150u && control <= 0x15fu) {
		const auto row    = state.builder.AllocateId();
		const auto target = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
		                          ConstantU32(state, 0xfffffff0u));
		state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row,
		                          ConstantU32(state, control & 0xfu));
		return {target, ConstantBool(state, true)};
	}
	if (control >= 0x160u && control <= 0x16fu) {
		const auto target = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), target, subid,
		                          ConstantU32(state, control & 0xfu));
		return {target, ConstantBool(state, true)};
	}
	return {subid, ConstantBool(state, true)};
}

bool UsesIdBuffer(const EmitterState& state) {
	return state.program.stage == ShaderType::Pixel && state.input_info.pixel != nullptr &&
	       state.input_info.pixel->id_buffer_slot != UINT32_MAX;
}

bool WaveHalvesInHostSubgroup(const EmitterState& state) {
	// KYTY_DEBUG_WAVE_HALVES=0 emits as if the host subgroup matched the guest wave (A/B).
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_WAVE_HALVES");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	const auto stage = state.program.stage;
	return enabled && state.program.wave_size == 32u && state.lane_count == 1 &&
	       (stage == ShaderType::Vertex || stage == ShaderType::Local ||
	        stage == ShaderType::TessellationEvaluation);
}

uint32_t EmitOwnWaveHalfBase(EmitterState& state) {
	return EmitBinaryU32(state, spv::OpBitwiseAnd, EmitSubgroupLocalInvocationId(state),
	                     ConstantU32(state, 32));
}

uint32_t EmitOwnWaveHalfWord(EmitterState& state, uint32_t ballot) {
	const auto low   = state.builder.AllocateId();
	const auto high  = state.builder.AllocateId();
	const auto upper = state.builder.AllocateId();
	const auto word  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), upper, EmitOwnWaveHalfBase(state),
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), word, upper, high, low);
	return word;
}

uint32_t EmitLaunchedLaneAtOrBelow(EmitterState& state, uint32_t lane) {
	const auto ballot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
	                          ConstantU32(state, spv::ScopeSubgroup), ConstantBool(state, true));
	const auto low      = state.builder.AllocateId();
	const auto in_row   = EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
	const auto excluded = EmitBinaryU32(state, spv::OpShiftLeftLogical,
	                                    ConstantU32(state, 0xfffffffeu), in_row);
	const auto below    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpNot, TypeU32(state), below, excluded);
	uint32_t word = 0;
	uint32_t base = 0;
	if (state.program.wave_size == 64u && state.lane_count == 1) {
		// A search from the high word falls back to the low one when no high lane launched.
		const auto high      = state.builder.AllocateId();
		const auto in_high   = state.builder.AllocateId();
		const auto high_hit  = state.builder.AllocateId();
		const auto use_high  = state.builder.AllocateId();
		const auto low_word  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
		const auto high_below = EmitBinaryU32(state, spv::OpBitwiseAnd, high, below);
		state.builder.AddFunction(
		    spv::OpINotEqual, TypeBool(state), in_high,
		    EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 32)),
		    ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), high_hit, high_below,
		                          ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), use_high, in_high, high_hit);
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), low_word, in_high, low,
		                          EmitBinaryU32(state, spv::OpBitwiseAnd, low, below));
		word = state.builder.AllocateId();
		base = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), word, use_high, high_below,
		                          low_word);
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), base, use_high,
		                          ConstantU32(state, 32), ConstantU32(state, 0));
	} else {
		// One word: a wave32 in its own subgroup, the invocation's own half of a 64-lane one, or
		// both halves of a wave64 emulated two lanes per invocation (launched together).
		const auto launched = WaveHalvesInHostSubgroup(state) ? EmitOwnWaveHalfWord(state, ballot)
		                                                      : low;
		word = EmitBinaryU32(state, spv::OpBitwiseAnd, launched, below);
		base = state.program.wave_size == 64u || WaveHalvesInHostSubgroup(state)
		           ? EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 32))
		           : ConstantU32(state, 0);
	}
	const auto highest = state.builder.AllocateId();
	const auto any     = state.builder.AllocateId();
	const auto result  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), highest, GlslStd450(state),
	                          GLSLstd450FindUMsb, word);
	const auto launched = EmitAddU32(state, base, highest);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), any, word, ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, any, launched, lane);
	return result;
}

uint32_t EmitSubgroupLocalInvocationId(EmitterState& state) {
	if (state.split_wave) {
		// The workgroup is the wave: its invocation index is the guest lane.
		return EmitLocalInvocationIndex(state);
	}
	return state.lane_half == 0 ? EmitHostSubgroupLane(state)
	                            : EmitAddU32(state, EmitHostSubgroupLane(state),
	                                         ConstantU32(state, 32));
}

uint32_t EmitHostSubgroupLane(EmitterState& state) {
	if (state.subgroup_local_invocation_id_variable == 0) {
		EXIT("SubgroupLocalInvocationId was not declared before SPIR-V function emission\n");
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value,
	                          state.subgroup_local_invocation_id_variable);
	return value;
}

void NoteWaveSync(EmitterState& state) {
	if (state.wave_sync_blocks != nullptr && state.current_block != nullptr) {
		state.wave_sync_blocks->insert(state.current_block);
	}
}

void EmitWaveBarrier(EmitterState& state) {
	NoteWaveSync(state);
	state.builder.AddFunction(
	    spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup),
	    ConstantU32(state, spv::ScopeWorkgroup),
	    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask |
	                           spv::MemorySemanticsWorkgroupMemoryMask));
}

namespace {

// The exchange slot of `lane` in the half of the exchange buffer this exchange uses. Exchanges
// alternate halves, so a split wave half that runs ahead into the next exchange does not
// overwrite slots the other half still reads: it waits at that exchange's barrier first.
uint32_t WaveExchangeSlot(EmitterState& state, uint32_t parity, uint32_t lane) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassWorkgroup, TypeU32(state)), pointer,
	                          state.wave_exchange_variable,
	                          EmitAddU32(state, EmitBinaryU32(state, spv::OpShiftLeftLogical, parity,
	                                                          ConstantU32(state, 6)),
	                                     EmitBinaryU32(state, spv::OpBitwiseAnd, lane,
	                                                   ConstantU32(state, 63))));
	return pointer;
}

uint32_t NextWaveExchangeParity(EmitterState& state) {
	const auto parity = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), parity, state.wave_exchange_parity);
	state.builder.AddFunction(spv::OpStore, state.wave_exchange_parity,
	                          EmitBinaryU32(state, spv::OpBitwiseXor, parity, ConstantU32(state, 1)));
	return parity;
}

} // namespace

uint32_t ExchangeWaveValue(EmitterState& state, uint32_t value, uint32_t lane) {
	const auto parity = NextWaveExchangeParity(state);
	state.builder.AddFunction(spv::OpStore,
	                          WaveExchangeSlot(state, parity, EmitLocalInvocationIndex(state)), value);
	EmitWaveBarrier(state);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), result,
	                          WaveExchangeSlot(state, parity, lane));
	return result;
}

uint32_t EmitHostBallotWord(EmitterState& state, uint32_t predicate) {
	const auto ballot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
	                          ConstantU32(state, spv::ScopeSubgroup), predicate);
	const auto word = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), word, ballot, 0);
	return word;
}

uint32_t ExchangeWaveBallot(EmitterState& state, uint32_t predicate) {
	const auto word = EmitHostBallotWord(state, predicate);
	// Each lane publishes its half's ballot word; lanes 0 and 32 hold the two halves'.
	const auto parity = NextWaveExchangeParity(state);
	state.builder.AddFunction(spv::OpStore,
	                          WaveExchangeSlot(state, parity, EmitLocalInvocationIndex(state)), word);
	EmitWaveBarrier(state);
	std::array<uint32_t, 2> halves {};
	for (uint32_t half = 0; half < 2; half++) {
		halves[half] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), halves[half],
		                          WaveExchangeSlot(state, parity, ConstantU32(state, half * 32)));
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result,
	                          halves[0], halves[1], ConstantU32(state, 0), ConstantU32(state, 0));
	return result;
}

uint32_t InputVariableForKind(const EmitterState& state, IR::StageInputKind kind) {
	for (const auto& input: state.inputs) {
		if (input.kind == kind) {
			return input.variable_id;
		}
	}
	return 0;
}

const InputBinding* InputBindingForParameter(const EmitterState& state, uint32_t location) {
	for (const auto& input: state.inputs) {
		if (input.kind == IR::StageInputKind::Parameter && input.location == location) {
			return &input;
		}
	}
	return nullptr;
}

uint32_t EmitInputComponentU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	const auto variable = InputVariableForKind(state, kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassInput, TypeU32(state)), pointer,
	                          variable, ConstantU32(state, component));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

uint32_t EmitLocalInvocationIndex(EmitterState& state) {
	const auto variable = InputVariableForKind(state, IR::StageInputKind::LocalInvocationIndex);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, variable);
	if (state.lane_count == 2) {
		const auto wave_base =
		    EmitBinaryU32(state, spv::OpBitwiseAnd, value, ConstantU32(state, ~31u));
		return EmitAddU32(state, EmitAddU32(state, value, wave_base),
		                  ConstantU32(state, state.lane_half * 32));
	}
	return value;
}

uint32_t EmitVertexParameterComponentU32(EmitterState& state, const InputBinding& input,
                                         uint32_t component) {
	const auto count       = VertexParameterComponentCount(input);
	const auto kind        = VertexParameterScalarKind(state, input.location);
	const auto scalar_type = VertexParameterScalarType(state, kind);
	uint32_t   raw         = state.builder.AllocateId();
	if (count == 1u) {
		state.builder.AddFunction(spv::OpLoad, scalar_type, raw, input.variable_id);
	} else {
		const auto pointer_type = TypePointer(state, spv::StorageClassInput, scalar_type);
		const auto pointer      = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer, input.variable_id,
		                          ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, scalar_type, raw, pointer);
	}

	if (kind == VertexInputScalarKind::Uint) {
		return raw;
	}

	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, raw);
	return bits;
}

uint32_t EmitSubgroupLaneActiveBool(EmitterState& state, uint32_t lane) {
	uint32_t predicate = ConstantBool(state, true);
	if (state.helper_invocation_variable != 0) {
		const auto helper = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeBool(state), helper,
		                          state.helper_invocation_variable);
		predicate = EmitLogicalNotBool(state, helper);
	}
	const auto active_ballot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), active_ballot,
	                          ConstantU32(state, spv::ScopeSubgroup), predicate);
	return EmitBallotLaneActiveBool(state, active_ballot, lane);
}
uint32_t EmitBallotLaneActiveBool(EmitterState& state, uint32_t active_ballot, uint32_t lane) {
	const auto low = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, active_ballot, 0);
	uint32_t mask = low;
	if (state.program.wave_size == 64u) {
		const auto high     = state.builder.AllocateId();
		const auto in_high  = state.builder.AllocateId();
		const auto selected = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, active_ballot, 1);
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), in_high, lane,
		                          ConstantU32(state, 32));
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, in_high, high, low);
		mask = selected;
	}

	const auto lane_low = state.builder.AllocateId();
	const auto bit      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane_low, lane,
	                          ConstantU32(state, 31));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), bit, ConstantU32(state, 1),
	                          lane_low);

	const auto hit      = state.builder.AllocateId();
	const auto active   = state.builder.AllocateId();
	const auto in_range = state.builder.AllocateId();
	const auto ret      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), hit, mask, bit);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), in_range, lane,
	                          ConstantU32(state, state.program.wave_size));
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), ret, active, in_range);
	return ret;
}

void CountSelectSite(const std::source_location& where) {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_SELECT_SITES") != nullptr;
	if (!enabled) return;
	static std::mutex                      mutex;
	static std::map<std::string, uint64_t> counts;
	static const bool                      registered = [] {
		std::atexit([] {
			std::lock_guard lock(mutex);
			for (const auto& [site, count]: counts)
				std::fprintf(stderr, "select site %8llu %s\n", static_cast<unsigned long long>(count),
				             site.c_str());
		});
		return true;
	}();
	(void)registered;
	std::string site = where.file_name();
	site = site.substr(site.find_last_of("/\\") + 1) + ":" + std::to_string(where.line()) + " " +
	       where.function_name();
	std::lock_guard lock(mutex);
	counts[site]++;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
