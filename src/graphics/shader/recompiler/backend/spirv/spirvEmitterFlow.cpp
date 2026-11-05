#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/logging/log.h"
#include "common/switches.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <algorithm>
#include <atomic>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

bool UserDataDwordIndex(const EmitterState& state, IR::ScalarReg reg, uint32_t& dword_index) {
	const auto register_index = IR::RegIndex(reg);
	const auto& registers = state.program.bindings.user_data_registers;
	const auto  found     = std::lower_bound(registers.begin(), registers.end(), register_index);
	if (found == registers.end() || *found != register_index) {
		return false;
	}
	dword_index = static_cast<uint32_t>(found - registers.begin());
	return true;
}

uint32_t EmitBuiltinU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	if (kind == IR::StageInputKind::LocalInvocationIndex) {
		return EmitLocalInvocationIndex(state);
	}
	if (state.lane_count == 2 && (kind == IR::StageInputKind::LocalInvocationId ||
	                              kind == IR::StageInputKind::GlobalInvocationId)) {
		const auto* cs      = ShaderWorkgroupInput(state.program.stage, state.input_info);
		uint32_t    divisor = 1;
		for (uint32_t axis = 0; axis < component; axis++) {
			divisor *= std::max(cs->threads_num[axis], 1u);
		}
		const auto size    = std::max(cs->threads_num[component], 1u);
		const auto divided = EmitBinaryU32(state, spv::OpUDiv, EmitLocalInvocationIndex(state),
		                                   ConstantU32(state, divisor));
		const auto local   = EmitBinaryU32(state, spv::OpUMod, divided, ConstantU32(state, size));
		if (kind == IR::StageInputKind::LocalInvocationId) {
			return local;
		}
		const auto group = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, component);
		return EmitAddU32(state, local,
		                  EmitBinaryU32(state, spv::OpIMul, group, ConstantU32(state, size)));
	}
	const bool centroid = kind == IR::StageInputKind::BaryCoordSmoothCentroid;
	const bool sample = kind == IR::StageInputKind::BaryCoordSmoothSample;
	const auto variable = InputVariableForKind(
	    state, centroid || sample ? IR::StageInputKind::BaryCoordSmooth : kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	if (kind == IR::StageInputKind::FrontFacing) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeBool(state), value, variable);
		// SPI_BARYC_CNTL.FRONT_FACE_ALL_BITS selects uint masks or float +/-1.
		const bool all_bits =
		    state.input_info.pixel != nullptr && state.input_info.pixel->ps_front_face_all_bits;
		const auto front = ConstantU32(state, all_bits ? 0xffffffffu : 0x3f800000u);
		const auto back  = ConstantU32(state, all_bits ? 0u : 0xbf800000u);
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), bits, value, front, back);
		return bits;
	}
	if (kind == IR::StageInputKind::VertexIndex || kind == IR::StageInputKind::InstanceIndex ||
	    kind == IR::StageInputKind::InvocationId || kind == IR::StageInputKind::PrimitiveId ||
	    kind == IR::StageInputKind::Layer || kind == IR::StageInputKind::SampleId) {
		const auto value = state.builder.AllocateId();
		const auto bits  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeI32(state), value, variable);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (kind == IR::StageInputKind::FragCoord || kind == IR::StageInputKind::TessCoord) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
		                          pointer, variable, ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	if (centroid || sample || kind == IR::StageInputKind::BaryCoordSmooth ||
	    kind == IR::StageInputKind::BaryCoordNoPerspective) {
		const auto value   = state.builder.AllocateId();
		const auto bits    = state.builder.AllocateId();
		if (centroid || sample) {
			const auto coordinates = state.builder.AllocateId();
			state.builder.RequireCapability(spv::CapabilityInterpolationFunction);
			if (sample) {
				const auto sample_id = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLoad, TypeI32(state), sample_id,
				                          InputVariableForKind(state, IR::StageInputKind::SampleId));
				state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 3), coordinates,
				                          GlslStd450(state), GLSLstd450InterpolateAtSample,
				                          variable, sample_id);
			} else {
				state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 3), coordinates,
				                          GlslStd450(state), GLSLstd450InterpolateAtCentroid, variable);
			}
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), value,
			                          coordinates, component + 1u);
		} else {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, variable, ConstantU32(state, component + 1u));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		}
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
		return bits;
	}
	return EmitInputComponentU32(state, kind, component);
}

uint32_t EmitDppWriteCondition(ValueEmitContext& ctx, const IR::DppMoveFlags& flags,
                               uint32_t exec) {
	auto&      state      = ctx.state;
	const auto lane       = EmitSubgroupLocalInvocationId(state);
	const auto bank_shift = state.builder.AllocateId();
	const auto row_shift  = state.builder.AllocateId();
	const auto bank       = state.builder.AllocateId();
	const auto row        = state.builder.AllocateId();
	const auto bank_bit   = state.builder.AllocateId();
	const auto row_bit    = state.builder.AllocateId();
	const auto bank_hit   = state.builder.AllocateId();
	const auto row_hit    = state.builder.AllocateId();
	const auto bank_ok    = state.builder.AllocateId();
	const auto row_ok     = state.builder.AllocateId();
	const auto masks_ok   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), bank_shift, lane,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), row_shift, lane,
	                          ConstantU32(state, 4));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), bank, bank_shift,
	                          ConstantU32(state, 3));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, row_shift,
	                          ConstantU32(state, 3));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), bank_bit,
	                          ConstantU32(state, 1), bank);
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), row_bit,
	                          ConstantU32(state, 1), row);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), bank_hit,
	                          ConstantU32(state, flags.bank_mask), bank_bit);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row_hit,
	                          ConstantU32(state, flags.row_mask), row_bit);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), bank_ok, bank_hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), row_ok, row_hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), masks_ok, bank_ok, row_ok);
	uint32_t writable = masks_ok;
	if (!flags.bound_control) {
		const auto target  = EmitDppTargetLane(state, flags);
		const auto bounded = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), bounded, writable,
		                          target.valid);
		writable = bounded;
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), result, exec, writable);
	return result;
}

uint32_t EmitAttribute(EmitterState& state, uint32_t attr, uint32_t chan) {
	const auto* input = InputBindingForParameter(state, attr);
	if (input == nullptr || input->variable_id == 0) {
		return ConstantU32(state, 0);
	}
	if (state.program.stage == ShaderType::Vertex || state.program.stage == ShaderType::Local) {
		return EmitVertexParameterComponentU32(state, *input, chan & 3u);
	}
	const auto load_per_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
		    input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		return value;
	};
	if (input->per_vertex && PixelParameterIsFlat(state, attr)) {
		// A flat input read through raw vertices (it shares its vertex output with a smooth
		// input): the value of vertex 0, as the flat interpolation would give.
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, load_per_vertex(0));
		return bits;
	}
	if (input->per_vertex) {
		const auto barycentric_kind = state.input_info.pixel->ps_no_perspective
		                                  ? IR::StageInputKind::BaryCoordNoPerspective
		                                  : IR::StageInputKind::BaryCoordSmooth;
		const auto barycentric      = InputVariableForKind(state, barycentric_kind);
		uint32_t   sum              = 0;
		for (uint32_t vertex = 0; vertex < 3u; vertex++) {
			const auto pointer = state.builder.AllocateId();
			const auto weight  = state.builder.AllocateId();
			const auto product = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain,
			                          TypePointer(state, spv::StorageClassInput, TypeF32(state)),
			                          pointer, barycentric, ConstantU32(state, vertex));
			state.builder.AddFunction(spv::OpLoad, TypeF32(state), weight, pointer);
			state.builder.AddFunction(spv::OpFMul, TypeF32(state), product, load_per_vertex(vertex),
			                          weight);
			if (vertex == 0u) {
				sum = product;
			} else {
				const auto next = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFAdd, TypeF32(state), next, sum, product);
				sum = next;
			}
		}
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, sum);
		return bits;
	}
	const auto vector    = state.builder.AllocateId();
	const auto component = state.builder.AllocateId();
	const auto bits      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), vector, input->variable_id);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, vector,
	                          chan & 3u);
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, component);
	return bits;
}

uint32_t EmitInterpolationParameter(ValueEmitContext& ctx, uint32_t attr, uint32_t chan,
                                    uint32_t mode) {
	auto&       state = ctx.state;
	const auto* input = InputBindingForParameter(state, attr);
	if (!input->per_vertex) {
		return EmitAttribute(ctx.state, attr, chan);
	}
	const auto load_vertex = [&](uint32_t vertex) {
		const auto pointer = state.builder.AllocateId();
		const auto value   = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeF32(state)), pointer,
		    input->variable_id, ConstantU32(state, vertex), ConstantU32(state, chan & 3u));
		state.builder.AddFunction(spv::OpLoad, TypeF32(state), value, pointer);
		return value;
	};

	const auto selected_vertex = (mode + 1u) % 3u;
	uint32_t   value           = load_vertex(selected_vertex);
	if (!PixelParameterIsCustom(state, attr) && mode < 2u) {
		const auto delta = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFSub, TypeF32(state), delta, value, load_vertex(0));
		value = delta;
	}
	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
	return bits;
}

uint32_t MrtOutputMode(const EmitterState& state, const IR::ExportInfo& exp) {
	if (state.program.stage != ShaderType::Pixel || exp.kind != IR::ExportTargetKind::Mrt ||
	    exp.index >= std::size(state.input_info.pixel->target_output_mode)) {
		return 0;
	}
	return state.input_info.pixel->target_output_mode[exp.index];
}

uint32_t ExportRawComponent(ValueEmitContext& ctx, uint32_t vector, uint32_t component) {
	const auto value = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), value, vector,
	                              component);
	return value;
}

uint32_t ExportVector(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp,
                      bool uint_output) {
	auto& state = ctx.state;
	if (exp.compr && !uint_output) {
		auto unpack = GLSLstd450UnpackHalf2x16;
		switch (MrtOutputMode(state, exp)) {
			case 5: unpack = GLSLstd450UnpackUnorm2x16; break;
			case 6: unpack = GLSLstd450UnpackSnorm2x16; break;
			default: break;
		}
		uint32_t f32[4] = {ConstantF32(state, 0), ConstantF32(state, 0), ConstantF32(state, 0),
		                   ConstantF32(state, 0x3f800000u)};
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed   = state.builder.AllocateId();
			const auto unpacked = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), packed, data, pair);
			state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), unpacked,
			                          GlslStd450(state), unpack, packed);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) != 0u) {
					f32[component] = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state),
					                          f32[component], unpacked, lane);
				}
			}
		}
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector,
		                          f32[0], f32[1], f32[2], f32[3]);
		return vector;
	}
	uint32_t raw[4] = {
	    ConstantU32(state, 0),
	    ConstantU32(state, 0),
	    ConstantU32(state, 0),
	    ConstantU32(state, uint_output ? 1u : 0x3f800000u),
	};
	if (exp.compr) {
		for (uint32_t pair = 0; pair < 2u; pair++) {
			if ((exp.en & (3u << (pair * 2u))) == 0u) {
				continue;
			}
			const auto packed = ExportRawComponent(ctx, data, pair);
			for (uint32_t lane = 0; lane < 2u; lane++) {
				const auto component = pair * 2u + lane;
				if (((exp.en >> component) & 1u) == 0u) {
					continue;
				}
				raw[component] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), raw[component],
				                          packed, ConstantU32(state, lane * 16u),
				                          ConstantU32(state, 16));
			}
		}
	} else {
		for (uint32_t component = 0; component < 4u; component++) {
			if (((exp.en >> component) & 1u) != 0u) {
				raw[component] = ExportRawComponent(ctx, data, component);
			}
		}
	}
	if (uint_output) {
		const auto vector = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), vector,
		                          raw[0], raw[1], raw[2], raw[3]);
		return vector;
	}
	uint32_t f32[4] {};
	for (uint32_t component = 0; component < 4u; component++) {
		f32[component] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32[component], raw[component]);
	}
	const auto vector = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), vector, f32[0],
	                          f32[1], f32[2], f32[3]);
	return vector;
}

void EmitAuxPositionExport(ValueEmitContext& ctx, uint32_t data, const IR::ExportInfo& exp) {
	auto& state = ctx.state;
	for (uint32_t component = 0; component < 4; component++) {
		if ((exp.en & (1u << component)) == 0) {
			continue;
		}
		const auto output = IR::DecodePositionExportComponent(
		    state.input_info.vertex->pa_cl_vs_out_cntl, exp.index, component);
		if (output.layer || output.viewport) {
			const auto raw = ExportRawComponent(ctx, data, component);
			if (output.layer) {
				const auto layer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), layer, raw,
				                          ConstantU32(state, 0x7ffu));
				const auto pointer = state.program.stage == ShaderType::Mesh
				                         ? MeshOutputPointer(state, IR::StageOutputKind::Layer)
				                         : state.layer_variable;
				state.builder.AddFunction(spv::OpStore, pointer, layer);
			}
			if (output.viewport) {
				// GFX10 MISC.z packs the viewport index in bits 16..19 alongside the layer.
				const auto viewport = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), viewport, raw,
				                          ConstantU32(state, 16), ConstantU32(state, 4));
				state.builder.AddFunction(spv::OpStore, state.viewport_index_variable, viewport);
			}
			continue;
		}
		if (!output.point_size && output.clip_distance == UINT32_MAX &&
		    output.cull_distance == UINT32_MAX) {
			continue;
		}

		const auto raw = ExportRawComponent(ctx, data, component);
		const auto f32 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
		if (output.point_size) {
			state.builder.AddFunction(spv::OpStore, state.point_size_variable, f32);
			continue;
		}
		auto StoreDistance = [&](IR::StageOutputKind kind, uint32_t variable, uint32_t index) {
			if (index == UINT32_MAX) {
				return;
			}
			uint32_t pointer;
			if (state.program.stage == ShaderType::Mesh) {
				pointer = MeshOutputPointer(state, kind, index);
			} else {
				pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpAccessChain,
				                          TypePointer(state, spv::StorageClassOutput, TypeF32(state)),
				                          pointer, variable, ConstantU32(state, index));
			}
			state.builder.AddFunction(spv::OpStore, pointer, f32);
		};
		StoreDistance(IR::StageOutputKind::ClipDistance, state.clip_distance_variable, output.clip_distance);
		StoreDistance(IR::StageOutputKind::CullDistance, state.cull_distance_variable, output.cull_distance);
	}
}

uint32_t ConvertClipCoordinate(EmitterState& state, uint32_t coordinate, float scale,
                               float offset, float half_extent) {
	const auto window  = state.builder.AllocateId();
	const auto biased  = state.builder.AllocateId();
	const auto divided = state.builder.AllocateId();
	const auto ndc     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFMul, TypeF32(state), window, coordinate,
	                          ConstantF32Value(state, scale));
	state.builder.AddFunction(spv::OpFAdd, TypeF32(state), biased, window,
	                          ConstantF32Value(state, offset));
	state.builder.AddFunction(spv::OpFDiv, TypeF32(state), divided, biased,
	                          ConstantF32Value(state, half_extent));
	state.builder.AddFunction(spv::OpFSub, TypeF32(state), ndc, divided,
	                          ConstantF32Value(state, 1.0f));
	return ndc;
}

uint32_t ConvertPositionToClipSpace(EmitterState& state, uint32_t position) {
	const auto& transform = state.input_info.vertex->clip_space;
	uint32_t    components[4] {};
	for (uint32_t i = 0; i < 4; i++) {
		components[i] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), components[i], position,
		                          i);
	}
	components[0] = ConvertClipCoordinate(state, components[0], transform.scale[0],
	                                      transform.offset[0], transform.half_extent[0]);
	components[1] = ConvertClipCoordinate(state, components[1], transform.scale[1],
	                                      transform.offset[1], transform.half_extent[1]);
	const auto converted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), converted,
	                          components[0], components[1], components[2], components[3]);
	return converted;
}

} // namespace

// A float bound for a 16-bit float target, the guest GPU's way: a finite value beyond the half
// range becomes the largest half of its sign (D3D's conversion rounds toward zero), where the
// host's conversion can give infinity, which bloom and exposure spread over a frame. Infinity and
// NaN pass unchanged (usa-ratt's GT7 fork, cfa8a479e; KYTY_FLOAT_STORE_CLAMP=0 turns it off).
uint32_t EmitClampToHalfRange(EmitterState& state, uint32_t f32) {
	const auto bits      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, f32);
	const auto magnitude = Binary(state, spv::OpBitwiseAnd, TypeU32(state), bits,
	                              ConstantU32(state, 0x7fffffffu));
	// 0x477fe000 is 65504, the largest half; 0x7f800000 and above are infinity and NaN.
	const auto above   = Binary(state, spv::OpUGreaterThan, TypeBool(state), magnitude,
	                            ConstantU32(state, 0x477fe000u));
	const auto finite  = Binary(state, spv::OpULessThan, TypeBool(state), magnitude,
	                            ConstantU32(state, 0x7f800000u));
	const auto beyond  = Binary(state, spv::OpLogicalAnd, TypeBool(state), above, finite);
	const auto sign    = Binary(state, spv::OpBitwiseAnd, TypeU32(state), bits,
	                            ConstantU32(state, 0x80000000u));
	const auto largest = Binary(state, spv::OpBitwiseOr, TypeU32(state), sign,
	                            ConstantU32(state, 0x477fe000u));
	const auto chosen  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), chosen, beyond, largest, bits);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeF32(state), result, chosen);
	return result;
}
uint32_t EmitWqmU64(EmitterState& state, uint32_t value) {
	const auto shifted_one = state.builder.AllocateId();
	const auto merged_one  = state.builder.AllocateId();
	const auto shifted_two = state.builder.AllocateId();
	const auto merged_two  = state.builder.AllocateId();
	const auto quad_bits   = state.builder.AllocateId();
	const auto result      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_one, value,
	                          ConstantU64(state, 0x0000000100000001ull));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_one, value, shifted_one);
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU64(state), shifted_two, merged_one,
	                          ConstantU64(state, 0x0000000200000002ull));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU64(state), merged_two, merged_one,
	                          shifted_two);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU64(state), quad_bits, merged_two,
	                          ConstantU64(state, 0x1111111111111111ull));
	state.builder.AddFunction(spv::OpIMul, TypeU64(state), result, quad_bits,
	                          ConstantU64(state, 0x0000000f0000000full));
	return result;
}

void EmitSetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state = ctx.state;
	const auto& exp   = ctx.Export(inst);
	const auto  exec  = ctx.Arg(inst, 1);
	if (state.program.stage == ShaderType::Pixel && exp.vm && state.requirements.pixel_valid_mask &&
	    state.pixel_valid_mask_variable != 0) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, exec, ConstantU32(state, 1),
		                          ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable, value);
	}
	if (exp.kind == IR::ExportTargetKind::Null || exp.en == 0u) {
		return;
	}
	// Skip dormant color exports after their valid mask; MRT1 is reserved for logical alpha.
	if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
	    exp.index != 0 && state.input_info.pixel->alpha_blend_source != ShaderAlphaBlendSource::None) {
		return;
	}
	EmitIfCondition(state, exec, [&]() {
		const auto data = ctx.Arg(inst, 0);
		if (exp.kind == IR::ExportTargetKind::Primitive) {
			if (state.program.stage == ShaderType::Mesh) {
				state.builder.AddFunction(spv::OpStore, MeshPrimitivePointer(state),
				                          ExportRawComponent(ctx, data, 0));
			}
			return;
		}
		if (exp.kind == IR::ExportTargetKind::Position && exp.index != 0) {
			EmitAuxPositionExport(ctx, data, exp);
			return;
		}
		if (exp.kind == IR::ExportTargetKind::MrtZ) {
			if ((exp.en & 1u) != 0u && state.depth_variable != 0) {
				const auto raw = ExportRawComponent(ctx, data, 0);
				const auto f32 = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeF32(state), f32, raw);
				state.builder.AddFunction(spv::OpStore, state.depth_variable, f32);
			}
			if ((exp.en & 4u) != 0u && state.sample_mask_variable != 0) {
				const auto raw     = ExportRawComponent(ctx, data, 2);
				const auto value   = state.builder.AllocateId();
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitcast, TypeI32(state), value, raw);
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeI32(state)),
				    pointer, state.sample_mask_variable, ConstantU32(state, 0));
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
			return;
		}
		const auto variable =
		    state.program.stage == ShaderType::Mesh ? 0u : OutputVariableForExport(state, exp);
		if (state.program.stage != ShaderType::Mesh && variable == 0) {
			return;
		}
		const bool uint_output = MrtOutputMode(state, exp) == 7u;
		const auto vector_type = uint_output ? TypeU32Vector(state, 4) : TypeF32Vector(state, 4);
		auto       value       = ExportVector(ctx, data, exp, uint_output);
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    !uint_output && !exp.compr && exp.index < 8u &&
		    (state.input_info.pixel->target_half_float_mask & (1u << exp.index)) != 0 &&
		    FloatStoreClampEnabled()) {
			// A 32-bit export to a half-float target (packed half exports are in range already).
			uint32_t components[4] {};
			for (uint32_t component = 0; component < 4u; component++) {
				const auto scalar = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), scalar, value,
				                          component);
				components[component] = EmitClampToHalfRange(state, scalar);
			}
			value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, value, components[0],
			                          components[1], components[2], components[3]);
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index == 0 && !uint_output &&
		    state.input_info.pixel->alpha_blend_source != ShaderAlphaBlendSource::None) {
			// Broadcast logical alpha before swizzling the primary output.
			const auto blend_output =
			    OutputVariableForExport(state, {.kind = IR::ExportTargetKind::Mrt, .index = 1});
			if (blend_output != 0) {
				auto alpha = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, alpha, value, value,
				                          3u, 3u, 3u, 3u);
				uint32_t alpha_factor = 0;
				switch (state.input_info.pixel->alpha_blend_source) {
					case ShaderAlphaBlendSource::SourceAlphaOne:
						alpha_factor = ConstantF32Value(state, 1.0f);
						break;
					case ShaderAlphaBlendSource::SourceAlphaZero:
						alpha_factor = ConstantF32Value(state, 0.0f);
						break;
					default: break;
				}
				if (alpha_factor != 0) {
					const auto mapping = state.input_info.pixel->target_export_mapping[0];
					for (uint32_t component = 0; component < 4; component++) {
						if (mapping.Map(component) != 3u) continue;
						const auto factors = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpCompositeInsert, vector_type, factors,
						                          alpha_factor, alpha, component);
						alpha = factors;
						break;
					}
				}
				state.builder.AddFunction(spv::OpStore, blend_output, alpha);
			}
		}
		if (state.program.stage == ShaderType::Pixel && exp.kind == IR::ExportTargetKind::Mrt &&
		    exp.index < state.input_info.pixel->target_export_mapping.size()) {
			const auto mapping = state.input_info.pixel->target_export_mapping[exp.index];
			if (!mapping.IsIdentity()) {
				const auto mapped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVectorShuffle, vector_type, mapped, value, value,
				                          mapping.Map(0), mapping.Map(1), mapping.Map(2),
				                          mapping.Map(3));
				value = mapped;
			}
		}
		if (exp.kind == IR::ExportTargetKind::Position &&
		    state.input_info.vertex->clip_space.enabled) {
			value = ConvertPositionToClipSpace(state, value);
		}
		if (state.program.stage == ShaderType::Mesh) {
			const auto kind = exp.kind == IR::ExportTargetKind::Position
			                      ? IR::StageOutputKind::Position
			                      : IR::StageOutputKind::Parameter;
			state.builder.AddFunction(spv::OpStore, MeshOutputPointer(state, kind, exp.index),
			                          value);
		} else if (exp.kind == IR::ExportTargetKind::Position) {
			if (state.invalid_position_clip_distance != UINT32_MAX) {
				const auto zero = state.builder.Constant(spv::OpConstantNull, TypeF32Vector(state, 4));
				const auto equal = state.builder.AllocateId();
				const auto all_zero         = state.builder.AllocateId();
				const auto distance = state.builder.AllocateId();
				const auto distance_pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFOrdEqual, TypeBoolVector(state, 4), equal,
				                          value, zero);
				state.builder.AddFunction(spv::OpAll, TypeBool(state), all_zero, equal);
				// A NaN component is a clipping error as well: the guest kills primitives by
				// exporting NaN positions, which the native clipper discards but a Vulkan
				// rasterizer may stretch across the screen. Integer compares keep the test
				// from being folded away by a fast-math driver compiler.
				const auto u32x4     = TypeU32Vector(state, 4);
				const auto magnitude = Binary(
				    state, spv::OpBitwiseAnd, u32x4, Unary(state, spv::OpBitcast, u32x4, value),
				    state.builder.Constant(
				        spv::OpConstantComposite, u32x4, ConstantU32(state, 0x7fffffffu),
				        ConstantU32(state, 0x7fffffffu), ConstantU32(state, 0x7fffffffu),
				        ConstantU32(state, 0x7fffffffu)));
				const auto infinity = ConstantU32(state, 0x7f800000u);
				const auto nan =
				    Binary(state, spv::OpUGreaterThan, TypeBoolVector(state, 4), magnitude,
				           state.builder.Constant(spv::OpConstantComposite, u32x4, infinity,
				                                  infinity, infinity, infinity));
				const auto invalid = Binary(state, spv::OpLogicalOr, TypeBool(state), all_zero,
				                            Unary(state, spv::OpAny, TypeBool(state), nan));
				// The rasterizer then sees a zero position: a NaN would reach the clipper's
				// interpolation below even at a zero distance.
				const auto invalid_vector = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeConstruct, TypeBoolVector(state, 4),
				                          invalid_vector, invalid, invalid, invalid, invalid);
				const auto culled = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeF32Vector(state, 4), culled,
				                          invalid_vector, zero, value);
				value = culled;
				// Zero at valid vertices makes a primitive containing an invalid position
				// collapse to its remaining edge, before the undefined 0/0 perspective divide.
				state.builder.AddFunction(spv::OpSelect, TypeF32(state), distance, invalid,
				                          ConstantF32Value(state, -1.0f),
				                          ConstantF32Value(state, 0.0f));
				state.builder.AddFunction(
				    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, TypeF32(state)),
				    distance_pointer, state.clip_distance_variable,
				    ConstantU32(state, state.invalid_position_clip_distance));
				state.builder.AddFunction(spv::OpStore, distance_pointer, distance);
				static std::atomic_bool logged = false;
				if (!logged.exchange(true, std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog("Shader: emitted zero or NaN position clip guard\n");
				}
			}
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(
			    spv::OpAccessChain,
			    TypePointer(state, spv::StorageClassOutput, TypeF32Vector(state, 4)), pointer,
			    variable, ConstantU32(state, 0));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		} else {
			state.builder.AddFunction(spv::OpStore, variable, value);
		}
	});
}

uint32_t EmitIdentity(ValueEmitContext&, uint32_t value) {
	return value;
}

void EmitVoid(ValueEmitContext&) {}

void EmitStoreCompletion(EmitterState& state) {
	// A split wave spans two host subgroups of its workgroup.
	NoteWaveSync(state);
	state.builder.AddFunction(
	    spv::OpControlBarrier,
	    ConstantU32(state, state.split_wave ? spv::ScopeWorkgroup : spv::ScopeSubgroup),
	    ConstantU32(state, spv::ScopeDevice),
	    ConstantU32(state, spv::MemorySemanticsReleaseMask | spv::MemorySemanticsUniformMemoryMask |
	                          spv::MemorySemanticsImageMemoryMask));
}

// KYTY_LDS_WAIT_BARRIER=1: S_WAITCNT lgkmcnt(0) in a shader that writes LDS waits for the whole
// subgroup. The guest wave runs its lanes in lockstep, so a lane reads what another lane of the
// wave wrote right after the wait; the driver schedules a subgroup's invocations independently, and
// without the barrier such a read can come first (an LDS race GPU-assisted validation reported in
// The title, KytyPS5 issue 1458). Split waves already synchronize before each such DS access.
bool LdsWaitBarrierEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_LDS_WAIT_BARRIER");
		return text != nullptr && text[0] == '1';
	}();
	return enabled;
}

// KYTY_LDS_ACCESS_BARRIER=1: in a shader that writes LDS, every DS access waits for the subgroup
// first, as split waves do (EmitBlock). The guest's LDS executes a wave's DS operations in order,
// so the compiler never waits between a DS write and a DS read of what another lane wrote (it
// waits only before the read value is used, where KYTY_LDS_WAIT_BARRIER's barrier comes one
// instruction too late). The title cloth sim cs a31f7bf9cc65e227: one lane writes the wave minimum and
// the chosen contact, all lanes read them back at once; a stale read exploded the cloth.
bool LdsAccessBarrierEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_LDS_ACCESS_BARRIER");
		return text != nullptr && text[0] == '1';
	}();
	return enabled;
}

void EmitLdsWaitBarrier(EmitterState& state) {
	if (!LdsWaitBarrierEnabled() || !state.ds_writes_lds || state.split_wave ||
	    ShaderWorkgroupInput(state.program.stage, state.input_info) == nullptr) {
		return;
	}
	EmitLdsSubgroupBarrier(state);
}

void EmitLdsSubgroupBarrier(EmitterState& state) {
	const auto memory = state.lds_storage_class == spv::StorageClassStorageBuffer
	                        ? spv::MemorySemanticsUniformMemoryMask
	                        : spv::MemorySemanticsWorkgroupMemoryMask;
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeSubgroup),
	                          ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | memory));
}

void EmitBarrier(EmitterState& state) {
	const auto tessellation = state.program.stage == ShaderType::TessellationControl;
	if (!tessellation && ShaderWorkgroupInput(state.program.stage, state.input_info) == nullptr) {
		// Independent graphics invocations have no native workgroup left to synchronize.
		return;
	}
	const auto memory_scope = tessellation ? spv::ScopeInvocation : spv::ScopeWorkgroup;
	const auto memory = state.lds_storage_class == spv::StorageClassStorageBuffer
	                        ? spv::MemorySemanticsUniformMemoryMask
	                        : spv::MemorySemanticsWorkgroupMemoryMask;
	auto semantics = static_cast<uint32_t>(tessellation ? spv::MemorySemanticsMaskNone
	                                                    : spv::MemorySemanticsAcquireReleaseMask | memory);
	// A compute shader that stores to buffers or images and reads them back in other waves after
	// S_BARRIER (the guest waits vscnt first; its stores are then visible in the workgroup) needs the
	// barrier to order those memories too. The title hair cs 2ef93a49 builds per-pixel fragment lists in
	// buffers and walks them after the barrier: with workgroup memory alone a lane may read an older
	// next pointer (a list walk that never ends = the fight-load hang?).
	// KYTY_BARRIER_MEMORY=0 orders workgroup memory only, as before.
	static const bool barrier_memory = [] {
		const char* text = KYTY_GETENV("KYTY_BARRIER_MEMORY");
		return text == nullptr || text[0] != '0';
	}();
	if (barrier_memory && !tessellation && state.program.stage == ShaderType::Compute) {
		const auto& info   = state.program.info;
		const bool  stores = state.program.has_address_writes ||
		                    std::ranges::any_of(info.buffers, [](const auto& b) { return b.written; }) ||
		                    std::ranges::any_of(info.images, [](const auto& i) { return i.written; });
		if (stores) {
			semantics |= static_cast<uint32_t>(spv::MemorySemanticsUniformMemoryMask) |
			             static_cast<uint32_t>(spv::MemorySemanticsImageMemoryMask);
		}
	}
	NoteWaveSync(state);
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, memory_scope), ConstantU32(state, semantics));
}

uint32_t EmitLaneId(EmitterState& state) {
	return state.program.stage == ShaderType::TessellationControl
	           ? EmitBuiltinU32(state, IR::StageInputKind::InvocationId, 0)
	           : EmitSubgroupLocalInvocationId(state);
}

uint32_t EmitIsHelperInvocation(EmitterState& state) {
	if (state.helper_invocation_variable == 0) {
		return ConstantBool(state, false);
	}
	const auto helper = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeBool(state), helper,
	                          state.helper_invocation_variable);
	return helper;
}

uint32_t EmitMeshDrawParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto result = state.builder.AllocateId();
	const auto index  = inst.Arg(0).U32();
	if (state.program.stage != ShaderType::Mesh || index >= IR::PushData::MeshDrawDwordCount) {
		ctx.Fail(inst, "invalid mesh draw parameter");
	}
	// Push-constant dwords 0 and 1 hold the device address of the draw's parameter record, so
	// that an indirect draw can have the GPU write the record from its arguments.
	const auto push_word = [&](uint32_t word) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state),
		                          pointer, state.push_constant_variable, ConstantU32(state, 0),
		                          ConstantU32(state, word));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	const auto u64      = TypeScalarU64(state);
	const auto constant = [&](uint64_t value) {
		return state.builder.Constant(spv::OpConstant, u64, static_cast<uint32_t>(value),
		                              static_cast<uint32_t>(value >> 32u));
	};
	const auto low  = Unary(state, spv::OpUConvert, u64, push_word(0));
	const auto high = Binary(state, spv::OpShiftLeftLogical, u64,
	                         Unary(state, spv::OpUConvert, u64, push_word(1)), constant(32));
	const auto address =
	    Binary(state, spv::OpIAdd, u64, Binary(state, spv::OpBitwiseOr, u64, low, high),
	           constant(uint64_t {index} * sizeof(uint32_t)));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
	                          address);
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), result, pointer,
	                          spv::MemoryAccessAlignedMask, static_cast<uint32_t>(sizeof(uint32_t)));
	return result;
}

uint32_t EmitGetUserData(EmitterState& state, IR::ScalarReg reg) {

	uint32_t dword = 0;
	if (!UserDataDwordIndex(state, reg, dword)) {
		return ConstantU32(state, 0);
	} else {
		return EmitShaderDataDwordLoad(state, dword);
	}
}

uint32_t EmitGetBuiltin(ValueEmitContext& ctx, IR::Value kind, IR::Value index) {
	return EmitBuiltinU32(ctx.state, static_cast<IR::StageInputKind>(kind.U32()), index.U32());
}

uint32_t EmitGetDispatchThreadExtent(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto start = ctx.state.program.bindings.dispatch_thread_dword;
	if (start == IR::PushData::NoStart || !inst.Arg(0).IsImmediate() || inst.Arg(0).U32() >= 3u) {
		ctx.Fail(inst, "invalid dispatch thread extent");
	}
	return EmitShaderDataDwordLoad(ctx.state, start + inst.Arg(0).U32());
}

uint32_t EmitUndefU1(EmitterState& state, const IR::Inst& inst) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUndef, TypeId(state, inst.GetType()), result);
	return result;
}

// Reads `value` from `lane` of a split wave's 32-lane half: within the invocation's own host
// subgroup, for operations that stay within rows or halves of the wave.
uint32_t ReadHostSubgroupLane(EmitterState& state, uint32_t type, uint32_t value, uint32_t lane) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, result,
	                          ConstantU32(state, spv::ScopeSubgroup), value,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31)));
	return result;
}

uint32_t EmitDppMoveU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state    = ctx.state;
	const auto flags    = inst.Flags<IR::DppMoveFlags>();
	const auto target   = EmitDppTargetLane(state, flags);
	// DPP reads within rows of 16 lanes, which never leave a split wave's half.
	const auto shuffled = state.split_wave
	                          ? ReadHostSubgroupLane(state, TypeU32(state), ctx.Arg(inst, 0),
	                                                 target.lane)
	                          : ctx.Shuffle(inst, 0, target.lane);
	if (flags.fetch_inactive) {
		return shuffled;
	}
	uint32_t ballot = 0, source_lane = target.lane;
	if (state.split_wave) {
		// DPP rows stay within the invocation's own host subgroup.
		ballot = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), ctx.Arg(inst, 1));
		source_lane = EmitBinaryU32(state, spv::OpBitwiseAnd, target.lane, ConstantU32(state, 31));
	} else {
		ballot = ctx.Ballot(inst.Arg(1));
	}
	auto source_active = EmitBallotLaneActiveBool(state, ballot, source_lane);
	if (state.helper_invocation_variable != 0) {
		// Ballots leave helper invocations out (they are not in the guest's initial EXEC), but
		// after S_WQM they are active lanes whose values a DPP read gets, as on the GPU (the min
		// scan of a material loop otherwise reads 0 from a helper and never ends). A helper source
		// counts by its own EXEC bit.
		const auto helper = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeBool(state), helper,
		                          state.helper_invocation_variable);
		const auto helpers = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), helpers,
		                          ConstantU32(state, spv::ScopeSubgroup), helper);
		const auto helper_source = EmitBallotLaneActiveBool(state, helpers, target.lane);
		const auto helper_active = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), helper_active, helper_source,
		                          ctx.Shuffle(inst, 1, target.lane));
		const auto either = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), either, source_active,
		                          helper_active);
		source_active = either;
	}
	const auto can_fetch     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), can_fetch, target.valid,
	                          source_active);
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, can_fetch, shuffled,
	                                                ConstantU32(state, 0));
}

uint32_t EmitDppUpdateU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto flags = inst.Flags<IR::DppMoveFlags>();
	const auto write = EmitDppWriteCondition(ctx, flags, ctx.Arg(inst, 2));
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, write, ctx.Arg(inst, 0),
	                                                ctx.Arg(inst, 1));
}

uint32_t EmitConditionRef(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (ctx.state.wave_local_insts.contains(&inst)) {
		// Split wave, and no block this branch controls synchronizes the halves: a half may
		// branch on its own lanes, so one whose lanes are all inactive skips code the guest ran
		// with them masked. Argument 0 is the lane predicate (zero kinds are already inverted).
		const auto kind = inst.Flags<CFG::BranchCondition>();
		const bool zero = kind == CFG::BranchCondition::ExecZero ||
		                  kind == CFG::BranchCondition::VccZero || kind == CFG::BranchCondition::SccZero;
		const auto word = EmitHostBallotWord(ctx.state, ctx.Arg(inst, 0));
		return Binary(ctx.state, zero ? spv::OpIEqual : spv::OpINotEqual, TypeBool(ctx.state),
		              word, ConstantU32(ctx.state, zero ? ~0u : 0u));
	}
	// Argument 1 is the branch's emitted condition; argument 0 is only for analysis.
	if (ctx.other_half == nullptr) return ctx.Arg(inst, 1);
	// A native scalar branch makes one decision for both emulated wave halves.
	if (ctx.half != 0) return ctx.other_half->Def(IR::Value(&inst));
	const auto kind = inst.Flags<CFG::BranchCondition>();
	if (kind == CFG::BranchCondition::ScalarInstruction) return ctx.Arg(inst, 1);
	const auto ballot = ctx.Ballot(inst.Arg(1));
	const auto low = ctx.state.builder.AllocateId();
	const auto high = ctx.state.builder.AllocateId();
	const auto combined = ctx.state.builder.AllocateId();
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), low, ballot, 0);
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), high, ballot, 1);
	const bool zero = kind == CFG::BranchCondition::ExecZero ||
	                  kind == CFG::BranchCondition::VccZero || kind == CFG::BranchCondition::SccZero;
	if (!zero) {
		ctx.state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(ctx.state), combined, low, high);
		ctx.state.builder.AddFunction(spv::OpINotEqual, TypeBool(ctx.state), result, combined,
		                              ConstantU32(ctx.state, 0u));
		return result;
	}
	// "Zero" holds when every active lane of both halves says so. Invocations that are not running
	// (a partial wave, lanes that left the shader) have no ballot bit; comparing with all ones
	// instead of the active mask kept such waves in their loop forever (Ghost of Yotei cutscene
	// compute b5d1ae71f2600147, a merge-path search, hung the GPU).
	const auto active      = ctx.Ballot(IR::Value(true));
	const auto active_word = ctx.state.builder.AllocateId();
	const auto low_all     = ctx.state.builder.AllocateId();
	const auto high_all    = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), active_word, active, 0);
	ctx.state.builder.AddFunction(spv::OpIEqual, TypeBool(ctx.state), low_all, low, active_word);
	ctx.state.builder.AddFunction(spv::OpIEqual, TypeBool(ctx.state), high_all, high, active_word);
	ctx.state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(ctx.state), combined, low_all, high_all);
	return combined;
}

uint32_t EmitBallot(ValueEmitContext& ctx, IR::Value predicate) {
	return ctx.Ballot(predicate);
}

uint32_t EmitReadFirstLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (ctx.state.wave_local_insts.contains(&inst)) {
		// A waterfall loop handles the lanes whose value matches the one read and repeats for
		// the rest, so each half of a split wave can read its own first active lane.
		auto&      state = ctx.state;
		const auto first = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeU32(state), first, GlslStd450(state),
		                          GLSLstd450FindILsb, EmitHostBallotWord(state, ctx.Arg(inst, 1)));
		return ReadHostSubgroupLane(state, TypeU32(state), ctx.Arg(inst, 0), first);
	}
	const auto ballot   = ctx.Ballot(inst.Arg(1));
	const auto lane     = ctx.FirstLane(ballot);
	const auto shuffled = ctx.Shuffle(inst, 0, lane);
	if (WaveHalvesInHostSubgroup(ctx.state)) {
		// Each half is its own guest wave: the value is uniform per half only.
		return shuffled;
	}
	// Every lane now holds the same value, so reading the first active lane changes nothing but
	// tells the driver the result is uniform. AMD compiles the shuffle alone to ds_bpermute and
	// treats its result as per-lane, which turned everything derived from a waterfall key
	// (scalar loads, address math, loop exits) into vector code.
	auto&      state  = ctx.state;
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBroadcastFirst,
	                          TypeId(state, inst.Arg(0).GetType()), result,
	                          ConstantU32(state, spv::ScopeSubgroup), shuffled);
	return result;
}

uint32_t EmitReadLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	// V_READLANE's lane is an SGPR or a constant, so the result is uniform too; see
	// EmitReadFirstLane.
	auto& state = ctx.state;
	if (WaveHalvesInHostSubgroup(state)) {
		// The lane is the guest wave's (wave32 reads its low 5 bits): offset it into the
		// invocation's own half.
		const auto lane =
		    EmitBinaryU32(state, spv::OpBitwiseAnd, ctx.Arg(inst, 1), ConstantU32(state, 31));
		return ctx.Shuffle(inst, 0,
		                   EmitAddU32(state, EmitLaunchedLaneAtOrBelow(state, lane),
		                              EmitOwnWaveHalfBase(state)));
	}
	const auto shuffled = ctx.Shuffle(inst, 0, EmitLaunchedLaneAtOrBelow(state, ctx.Arg(inst, 1)));
	const auto result   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBroadcastFirst,
	                          TypeId(state, inst.Arg(0).GetType()), result,
	                          ConstantU32(state, spv::ScopeSubgroup), shuffled);
	return result;
}

uint32_t EmitWriteLane(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto hit   = state.builder.AllocateId();
	const auto lane  = WaveHalvesInHostSubgroup(state)
	                       ? EmitBinaryU32(state, spv::OpBitwiseAnd,
	                                       EmitSubgroupLocalInvocationId(state),
	                                       ConstantU32(state, 31))
	                       : EmitSubgroupLocalInvocationId(state);
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), hit, lane, ctx.Arg(inst, 2));
	return EmitNative<spv::OpSelect, IR::Type::U32>(ctx.state, hit, ctx.Arg(inst, 1),
	                                                ctx.Arg(inst, 0));
}

uint32_t EmitPermlane16U32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state     = ctx.state;
	const auto flags     = inst.Flags<IR::PermlaneFlags>();
	const auto subid     = EmitSubgroupLocalInvocationId(state);
	const auto row       = state.builder.AllocateId();
	const auto row_value = state.builder.AllocateId();
	const auto lane      = state.builder.AllocateId();
	const auto lane8     = state.builder.AllocateId();
	const auto shift     = state.builder.AllocateId();
	const auto upper     = state.builder.AllocateId();
	const auto selected  = state.builder.AllocateId();
	const auto shifted   = state.builder.AllocateId();
	const auto index     = state.builder.AllocateId();
	const auto target    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	if (flags.x16) {
		state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), row_value, row,
		                          ConstantU32(state, 16));
	} else {
		state.builder.AddFunction(spv::OpCopyObject, TypeU32(state), row_value, row);
	}
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane8, lane,
	                          ConstantU32(state, 7));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), shift, lane8,
	                          ConstantU32(state, 2));
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), upper, lane,
	                          ConstantU32(state, 8));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, upper, ctx.Arg(inst, 2),
	                          ctx.Arg(inst, 1));
	state.builder.AddFunction(spv::OpShiftRightLogical, TypeU32(state), shifted, selected, shift);
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), index, shifted,
	                          ConstantU32(state, 15));
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row_value, index);
	const auto launched = EmitLaunchedLaneAtOrBelow(state, target);
	// Permlane16 reads within the caller's 32-lane half, a split wave's host subgroup.
	const auto read = [&](size_t argument, uint32_t lane) {
		return state.split_wave
		           ? ReadHostSubgroupLane(state, TypeId(state, inst.Arg(argument).GetType()),
		                                  ctx.Arg(inst, argument), lane)
		           : ctx.Shuffle(inst, argument, lane);
	};
	const auto shuffled = read(0, launched);
	uint32_t   result   = shuffled;
	if (!flags.fetch_inactive) {
		// A lane the host did not launch is active if the guest switched it on, which only the
		// scalar EXEC copy records.
		const auto in_high = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpLogicalAnd, TypeBool(state), in_high,
		    ConstantBool(state, state.program.wave_size == 64u),
		    Binary(state, spv::OpINotEqual, TypeBool(state),
		                   EmitBinaryU32(state, spv::OpBitwiseAnd, target, ConstantU32(state, 32)),
		                   ConstantU32(state, 0)));
		const auto exec_word = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), exec_word, in_high,
		                          ctx.Arg(inst, 5), ctx.Arg(inst, 4));
		const auto exec_bit = EmitBinaryU32(
		    state, spv::OpBitwiseAnd,
		    EmitBinaryU32(state, spv::OpShiftRightLogical, exec_word,
		                  EmitBinaryU32(state, spv::OpBitwiseAnd, target, ConstantU32(state, 31))),
		    ConstantU32(state, 1));
		const auto was_launched = state.builder.AllocateId();
		const auto source_exec  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpIEqual, TypeBool(state), was_launched, launched, target);
		state.builder.AddFunction(
		    spv::OpSelect, TypeBool(state), source_exec, was_launched, read(3, launched),
		    Binary(state, spv::OpINotEqual, TypeBool(state), exec_bit, ConstantU32(state, 0)));
		result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, source_exec, shuffled,
		                          ConstantU32(state, 0));
	}
	return result;
}

uint32_t EmitGetAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitAttribute(ctx.state, inst.Arg(0).U32(), inst.Arg(1).U32());
}

uint32_t EmitGetInterpolationParameter(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitInterpolationParameter(ctx, inst.Arg(0).U32(), inst.Arg(1).U32(), inst.Arg(2).U32());
}

uint32_t EmitGetShaderBase(ValueEmitContext& ctx) {
	// Guest S_GETPC values stay shader-relative in SPIR-V, matching the runtime ABI. The
	// runtime descriptor evaluator supplies the mapped shader base for host-side planning.
	return ctx.Def(IR::Value(uint64_t {0}));
}

void EmitUnreachable(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.Fail(inst, "must be lowered before SPIR-V emission");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
