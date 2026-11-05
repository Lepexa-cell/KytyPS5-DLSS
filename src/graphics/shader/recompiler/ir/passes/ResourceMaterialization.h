#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          bindless                   = false;
		// A written storage image of a 16-bit float format: stores clamp to the half range
		// (ShaderRecompiler::FloatStoreClampEnabled).
		bool                          clamp_half_float           = false;
		bool                          operator==(const Image&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Inputs of the previous refresh of one plan into one snapshot and specialization. Pass the same
// memo only with the same plan, snapshot and specialization objects, and do not modify the
// snapshot's descriptors or the specialization between refreshes.
struct MaterializationMemo {
	std::vector<uint32_t> key;
	bool                  valid  = false;
	// The last refresh kept the previous descriptors and specialization.
	bool                  reused = false;
};

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
// With a memo, a memoizable plan whose descriptor inputs (active sources, user data, shader base
// and flat SRT slots) are unchanged skips descriptor evaluation and specialization.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          MaterializationMemo* memo = nullptr);

// The guest reads one refresh made through its runtime's readers, in order, with their results.
struct ReadLog {
	enum class Reader : uint8_t { Ordinary, OrdinaryBlock, Specialization };
	struct Entry {
		uint64_t address = 0;
		uint32_t first   = 0; // Into words.
		uint32_t count   = 0; // Dwords.
		Reader   reader  = Reader::Ordinary;
		bool     ok      = false;
	};

	std::vector<Entry>    entries;
	std::vector<uint32_t> words;

	void Clear() {
		entries.clear();
		words.clear();
	}
};

// MaterializeResources without a memo, logging every read it makes through runtime's readers.
// Plans that capture their reads (masked indirect images beside written buffers) are refused.
bool MaterializeResourcesLogged(const ResourcePlan& program, const SrtRuntime& runtime,
                                ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                                ReadLog& log);

// Whether every logged read, made again through runtime's readers, fails as it failed or reads
// the same dwords. A refresh is a function of its plan, user data, shader base and read results,
// so a refresh with this runtime (the same plan, user data and shader base) would then produce
// the logged refresh's snapshot and specialization.
bool ReadsUnchanged(const ReadLog& log, const SrtRuntime& runtime);

// The SrtWalker implementation that MaterializeResources must match.
bool MaterializeResourcesReference(const ResourcePlan& program, const SrtRuntime& runtime,
                                   ResourceSnapshot&       snapshot,
                                   ResourceSpecialization& specialization);

// Debugging aid: compare every refresh with MaterializeResourcesReference and abort on a
// difference. GPU thread only.
void SetResourceMaterializationVerification(bool enabled);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

// Marks a buffer's packed_stride whose nonzero stride the shader reads from shader data. The
// swizzle bit, index stride and ADD_TID (bits 14, 16-17, 20) stay in the permutation key. Ghost of Yotei
// binds buffers of different strides to the same slots, so baking the stride in compiled one
// pipeline per combination (3e8244db54b38b0c: 7, at 20-60 s each in the driver).
constexpr uint32_t DynamicBufferStride = 1u << 31u;

// KYTY_BUFFER_SPECIALIZE=1: strides are baked into the module again (A/B runs).
bool BufferStrideSpecialization();

// The descriptor's packed stride (stride, swizzle bit 14, index stride bits 16-17, ADD_TID bit
// 20) with the fields a zero stride or a disabled swizzle ignores cleared.
uint32_t CanonicalPackedStride(const DescriptorValue& descriptor);

// Whether a stage's buffers take DynamicBufferStride and GenericBufferFormat: compute shaders
// only (their permutations multiplied; a vertex shader's many vertex buffers would fill the push
// data with stride and format words and move its shader data to storage).
bool BufferGenericStage(ShaderType stage);

// descriptor_format of a buffer whose formatted loads decode the V#'s format and dst_sel from
// shader data. The same slots of Ghost of Yotei's shaders see 8-bit UInt, 8-bit UNorm and 8_8
// UNorm data. Only read-only buffers of an 8/16/32-bit per-component format without zero-stride
// OOB folding are generic; packed formats (10_11_11, 2_10_10_10, 5_6_5, ...) stay specialized.
constexpr auto GenericBufferFormat = static_cast<Prospero::BufferFormat>(0x1ffu);

// The shader-data word of a GenericBufferFormat buffer: the V#'s stride (bits 0-13), format
// (bits 14-20) and dst_sel (bits 21-31: the four 3-bit selectors, each 0, 1 or 4-7, as base-6
// digits, X lowest); 0 (kInvalid) for an invalid or non-buffer descriptor.
uint32_t RuntimeBufferSlotWord(const DescriptorValue& descriptor);

// Whether a buffer slot (base) of a stage with this specialization takes GenericBufferFormat.
bool GenericBufferFormatEligible(ShaderType stage, const BufferResource& base,
                                 const ResourceSpecialization::Buffer& buffer);

// Replaces baked nonzero strides by DynamicBufferStride and the formats of eligible formatted
// slots by GenericBufferFormat when the resulting stride and format words still fit in push data
// with `base_dwords` (ShaderDataBaseDwords) before them; otherwise the specialization stays baked,
// so no shader data moves to storage. Specializations are made this way, and CompileProgram
// applies it to records made with KYTY_BUFFER_SPECIALIZE=1 or by an older build. `info`: the
// shader's untranslated info.
void NormalizeBufferSpecialization(ResourceSpecialization& specialization, ShaderType stage,
                                   const ShaderInfo& info, uint32_t base_dwords);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
