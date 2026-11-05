#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/assert.h"
#include "common/switches.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;
constexpr uint64_t MaxIndirectDescriptorProbes = 65536u;

bool SpecializationFail(std::string_view message) {
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	// Reject texture descriptors with nonzero reserved bits.
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto base_level = (descriptor.dwords[3] >> 12u) & 0xfu;
		const auto fragments  = (descriptor.dwords[3] >> 16u) & 0xfu;
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && fragments >= 1 && fragments <= 3 &&
		       (r128 || max_mip == fragments);
	}
	return true;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

enum class SamplerClass : uint8_t { Float, Integer, PointInteger };

SamplerClass ClassifySampler(const ImageResource& image) {
	if (image.numeric_class == Prospero::TextureNumericClass::Sint ||
	    (image.numeric_class == Prospero::TextureNumericClass::Uint &&
	     image.conversion_format != Prospero::BufferFormat::kInvalid)) {
		return SamplerClass::PointInteger;
	}
	return image.numeric_class == Prospero::TextureNumericClass::Uint ? SamplerClass::Integer
	                                                               : SamplerClass::Float;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return base <= last ? last - base + 1u : 0u;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

struct ReadCapture {
	SrtRuntime                                  source;
	std::vector<std::pair<uint64_t, uint64_t>>& ranges;
};

bool CaptureStrictRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (!capture.source.read_specialization_memory(SpecializationUserdata(capture.source), address,
	                                               values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool CaptureOrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<ReadCapture*>(userdata);
	if (!capture.source.read_memory(capture.source.userdata, address, values)) {
		return false;
	}
	capture.ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool WrittenBuffersDisjoint(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                            std::span<const std::pair<uint64_t, uint64_t>> reads) {
	for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
		if (!program.info.buffers[i].written) continue;
		ShaderBufferResource buffer;
		if (!DecodeBufferDescriptor(snapshot.buffers[i], buffer)) return false;
		const auto base = buffer.Base48();
		const auto size = buffer.GetSize();
		if (size == 0u) continue;
		if (size - 1u > AddressMask - base) return false;
		for (const auto [address, bytes]: reads) {
			if (bytes == 0u) continue;
			if (address > AddressMask || bytes - 1u > AddressMask - address ||
			    (address <= base ? base - address < bytes : address - base < size)) {
				return false;
			}
		}
	}
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

void MarkCleanFlatSlots(const ResourcePlan& program, const DescriptorSource* source,
                        std::span<uint8_t> slots, Value extra = {}) {
	if (source == nullptr && extra.IsEmpty()) {
		return;
	}
	std::vector<Value>       pending;
	if (source != nullptr) {
		pending.assign(source->dwords.begin(), source->dwords.begin() + source->dword_count);
	}
	if (!extra.IsEmpty()) pending.push_back(extra);
	std::vector<const Inst*> visited;
	while (!pending.empty()) {
		auto value = pending.back().Resolve();
		pending.pop_back();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || std::ranges::find(visited, inst) != visited.end()) {
			continue;
		}
		visited.push_back(inst);
		if (inst->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = inst->Arg(1).Resolve();
			if (slot.IsImmediate() && slot.GetType() == Type::U32 && slot.U32() < slots.size()) {
				slots[slot.U32()] = 1u;
				pending.push_back(program.srt_reads[slot.U32()].value);
			}
			continue;
		}
		for (size_t arg = 0; arg < inst->NumArgs(); arg++) {
			pending.push_back(inst->Arg(arg));
		}
	}
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t dynamic_offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto offset = dynamic_offset & ~uint64_t {3};
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	base &= AddressMask & ~uint64_t {3};
	if (offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(SpecializationUserdata(runtime), address, prefix);
}

// Plan roots for the Value-based SrtWalker and the node-based SrtEvaluator.
struct WalkerRoots {
	using Walker = SrtWalker;
	const ResourcePlan& program;

	Value Fill(uint32_t index) const { return program.uniform_fill.values[index]; }
	Value KeyCount(uint32_t source) const {
		return program.descriptor_sources[source].indirect_descriptor->key_count;
	}
	Value SelectorMask(uint32_t source) const {
		return program.descriptor_sources[source].indirect_descriptor->selector_mask;
	}
	Value SelectorFirst(uint32_t source) const {
		return program.descriptor_sources[source].indirect_descriptor->selector_first;
	}
};

struct CompiledRoots {
	using Walker = SrtEvaluator;
	const CompiledResourcePlan& compiled;

	uint32_t Fill(uint32_t index) const { return compiled.fill[index]; }
	uint32_t KeyCount(uint32_t source) const { return compiled.key_counts[source]; }
	uint32_t SelectorMask(uint32_t source) const { return compiled.selector_masks[source]; }
	uint32_t SelectorFirst(uint32_t source) const { return compiled.selector_firsts[source]; }
};

bool NormalizeIndirectStoreBuffer(DescriptorValue& value) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    descriptor.Base48() == 0u || descriptor.GetSize() == 0u) {
		value.dwords.fill(0u);
	} else if (descriptor.Stride() != 1u || descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	           descriptor.OutOfBounds() != 3u) {
		return SpecializationFail("indirect buffer stores require linear byte bounds");
	}
	return true;
}

// A GPU-selected descriptor: an image or buffer the shader picks from a table by a key it reads
// at run time. Every candidate becomes a resource of its own (the first in place, the others
// appended to `descriptors` and `specializations` as children of `resource_index`), and the
// flattened SRT gets a sorted key -> ordinal table the shader searches. The caller evaluates the
// material and table descriptors; a finite list of candidate sources needs neither.
template <typename Roots, typename Specialization, typename Normalize>
bool MaterializeIndirectDescriptor(const ResourcePlan& program, const Roots& roots, uint32_t source,
                                   const DescriptorSource::IndirectDescriptor& indirect,
                                   const DescriptorValue& material_value,
                                   const DescriptorValue& table_value, uint32_t resource_index,
                                   uint32_t dword_count, const SrtRuntime& runtime,
                                   typename Roots::Walker& clean, ResourceSnapshot& snapshot,
                                   std::vector<DescriptorValue>& descriptors,
                                   std::vector<Specialization>&  specializations,
                                   uint32_t maximum_resources, Normalize&& normalize) {
	const auto  descriptor_bytes = dword_count * sizeof(uint32_t);
	const auto& listed           = indirect.sources;
	auto&       keys             = program.ThreadScratch().material_keys;
	const auto  read_keys = [&](const ShaderBufferResource& material, uint64_t first, uint64_t step,
	                           uint64_t count) {
		keys.resize(count);
		if (step == 4u) {
			return ReadScalarTable(material.Base48(), material.GetSize(), first, runtime, keys);
		}
		for (auto& key: keys) {
			if (!ReadScalarTable(material.Base48(), material.GetSize(), first, runtime, {&key, 1}))
				return false;
			first += step;
		}
		return true;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX; // Scalar addresses have no buffer descriptor bounds.
	if (listed.empty()) {
		keys.clear();
		ShaderBufferResource table;
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return false;
		}
		if (indirect.material_source == UINT32_MAX) {
			uint32_t key_count = 0;
			if (indirect.workgroup_axis != UINT32_MAX) {
				if (indirect.workgroup_axis >= runtime.workgroup_counts.size()) return false;
				key_count = runtime.workgroup_counts[indirect.workgroup_axis];
				if (key_count == 0u) return false;
			} else {
				if (table_value.dword_count != 2u ||
				    !clean.Evaluate(roots.KeyCount(source), key_count))
					return false;
				if (std::bit_cast<int32_t>(key_count) <= 0) key_count = 0;
			}
			if (indirect.table_stride == 0u || key_count > MaxIndirectDescriptorProbes ||
			    (key_count != 0u && uint64_t {indirect.table_offset} +
			                                uint64_t {key_count - 1u} * indirect.table_stride +
			                                descriptor_bytes >
			                            UINT32_MAX + 1ull)) {
				return false;
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), 0u);
		} else if (!indirect.selector_first.IsEmpty()) {
			ShaderBufferResource material;
			uint32_t             first = 0, count = 0;
			if (!DecodeBufferDescriptor(material_value, material) || material.Type() != 0u ||
			    ((indirect.selector_shift != 0u || indirect.selector_bits != UINT32_MAX) &&
			     (material.Base48() & 3u) != 0u) ||
			    material.SwizzleEnabled() || material.AddTid() || material.OutOfBounds() != 0u ||
			    uint64_t {indirect.selector_offset} + 4u > material.Stride() ||
			    !clean.Evaluate(roots.SelectorFirst(source), first) ||
			    !clean.Evaluate(roots.KeyCount(source), count) ||
			    count > MaxIndirectDescriptorProbes || uint64_t {first} + count > material.NumRecords())
				return false;
			if (count != 0u && (uint64_t {first} + count - 1u) * material.Stride() +
			                           indirect.selector_offset + 4u >
			                       uint64_t {UINT32_MAX} + 1u)
				return false;
			if (!read_keys(material, uint64_t {first} * material.Stride() + indirect.selector_offset,
			               material.Stride(), count))
				return false;
		} else if (!indirect.selector_mask.IsEmpty()) {
			uint32_t mask  = 0;
			uint32_t count = 0;
			if (material_value.dword_count != 2u || table_value.dword_count != 2u ||
			    !clean.Evaluate(roots.SelectorMask(source), mask) ||
			    !clean.Evaluate(roots.KeyCount(source), count) || count == 0u || count > 32u) {
				return false;
			}
			if (count < 32u) mask &= (1u << count) - 1u;
			const auto material_base =
			    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
			keys.reserve(std::popcount(mask));
			while (mask != 0u) {
				const auto index  = std::countr_zero(mask);
				const auto offset = static_cast<uint64_t>(indirect.selector_offset) +
				                    static_cast<uint64_t>(index) * indirect.selector_stride;
				if (offset > UINT32_MAX) return false;
				uint32_t key = 0;
				if (!ReadScalarTable(material_base, UINT64_MAX, static_cast<uint32_t>(offset),
				                     runtime, {&key, 1}))
					return false;
				keys.push_back(key);
				mask &= mask - 1u;
			}
		} else {
			ShaderBufferResource material;
			if (!DecodeBufferDescriptor(material_value, material) || table_value.dword_count != 4u) {
				return false;
			}
			// The first aligned offset includes the immediate added after shader U32 arithmetic.
			const auto step = std::max<uint64_t>(
			    4u, std::gcd<uint64_t>(indirect.selector_stride, uint64_t {1} << 32u));
			const uint64_t first = indirect.selector_offset;
			const auto     size  = material.GetSize();
			const auto     limit =
			    std::min(first + (uint64_t {1} << 32u) - step, size >= 4u ? size - 4u : 0u);
			const auto probe_count =
			    size >= 4u && first <= limit ? (limit - first) / step + 1u : 0u;
			if (probe_count > MaxIndirectDescriptorProbes) {
				return false;
			}
			if (!read_keys(material, first, step, probe_count)) return false;
			if (indirect.selector_shift == 0u) keys.push_back(0u);
		}
		if (indirect.material_source != UINT32_MAX) {
			for (auto& key: keys) key = (key >> indirect.selector_shift) & indirect.selector_bits;
			if (indirect.selector_shift != 0u || indirect.selector_bits != UINT32_MAX) {
				keys.push_back(0u); // An out-of-range material load returns zero.
			}
			std::ranges::sort(keys);
			keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
		}
	}

	const auto children_begin = descriptors.size();
	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto root_resource  = specializations[resource_index];
	const auto key_count      = listed.empty() ? keys.size() : listed.size();
	snapshot.flattened_srt.resize(mapping_offset + 1u + key_count * 2u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto      key = listed.empty() ? keys[entry] : entry;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (listed.empty()) {
			const auto table_offset =
			    static_cast<uint32_t>(key * indirect.table_stride) + indirect.table_offset;
			if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
			                     std::span(candidate.dwords).first(dword_count))) {
				return false;
			}
		} else if (!clean.EvaluateDescriptor(listed[entry], candidate)) {
			return false;
		}
		if (!normalize(candidate)) return false;
		uint32_t ordinal = 0;
		if (entry == 0) {
			descriptors[resource_index] = candidate;
		} else if (descriptors[resource_index] != candidate) {
			const auto found =
			    std::find(descriptors.begin() + children_begin, descriptors.end(), candidate);
			ordinal = static_cast<uint32_t>(found - descriptors.begin() - children_begin + 1u);
			if (found == descriptors.end()) {
				if (descriptors.size() >= maximum_resources) {
					return false;
				}
				descriptors.push_back(candidate);
				auto child          = root_resource;
				child.indirect_root = resource_index;
				specializations.push_back(child);
			}
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	if (descriptors.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		root.indirect_mapping_offset    = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(key_count);
	}
	return true;
}

// The numeric class of the T#s of a bindless heap, from its first records: integer when most of
// the non-null ones are 2D or 2D-array textures of one integer class (Ghost of Yotei: heaps of
// R8_UINT terrain maps, whose first records can be null), so the shader reads them with an
// integer result type. Float otherwise, or when the records cannot be read. Two float T#s before
// any integer one end the probe early, as do four integer ones before any float one.
Prospero::TextureNumericClass ProbeBindlessHeapClass(const SrtRuntime& runtime, uint64_t heap_base,
                                                     uint64_t heap_size, uint32_t record_stride,
                                                     const DescriptorSource::IndirectDescriptor& indirect) {
	constexpr uint32_t ProbedRecords = 32;
	// Every read goes through the strict reader, which fails on unbacked (and GPU-written)
	// memory: the probed records lie past what the heap's descriptor bounds (a bare address has
	// none) and a followed pointer can be garbage. The ordinary reader reads the guest address
	// and faulted on one (a fresh Ghost of Yotei start crashed here).
	if (runtime.read_specialization_memory == nullptr || record_stride == 0u) {
		return Prospero::TextureNumericClass::Float;
	}
	const auto read = [&](uint64_t address, std::span<uint32_t> values) {
		return address >= 0x100000u &&
		       runtime.read_specialization_memory(SpecializationUserdata(runtime), address, values);
	};
	auto     integer_class = Prospero::TextureNumericClass::Float;
	uint32_t integers = 0, floats = 0, others = 0, nulls = 0, probed = 0, unreadable = 0;
	for (uint32_t key = 0; key < ProbedRecords; key++) {
		const auto offset = static_cast<uint64_t>(key) * record_stride;
		if (heap_size != 0u && offset + indirect.table_offset + 16u > heap_size &&
		    indirect.pointer_offset == UINT32_MAX) {
			break;
		}
		const auto record  = heap_base + offset;
		auto       address = record + indirect.table_offset;
		if (indirect.pointer_offset != UINT32_MAX) {
			std::array<uint32_t, 2> pointer {};
			if (!read(record + indirect.pointer_offset, pointer)) {
				unreadable++; // GPU-written or unbacked: look at the next record
				continue;
			}
			const auto target =
			    ((static_cast<uint64_t>(pointer[1]) << 32u) | pointer[0]) & AddressMask;
			if (target == 0u) {
				nulls++;
				continue;
			}
			address = target + indirect.table_offset;
		}
		std::array<uint32_t, 4> dwords {};
		if (!read(address, dwords)) {
			unreadable++;
			continue;
		}
		probed++;
		const auto format = static_cast<Prospero::BufferFormat>((dwords[1] >> 20u) & 0x1ffu);
		if (format == Prospero::BufferFormat::kInvalid) {
			nulls++;
			continue;
		}
		const auto type          = static_cast<Prospero::ImageType>((dwords[3] >> 28u) & 0xfu);
		const auto numeric_class = Prospero::SampledTextureNumericClass(format);
		const bool flat = type == Prospero::ImageType::kColor2D ||
		                  type == Prospero::ImageType::kColor2DArray;
		if (numeric_class == Prospero::TextureNumericClass::Float) {
			floats++;
		} else if (flat && (numeric_class == Prospero::TextureNumericClass::Uint ||
		                    numeric_class == Prospero::TextureNumericClass::Sint) &&
		           (integers == 0u || numeric_class == integer_class)) {
			integers++;
			integer_class = numeric_class;
		} else {
			others++;
		}
		if ((integers == 0u && floats >= 2u) || (floats == 0u && others == 0u && integers >= 4u)) {
			break;
		}
	}
	const auto result = integers > floats + others ? integer_class
	                                               : Prospero::TextureNumericClass::Float;
	return result;
}

// The shader looks the key up in the bindless translation table: nothing to enumerate. Record
// the heap for the host and reserve the two words it patches (region base, entry count); zeros
// resolve every key to the placeholder in slot 0.
template <typename Specialization>
bool MaterializeBindlessImage(const DescriptorSource::IndirectDescriptor& indirect,
                              const DescriptorValue& table_value, uint32_t image_index,
                              const SrtRuntime& runtime, ResourceSnapshot& snapshot,
                              std::vector<Specialization>& specializations) {
	uint64_t heap_base     = 0;
	uint64_t heap_size     = 0;
	uint32_t record_stride = indirect.table_stride;
	if (indirect.stride_from_buffer && table_value.dword_count != 4u) {
		return false;
	}
	if (table_value.dword_count == 2u) {
		// A bare address has no bounds: the heap covers a fixed number of records. Keys are
		// resolved when the shader names them, so only those records are read.
		constexpr uint64_t AddressHeapKeys = 8192;
		heap_base = ((static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0]) &
		            AddressMask;
		heap_size = indirect.table_offset + AddressHeapKeys * std::max(indirect.table_stride, 32u);
		if (heap_base == 0u) {
			return false;
		}
	} else {
		ShaderBufferResource heap;
		if (table_value.dword_count != 4u || !DecodeBufferDescriptor(table_value, heap)) {
			return false;
		}
		heap_base = heap.Base48();
		heap_size = heap.GetSize();
		if (indirect.stride_from_buffer) {
			// Records of an indexed vector load: the V# gives their stride.
			record_stride = heap.Stride();
			if (record_stride < 8u) {
				return false;
			}
		}
	}
	const auto mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
	snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
	snapshot.images[image_index] = {.dword_count = 8u};
	snapshot.bindless_heaps.push_back({.base           = heap_base,
	                                   .size           = heap_size,
	                                   .table_offset   = indirect.table_offset,
	                                   .record_stride  = record_stride,
	                                   .image          = image_index,
	                                   .mapping_offset = mapping_offset,
	                                   .pointer_offset = indirect.pointer_offset,
	                                   .sub_stride     = indirect.sub_stride,
	                                   .sub_keys       = indirect.sub_keys});
	auto& root                      = specializations[image_index];
	root.indirect_root              = image_index;
	root.indirect_mapping_offset    = mapping_offset;
	root.indirect_search_iterations = 0;
	root.bindless                   = true;
	root.numeric_class              = ProbeBindlessHeapClass(runtime, heap_base, heap_size, record_stride,
	                                                    indirect);
	return true;
}

} // namespace

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::array<std::array<uint32_t, 3>, ShaderInfo::MaxSamplers> mapping;
	std::array<Binding, ShaderInfo::MaxSamplers>                bindings;
	uint32_t                                                  sampler_count = 0;
};

template <typename T, typename Keep>
void CompactImages(std::vector<T>& images, Keep&& keep) {
	size_t count = 0;
	for (size_t index = 0; index < images.size(); ++index) {
		if (!keep(index)) continue;
		if (count != index) images[count] = std::move(images[index]);
		++count;
	}
	images.resize(count);
}

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : indices(specialization.images.size()),
	      source_count(static_cast<uint32_t>(specialization.images.size())) {
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		CompactImages(images, [&](size_t index) { return indices[index] != UINT32_MAX; });
	}

private:
	std::vector<uint32_t> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization) {
	if (specialization.buffers.size() != snapshot.buffers.size()) {
		return SpecializationFail("buffer specialization does not match the snapshot");
	}
	for (uint32_t i = 0; i < snapshot.buffers.size(); i++) {
		auto&      buffer     = specialization.buffers[i];
		const auto base_index = i < program.info.buffers.size() ? i : buffer.indirect_root;
		if (base_index >= program.info.buffers.size()) {
			return SpecializationFail(fmt::format("buffer resource {} has an invalid root", i));
		}
		const auto&          base             = program.info.buffers[base_index];
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		const auto packed_stride = CanonicalPackedStride(descriptor_value);
		const auto stride        = packed_stride & 0x3fffu;
		buffer.packed_stride = packed_stride;
		buffer.descriptor_format =
		    base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
	}
	NormalizeBufferSpecialization(specialization, program.stage, program.info,
	                              program.shader_data_base_dwords);
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = ImageMipCount(base, descriptor);
		if (image.mip_count == 0u) {
			return SpecializationFail(
			    fmt::format("image descriptor {} has an invalid mip range", i));
		}
		if (NullImageDescriptor(descriptor)) {
			// A bindless image keeps the class its heap's records showed.
			const auto heap_class = image.numeric_class;
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			image.dimension     = Decoder::ImageDimension::Dim2D;
			image.cube          = false;
			if (image.bindless) {
				// No one descriptor describes a bindless image: keep the dimension the
				// instruction samples, which picks the bindless array.
				image.dimension = base.dimension == Decoder::ImageDimension::Unknown
				                      ? Decoder::ImageDimension::Dim2D
				                      : base.dimension;
				image.cube      = base.cube;
				// Integer arrays exist for 2D and 2D-array images only, and a depth compare
				// samples float.
				const bool integer_array =
				    !image.cube && !base.depth_compare && !base.atomic &&
				    base.resource_class == ImageResourceClass::Sampled &&
				    (image.dimension == Decoder::ImageDimension::Dim2D ||
				     image.dimension == Decoder::ImageDimension::Dim2DArray);
				if (integer_array && (heap_class == Prospero::TextureNumericClass::Uint ||
				                      heap_class == Prospero::TextureNumericClass::Sint)) {
					image.numeric_class = heap_class;
				}
			}
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32Float)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		image.conversion_format = ImageConversionFormat(format);
		image.clamp_half_float =
		    storage && base.written && !base.atomic && FloatStoreClampEnabled() &&
		    (format == Prospero::BufferFormat::k16Float || format == Prospero::BufferFormat::k16_16Float ||
		     format == Prospero::BufferFormat::k16_16_16_16Float);
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		image.numeric_class         = Prospero::SampledTextureNumericClass(format);
		// A write-only image of a signed integer format is written through its unsigned view
		// (the stored bits are the same).
		const bool raw_sint_storage = storage &&
		                              image.numeric_class == Prospero::TextureNumericClass::Sint &&
		                              base.written && !base.read && !base.atomic;
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {} (read={} "
				                "written={} atomic={})",
				                i, static_cast<uint32_t>(format), base.read, base.written,
				                base.atomic));
			}
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			return SpecializationFail(
			    fmt::format("sampled image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index || root.bindless) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		if (root.indirect_search_iterations == 0u || key_count < 2u ||
		    static_cast<size_t>(root.indirect_mapping_offset) + 1u +
		            static_cast<size_t>(key_count) * 2u >
		        snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			if (NullImageDescriptor(snapshot.images[candidate])) {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			if (image.numeric_class != image_class.numeric_class ||
			    (!same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension))) ||
			    image.mip_count != image_class.mip_count ||
			    image.conversion_format != image_class.conversion_format ||
			    image.shader_swizzle != image_class.shader_swizzle) {
				// A table whose other records hold unrelated data (Ghost of Yotei 01a51b1204e3be1b
				// enumerates all 32 records of a 5-bit key): those candidates become null images,
				// so a key that selects one reads zero or drops the write.
				// KYTY_STRICT_INDIRECT_CANDIDATES=1 fails the specialization instead (the old way).
				static const bool strict = KYTY_GETENV("KYTY_STRICT_INDIRECT_CANDIDATES") != nullptr;
				if (strict) {
					return SpecializationFail(
					    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
					                program.info.images[root_index].first_use_pc));
				}
				snapshot.images[candidate].dwords[0] = 0;
				snapshot.images[candidate].dwords[1] &= ~0xffu;
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			}
		}
	}
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

// `images`: the specialized images, which ApplyResourceSpecialization stores into the program
// after the plan is made.
bool BuildSamplerPlan(const ShaderInfo& base, std::span<const ImageResource> images,
                      SamplerPlan& plan) {
	if (base.samplers.size() > plan.mapping.size()) {
		return false;
	}
	std::array<uint8_t, ShaderInfo::MaxSamplers> usage {};
	plan.sampler_count = static_cast<uint32_t>(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(images[pair.image]));
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			const auto target = first ? index : plan.sampler_count++;
			if (target >= ShaderInfo::MaxSamplers) return false;
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

static std::vector<ResourceBlock> ResourceControlFlow(const Program& program) {
	// Any shader write may alias a scalar predicate read, including on a later loop visit.
	if (program.blocks.size() != program.block_info.size() || HasShaderMemoryWrites(program)) {
		return {};
	}
	if (program.blocks.size() > (1u << 20u)) {
		std::printf("ResourceControlFlow: hash=0x%016llx implausible block count %zu\n",
		            static_cast<unsigned long long>(program.shader_hash), program.blocks.size());
		std::fflush(stdout);
		return {};
	}
	std::unordered_map<uint32_t, uint32_t> indices;
	for (uint32_t i = 0; i < program.block_info.size(); i++) {
		if (!indices.emplace(program.block_info[i].id, i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&           info       = program.block_info[i];
		const auto&           terminator = info.terminator;
		std::vector<uint32_t> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors.push_back(terminator.true_block); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors = {terminator.true_block, terminator.false_block};
				if (ValidateRuntimeValue(program, info.condition, RuntimeValueType::Integer)) {
					block.condition = info.condition;
				}
				break;
			case CFG::TerminatorKind::IndirectBranch:
				successors = terminator.indirect_targets;
				break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			// A resource this plan cannot name (seen: a sampler index past the sampler list in a
			// Ghost of Yotei shader after the calligraphy scene) gives up the control-flow plan,
			// which only refines materialization, instead of throwing out of the session.
			const auto memory_index = inst.Flags<MemoryFlags>().index;
			if (memory_index >= program.memory_info.size()) {
				return {};
			}
			const auto& memory = program.memory_info[memory_index];
			if (memory.planning_only) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				if (memory.resource >= program.info.buffers.size()) {
					return {};
				}
				block.sources.push_back(program.info.buffers[memory.resource].source);
			} else {
				if (memory.resource >= program.info.images.size() ||
				    (image.needs_sampler && memory.sampler >= program.info.samplers.size())) {
					std::printf("ResourceControlFlow: hash=0x%016llx image %u/%zu sampler %u/%zu out of "
					            "range; no control-flow plan\n",
					            static_cast<unsigned long long>(program.shader_hash), memory.resource,
					            program.info.images.size(), memory.sampler,
					            program.info.samplers.size());
					return {};
				}
				block.sources.push_back(program.info.images[memory.resource].source);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers[memory.sampler].source);
				}
			}
		}
		std::ranges::sort(block.sources);
		block.sources.erase(std::unique(block.sources.begin(), block.sources.end()),
		                    block.sources.end());
	}
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

// ResolveComputeBufferFill proves complete workgroups and matching requested extents, so every
// dispatch-bound predicate accepted here is true for every invocation covered by the fill.
static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

// Nonnegative affine coefficients for constant, local and workgroup coordinates. Reject modular
// arithmetic that could wrap; runtime coverage also bounds the largest invocation index.
static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard,
                                                      uint32_t depth = 0) {
	value = ResolveActiveU32(value, guard);
	if (depth > 32 || value.GetType() != Type::U32) {
		return {};
	}
	if (value.IsImmediate()) {
		return std::array<uint64_t, 3> {value.U32(), 0, 0};
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return {};
	}
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::GetBuiltin && inst->Arg(1) == Value(axis)) {
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) {
			return std::array<uint64_t, 3> {0, 1, 0};
		}
		if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) {
			return std::array<uint64_t, 3> {0, 0, 1};
		}
	}
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) {
		return {};
	}
	auto left  = FillIndex(inst->Arg(0), axis, guard, depth + 1);
	auto right = FillIndex(inst->Arg(1), axis, guard, depth + 1);
	if (!left || !right) {
		return {};
	}
	if (op == ValueOpcode::IMul32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		std::swap(left, right);
	}
	if (op != ValueOpcode::IAdd32 && ((*right)[1] != 0 || (*right)[2] != 0)) {
		return {};
	}
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (uint32_t i = 0; i < left->size(); ++i) {
		(*left)[i] =
		    op == ValueOpcode::IAdd32 ? (*left)[i] + (*right)[i] : (*left)[i] * (*right)[0];
		if ((*left)[i] > UINT32_MAX) return {};
	}
	return left;
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() ||
	    program.blocks.size() != program.block_info.size() || program.info.uses_dma ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<uint32_t> visited;
	uint32_t                     index = 0;
	const Inst*                  store = nullptr;
	for (;;) {
		if (!visited.insert(index).second) return {};
		for (const auto& inst: *program.blocks[index]) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = program.block_info[index].terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		const auto next = std::ranges::find(program.block_info, term.true_block, &BlockInfo::id);
		if (next == program.block_info.end()) return {};
		index = static_cast<uint32_t>(next - program.block_info.begin());
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    store->Arg(3).Resolve() != Value(true)) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.shader_data_base_dwords    = ShaderDataBaseDwords(program);
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    ValidateRuntimeValue(program, source->Arg(0))) {
			// Uniform values need no new EXEC context; preserve their shared evaluation memo.
			return Clone(source->Arg(0));
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		if (source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			plan.requires_specialization_memory = true;
			plan.capture_specialization_reads   = true;
		}
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	// KYTY_DEBUG_PLAN_VOID=1: plan values that point at removed (Void) instructions, by shader and
	// place (see PlanRoots in DeadCodeElimination.cpp, KYTY_DCE_PLAN_ROOTS).
	static const bool void_log = KYTY_GETENV("KYTY_DEBUG_PLAN_VOID") != nullptr;
	const auto        note_void = [&](Value value, const char* where, size_t index) {
		const auto* inst = value.Resolve().TryInstruction();
		if (void_log && inst != nullptr && inst->GetOpcode() == ValueOpcode::Void) [[unlikely]] {
			std::printf("plan-void: shader=0x%016llx stage=%u %s %zu\n",
			            static_cast<unsigned long long>(program.shader_hash),
			            static_cast<uint32_t>(program.stage), where, index);
			std::fflush(stdout);
		}
	};
	for (size_t i = 0; i < program.descriptor_sources.size(); i++) {
		const auto& source = program.descriptor_sources[i];
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			note_void(source.dwords[dword], "descriptor-dword", i * 8u + dword);
		}
		if (source.indirect_descriptor.has_value()) {
			note_void(source.indirect_descriptor->key_count, "indirect-key-count", i);
			note_void(source.indirect_descriptor->selector_first, "indirect-selector-first", i);
			note_void(source.indirect_descriptor->selector_mask, "indirect-selector-mask", i);
		}
	}
	for (size_t i = 0; i < program.srt_reads.size(); i++) {
		note_void(program.srt_reads[i].value, "srt-read", i);
	}

	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		if (target.indirect_descriptor.has_value()) {
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	plan.control_flow = ResourceControlFlow(program);
	for (auto& block: plan.control_flow) {
		block.condition = Clone(block.condition);
	}
	plan.uniform_fill = AnalyzeUniformFill(program);
	for (uint32_t i = 0; i < plan.uniform_fill.fill.words; ++i) {
		plan.uniform_fill.values[i] = Clone(plan.uniform_fill.values[i]);
	}
	plan.clean_flat_slots.resize(plan.srt_reads.size());
	bool masked_image     = false;
	bool capture_indirect = false;
	for (const auto& source: plan.descriptor_sources) {
		if (!source.indirect_descriptor.has_value()) continue;
		const auto& indirect                = *source.indirect_descriptor;
		plan.requires_specialization_memory = true;
		masked_image |= !indirect.selector_mask.IsEmpty();
		// Keys read from a material buffer at a GPU-chosen first record (upstream 5c830604): the
		// reads are captured so that shader writes cannot change them.
		capture_indirect |= !indirect.selector_first.IsEmpty();
		MarkCleanFlatSlots(plan, Source(plan, indirect.material_source), plan.clean_flat_slots,
		                   indirect.selector_mask);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.selector_first);
		MarkCleanFlatSlots(plan, nullptr, plan.clean_flat_slots, indirect.key_count);
		if (indirect.sources.empty()) {
			MarkCleanFlatSlots(plan, Source(plan, indirect.table_source), plan.clean_flat_slots);
		} else {
			for (const auto candidate: indirect.sources) {
				MarkCleanFlatSlots(plan, Source(plan, candidate), plan.clean_flat_slots);
			}
		}
	}
	if (masked_image) {
		plan.resource_tracking_complete &= !program.has_address_writes &&
		    !std::ranges::any_of(plan.info.images, &ImageResource::written);
		plan.capture_specialization_reads |=
		    std::ranges::any_of(plan.info.buffers, &BufferResource::written);
	}
	if (capture_indirect) {
		plan.capture_specialization_reads = true;
		plan.resource_tracking_complete &= !program.has_address_writes;
		// Clean writable addresses prove that shader writes cannot overlap captured reads.
		for (const auto& buffer: plan.info.buffers) {
			if (buffer.written) MarkCleanFlatSlots(plan, Source(plan, buffer.source), plan.clean_flat_slots);
		}
		for (const auto& image: plan.info.images) {
			if (image.written) MarkCleanFlatSlots(plan, Source(plan, image.source), plan.clean_flat_slots);
		}
	}
	return plan;
}

namespace {

bool g_verify_materialization = false;

bool MaterializationAllowed(const ResourcePlan& program, const SrtRuntime& runtime,
                            bool& capture_reads) {
	if (!program.resource_tracking_complete ||
	    (program.requires_specialization_memory && runtime.read_specialization_memory == nullptr)) {
		return false;
	}
	// ExtractResourcePlan folds the masked-image write checks into resource_tracking_complete.
	capture_reads = program.capture_specialization_reads;
	return true;
}

// Serves one refresh's strict reads from whole 64-byte blocks. Conditions and predicates usually
// read neighboring SRT or constant-buffer dwords, and each strict read pays for the GPU-dirty
// checks. A block that cannot be read whole falls back to exact reads.
class CleanBlockCache {
public:
	static constexpr uint64_t BlockBytes = 64;

	explicit CleanBlockCache(const SrtRuntime& source): m_source(source) {}

	// Direct reads keep their reader and userdata.
	SrtRuntime Runtime() {
		auto runtime                       = m_source;
		runtime.specialization_userdata    = this;
		runtime.read_specialization_memory = Read;
		return runtime;
	}

private:
	static bool Read(void* userdata, uint64_t address, std::span<uint32_t> values) {
		return static_cast<CleanBlockCache*>(userdata)->Read(address, values);
	}

	static constexpr uint32_t BlockWords = BlockBytes / sizeof(uint32_t);
	static constexpr uint32_t Capacity   = 8;

	struct Block {
		uint64_t                            address = 0;
		std::array<uint32_t, BlockWords>    words;
		bool                                clean = false;
	};

	bool Exact(uint64_t address, std::span<uint32_t> values) const {
		return m_source.read_specialization_memory(SpecializationUserdata(m_source), address, values);
	}

	bool Read(uint64_t address, std::span<uint32_t> values) {
		const auto block_address = address & ~(BlockBytes - 1u);
		const auto offset        = address - block_address;
		if (block_address == 0 || (address & 3u) != 0u || values.empty() ||
		    values.size_bytes() > BlockBytes - offset) {
			return Exact(address, values);
		}
		Block* block = nullptr;
		for (uint32_t index = 0; index < m_count; index++) {
			if (m_blocks[index].address == block_address) {
				block = &m_blocks[index];
				break;
			}
		}
		if (block == nullptr) {
			block  = &m_blocks[m_next];
			m_next = (m_next + 1u) % Capacity;
			if (m_count < Capacity) {
				m_count++;
			}
			block->address = block_address;
			block->clean   = Exact(block_address, block->words);
		}
		if (!block->clean) {
			return Exact(address, values);
		}
		// Usually one dword: avoid a memmove call.
		const auto* words = block->words.data() + offset / sizeof(uint32_t);
		for (size_t index = 0; index < values.size(); index++) {
			values[index] = words[index];
		}
		return true;
	}

	SrtRuntime                   m_source;
	std::array<Block, Capacity>  m_blocks;
	uint32_t                     m_count = 0;
	uint32_t                     m_next  = 0;
};

// Ordinary reads by 64-byte block: descriptors and tables are runs of dwords, so a refresh reads
// each block once. A block the runtime cannot read whole is read dword by dword.
class OrdinaryBlockCache {
public:
	static constexpr uint64_t BlockBytes = 64;

	explicit OrdinaryBlockCache(const SrtRuntime& source): m_source(source) {}

	// Specialization reads keep their reader and userdata.
	SrtRuntime Runtime() {
		auto runtime                    = m_source;
		runtime.specialization_userdata = SpecializationUserdata(m_source);
		runtime.userdata                = this;
		runtime.read_memory             = Read;
		runtime.read_memory_block       = nullptr;
		return runtime;
	}

private:
	static bool Read(void* userdata, uint64_t address, std::span<uint32_t> values) {
		return static_cast<OrdinaryBlockCache*>(userdata)->Read(address, values);
	}

	static constexpr uint32_t BlockWords = BlockBytes / sizeof(uint32_t);
	static constexpr uint32_t Capacity   = 8;

	struct Block {
		uint64_t                         address = 0;
		std::array<uint32_t, BlockWords> words;
		bool                             whole = false;
	};

	bool Exact(uint64_t address, std::span<uint32_t> values) const {
		return m_source.read_memory(m_source.userdata, address, values);
	}

	bool Read(uint64_t address, std::span<uint32_t> values) {
		const auto block_address = address & ~(BlockBytes - 1u);
		const auto offset        = address - block_address;
		if (block_address == 0 || (address & 3u) != 0u || values.empty() ||
		    values.size_bytes() > BlockBytes - offset) {
			return Exact(address, values);
		}
		Block* block = nullptr;
		for (uint32_t index = 0; index < m_count; index++) {
			if (m_blocks[index].address == block_address) {
				block = &m_blocks[index];
				break;
			}
		}
		if (block == nullptr) {
			block  = &m_blocks[m_next];
			m_next = (m_next + 1u) % Capacity;
			if (m_count < Capacity) {
				m_count++;
			}
			block->address = block_address;
			block->whole   = m_source.read_memory_block(m_source.userdata, block_address, block->words);
		}
		if (!block->whole) {
			return Exact(address, values);
		}
		const auto* words = block->words.data() + offset / sizeof(uint32_t);
		for (size_t index = 0; index < values.size(); index++) {
			values[index] = words[index];
		}
		return true;
	}

	SrtRuntime                  m_source;
	std::array<Block, Capacity> m_blocks;
	uint32_t                    m_count = 0;
	uint32_t                    m_next  = 0;
};

SrtRuntime ObservedRuntime(const SrtRuntime& runtime, bool capture_reads, ReadCapture& capture) {
	SrtRuntime observed = runtime;
	if (capture_reads) {
		capture.ranges.clear();
		observed.userdata = &capture;
		observed.specialization_userdata = nullptr;
		observed.read_specialization_memory = CaptureStrictRead;
		if (observed.read_memory != nullptr) observed.read_memory = CaptureOrdinaryRead;
	}
	return observed;
}

// Stores the descriptor inputs of a memoizable plan (see CompiledResourcePlan::memoizable) and
// reports whether a valid key already held them.
bool UpdateMemoKey(const CompiledResourcePlan& compiled, const SrtRuntime& runtime,
                   std::span<const uint8_t> active, std::span<const uint32_t> flat,
                   MaterializationMemo& memo) {
	const auto size = 2u + (active.size() + 31u) / 32u + (compiled.memo_shader_base ? 2u : 0u) +
	                  compiled.memo_user_data.size() + compiled.memo_slots.size();
	auto& key  = memo.key;
	bool  same = memo.valid && key.size() == size;
	key.resize(size);
	auto*      out = key.data();
	const auto put = [&](uint32_t value) {
		same &= *out == value;
		*out++ = value;
	};
	put(static_cast<uint32_t>(runtime.user_data.size()));
	put(static_cast<uint32_t>(active.size()));
	for (size_t base = 0; base < active.size(); base += 32u) {
		uint32_t bits = 0;
		for (size_t index = base; index < active.size() && index < base + 32u; index++) {
			bits |= static_cast<uint32_t>(active[index] != 0u) << (index - base);
		}
		put(bits);
	}
	if (compiled.memo_shader_base) {
		put(static_cast<uint32_t>(runtime.shader_base));
		put(static_cast<uint32_t>(runtime.shader_base >> 32u));
	}
	for (const auto index: compiled.memo_user_data) {
		put(index < runtime.user_data.size() ? runtime.user_data[index] : 0u);
	}
	for (const auto slot: compiled.memo_slots) {
		put(flat[slot]);
	}
	return same;
}

// KYTY_DEBUG_SRT_FAIL=1 reports the first 32 failed materializations by stage.
bool ReportMaterializeFailure(const ResourcePlan& program, const char* stage, uint32_t index) {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_SRT_FAIL") != nullptr;
	static std::atomic<uint32_t> reported {0};
	if (enabled && reported.fetch_add(1, std::memory_order_relaxed) < 32u) {
		std::printf("materialize-failed: shader=0x%016llx stage=%s index=%u\n",
		            static_cast<unsigned long long>(program.shader_hash), stage, index);
		std::fflush(stdout);
	}
	return false;
}

template <typename Roots>
bool Materialize(const ResourcePlan& program, const Roots& roots, const SrtRuntime& runtime,
                 const SrtRuntime& observed, bool capture_reads, typename Roots::Walker& clean,
                 typename Roots::Walker& walker, ResourceSnapshot& snapshot,
                 ResourceSpecialization& specialization, MaterializationMemo* memo) {
	auto& reads = program.ThreadScratch().specialization_reads;
	// The ordinary walker sends conditions that are not direct to its strict walker.
	const auto active = walker.FindActiveSources();
	// KYTY_DEBUG_ACTIVE_SOURCES=<hex hash>: the descriptor sources this shader's control flow keeps
	// active, per materialization (first 400), to find a path whose resources are pruned wrongly.
	{
		static const uint64_t watch = [] {
			const char* text = KYTY_GETENV("KYTY_DEBUG_ACTIVE_SOURCES");
			return text != nullptr ? std::strtoull(text, nullptr, 16) : 0ull;
		}();
		static std::atomic<uint32_t> printed {0};
		if (watch != 0 && program.shader_hash == watch && printed.fetch_add(1) < 400) [[unlikely]] {
			std::string bits;
			for (const auto a: active) {
				bits += a != 0 ? '1' : '0';
			}
			std::printf("active-sources: shader=0x%016llx sources=%zu active=%s%s\n",
			            static_cast<unsigned long long>(program.shader_hash), program.descriptor_sources.size(),
			            active.empty() ? "(all)" : bits.c_str(), memo != nullptr ? " memo" : "");
			std::fflush(stdout);
		}
	}
	if (!walker.RefreshFlatBuffer(snapshot.flattened_srt)) {
		return ReportMaterializeFailure(program, "flat refresh", 0);
	}
	snapshot.uniform_fill = {};
	const auto& fill = program.uniform_fill;
	const auto words = fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool evaluated = words != 0;
	bool uniform   = true;
	for (uint32_t i = 0; i < words && evaluated; ++i) {
		evaluated = clean.Evaluate(roots.Fill(i), stored[i]);
		uniform   = uniform && stored[i] == stored[0];
	}
	if (evaluated) {
		snapshot.uniform_fill         = fill.fill;
		snapshot.uniform_fill.value   = stored[0];
		snapshot.uniform_fill.uniform = uniform;
	}
	if constexpr (std::is_same_v<Roots, CompiledRoots>) {
		if (memo != nullptr) {
			memo->reused = false;
			if (roots.compiled.memoizable &&
			    UpdateMemoKey(roots.compiled, runtime, active, snapshot.flattened_srt, *memo)) {
				snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
				memo->reused = true;
				return true;
			}
			// The key now holds this refresh's inputs; it becomes valid only if the refresh
			// succeeds.
			memo->valid = false;
		}
	}
	snapshot.bindless_heaps.clear();
	snapshot.bindless_sampler_heaps.clear();
	const auto evaluate = [&](uint32_t source, DescriptorValue& value) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (active.empty() || active[source]) {
			return walker.EvaluateDescriptor(source, value);
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	// The material and table descriptors of a GPU-selected resource, evaluated strictly.
	const auto indirect_inputs = [&](const DescriptorSource::IndirectDescriptor& indirect,
	                                 DescriptorValue& material, DescriptorValue& table) {
		return !indirect.sources.empty() ||
		       ((indirect.material_source == UINT32_MAX ||
		         clean.EvaluateDescriptor(indirect.material_source, material)) &&
		        clean.EvaluateDescriptor(indirect.table_source, table));
	};
	snapshot.buffers.resize(program.info.buffers.size());
	// Children of GPU-selected buffers are appended after the program's own buffers.
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
		const auto& buffer = program.info.buffers[i];
		const auto* source = Source(program, buffer.source);
		if (source == nullptr) {
			return ReportMaterializeFailure(program, "buffer", i);
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.buffers[i] = {.dword_count = 4u};
			if (!active.empty() && !active[buffer.source]) {
				continue;
			}
			const auto&     indirect = *source->indirect_descriptor;
			DescriptorValue material;
			DescriptorValue table;
			if (!indirect_inputs(indirect, material, table) ||
			    !MaterializeIndirectDescriptor(program, roots, buffer.source, indirect, material,
			                                   table, i, 4u, observed, clean, snapshot,
			                                   snapshot.buffers, specialization.buffers,
			                                   ShaderInfo::MaxBuffers, NormalizeIndirectStoreBuffer)) {
				return ReportMaterializeFailure(program, "indirect buffer", i);
			}
		} else if (!evaluate(buffer.source, snapshot.buffers[i])) {
			return ReportMaterializeFailure(program, "buffer", i);
		}
	}
	if (capture_reads) {
		for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
			const auto& buffer = program.info.buffers[i];
			if (!buffer.written || (!active.empty() && !active[buffer.source])) continue;
			if (const auto* source = Source(program, buffer.source);
			    source == nullptr || source->indirect_descriptor.has_value()) continue;
			DescriptorValue strict;
			if (!clean.EvaluateDescriptor(buffer.source, strict) ||
			    strict != snapshot.buffers[i]) return ReportMaterializeFailure(program, "strict buffer", i);
		}
	}
	snapshot.images.resize(program.info.images.size());
	specialization.images.resize(program.info.images.size());
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class = image.numeric_class,
		    .dimension = image.dimension,
		    .mip_count = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle = image.shader_swizzle,
		    .indirect_root = image.indirect_root,
		    .indirect_mapping_offset = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .cube = image.cube,
		    .clamp_half_float = image.clamp_half_float,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return false;
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if (!active.empty() && !active[image.source]) {
				continue;
			}
			const auto&     indirect = *source->indirect_descriptor;
			DescriptorValue material;
			DescriptorValue table;
			if (indirect.bindless) {
				if (!clean.EvaluateDescriptor(indirect.table_source, table) ||
				    !MaterializeBindlessImage(indirect, table, i, runtime, snapshot,
				                              specialization.images)) {
					return ReportMaterializeFailure(program, "bindless image", i);
				}
				continue;
			}
			const auto      normalize = [&](DescriptorValue& value) {
				if (NullImageDescriptor(value) || !ValidImageDescriptor(value, image.r128)) {
					value.dwords.fill(0u);
				}
				return true;
			};
			if (!indirect_inputs(indirect, material, table) ||
			    !MaterializeIndirectDescriptor(program, roots, image.source, indirect, material,
			                                   table, i, 8u, observed, clean, snapshot,
			                                   snapshot.images, specialization.images, UINT32_MAX,
			                                   normalize)) {
				return ReportMaterializeFailure(program, "indirect image", i);
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i])) {
				return ReportMaterializeFailure(program, "image", i);
			}
			if (!ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	snapshot.samplers.resize(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); ++i) {
		if (!evaluate(program.info.samplers[i].source, snapshot.samplers[i])) {
			return ReportMaterializeFailure(program, "sampler", i);
		}
		if (program.info.samplers[i].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			// MipNone always selects the base level. Explicit point gathers currently require
			// encoded-zero primary and secondary bias; linear primary-mip selection is unsupported.
			if (filter > 1u || (filter == 1u && (control & 0xfffffu) != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires mip filtering None or Point with zero LOD biases");
			}
		}
	}
	if (capture_reads && !WrittenBuffersDisjoint(program, snapshot, reads))
		return ReportMaterializeFailure(program, "written buffers overlap reads", 0);
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	if (!BuildResourceSpecialization(program, snapshot, specialization)) {
		return false;
	}
	if constexpr (std::is_same_v<Roots, CompiledRoots>) {
		if (memo != nullptr && roots.compiled.memoizable) {
			memo->valid = true;
		}
	}
	return true;
}

bool MaterializeCompiled(const ResourcePlan& program, const SrtRuntime& runtime,
                         ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                         MaterializationMemo* memo) {
	bool capture_reads = false;
	bool result        = false;
	if (MaterializationAllowed(program, runtime, capture_reads)) {
		const auto&  compiled = CompileResourcePlan(program);
		// Read capture must see exact ranges.
		CleanBlockCache blocks(runtime);
		const bool      use_blocks = runtime.specialization_block_reads && !capture_reads &&
		                        runtime.read_specialization_memory != nullptr;
		const auto         clean_source = use_blocks ? blocks.Runtime() : runtime;
		OrdinaryBlockCache ordinary(clean_source);
		const bool         ordinary_blocks = !capture_reads && runtime.read_memory != nullptr &&
		                             runtime.read_memory_block != nullptr;
		const auto   source = ordinary_blocks ? ordinary.Runtime() : clean_source;
		ReadCapture  capture {source, program.ThreadScratch().specialization_reads};
		const auto   observed = ObservedRuntime(source, capture_reads, capture);
		SrtEvaluator clean(program, compiled, CleanRuntime(observed));
		SrtEvaluator walker(program, compiled, observed, true, &clean);
		result = Materialize(program, CompiledRoots {compiled}, runtime, observed, capture_reads,
		                     clean, walker, snapshot, specialization, memo);
	}
	if (!result && memo != nullptr) {
		memo->valid  = false;
		memo->reused = false;
	}
	return result;
}

void VerifyMaterialization(const ResourcePlan& program, const SrtRuntime& runtime,
                           const ResourceSnapshot&       snapshot,
                           const ResourceSpecialization& specialization, bool result,
                           bool reused) {
	ResourceSnapshot       expected_snapshot;
	ResourceSpecialization expected_specialization;
	const bool             expected = MaterializeResourcesReference(program, runtime,
	                                                                expected_snapshot,
	                                                                expected_specialization);
	const char* field = nullptr;
	if (expected != result) {
		field = "result";
	} else if (!result) {
		return;
	} else if (expected_snapshot.buffers != snapshot.buffers) {
		field = "buffers";
	} else if (expected_snapshot.images != snapshot.images) {
		field = "images";
	} else if (expected_snapshot.samplers != snapshot.samplers) {
		field = "samplers";
	} else if (expected_snapshot.flattened_srt != snapshot.flattened_srt) {
		field = "flattened SRT";
	} else if (expected_snapshot.user_data != snapshot.user_data) {
		field = "user data";
	} else if (expected_snapshot.uniform_fill != snapshot.uniform_fill) {
		field = "uniform fill";
	} else if (expected_specialization != specialization) {
		field = "specialization";
	}
	if (field != nullptr) {
		EXIT("shader resource materialization mismatch: hash=0x%016" PRIx64
		     " stage=%u field=%s memo=%s\n",
		     program.shader_hash, static_cast<uint32_t>(program.stage), field,
		     reused ? "reused" : "evaluated");
	}
}

} // namespace

namespace {

// Forwards reads to a runtime's readers and logs them.
class ReadLogger {
public:
	ReadLogger(const SrtRuntime& source, ReadLog& log): m_source(source), m_log(log) {}

	SrtRuntime Runtime() {
		auto runtime                    = m_source;
		runtime.userdata                = this;
		runtime.specialization_userdata = this;
		if (m_source.read_memory != nullptr) {
			runtime.read_memory = Ordinary;
		}
		if (m_source.read_memory_block != nullptr) {
			runtime.read_memory_block = OrdinaryBlock;
		}
		if (m_source.read_specialization_memory != nullptr) {
			runtime.read_specialization_memory = Specialization;
		}
		return runtime;
	}

private:
	static bool Ordinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
		auto&      self = *static_cast<ReadLogger*>(userdata);
		const bool ok   = self.m_source.read_memory(self.m_source.userdata, address, values);
		self.Log(ReadLog::Reader::Ordinary, address, values, ok);
		return ok;
	}
	static bool OrdinaryBlock(void* userdata, uint64_t address, std::span<uint32_t> values) {
		auto&      self = *static_cast<ReadLogger*>(userdata);
		const bool ok   = self.m_source.read_memory_block(self.m_source.userdata, address, values);
		self.Log(ReadLog::Reader::OrdinaryBlock, address, values, ok);
		return ok;
	}
	static bool Specialization(void* userdata, uint64_t address, std::span<uint32_t> values) {
		auto&      self = *static_cast<ReadLogger*>(userdata);
		const bool ok   = self.m_source.read_specialization_memory(
            SpecializationUserdata(self.m_source), address, values);
		self.Log(ReadLog::Reader::Specialization, address, values, ok);
		return ok;
	}

	void Log(ReadLog::Reader reader, uint64_t address, std::span<const uint32_t> values, bool ok) {
		m_log.entries.push_back({.address = address,
		                         .first   = static_cast<uint32_t>(m_log.words.size()),
		                         .count   = static_cast<uint32_t>(values.size()),
		                         .reader  = reader,
		                         .ok      = ok});
		// A failed read's values are not compared (see ReadsUnchanged).
		if (ok) {
			m_log.words.insert(m_log.words.end(), values.begin(), values.end());
		}
	}

	SrtRuntime m_source;
	ReadLog&   m_log;
};

} // namespace

bool MaterializeResourcesLogged(const ResourcePlan& program, const SrtRuntime& runtime,
                                ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                                ReadLog& log) {
	log.Clear();
	bool capture_reads = false;
	if (!MaterializationAllowed(program, runtime, capture_reads) || capture_reads) {
		return false;
	}
	ReadLogger logger(runtime, log);
	return MaterializeCompiled(program, logger.Runtime(), snapshot, specialization, nullptr);
}

bool ReadsUnchanged(const ReadLog& log, const SrtRuntime& runtime) {
	std::array<uint32_t, 64> small {};
	std::vector<uint32_t>    large;
	for (const auto& entry: log.entries) {
		std::span<uint32_t> values;
		if (entry.count <= small.size()) {
			values = std::span(small).first(entry.count);
		} else {
			large.resize(entry.count);
			values = large;
		}
		bool ok = false;
		switch (entry.reader) {
			case ReadLog::Reader::Ordinary:
				ok = runtime.read_memory != nullptr &&
				     runtime.read_memory(runtime.userdata, entry.address, values);
				break;
			case ReadLog::Reader::OrdinaryBlock:
				ok = runtime.read_memory_block != nullptr &&
				     runtime.read_memory_block(runtime.userdata, entry.address, values);
				break;
			case ReadLog::Reader::Specialization:
				ok = runtime.read_specialization_memory != nullptr &&
				     runtime.read_specialization_memory(SpecializationUserdata(runtime),
				                                        entry.address, values);
				break;
		}
		if (ok != entry.ok ||
		    (ok && !std::equal(values.begin(), values.end(), log.words.begin() + entry.first))) {
			return false;
		}
	}
	return true;
}

bool MaterializeResourcesReference(const ResourcePlan& program, const SrtRuntime& runtime,
                                   ResourceSnapshot&       snapshot,
                                   ResourceSpecialization& specialization) {
	bool capture_reads = false;
	if (!MaterializationAllowed(program, runtime, capture_reads)) {
		return false;
	}
	ReadCapture capture {runtime, program.ThreadScratch().specialization_reads};
	const auto  observed = ObservedRuntime(runtime, capture_reads, capture);
	SrtWalker   clean(program, CleanRuntime(observed));
	SrtWalker   walker(program, observed, program.clean_flat_slots, &clean);
	return Materialize(program, WalkerRoots {program}, runtime, observed, capture_reads, clean,
	                   walker, snapshot, specialization, nullptr);
}

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          MaterializationMemo* memo) {
	bool result = MaterializeCompiled(program, runtime, snapshot, specialization, memo);
	if (!result && !g_verify_materialization) {
		// The compiled evaluator lacks some IR operations the walker knows (new resource tracking
		// may bring more); the walker gives the draw its resources instead of skipping it.
		result = MaterializeResourcesReference(program, runtime, snapshot, specialization);
	}
	if (g_verify_materialization) {
		VerifyMaterialization(program, runtime, snapshot, specialization, result,
		                      memo != nullptr && memo->reused);
	}
	return result;
}

void SetResourceMaterializationVerification(bool enabled) {
	g_verify_materialization = enabled;
}

bool BufferStrideSpecialization() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BUFFER_SPECIALIZE");
		return text != nullptr && std::strcmp(text, "0") != 0 && std::strcmp(text, "2") != 0;
	}();
	return enabled;
}

uint32_t CanonicalPackedStride(const DescriptorValue& descriptor) {
	ShaderBufferResource buffer;
	if (!DecodeBufferDescriptor(descriptor, buffer) || buffer.Type() != 0) {
		return 0;
	}
	auto       packed_stride = buffer.PackedStride();
	const auto stride        = packed_stride & 0x3fffu;
	const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
	if (stride == 0u) {
		packed_stride &= ~((1u << 14u) | (3u << 16u));
	} else if (!swizzle) {
		packed_stride &= ~(3u << 16u);
	}
	return packed_stride;
}

bool BufferGenericStage(ShaderType stage) {
	return stage == ShaderType::Compute && !BufferStrideSpecialization();
}

uint32_t RuntimeBufferSlotWord(const DescriptorValue& descriptor) {
	ShaderBufferResource buffer;
	if (!DecodeBufferDescriptor(descriptor, buffer) || buffer.Type() != 0) {
		return 0;
	}
	// Reserved selectors (2, 3) read as 0.
	uint32_t selectors = 0;
	for (uint32_t component = 4; component-- > 0;) {
		const auto selector = (buffer.DstSelXYZW() >> (component * 3u)) & 7u;
		selectors = selectors * 6u + (selector < 2u ? selector : selector < 4u ? 0u : selector - 2u);
	}
	return buffer.Stride() | (static_cast<uint32_t>(buffer.RawFormat()) << 14u) | (selectors << 21u);
}

bool GenericBufferFormatEligible(ShaderType stage, const BufferResource& base,
                                 const ResourceSpecialization::Buffer& buffer) {
	// KYTY_BUFFER_SPECIALIZE=2: strides dynamic, formats baked (A/B runs).
	static const bool formats_baked = [] {
		const char* text = KYTY_GETENV("KYTY_BUFFER_SPECIALIZE");
		return text != nullptr && std::strcmp(text, "2") == 0;
	}();
	if (!BufferGenericStage(stage) || formats_baked || !base.formatted || base.written ||
	    buffer.zero_stride_oob) {
		return false;
	}
	if (buffer.descriptor_format == GenericBufferFormat) {
		return true;
	}
	const auto info = Format::GetFormatInfo(buffer.descriptor_format);
	if (info.type == Format::ComponentType::Unknown || info.packed_bitfield ||
	    info.component_count == 0u) {
		return false;
	}
	for (uint32_t component = 0; component < info.component_count; component++) {
		const auto bits = info.component_bits[component];
		if (bits != info.component_bits[0] || (bits != 8u && bits != 16u && bits != 32u)) {
			return false;
		}
	}
	return true;
}

void NormalizeBufferSpecialization(ResourceSpecialization& specialization, ShaderType stage,
                                   const ShaderInfo& info, uint32_t base_dwords) {
	if (!BufferGenericStage(stage)) {
		return;
	}
	auto generic = specialization.buffers;
	for (uint32_t i = 0; i < generic.size(); i++) {
		auto& buffer = generic[i];
		if ((buffer.packed_stride & 0x3fffu) != 0u) {
			buffer.packed_stride = DynamicBufferStride | (buffer.packed_stride & ~0x3fffu);
		}
		const auto base_index = i < info.buffers.size() ? i : buffer.indirect_root;
		if (base_index < info.buffers.size() &&
		    GenericBufferFormatEligible(stage, info.buffers[base_index], buffer)) {
			buffer.descriptor_format  = GenericBufferFormat;
			buffer.descriptor_swizzle = DstSel(4, 5, 6, 7);
		}
	}
	// The words must fit in push data after the base and the memory offsets (one byte per buffer).
	uint32_t strides = 0;
	uint32_t formats = 0;
	for (const auto& buffer: generic) {
		formats += buffer.descriptor_format == GenericBufferFormat ? 1u : 0u;
		strides += buffer.descriptor_format != GenericBufferFormat &&
		                   (buffer.packed_stride & DynamicBufferStride) != 0u
		               ? 1u
		               : 0u;
	}
	const auto dwords = base_dwords + static_cast<uint32_t>(generic.size() + 3u) / 4u +
	                    (strides + 1u) / 2u + formats;
	if (dwords <= PushData::DwordCount) {
		specialization.buffers = std::move(generic);
	}
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete ||
	        program.binding_layout_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size());

	auto buffers = program.info.buffers;
	const auto original_buffer_count = buffers.size();
	buffers.reserve(specialization.buffers.size());
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= original_buffer_count);
			buffers.push_back(program.info.buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto images = program.info.images;
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= program.info.images.size());
			images.push_back(program.info.images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.cube                       = source.cube;
		image.clamp_half_float           = source.clamp_half_float;
		image.bindless                   = source.bindless;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, images, sampler_plan));
	auto samplers      = program.info.samplers;
	auto sampled_pairs = program.info.sampled_pairs;
	samplers.reserve(sampler_plan.sampler_count);
	for (uint32_t index = 0; index < sampler_plan.sampler_count; index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= program.info.samplers.size()) {
			samplers.push_back(program.info.samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::PointInteger;
		samplers[index].integer_border        = binding.type != SamplerClass::Float;
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
	}

	auto memory_info = program.memory_info;
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				if (memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob) {
					// Bounds mode 0 checks offset >= stride, so zero stride
					// makes every vector read out of bounds regardless of its address.
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						for (uint32_t component = 0; component < count; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							const auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(one), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				// Vulkan MSAA stores each sample directly; FMASK's four-bit fragment indices
				// therefore map each coverage sample to the same host sample.
				constexpr uint32_t indices[] = {0x76543210u, 0xfedcba98u};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < program.info.samplers.size()) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			// Enumerated candidates sample, load and store (a switch over the candidates); a
			// bindless image is loaded per instruction and serves every image operation.
			EXIT_IF(image.indirect_root == memory.resource && !image.bindless &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageWrite &&
			        inst.GetOpcode() != ValueOpcode::ImageRead);
		}
	}
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
			}
		}
	}
	for (auto& memory: memory_info) {
		if (memory.kind == ResourceKind::Image && !memory.planning_only) {
			memory.resource = image_remap[memory.resource];
		}
	}
	for (auto& buffer: buffers) {
		if (buffer.image_alias != BufferResource::NoImageAlias) {
			buffer.image_alias = image_remap[buffer.image_alias];
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap[pair.image];
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap[image.indirect_root];
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap[resource];
		}
	}
	image_remap.Apply(images);
	program.info.buffers       = std::move(buffers);
	program.info.images        = std::move(images);
	program.info.samplers      = std::move(samplers);
	program.info.sampled_pairs = std::move(sampled_pairs);
	program.memory_info        = std::move(memory_info);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
