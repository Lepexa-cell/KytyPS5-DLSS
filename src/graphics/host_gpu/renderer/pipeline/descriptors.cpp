#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/handleRegistry.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/switches.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandHooks.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <vector>
#include <xxhash.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

uint32_t NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding) {
	return binding.resources.empty() ? 1u : static_cast<uint32_t>(binding.resources.size());
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

static Prospero::BufferFormat RuntimeBufferFormat(const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                                           uint32_t                                     index) {
	return index < snapshot.buffers.size()
	           ? static_cast<Prospero::BufferFormat>(
	                 (ShaderRecompiler::IR::RuntimeBufferSlotWord(snapshot.buffers[index]) >> 14u) &
	                 0x7fu)
	           : Prospero::BufferFormat::kInvalid;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset,
                    vk::DeviceAddress* device_address = nullptr,
                    TextureCache::TrackWrite* track = nullptr) {
	buffer_offset = 0;
	// Table mode: the device address of the returned range; the null buffer's is the zero page.
	const auto null_address = [&] {
		if (device_address != nullptr) {
			*device_address = context.GetBufferCache().ZeroPageAddress();
		}
	};

	const auto address = source.address;
	const auto size    = source.size;
	const auto id      = source.id;
	if (address == 0 || size == 0) {
		null_address();
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	std::optional<DrawPhaseTimer::ProbeScope> obtain_probe;
	obtain_probe.emplace(g_draw_phases, resource.written ? DrawPhaseTimer::BufferWritten
	                                                     : DrawPhaseTimer::BufferRead);
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(address, size, resource.written,
	                                                              resource.formatted, id);
	obtain_probe.reset();
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	// The shader adds the adjustment to its byte addresses. Word accesses need it dword-aligned;
	// a read-only buffer the shader reads only as 8- or 16-bit formatted elements (Ghost of Yotei
	// binds such vertex data 2 bytes off a dword) loads those elements by their byte address.
	const auto element_bytes = [&]() -> uint64_t {
		// Every access must be a formatted load of the descriptor's 8/16-bit format: a raw word,
		// typed or scalar load of the same buffer would need dword alignment (an earlier version
		// relaxed those too, and Ghost of Yotei's hand mesh lost its data).
		// KYTY_STRICT_BUFFER_ALIGN=1 binds the null buffer for these too (the old behavior).
		// Formatted 8/16-bit stores go through StoreSubwordInBounds (byte address, atomic AND/OR
		// on the dword), so a written buffer qualifies as well when every access is formatted
		// and none is atomic. The title's cloth init (cs 4a674edae1971d9c) writes such a 16-bit table
		// 2 or 14 bytes off a dword in some runs; with the null buffer the cloth exploded.
		// KYTY_FORMATTED_WRITE_ALIGN=0 keeps written buffers dword-aligned (the old behavior).
		static const bool strict = KYTY_GETENV("KYTY_STRICT_BUFFER_ALIGN") != nullptr;
		static const bool written_ok = [] {
			const char* text = KYTY_GETENV("KYTY_FORMATTED_WRITE_ALIGN");
			return text == nullptr || text[0] != '0';
		}();
		if (strict || (resource.written && (!written_ok || resource.atomic)) || !resource.formatted ||
		    resource.unformatted || resource.scalar) {
			return sizeof(uint32_t);
		}
		const auto info = ShaderRecompiler::Format::GetFormatInfo(resource.descriptor_format);
		if (info.type == ShaderRecompiler::Format::ComponentType::Unknown || info.packed_bitfield ||
		    (info.component_bits[0] != 8u && info.component_bits[0] != 16u)) {
			return sizeof(uint32_t);
		}
		return info.component_bits[0] / 8u;
	}();
	if (adjustment % element_bytes != 0 || adjustment >= 256 || size > max_range - adjustment) {
		// The shader takes the adjustment in whole dwords. Bind the null buffer instead of ending
		// the session (Ghost of Yotei binds byte-aligned buffers); its accesses then read zero.
		static std::atomic<uint32_t> warned {0};
		if (warned.fetch_add(1, std::memory_order_relaxed) < 24u) {
			std::printf("Warning: unsupported storage buffer offset adjustment (address=0x%016" PRIx64
			            " size=0x%" PRIx64 " adjustment=%" PRIu64 " written=%d formatted=%d "
			            "format=%u); binding a null buffer\n",
			            static_cast<uint64_t>(address), static_cast<uint64_t>(size),
			            static_cast<uint64_t>(adjustment), resource.written ? 1 : 0,
			            resource.formatted ? 1 : 0,
			            static_cast<uint32_t>(resource.descriptor_format));
		}
		null_address();
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	auto range = size + adjustment;
	if (graphics.hardware_storage_buffer_bounds) {
		// The device checks word accesses against the range, which the shader's own checks took
		// in whole dwords: a trailing partial dword is out of range. A range without one whole
		// dword is a null descriptor, which the device treats as empty.
		range &= ~vk::DeviceSize {sizeof(uint32_t) - 1};
		if (range == 0) {
			buffer_offset = 0;
			null_address();
			return {nullptr, 0, VK_WHOLE_SIZE};
		}
	}
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, range};
	if (device_address != nullptr) {
		EXIT_IF(!buffer->HasDeviceAddress());
		*device_address = buffer->BufferDeviceAddress() + aligned_offset;
	}
	if (resource.written) {
		DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::BufferInvalidate);
		context.GetTextureCache().InvalidateMemoryFromGPU(
		    address, size, TextureCache::GpuWriteSource::Shader, track);
	}
	return result;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	const bool         common          = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                                     (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                                     descriptor.fields[3] == field3_expected;
	if (r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	const bool full = common && descriptor.fields[4] == field4_expected &&
	                  descriptor.fields[5] == field5_expected;
	if (!full ||
	    (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type()))) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

// A bindless key resolves softly (ResolveBindlessKey): the reason becomes a red placeholder line.
// Defined with ReportPlaceholder below; false when no soft resolution is running.
static bool SoftTextureFailure(const char* reason);

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	// A bindless key whose T# is a depth surface the shader cannot sample this way (Ghost of Yotei:
	// a 2D depth T# in a volume heap ended the session at the Torii) keeps the placeholder.
	if (SoftTextureFailure(fmt::format("unsupported sampled depth image: resource={} encoding={} view={} "
	                                   "dimension={} guest_format={} type={}\n",
	                                   resource_ok, encoding_ok, view_ok,
	                                   static_cast<uint32_t>(resource.dimension),
	                                   static_cast<uint32_t>(descriptor.Format()),
	                                   static_cast<uint32_t>(descriptor.Type()))
	                           .c_str())) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	// Bit 31 (resource level) is set by some storage T#s, e.g. Yotei's terrain tiles.
	constexpr uint32_t field2_reserved_mask = 0x70003000u;
	constexpr uint32_t field5_expected      = 0x00700000u;
	constexpr uint32_t field5_max_mip_mask  = 0x000000f0u;
	const uint32_t     expected_field3 = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     expected_field4 =
	    descriptor.Depth() | (static_cast<uint32_t>(descriptor.BaseArray5()) << 16u);
	const bool common = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                    (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                    descriptor.fields[3] == expected_field3;
	if (resource.r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	// An upper half that is all zero (a 128-bit T# in a 256-bit slot, e.g. Yotei's 132x132
	// terrain tiles) means one level, no array, defaults.
	const bool zero_upper = descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
	                        descriptor.fields[6] == 0 && descriptor.fields[7] == 0 &&
	                        descriptor.BaseLevel() == 0 && descriptor.LastLevel() == 0;
	return common && (zero_upper || (descriptor.fields[4] == expected_field4 &&
	                                 (descriptor.fields[5] & ~field5_max_mip_mask) == field5_expected));
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	// A write-only image of a signed integer format goes through its unsigned view.
	const bool raw_sint_storage = numeric_class == Prospero::TextureNumericClass::Sint &&
	                              uint_resource && resource.written && !resource.read &&
	                              !resource.atomic;
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static void PopulateTextureMipLayout(ImageInfo& info) {
	if (info.IsVolume() && info.tile_mode != Prospero::TileMode::kLinear) {
		TileSurfaceLayout            surface {};
		const TileSurfaceDescription description {
		    info.guest_format,  info.tile_mode,    TileSurfaceDimension::Dim3D, info.extent.width,
		    info.extent.height, info.extent.depth, info.resources.levels,       1};
		if (!TileGetTiledTextureLayout(description, surface)) {
			EXIT("unsupported normalized volume texture layout\n");
		}
		for (uint32_t level = 0; level < info.resources.levels; level++) {
			const auto& mip        = surface.mips[level];
			info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
		}
		return;
	}

	TileSizeOffset levels[16] {};
	TilePaddedSize padded[16] {};
	TileGetTextureSize(info.guest_format, info.extent.width, info.extent.height,
	                   info.resources.levels, info.tile_mode, nullptr, levels, padded);
	const auto texel_shift = info.IsBlock() ? 2u : 0u;
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		const auto offset =
		    levels[level].src_size != 0 ? levels[level].src_offset : levels[level].offset;
		auto size = static_cast<uint64_t>(levels[level].src_size != 0 ? levels[level].src_size
		                                                              : levels[level].size);
		if (info.IsVolume()) {
			size *= std::max(info.extent.depth >> level, 1u);
		} else {
			size *= info.resources.layers;
		}
		info.mip_layout[level] = {
		    offset,
		    size,
		    padded[level].width >> texel_shift,
		    padded[level].height >> texel_shift,
		};
	}
}

// A texture the host cannot represent ends the session, except while a bindless key is
// resolved: that key then samples the placeholder (the shader keeps running).
static thread_local bool g_soft_texture_failures = false;
static thread_local bool g_soft_texture_failed   = false;
// The bindless key being resolved, for the placeholder message: its heap, key, and whether a
// shader asked for it (otherwise a small heap is resolved completely, unused slots included).
struct BindlessFailContext {
	uint64_t heap_base = 0;
	uint32_t binding   = 0;
	uint32_t key       = 0;
	bool     requested = false;
};
static thread_local BindlessFailContext g_bindless_fail_context;
// Every distinct placeholder is printed in red (no shader may lose a texture unnoticed, user
// 2026-10-09); the same texture failing again in later frames is not repeated.
// KYTY_STRICT_TEXTURES=1 (test runs) ends the session at the first one a shader asked for.
static void ReportPlaceholder(const char* reason, bool quiet = false) {
	static std::mutex                      mutex;
	static std::unordered_set<std::string> seen;
	static const bool strict   = KYTY_GETENV("KYTY_STRICT_TEXTURES") != nullptr;
	const auto&       context  = g_bindless_fail_context;
	auto line = fmt::format("bindless: texture kept as placeholder ({} heap=0x{:x} binding={} key={}): {}",
	                        context.requested ? "requested by a shader" : "unrequested slot",
	                        context.heap_base, context.binding, context.key, reason);
	if (strict && context.requested) {
		EXIT("%s", line.c_str());
	}
	const std::lock_guard lock(mutex);
	// Records that hold no image T# at all (null, other data: shader feedback also names keys of
	// lanes that never sample) are counted, not listed: thousands per scene would bury the rest.
	if (quiet) {
		static uint64_t counted = 0;
		static auto     next    = std::chrono::steady_clock::now();
		counted++;
		if (const auto now = std::chrono::steady_clock::now(); now >= next) {
			next = now + std::chrono::seconds(30);
			std::printf("\x1b[91mbindless: %llu table records hold no image T# (null or other data; they read "
			            "the zero placeholder)\x1b[0m\n",
			            static_cast<unsigned long long>(counted));
			std::fflush(stdout);
		}
		return;
	}
	// Once per distinct T# and reason: the same record shows up in many heaps and keys.
	if (seen.size() < 4096 && seen.insert(reason).second) {
		std::printf("\x1b[91m%s\x1b[0m", line.c_str());
		std::fflush(stdout);
	}
}
static bool SoftTextureFailure(const char* reason) {
	if (!g_soft_texture_failures) {
		return false;
	}
	g_soft_texture_failed = true;
	ReportPlaceholder(reason);
	return true;
}

#define KYTY_TEXTURE_FAIL(...)                                                                   \
	do {                                                                                         \
		if (g_soft_texture_failures) {                                                           \
			g_soft_texture_failed = true;                                                        \
			char reason_text[512];                                                               \
			std::snprintf(reason_text, sizeof(reason_text), __VA_ARGS__);                        \
			ReportPlaceholder(reason_text);                                                      \
			return {};                                                                           \
		}                                                                                        \
		EXIT(__VA_ARGS__);                                                                       \
	} while (false)

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers,
                                     uint32_t layer_offset = 0) {
	ImageViewInfo view {};
	// The image may start at the view's first layer (see DescribeTexture): layers count from there.
	const uint32_t base_array = descriptor.BaseArray5() - layer_offset;
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		KYTY_TEXTURE_FAIL("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || surface_format.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = base_array;
			if (view.base_layer >= image_layers) {
				KYTY_TEXTURE_FAIL("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = base_array;
			if (view.base_layer >= image_layers) {
				KYTY_TEXTURE_FAIL("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = base_array;
			if (view.base_layer >= image_layers) {
				KYTY_TEXTURE_FAIL("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = base_array;
			if (view.base_layer >= image_layers) {
				KYTY_TEXTURE_FAIL("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: KYTY_TEXTURE_FAIL("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(const TileSurfaceDescription& description, bool metadata,
                                   uint32_t view_levels, uint32_t& levels, uint32_t& base_level) {
	TileSurfaceLayout physical {};
	TileSurfaceLayout view {};
	auto              view_description = description;
	view_description.levels            = levels;
	if (!TileGetTiledTextureLayout(description, physical) ||
	    !TileGetTiledTextureLayout(view_description, view)) {
		return false;
	}
	if (physical.first_tail_level == view.first_tail_level &&
	    physical.block_slice_size == view.block_slice_size &&
	    physical.total_size == view.total_size &&
	    std::equal(std::begin(physical.mips), std::begin(physical.mips) + description.levels,
	               std::begin(view.mips))) {
		return true;
	}
	if (metadata || ((description.layers > 1 || description.depth > 1) &&
	                 physical.block_slice_size != view.block_slice_size)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= description.levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (physical.mips[target] != view.mips[source] ||
			    (target >= physical.first_tail_level) != (source >= view.first_tail_level) ||
			    std::max(description.width >> target, 1u) != std::max(description.width >> source, 1u) ||
			    std::max(description.height >> target, 1u) != std::max(description.height >> source, 1u) ||
			    std::max(description.depth >> target, 1u) != std::max(description.depth >> source, 1u)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			levels     = description.levels;
			base_level = base;
			return true;
		}
	}
	return false;
}

namespace {

// Everything ResolveTexture derives from an image resource and its descriptor alone.
struct TextureDescription {
	TextureCache::ImageDesc desc;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint64_t                size              = 0;
	bool                    shader_conversion = false;
};


TextureDescription DescribeTexture(const ShaderRecompiler::IR::ImageResource& resource,
                                   const ShaderTextureResource&               descriptor) {
	const bool storage         = resource.written;
	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	auto view_levels = multisampled || single_storage_mip
	                       ? 1u
	                       : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		KYTY_TEXTURE_FAIL("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = descriptor.Format();
	if (!TextureHasSurfaceFormat(format)) {
		KYTY_TEXTURE_FAIL("unknown texture format: %u\n", static_cast<uint32_t>(format));
	}
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class == Prospero::SampledTextureNumericClass(format);
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		KYTY_TEXTURE_FAIL("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	auto          image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto          image_address = address;
	uint32_t      layer_offset  = 0;
	if (levels > physical_levels) {
		const TileSurfaceDescription physical {
		    format, tile, volume ? TileSurfaceDimension::Dim3D : TileSurfaceDimension::Dim2D,
		    width, height, volume ? depth : 1u, physical_levels, image_layers};
		if (ResolveTextureMipView(physical, !resource.r128 && descriptor.MetaCompress(),
		                          view_levels, levels, view_base)) {
			// The view maps onto the stored levels.
		} else if (!resource.written && base_level < physical_levels) {
			// The memory holds MAX_MIP + 1 levels; LAST_LEVEL beyond them names mips that do not
			// exist (Ghost of Yotei 512x512x33 arrays: max 2, view 1..9). The GPU reads the stored
			// levels then, so the view keeps the stored ones instead of the whole texture failing.
			levels      = physical_levels;
			view_levels = physical_levels - base_level;
			view_base   = base_level;
		} else {
			KYTY_TEXTURE_FAIL("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	}
	uint32_t      pitch = 0;
	TileSizeAlign size {};
	if (multisampled) {
		const auto bytes = Prospero::NumBytesPerElement(format);
		pitch            = depth_tile ? TileGetDepthPitch(width, bytes, last_level)
		                              : TileGetRenderTargetPitch(width, bytes, last_level);
		if (pitch == 0 || !TileGetRenderTargetSize(width, height, pitch, bytes, size, last_level) ||
		    size.size > UINT32_MAX / image_layers) {
			KYTY_TEXTURE_FAIL("unsupported multisample texture layout\n");
		}
		size.size *= image_layers;
	} else {
		// A format and tile mode the tiler has no layout for would end the session there (seen
		// with Yotei bindless keys, e.g. format 128 with tile 31).
		TileTextureElementLayout element {};
		TileSurfaceLayout        layout {};
		if (!volume &&
		    !(tile == Prospero::TileMode::kLinear && TileGetTextureElementLayout(format, element)) &&
		    !TileGetTiledTextureLayout({format, tile, TileSurfaceDimension::Dim2D, width, height, 1u,
		                                physical_levels, 1u},
		                               layout)) {
			KYTY_TEXTURE_FAIL("unsupported texture format for its tile mode: format=%u tile=%u "
			                  "extent=%ux%u levels=%u\n",
			                  static_cast<uint32_t>(format), static_cast<uint32_t>(tile), width, height,
			                  physical_levels);
		}
		pitch = TileGetTexturePitch(format, width, tile);
		if (volume) {
			TileGetTextureTotalSize(format, width, height, depth, physical_levels, tile, true, size);
		} else {
			// One slice, then the layers: an array over 4 GiB (Ghost of Yotei after the
			// calligraphy scene) would end the session in the tiler.
			TileGetTextureTotalSize(format, width, height, 1u, physical_levels, tile, false, size);
			if (size.size > UINT32_MAX / image_layers && layered && !descriptor.MetaCompress() &&
			    descriptor.BaseArray5() < image_layers &&
			    size.size <= UINT32_MAX / (image_layers - descriptor.BaseArray5())) {
				// DEPTH is the view's last layer, BASE_ARRAY its first: a pool of hundreds of layers
				// (Ghost of Yotei's calligraphy cloth, 2048x2048x515 RG8) is seen one range at a time,
				// so the image covers that range only, from its first layer on.
				layer_offset  = descriptor.BaseArray5();
				image_address = address + uint64_t {layer_offset} * size.size;
				image_layers -= layer_offset;
			}
			if (size.size > UINT32_MAX / image_layers) {
				KYTY_TEXTURE_FAIL("texture array larger than 4 GiB: format=%u tile=%u type=%u "
				                  "extent=%ux%u layers=%u levels=%u address=0x%016" PRIx64 "\n",
				                  static_cast<uint32_t>(format), static_cast<uint32_t>(tile),
				                  static_cast<uint32_t>(type), width, height, image_layers,
				                  physical_levels, address);
			}
			size.size *= image_layers;
		}
	}
	// Ghost of Yotei places small swizzled textures (16x16 in a 4 KiB mode) inside a block, so
	// only the descriptor's own 256-byte address granularity is required.
	if (size.size == 0 || size.align == 0 || (image_address & 0xffu) != 0) {
		KYTY_TEXTURE_FAIL("unsupported texture layout: format=%u tile=%u type=%u extent=%ux%ux%u levels=%u "
		     "address=0x%016" PRIx64 " size=%u align=%u\n",
		     static_cast<uint32_t>(format), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(type), width, height, depth, physical_levels, address,
		     static_cast<uint32_t>(size.size), static_cast<uint32_t>(size.align));
	}
	// A descriptor read from guest data at run time (a bindless key) can name memory outside
	// the guest's address space; the texture cache must not see it.
	if (!GuestRange {image_address, size.size}.Valid()) {
		KYTY_TEXTURE_FAIL("texture outside guest memory: address=0x%016" PRIx64 " size=0x%x\n",
		                  image_address, static_cast<uint32_t>(size.size));
	}
	if (storage) {
		ValidateStorageTexture(resource, descriptor, size.size);
	}

	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	if (pixel_format == vk::Format::eUndefined) {
		KYTY_TEXTURE_FAIL("texture format has no host format: format=%u\n",
		                  static_cast<uint32_t>(format));
	}
	const bool sint_storage = storage && !resource.atomic &&
	                          Prospero::SampledTextureNumericClass(format) ==
	                              Prospero::TextureNumericClass::Sint;
	const auto storage_view_format = resource.atomic64            ? vk::Format::eR64Uint
	                                 : storage && resource.atomic ? vk::Format::eR32Uint
	                                 : sint_storage               ? SintStorageViewFormat(pixel_format)
	                                                              : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {image_address, size.size};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.kind          = ImageMetadataKind::Dcc;
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
		// A T# with compression on but its DCC address outside guest memory (most often 0): a
		// stale or not yet written bindless slot, or one the host cannot represent. ImageOps::
		// Validate would end the session; a bindless key samples the placeholder instead.
		if (!GuestRange {desc.info.metadata.range.address,
		                 std::max<uint64_t>(desc.info.metadata.range.size, 1)}
		         .Valid()) {
			KYTY_TEXTURE_FAIL("texture DCC metadata outside guest memory: meta=0x%016llx size=0x%llx "
			                  "base=0x%016llx %ux%u format=%u tile=%u levels=%u layers=%u "
			                  "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
			                  static_cast<unsigned long long>(desc.info.metadata.range.address),
			                  static_cast<unsigned long long>(desc.info.metadata.range.size),
			                  static_cast<unsigned long long>(address), width, height,
			                  static_cast<uint32_t>(format), static_cast<uint32_t>(tile),
			                  levels, image_layers, descriptor.fields[0], descriptor.fields[1],
			                  descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
			                  descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
		}
	}
	if (samples > 1) {
		desc.info.mip_layout[0] = {0, size.size, pitch, height};
	} else {
		PopulateTextureMipLayout(desc.info);
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers, layer_offset);
	if (g_soft_texture_failures && g_soft_texture_failed) {
		return {};
	}
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	return {std::move(desc), pixel_format, view_format, size.size, shader_conversion};
}

// A texture description is a pure function of its inputs, and draws bind the same textures over
// and over, so recent descriptions are kept. An entry holds copies of both inputs: a hit is exact.
// A scene can bind a few thousand distinct textures (Sky Garden missed half its lookups in a
// 256-entry direct-mapped table), so the table is set-associative and evicts the least recently
// used entry of a set.
class TextureDescriptionCache {
public:
	struct Entry {
		uint32_t                              hash     = 0;
		uint64_t                              last_use = 0; // 0: empty.
		ShaderRecompiler::IR::DescriptorValue value;
		ShaderRecompiler::IR::ImageResource   resource;
		TextureDescription                    description;
		// The description's last FindImage result (see TextureCache::RefindImage) and the image
		// ResolveTexture bound for it; generation 0: none.
		uint64_t                              generation = 0;
		ImageId                               found;
		ImageId                               bound;
		// The view RebindImages acquired for a binding of this description.
		PreparedBindings::ViewMemo            view;
	};

	// The entry stays valid until the next call.
	Entry& Get(const ShaderRecompiler::IR::ImageResource&   resource,
	           const ShaderRecompiler::IR::DescriptorValue& value,
	           const ShaderTextureResource&                 descriptor) {
		uint32_t hash = 2166136261u;
		for (const auto dword: value.dwords) {
			hash = (hash ^ dword) * 16777619u;
		}
		hash = (hash ^ resource.source) * 16777619u;
		// Mix the high bits into the set index (MurmurHash3's finalizer).
		hash ^= hash >> 16u;
		hash *= 0x85ebca6bu;
		hash ^= hash >> 13u;
		hash *= 0xc2b2ae35u;
		hash ^= hash >> 16u;
		auto* const set    = &m_entries[(hash % Sets) * Ways];
		Entry*      victim = set;
		for (size_t way = 0; way < Ways; way++) {
			auto& entry = set[way];
			if (entry.last_use != 0 && entry.hash == hash && entry.value == value &&
			    entry.resource == resource) {
				entry.last_use = ++m_clock;
				return entry;
			}
			if (entry.last_use < victim->last_use) {
				victim = &entry;
			}
		}
		victim->description = DescribeTexture(resource, descriptor);
		victim->hash        = hash;
		victim->last_use    = g_soft_texture_failures && g_soft_texture_failed ? 0 : ++m_clock;
		victim->value       = value;
		victim->resource    = resource;
		victim->generation  = 0;
		victim->view        = {};
		return *victim;
	}

private:
	static constexpr size_t Ways = 4;
	static constexpr size_t Sets = 1024;
	uint64_t                m_clock   = 0;
	std::vector<Entry>      m_entries = std::vector<Entry>(Sets * Ways);
};

// KYTY_DEBUG_TEXTURE_REUSE=0 resolves every texture binding from scratch, for A/B runs (see also
// AbFeatureOff).
bool TextureReuseEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_TEXTURE_REUSE");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("reuse");
	return enabled && !(ab && AbFeatureOff());
}

// A null texture binding keeps its image and view while its descriptor repeats;
// KYTY_DEBUG_AB=nullreuse alternates.
bool NullReuseEnabled() {
	static const bool ab = AbSelected("nullreuse");
	return TextureReuseEnabled() && !(ab && AbFeatureOff());
}

// A resolved binding starts from the view its description's last binding acquired;
// KYTY_DEBUG_AB=viewmemo alternates.
bool ViewMemoEnabled() {
	static const bool ab = AbSelected("viewmemo");
	return TextureReuseEnabled() && !(ab && AbFeatureOff());
}

} // namespace

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              PreparedBindings::ImageSource*               source) {
	TextureBinding binding;
	ResolveTextureInto(resource, value, source, binding);
	return binding;
}

void RenderExecutor::ResolveTextureInto(const ShaderRecompiler::IR::ImageResource&   resource,
                                        const ShaderRecompiler::IR::DescriptorValue& value,
                                        PreparedBindings::ImageSource*               source,
                                        TextureBinding&                              out) {
	if (source != nullptr) {
		source->generation = 0;
		source->null_image = false;
		source->view_memo  = nullptr;
	}
	out.image_view = nullptr;
	out.layout     = vk::ImageLayout::eUndefined;
	out.mip_views.clear();
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
		if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
			EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
		}
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		out.desc     = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                 : TextureCache::BindingType::Texture);
		out.image_id = texture_cache.FindImage(out.desc);
		if (source != nullptr && !storage) {
			source->resource   = resource;
			source->value      = value;
			source->found      = out.image_id;
			source->null_image = true;
		}
		return;
	}

	thread_local TextureDescriptionCache descriptions;
	std::optional<DrawPhaseTimer::ProbeScope> describe_probe;
	describe_probe.emplace(g_draw_phases, DrawPhaseTimer::TextureDescribe);
	auto&       entry     = descriptions.Get(resource, value, descriptor);
	const auto& described = entry.description;
	out.desc              = described.desc;
	auto&       desc      = out.desc;
	describe_probe.reset();
	if (g_soft_texture_failures && g_soft_texture_failed) {
		// The bindless key samples the placeholder (see ResolveBindlessKey).
		return;
	}
	if (desc.info.pixel_format == vk::Format::eUndefined) [[unlikely]] {
		std::printf("texture-desc undefined: soft=%d failed=%d last_use=%llu format=%u bindless=%d "
		            "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		            g_soft_texture_failures ? 1 : 0, g_soft_texture_failed ? 1 : 0,
		            static_cast<unsigned long long>(entry.last_use),
		            static_cast<uint32_t>(descriptor.Format()), resource.bindless ? 1 : 0,
		            value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
		            value.dwords[4], value.dwords[5], value.dwords[6], value.dwords[7]);
		std::fflush(stdout);
	}
	// Debug: KYTY_DEBUG_VOLUME_MIP0=1 views volume textures as their first level only, so every
	// sample clamps to it: tells wrong lower levels (their layout) from wrong level 0.
	static const bool volume_mip0 = KYTY_GETENV("KYTY_DEBUG_VOLUME_MIP0") != nullptr;
	if (volume_mip0 && desc.info.IsVolume()) [[unlikely]] {
		desc.view_info.level_count = 1;
	}

	const auto metadata_base_layer = desc.view_info.base_layer;
	const auto remember            = [&](ImageId found, uint64_t generation) {
		if (source != nullptr && generation != 0) {
			source->resource            = resource;
			source->value               = value;
			source->found               = found;
			source->generation          = generation;
			source->metadata_base_layer = metadata_base_layer;
		}
	};
	// The description's previous lookup still holds while the image set is unchanged, and the
	// validation below passed for the image it bound while that image's stencil association is
	// unchanged (see ReuseTexture).
	if (TextureReuseEnabled() && entry.generation != 0 &&
	    texture_cache.RefindImage(entry.found, entry.generation, desc, metadata_base_layer)) {
		const auto depth_id = texture_cache.GetImage(entry.found).depth_id;
		if ((depth_id ? depth_id : entry.found) == entry.bound) {
			remember(entry.found, entry.generation);
			out.image_id = entry.bound;
			if (source != nullptr) {
				source->view_memo = &entry.view;
				// A binding of this description acquired a view before, maybe in another slot.
				if (const auto& memo = entry.view;
				    ViewMemoEnabled() && memo.generation != 0 && memo.image == entry.bound) {
					source->view            = memo.view;
					source->view_image      = memo.image;
					source->view_info       = memo.info;
					source->view_generation = memo.generation;
				}
			}
			return;
		}
	}
	entry.generation = 0;
	if (source != nullptr) {
		source->view_memo = &entry.view;
	}

	uint64_t generation = 0;
	ImageId  id;
	{
		DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::TextureImage);
		id = texture_cache.FindImage(desc, described.shader_conversion, &generation);
	}
	remember(id, generation);
	const auto found = id;
	auto*      image = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		// A depth surface read through a color view Vulkan cannot make of it (the title after its
		// menus: D16 sampled as R16_UINT, which ended the session). Until a color alias with a
		// D16 copy exists, the binding reads an empty texture instead (that effect stays wrong).
		// KYTY_DEPTH_VIEW_FATAL=1 ends the session as before.
		static const bool fatal = KYTY_GETENV("KYTY_DEPTH_VIEW_FATAL") != nullptr;
		if (!fatal && !g_soft_texture_failures && IsSupportedSampledDepthResource(resource) &&
		    !IsSupportedSampledDepthView(image->info.pixel_format, described.pixel_format,
		                                 descriptor.DstSelXYZW())) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				std::printf("Warning: depth image 0x%016llx (format %d) sampled as format %d: not "
				            "supported yet, an empty texture is bound\n",
				            static_cast<unsigned long long>(descriptor.Base40()),
				            static_cast<int>(image->info.pixel_format),
				            static_cast<int>(described.pixel_format));
				std::fflush(stdout);
			}
			out.desc     = NullTextureDesc(resource, TextureCache::BindingType::Texture);
			out.image_id = texture_cache.FindImage(out.desc);
			if (source != nullptr) {
				source->found      = out.image_id;
				source->generation = 0;
				source->null_image = false;
			}
			return;
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, described.pixel_format,
		                            described.size);
		if (g_soft_texture_failures && g_soft_texture_failed) {
			return; // not cached: the key keeps the placeholder
		}
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, described.view_format,
		                         descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, described.pixel_format,
		                             descriptor.DstSelXYZW());
	}
	if (generation != 0) {
		entry.generation = generation;
		entry.found      = found;
		entry.bound      = id;
	}
	out.image_id = id;
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::Upload);
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

// A sampled texture whose descriptor enables mip statistics counts in its counter, which texture
// streamers read back (RenderContext::ReportMipStats).
// KYTY_DEBUG_MIP_STATS: every 2^20 sampled textures, how many had full eight-dword descriptors
// and how many enabled a mip statistics counter (with the last such descriptor's dwords 5 and 6).
static void NoteMipStatsDescriptor(const ShaderRecompiler::IR::DescriptorValue& value, bool enabled) {
	static const bool debug = KYTY_GETENV("KYTY_DEBUG_MIP_STATS") != nullptr;
	if (!debug) [[likely]] {
		return;
	}
	static std::atomic<uint64_t> seen {0};
	static std::atomic<uint64_t> full {0};
	static std::atomic<uint64_t> counted {0};
	static std::atomic<uint64_t> last {0};
	full.fetch_add(value.dword_count >= 8 ? 1u : 0u, std::memory_order_relaxed);
	if (enabled) {
		counted.fetch_add(1, std::memory_order_relaxed);
		last.store(value.dwords[5] | (uint64_t {value.dwords[6]} << 32u), std::memory_order_relaxed);
	}
	if ((seen.fetch_add(1, std::memory_order_relaxed) + 1) % (1u << 20u) == 0) {
		const auto words = last.load(std::memory_order_relaxed);
		std::printf("mip-stats: textures=%llu full=%llu counted=%llu last d5=%08x d6=%08x\n",
		            static_cast<unsigned long long>(seen.load()), static_cast<unsigned long long>(full.load()),
		            static_cast<unsigned long long>(counted.load()), static_cast<uint32_t>(words),
		            static_cast<uint32_t>(words >> 32u));
		std::fflush(stdout);
	}
}

static void MarkMipStats(RenderContext& context, const ShaderRecompiler::IR::ImageResource& resource,
                         const ShaderRecompiler::IR::DescriptorValue& value) {
	if (resource.written) {
		return;
	}
	if (value.dword_count < 8) {
		NoteMipStatsDescriptor(value, false);
		return;
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	NoteMipStatsDescriptor(value, descriptor.MipStatsCntEn());
	if (descriptor.MipStatsCntEn()) {
		context.MarkMipStatsCounter(descriptor.MipStatsCntId(), descriptor.MinLod() != 0);
	}
}

// Whether a stage slot's previous image binding stands for ResolveTexture(resource, value): the
// same resource and descriptor find the same image while the image set is unchanged, and the
// binding follows that image's stencil association, which must not have changed either.
static bool ReuseTexture(TextureCache& cache, const PreparedBindings::ImageSource& source,
                         const TextureBinding&                        binding,
                         const ShaderRecompiler::IR::ImageResource&   resource,
                         const ShaderRecompiler::IR::DescriptorValue& value) {
	if (source.null_image) {
		// The same resource's null descriptor finds the same null image, which never changes.
		return NullReuseEnabled() && source.value == value && source.resource == resource &&
		       source.found == binding.image_id;
	}
	if (!TextureReuseEnabled() || source.generation == 0 || !(source.value == value) ||
	    !(source.resource == resource) ||
	    !cache.RefindImage(source.found, source.generation, binding.desc,
	                       source.metadata_base_layer)) {
		return false;
	}
	const auto depth_id = cache.GetImage(source.found).depth_id;
	return (depth_id ? depth_id : source.found) == binding.image_id;
}

// KYTY_DEBUG_IMAGE_GROUPS=0 checks every image of every draw; KYTY_DEBUG_AB=imagegroups
// alternates.
static bool ImageGroupsEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_IMAGE_GROUPS");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("imagegroups");
	return enabled && TextureReuseEnabled() && !(ab && AbFeatureOff());
}

// KYTY_VERIFY_IMAGE_GROUPS=1: a draw keeping its stage's image group also resolves each image and
// acquires its view, and reports what came out differently.
static bool VerifyImageGroups() {
	static const bool enabled = KYTY_GETENV("KYTY_VERIFY_IMAGE_GROUPS") != nullptr;
	return enabled;
}

static void ReportImageGroupCheck(bool same, uint64_t address) {
	static std::atomic<uint64_t> checked {0};
	static std::atomic<uint64_t> missed {0};
	const auto                   count = checked.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!same && missed.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("image-group verify: 0x%016" PRIx64 " kept binding differs\n", address);
	}
	if (count % 100000 == 0) {
		std::printf("image-group verify: images=%" PRIu64 " missed=%" PRIu64 "\n", count,
		            missed.load(std::memory_order_relaxed));
		std::fflush(stdout);
	}
}

// Whether the stage's recorded images still stand: no image was registered or unregistered and
// the tick and GC tick are the ones they were checked under, so every lookup and view acquisition
// would repeat itself (RefindImage, IsTextureCurrent), and each image is still clean, bound for
// the image its lookup finds, and has no color clear to apply for its request.
// KYTY_CLEAN_EPOCH=0 checks a standing group's images every draw; KYTY_DEBUG_AB=cleanepoch
// alternates.
static bool CleanEpochEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_CLEAN_EPOCH");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("cleanepoch");
	return enabled && !(ab && AbFeatureOff());
}

static bool ImageGroupImagesClean(TextureCache& cache, const PreparedBindings& prepared) {
	for (size_t i = 0; i < prepared.images.size(); i++) {
		const auto& source = prepared.image_sources[i];
		const auto  bound  = prepared.images[i].image_id;
		if (source.null_image) {
			continue; // The null image never changes (see ReuseTexture).
		}
		const auto depth = cache.GetImage(source.found).depth_id;
		if ((depth ? depth : source.found) != bound || !cache.IsTextureClean(bound) ||
		    !cache.IsColorClearCurrent(source.found, prepared.images[i].desc,
		                               source.metadata_base_layer)) {
			return false;
		}
	}
	return true;
}

// With the target state of the last check, each image of a group whose requests carry no color
// metadata is as clean as it was then while g_image_clean_epoch is unchanged: every change that
// IsTextureClean or the bound image depends on bumps it. KYTY_VERIFY_CLEAN_EPOCH=1 checks the
// images anyway and reports a group the epoch kept that is not clean.
// KYTY_DEBUG_IMAGE_EPOCH_STATS=1: every 5 s, how ImageGroupStands decided (target state changed,
// kept by the epoch, epoch off for the group, checked clean, checked not clean) and the source
// lines that bumped g_image_clean_epoch most.
enum class GroupOutcome { State, Epoch, Off, Clean, Dirty, Count };
static void NoteGroupOutcome(GroupOutcome outcome) {
	static std::array<uint64_t, static_cast<size_t>(GroupOutcome::Count)> counts {};
	static auto start = std::chrono::steady_clock::now();
	counts[static_cast<size_t>(outcome)]++;
	if ((counts[0] + counts[1] + counts[2] + counts[3] + counts[4]) % 4096 != 0 ||
	    std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
		return;
	}
	start = std::chrono::steady_clock::now();
	std::vector<std::pair<uint64_t, uint32_t>> bumps;
	for (uint32_t line = 0; line < g_image_epoch_bumps.size(); line++) {
		if (const auto n = g_image_epoch_bumps[line].exchange(0, std::memory_order_relaxed); n != 0) {
			bumps.emplace_back(n, line);
		}
	}
	std::ranges::sort(bumps, std::greater {});
	std::string top;
	for (size_t i = 0; i < bumps.size() && i < 8; i++) {
		top += fmt::format(" {}{}:{}", bumps[i].second >= 8000 ? "image.h:" : "tc:",
		                   bumps[i].second >= 8000 ? bumps[i].second - 8000 : bumps[i].second,
		                   bumps[i].first / 5);
	}
	std::printf("image-epoch 5s: state=%" PRIu64 " epoch=%" PRIu64 " off=%" PRIu64 " clean=%" PRIu64
	            " dirty=%" PRIu64 " | bumps/s%s\n",
	            counts[0], counts[1], counts[2], counts[3], counts[4], top.c_str());
	std::fflush(stdout);
	counts.fill(0);
}

static bool ImageGroupStands(TextureCache& cache, PreparedBindings& prepared) {
	auto& group = prepared.image_group;
	static const bool stats = ImageEpochStatsEnabled();
	if (!(group.state == cache.CurrentTargetState())) {
		if (stats) [[unlikely]] {
			NoteGroupOutcome(GroupOutcome::State);
		}
		return false;
	}
	const bool epoch_on = group.clear_free && CleanEpochEnabled();
	const auto epoch    = g_image_clean_epoch.load(std::memory_order_acquire);
	if (stats) [[unlikely]] {
		NoteGroupOutcome(!epoch_on                                ? GroupOutcome::Off
		                 : group.clean_epoch == epoch             ? GroupOutcome::Epoch
		                 : ImageGroupImagesClean(cache, prepared) ? GroupOutcome::Clean
		                                                          : GroupOutcome::Dirty);
	}
	if (epoch_on && group.clean_epoch == epoch) {
		static const bool verify = KYTY_GETENV("KYTY_VERIFY_CLEAN_EPOCH") != nullptr;
		if (verify) [[unlikely]] {
			static uint64_t checked = 0;
			static uint64_t differ  = 0;
			checked++;
			if (!ImageGroupImagesClean(cache, prepared) && differ++ < 16) {
				std::printf("clean-epoch verify: a group kept by the epoch is not clean\n");
			}
			if (checked % 500000 == 0) {
				std::printf("clean-epoch verify: checked=%" PRIu64 " differ=%" PRIu64 "\n", checked,
				            differ);
				std::fflush(stdout);
			}
		}
		return true;
	}
	if (!ImageGroupImagesClean(cache, prepared)) {
		return false;
	}
	group.clean_epoch = epoch_on ? epoch : 0;
	return true;
}

// KYTY_STAGE_RECORDS=0: each stage slot keeps only its previous draw's bindings; KYTY_DEBUG_AB=records
// alternates. On: the bindings are kept per program and image descriptors (see
// RenderExecutor::StageRecord), and a group recorded under another tick or image set is checked
// image by image (ImageGroupRevalidates).
bool StageRecordsEnabled() noexcept {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_STAGE_RECORDS");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("records");
	return enabled && !(ab && AbFeatureOff());
}

// A group recorded under another target state (a newer submission tick, GC tick or image set):
// it still stands when each image's lookup would repeat itself (RefindImage, which also touches
// the image for this tick and finishes the find as a lookup would) and binds the same clean image
// with the view it kept, as ReuseTexture and RebindImages would conclude image by image.
static bool ImageGroupRevalidates(TextureCache& cache, PreparedBindings& prepared) {
	for (size_t i = 0; i < prepared.images.size(); i++) {
		const auto& source  = prepared.image_sources[i];
		const auto& binding = prepared.images[i];
		if (source.null_image) {
			continue;
		}
		if (source.generation == 0 ||
		    !cache.RefindImage(source.found, source.generation, binding.desc,
		                       source.metadata_base_layer)) {
			return false;
		}
		const auto depth = cache.GetImage(source.found).depth_id;
		if ((depth ? depth : source.found) != binding.image_id ||
		    source.view_image != binding.image_id || source.view != binding.image_view ||
		    !cache.IsTextureCurrent(binding.image_id, source.view_generation) ||
		    !cache.IsColorClearCurrent(source.found, binding.desc, source.metadata_base_layer)) {
			return false;
		}
	}
	prepared.image_group.state = cache.CurrentTargetState();
	return true;
}

// KYTY_IMAGE_RETICK=0 revalidates image by image after every submission tick; KYTY_DEBUG_AB=retick
// alternates.
static bool ImageRetickEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_IMAGE_RETICK");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("retick");
	return enabled && !(ab && AbFeatureOff());
}

// A group checked under the current image set and GC tick, only an earlier submission tick: no
// image came or went, so each image's range generation, lookup and view stay what they were when
// the group was last checked, and the LRU already holds each image for this GC tick. What
// ImageGroupRevalidates would still do is stamp each found image with this tick (RefindImage) and
// ask what ImageGroupStands asks; FinishFind's color clear check is IsColorClearCurrent.
// KYTY_VERIFY_IMAGE_RETICK=1 revalidates image by image as well and reports a disagreement.
static bool ImageGroupRetick(TextureCache& cache, PreparedBindings& prepared) {
	const auto current = cache.CurrentTargetState();
	auto&      state   = prepared.image_group.state;
	if (current.image_set != state.image_set || current.gc_tick != state.gc_tick ||
	    !ImageRetickEnabled()) {
		return false;
	}
	thread_local std::vector<ImageId> found;
	found.clear();
	for (size_t i = 0; i < prepared.images.size(); i++) {
		const auto& source  = prepared.image_sources[i];
		const auto& binding = prepared.images[i];
		if (source.null_image) {
			continue;
		}
		const auto depth = cache.GetImage(source.found).depth_id;
		if ((depth ? depth : source.found) != binding.image_id || source.view_image != binding.image_id ||
		    source.view != binding.image_view || !cache.IsTextureClean(binding.image_id) ||
		    !cache.IsColorClearCurrent(source.found, binding.desc, source.metadata_base_layer)) {
			return false;
		}
		found.push_back(source.found);
	}
	cache.MarkImagesAccessed(found);
	state = current;
	static const bool verify = KYTY_GETENV("KYTY_VERIFY_IMAGE_RETICK") != nullptr;
	if (verify) [[unlikely]] {
		static uint64_t checked = 0;
		static uint64_t differ  = 0;
		checked++;
		if (!ImageGroupRevalidates(cache, prepared) && differ++ < 16) {
			std::printf("image-retick verify: a reticked group does not revalidate\n");
		}
		if (checked % 100000 == 0) {
			std::printf("image-retick verify: checked=%" PRIu64 " differ=%" PRIu64 "\n", checked, differ);
			std::fflush(stdout);
		}
	}
	return true;
}

static bool ImageGroupStandsOrRevalidates(TextureCache& cache, PreparedBindings& prepared) {
	return ImageGroupStands(cache, prepared) ||
	       (StageRecordsEnabled() &&
	        (ImageGroupRetick(cache, prepared) || ImageGroupRevalidates(cache, prepared)));
}

// Records the stage's images as RebindImages left them, for the next draw of the same program and
// image descriptors (see PreparedBindings::ImageGroup). Only sampled textures that a lookup can
// repeat (or null textures) and that keep their views qualify; ImageGroupStands also asks each
// color-compressed surface whether its color clear check for the request still holds.
static void RecordImageGroup(RenderContext& context, PreparedBindings& prepared,
                             const ShaderRecompiler::IR::CompiledShaderInfo& program,
                             const std::vector<ShaderRecompiler::IR::DescriptorValue>& values) {
	auto& group   = prepared.image_group;
	auto& cache   = context.GetTextureCache();
	group.program = nullptr;
	if (!ImageGroupsEnabled()) {
		return;
	}
	group.mip_counters.clear();
	for (size_t i = 0; i < prepared.images.size(); i++) {
		const auto& binding = prepared.images[i];
		const auto& source  = prepared.image_sources[i];
		if (binding.desc.type != TextureCache::BindingType::Texture ||
		    (source.generation == 0 && !(source.null_image && NullReuseEnabled())) ||
		    source.view_generation == 0 || source.view_image != binding.image_id ||
		    source.view != binding.image_view ||
		    program.info.images[i].mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			group.refused =
			    binding.desc.type != TextureCache::BindingType::Texture ? BindStats::RefusedType
			    : program.info.images[i].mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic
			        ? BindStats::RefusedDynamicMip
			    : source.generation == 0 && !(source.null_image && NullReuseEnabled())
			        ? BindStats::RefusedLookup
			        : BindStats::RefusedView;
			return;
		}
		if (values[i].dword_count >= 8) {
			const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(values[i]);
			if (descriptor.MipStatsCntEn()) {
				// The counter id, and bit 8 when the descriptor raises the minimum LOD.
				group.mip_counters.push_back(descriptor.MipStatsCntId() |
				                             (descriptor.MinLod() != 0 ? 0x100u : 0u));
			}
		}
	}
	group.images.assign(values.begin(), values.end());
	group.state       = cache.CurrentTargetState();
	group.program     = &program;
	group.refused     = BindStats::RefusedNone;
	group.clean_epoch = 0;
	group.clear_free  = std::ranges::none_of(prepared.images, [](const TextureBinding& binding) {
		const auto kind = binding.desc.info.metadata.kind;
		return kind == ImageMetadataKind::Dcc || kind == ImageMetadataKind::Cmask;
	});
}

// KYTY_DEBUG_BIND_STATS: counts whether PrepareBindings kept the stage's image group, and if not,
// the first reason ImageGroupStands or the comparisons before it found.
static void NoteImageGroup(TextureCache& cache, const PreparedBindings& prepared,
                           const void*                                               program,
                           const std::vector<ShaderRecompiler::IR::DescriptorValue>& images,
                           bool                                                      kept) {
	auto&       stats   = g_bind_stats;
	const auto& group   = prepared.image_group;
	const auto  outcome = [&] {
		if (images.empty()) return BindStats::NoImages;
		if (kept) return BindStats::Kept;
		if (!ImageGroupsEnabled()) return BindStats::Disabled;
		if (group.program == nullptr) {
			stats.refused[group.refused]++;
			return BindStats::NotRecorded;
		}
		if (group.program != program) return BindStats::OtherProgram;
		if (!std::ranges::equal(images, group.images)) return BindStats::OtherImages;
		const auto state = cache.CurrentTargetState();
		if (state.image_set != group.state.image_set) return BindStats::ImageSet;
		if (state.tick != group.state.tick) return BindStats::Tick;
		if (state.gc_tick != group.state.gc_tick) return BindStats::GcTick;
		for (size_t i = 0; i < prepared.images.size(); i++) {
			const auto& source = prepared.image_sources[i];
			const auto  bound  = prepared.images[i].image_id;
			if (source.null_image) continue;
			const auto depth = cache.GetImage(source.found).depth_id;
			if ((depth ? depth : source.found) != bound) return BindStats::OtherImage;
			if (!cache.IsTextureClean(bound)) return BindStats::Dirty;
			if (!cache.IsColorClearCurrent(source.found, prepared.images[i].desc,
			                               source.metadata_base_layer)) {
				return BindStats::ColorClear;
			}
		}
		return BindStats::Kept; // Not reached: ImageGroupStands would have kept it.
	}();
	stats.group[outcome]++;
}

struct FrameRepeat {
	std::array<std::unordered_set<uint64_t>, 5> frames;
	size_t                                      current = 0;
	uint64_t                                    stages = 0;
	uint64_t                                    full = 0;
	uint64_t                                    without_buffers = 0;
	uint64_t                                    relocated = 0;
	uint64_t                                    user_data = 0;
	uint64_t                                    user_data_full = 0;
	uint64_t                                    constants = 0;
	uint64_t                                    flips = 0;
};
static FrameRepeat g_frame_repeat;

// Debugging aid for the draw-phases line: counts stages whose program and descriptors repeat the
// previous draw's.
static void NoteBindingRepeat(PreparedBindings& prepared, const ShaderStageRuntime& runtime) {
	const auto& snapshot = *runtime.resources;
	const auto  hash     = [](const std::vector<ShaderRecompiler::IR::DescriptorValue>& values) {
		return values.empty() ? uint64_t {0}
		                      : XXH3_64bits(values.data(), values.size() * sizeof(values[0]));
	};
	const PreparedBindings::RepeatKey key {runtime.program, hash(snapshot.images),
	                                      hash(snapshot.buffers), hash(snapshot.samplers)};
	const auto&                       last    = prepared.repeat_key;
	const bool                        program = last.program == key.program;
	auto&                             stats   = g_binding_repeats;
	stats.stages.fetch_add(1, std::memory_order_relaxed);
	stats.same_program.fetch_add(program ? 1 : 0, std::memory_order_relaxed);
	stats.same_images.fetch_add(program && last.images == key.images ? 1 : 0,
	                           std::memory_order_relaxed);
	stats.same_buffers.fetch_add(program && last.buffers == key.buffers ? 1 : 0,
	                            std::memory_order_relaxed);
	stats.same_all.fetch_add(program && last.images == key.images && last.buffers == key.buffers &&
	                                last.samplers == key.samplers
	                            ? 1
	                            : 0,
	                        std::memory_order_relaxed);
	prepared.repeat_key = key;
	// KYTY_DEBUG_FRAME_REPEAT=1 (with KYTY_DEBUG_DRAW_PHASES): how many stages bind a program with
	// descriptors (image, buffer and sampler values) that a stage of one of the last four frames
	// bound too, and how many of those have the same buffers but other images or the reverse:
	// what a cache of prepared bindings kept across frames could serve.
	static const bool frame_repeat = KYTY_GETENV("KYTY_DEBUG_FRAME_REPEAT") != nullptr;
	if (frame_repeat) {
		auto&          state = g_frame_repeat;
		const uint64_t program_key = reinterpret_cast<uint64_t>(key.program) * 0x9E3779B97F4A7C15ull;
		const uint64_t full        = program_key ^ key.images ^ (key.buffers * 3u) ^ (key.samplers * 7u);
		const uint64_t no_buffers  = program_key ^ key.images ^ (key.samplers * 7u) ^ 0x5555u;
		// The buffer descriptors without their base addresses (V# dword 0 and the low 16 bits of
		// dword 1): a record whose buffers only moved could be kept and relocated.
		uint64_t shape = 0;
		for (const auto& value: snapshot.buffers) {
			auto masked = value;
			if (masked.dword_count >= 2) {
				masked.dwords[0] = 0;
				masked.dwords[1] &= 0xffff0000u;
			}
			shape = XXH3_64bits_withSeed(masked.dwords.data(), sizeof(masked.dwords),
			                             shape ^ masked.dword_count);
		}
		const uint64_t relocated = no_buffers ^ (shape * 11u) ^ 0x3333u;
		// The Demon's Souls fork's record key: the program and the stage's user data registers.
		const uint64_t user_data =
		    program_key ^ 0x7777u ^
		    (snapshot.user_data.empty()
		         ? 0
		         : XXH3_64bits(snapshot.user_data.data(), snapshot.user_data.size() * sizeof(uint32_t)) *
		               13u);
		const uint64_t user_data_full = user_data ^ full;
		// ...and also the same constants read through the shader resource table.
		const uint64_t constants =
		    user_data_full ^ 0x1111u ^
		    (snapshot.flattened_srt.empty()
		         ? 0
		         : XXH3_64bits(snapshot.flattened_srt.data(), snapshot.flattened_srt.size() * sizeof(uint32_t)) *
		               17u);
		bool seen_constants = false;
		for (const auto& frame: state.frames) {
			seen_constants = seen_constants || frame.contains(constants);
		}
		state.constants += seen_constants ? 1 : 0;
		state.frames[state.current].insert(constants);
		state.stages++;
		bool seen_full           = false;
		bool seen_part           = false;
		bool seen_relocated      = false;
		bool seen_user_data      = false;
		bool seen_user_data_full = false;
		for (const auto& frame: state.frames) {
			seen_full           = seen_full || frame.contains(full);
			seen_part           = seen_part || frame.contains(no_buffers);
			seen_relocated      = seen_relocated || frame.contains(relocated);
			seen_user_data      = seen_user_data || frame.contains(user_data);
			seen_user_data_full = seen_user_data_full || frame.contains(user_data_full);
		}
		state.full += seen_full ? 1 : 0;
		state.without_buffers += seen_part ? 1 : 0;
		state.relocated += seen_relocated ? 1 : 0;
		state.user_data += seen_user_data ? 1 : 0;
		state.user_data_full += seen_user_data_full ? 1 : 0;
		auto& frame = state.frames[state.current];
		frame.insert(full);
		frame.insert(no_buffers);
		frame.insert(relocated);
		frame.insert(user_data);
		frame.insert(user_data_full);
	}
}

void DebugFrameRepeatFlip() {
	auto& state = g_frame_repeat;
	if (state.stages == 0) {
		return;
	}
	if (++state.flips % 120 == 0) {
		const auto percent = [&](uint64_t count) {
			return 100.0 * static_cast<double>(count) / static_cast<double>(state.stages);
		};
		std::printf("frame-repeat: stages/frame=%llu seen in last 4 frames: all=%.1f%% all but buffers=%.1f%% "
		            "buffers moved only=%.1f%% user data=%.1f%% user data+all=%.1f%% +constants=%.1f%%\n",
		            static_cast<unsigned long long>(state.stages / 120), percent(state.full),
		            percent(state.without_buffers), percent(state.relocated), percent(state.user_data),
		            percent(state.user_data_full), percent(state.constants));
		std::fflush(stdout);
		state.stages = state.full = state.without_buffers = state.relocated = state.user_data =
		    state.user_data_full = state.constants = 0;
	}
	state.current = (state.current + 1) % state.frames.size();
	state.frames[state.current].clear();
}

PreparedBindings& RenderExecutor::StageRecord(uint32_t slot, const ShaderStageRuntime& runtime,
                                              PreparedBindings& fallback) {
	if (!StageRecordsEnabled()) {
		return fallback;
	}
	// Draws of one program mostly cycle through a few materials: each program and image descriptor
	// set keeps its own bindings, so its images (the image group) and buffer and view sources stay
	// recorded while other draws come between. A key collision only shares one record.
	const auto& images  = runtime.resources->images;
	const auto  key     = XXH3_64bits_withSeed(images.data(), images.size() * sizeof(images[0]),
	                                           reinterpret_cast<uint64_t>(runtime.program));
	auto&       records = m_graphics_bindings.records[slot];
	if (const auto found = records.find(key); found != records.end()) {
		return *found->second;
	}
	// Bounded: a scene change starts over.
	if (records.size() >= 8192) {
		records.clear();
	}
	return *records.emplace(key, std::make_unique<PreparedBindings>()).first->second;
}

static uint32_t BindlessBindingFor(const ShaderRecompiler::IR::ImageResource& resource) {
	using ShaderRecompiler::Decoder::ImageDimension;
	if (resource.cube) {
		return ShaderRecompiler::IR::BindlessImagesCube;
	}
	if (resource.dimension == ImageDimension::Dim3D) {
		return ShaderRecompiler::IR::BindlessImages3D;
	}
	const bool arrayed = resource.dimension == ImageDimension::Dim2DArray ||
	                     resource.dimension == ImageDimension::Dim2DMsaaArray;
	if (resource.numeric_class == Prospero::TextureNumericClass::Uint) {
		return arrayed ? ShaderRecompiler::IR::BindlessImages2DArrayUint
		               : ShaderRecompiler::IR::BindlessImages2DUint;
	}
	if (resource.numeric_class == Prospero::TextureNumericClass::Sint) {
		return arrayed ? ShaderRecompiler::IR::BindlessImages2DArraySint
		               : ShaderRecompiler::IR::BindlessImages2DSint;
	}
	return arrayed ? ShaderRecompiler::IR::BindlessImages2DArray
	               : ShaderRecompiler::IR::BindlessImages2D;
}

// The numeric class of a bindless image array.
static Prospero::TextureNumericClass BindlessArrayClass(uint32_t binding) {
	switch (binding) {
		case ShaderRecompiler::IR::BindlessImages2DUint:
		case ShaderRecompiler::IR::BindlessImages2DArrayUint: return Prospero::TextureNumericClass::Uint;
		case ShaderRecompiler::IR::BindlessImages2DSint:
		case ShaderRecompiler::IR::BindlessImages2DArraySint: return Prospero::TextureNumericClass::Sint;
		default: return Prospero::TextureNumericClass::Float;
	}
}

// A heap entry the bindless array of this binding can hold: a single-sample texture of the
// array's numeric class (float, or the integer class of the integer arrays) and a matching shape.
// Anything else stays the placeholder rather than reaching ResolveTexture, which exits on
// descriptors a draw would never bind that way.
// An image T# (1D..2D array, a format the GPU knows): anything else in a texture table is other
// data or null.
static bool PlausibleImageDescriptor(const ShaderTextureResource& descriptor) {
	const auto type = static_cast<uint32_t>(descriptor.Type());
	return !descriptor.IsNull() && type >= 8u && type <= 13u && TextureHasSurfaceFormat(descriptor.Format());
}

static bool BindlessCompatible(const ShaderTextureResource& descriptor, uint32_t binding) {
	if (descriptor.IsNull() ||
	    Prospero::SampledTextureNumericClass(descriptor.Format()) != BindlessArrayClass(binding)) {
		return false;
	}
	const bool flat = binding == ShaderRecompiler::IR::BindlessImages2D ||
	                  binding == ShaderRecompiler::IR::BindlessImages2DArray ||
	                  binding == ShaderRecompiler::IR::BindlessImages2DUint ||
	                  binding == ShaderRecompiler::IR::BindlessImages2DArrayUint ||
	                  binding == ShaderRecompiler::IR::BindlessImages2DSint ||
	                  binding == ShaderRecompiler::IR::BindlessImages2DArraySint;
	switch (descriptor.Type()) {
		case Prospero::ImageType::kColor2D: return flat;
		case Prospero::ImageType::kColor2DArray: return flat;
		// A 1D texture is a 2D one of height 1 (sampled through the 2D arrays, see ResolveBindlessKey).
		case Prospero::ImageType::kColor1D:
		case Prospero::ImageType::kColor1DArray: return flat;
		// A cube is a 2D array of its faces (TextureType); shaders also read cube maps through the
		// 2D tables, which then see the face BASE_ARRAY names (Ghost of Yotei: 29 keys per Torii run).
		case Prospero::ImageType::kCube: return binding == ShaderRecompiler::IR::BindlessImagesCube || flat;
		case Prospero::ImageType::kColor3D: return binding == ShaderRecompiler::IR::BindlessImages3D;
		default: return false;
	}
}

// A pointer heap with layers (sub_keys): key = record * sub_keys + layer, and the layer's T# lies
// sub_stride bytes after the previous one behind the record's pointer.
static uint64_t BindlessPointerAddress(const BindlessTable::Heap& heap, uint32_t key) {
	const auto record = heap.sub_keys != 0u ? key / heap.sub_keys : key;
	return heap.base + static_cast<uint64_t>(record) * heap.record_stride + heap.pointer_offset;
}

static uint64_t BindlessLayerOffset(const BindlessTable::Heap& heap, uint32_t key) {
	return heap.sub_keys != 0u ? static_cast<uint64_t>(key % heap.sub_keys) * heap.sub_stride : 0u;
}

// The record of `key` as ResolveBindlessKey reads it (the pointer and the 32 descriptor bytes),
// hashed; distinct values for an unreadable record, a null pointer and an unreadable target.
static uint64_t BindlessRecordHash(const BindlessTable::Heap& heap, uint32_t key) {
	std::array<uint64_t, 5> words {};
	auto address = heap.base + heap.table_offset + static_cast<uint64_t>(key) * heap.record_stride;
	if (heap.pointer_offset != UINT32_MAX) {
		if (!Libs::LibKernel::Memory::TryReadBacking(BindlessPointerAddress(heap, key), &words[4],
		                                             sizeof(uint64_t))) {
			return 1;
		}
		if ((words[4] & 0x0000ffffffffffffull) == 0) {
			return 2;
		}
		address = (words[4] & 0x0000ffffffffffffull) + heap.table_offset +
		          BindlessLayerOffset(heap, key);
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, words.data(), 32u)) {
		return 3;
	}
	uint64_t hash = 0xcbf29ce484222325ull;
	for (const auto word: words) {
		hash = (hash ^ word) * 0x100000001b3ull;
	}
	return hash | 8u;
}

// KYTY_DEBUG_BINDLESS_STALE=1: every 300 frames, how many resolved keys were checked against
// their records and how many had changed.
static uint64_t g_bindless_rechecked = 0;
static uint64_t g_bindless_changed   = 0;

bool RenderExecutor::ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t key) {
	auto&                                 table = m_context.GetBindlessTable();
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count  = 8u;
	auto address =
	    heap.base + heap.table_offset + static_cast<uint64_t>(key) * heap.record_stride;
	heap.settled[key]  = 1;
	heap.record_hash[key] = BindlessRecordHash(heap, key);
	g_bindless_fail_context.heap_base = heap.base;
	g_bindless_fail_context.binding   = heap.binding;
	g_bindless_fail_context.key       = key;
	if (heap.listed[key] == 0) {
		heap.listed[key] = 1;
		heap.settled_keys.push_back(key);
	}
	if (heap.pointer_offset != UINT32_MAX) {
		// The record holds a pointer to the structure that contains the T#.
		uint64_t pointer = 0;
		if (!Libs::LibKernel::Memory::TryReadBacking(BindlessPointerAddress(heap, key), &pointer,
		                                             sizeof(pointer)) ||
		    (pointer & 0x0000ffffffffffffull) == 0) {
			// A null record pointer is the game's own empty slot (reads zero on the GPU too);
			// it only counts when a shader asked for it.
			if (g_bindless_fail_context.requested) {
				ReportPlaceholder("record pointer is null or unreadable\n");
			}
			table.SetTranslation(heap, key, 0u);
			return false;
		}
		address = (pointer & 0x0000ffffffffffffull) + heap.table_offset +
		          BindlessLayerOffset(heap, key);
	}
	if (!Libs::LibKernel::Memory::TryReadBacking(address, value.dwords.data(), 32u)) {
		ReportPlaceholder(fmt::format("T# at 0x{:x} is unreadable\n", address).c_str(), true);
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	// A 128-bit T#: the record's next 16 bytes are other data, not the descriptor's upper half.
	if (heap.resource.r128) {
		std::fill(value.dwords.begin() + 4, value.dwords.begin() + 8, 0u);
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	if (!BindlessCompatible(descriptor, heap.binding)) {
		ReportPlaceholder(fmt::format("T# does not fit the shader's texture array (format {} type {}) dwords={:08x},{:08x},{:08x},{:08x}\n",
		                              static_cast<uint32_t>(descriptor.Format()), static_cast<uint32_t>(descriptor.Type()),
		                              value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3]).c_str(),
		                  !PlausibleImageDescriptor(descriptor));
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	// The reserved component selectors 2 and 3 only appear in records that hold other data
	// (Ghost of Yotei tutorial: "unknown swizzle: 3" stopped the emulator on such a record).
	if (!IsValidImageSwizzle(descriptor.DstSelXYZW())) {
		ReportPlaceholder(fmt::format("T# has a reserved component swizzle 0x{:03x} dwords={:08x},{:08x},{:08x},{:08x}\n",
		                              descriptor.DstSelXYZW(), value.dwords[0], value.dwords[1],
		                              value.dwords[2], value.dwords[3]).c_str(),
		                  !g_bindless_fail_context.requested);
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	// Ghost of Yotei keeps 1x1 1D color textures (format 56) in its material tables and samples them
	// with 2D coordinates; the GPU reads a 1D texture as a 2D one of height 1, so it is resolved as
	// that (type 8 -> 9, 12 -> 13 in dword 3 bits 28-31).
	if (descriptor.Type() == Prospero::ImageType::kColor1D ||
	    descriptor.Type() == Prospero::ImageType::kColor1DArray) {
		value.dwords[3] += 1u << 28u;
	}
	g_soft_texture_failures = true;
	g_soft_texture_failed   = false;
	auto binding            = ResolveTexture(heap.resource, value);
	g_soft_texture_failures = false;
	const bool soft_failed  = g_soft_texture_failed;
	g_soft_texture_failed   = false;
	if (soft_failed) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	auto& texture_cache = m_context.GetTextureCache();
	auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty()) {
		ReportPlaceholder("the texture cache has no image for the T#\n");
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	BindImage(binding.image_id, false);
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
	const auto layout = image != nullptr && image->info.IsDepth()
	                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                        : vk::ImageLayout::eShaderReadOnlyOptimal;
	// One array slot per (array, view), shared by every heap and key showing that view:
	// per-frame material tables would otherwise use the array up.
	const auto slot = image != nullptr && view != nullptr
	                      ? table.SharedSlot(heap.binding, binding.image_id, view, layout)
	                      : 0u;
	if (image == nullptr || view == nullptr || slot == 0) {
		table.SetTranslation(heap, key, 0u);
		return false;
	}
	heap.slots[key]         = slot;
	image->bindless_pinned  = true;
	image->usage.texture    = true;
	heap.resolved.push_back(binding.image_id);
	table.AddImageReference(binding.image_id, heap, key);
	table.SetTranslation(heap, key, slot);
	return true;
}

// The Kombi build prints no diagnostic lines.
static constexpr bool EagerDiagnostics = false;

void RenderExecutor::ResolveBindlessHeapEagerly(BindlessTable::Heap& heap, uint32_t entries,
                                                uint64_t shader_hash) {
	const auto keys   = std::min<uint32_t>(entries, static_cast<uint32_t>(heap.settled.size()));
	uint32_t   images = 0, resolved = 0, other = 0;
	const auto started = std::chrono::steady_clock::now();
	// Every 10 s: how many heaps were resolved eagerly, by which shaders, and what it cost.
	struct Totals {
		uint64_t heaps = 0, keys = 0, images = 0, ns = 0;
	};
	static std::unordered_map<uint64_t, Totals> totals;
	static auto next_report = started + std::chrono::seconds(10);
	struct Charge {
		Totals& t;
		const std::chrono::steady_clock::time_point& start;
		uint32_t& images;
		uint32_t keys;
		~Charge() {
			t.heaps++;
			t.keys += keys;
			t.images += images;
			t.ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                  std::chrono::steady_clock::now() - start)
			                                  .count());
		}
	} charge {totals[shader_hash], started, resolved, keys};
	if (EagerDiagnostics && started >= next_report) {
		next_report = started + std::chrono::seconds(10);
		for (auto& [hash, t]: totals) {
			std::printf("\x1b[91mbindless-eager: last 10 s cs %016llx: %llu heaps, %llu keys read, %llu images "
			            "resolved, %.1f ms\x1b[0m\n",
			            static_cast<unsigned long long>(hash), static_cast<unsigned long long>(t.heaps),
			            static_cast<unsigned long long>(t.keys), static_cast<unsigned long long>(t.images),
			            static_cast<double>(t.ns) / 1e6);
		}
		std::fflush(stdout);
		// Reset in place: `charge` holds a reference to this shader's entry, and clear() freed it
		// under it (its destructor then wrote freed memory: heap corruption c0000374).
		for (auto& [hash, t]: totals) {
			t = {};
		}
	}
	std::vector<uint8_t> records(static_cast<size_t>(keys) * 32u);
	const auto base = heap.base + heap.table_offset;
	if (keys == 0 || !Libs::LibKernel::Memory::TryReadBacking(base, records.data(), records.size())) {
		return;
	}
	static std::unordered_set<uint64_t> reported;
	for (uint32_t key = 0; key < keys; key++) {
		ShaderRecompiler::IR::DescriptorValue value {};
		value.dword_count = 8u;
		std::memcpy(value.dwords.data(), records.data() + static_cast<size_t>(key) * 32u, 32u);
		const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
		if (descriptor.IsNull()) {
			continue; // the game's own empty slot
		}
		if (!PlausibleImageDescriptor(descriptor)) {
			other++;
			// Once per distinct record: other data, not an image.
			uint64_t hash = 0xcbf29ce484222325ull;
			for (const auto word: value.dwords) hash = (hash ^ word) * 0x100000001b3ull;
			if (EagerDiagnostics && reported.size() < 2048 && reported.insert(hash).second) {
				std::printf("\x1b[91mbindless-eager: cs %016llx heap 0x%llx key %u skipped, no image T#: "
				            "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\x1b[0m\n",
				            static_cast<unsigned long long>(shader_hash),
				            static_cast<unsigned long long>(heap.base), key, value.dwords[0],
				            value.dwords[1], value.dwords[2], value.dwords[3], value.dwords[4],
				            value.dwords[5], value.dwords[6], value.dwords[7]);
			}
			continue;
		}
		if (descriptor.Type() != Prospero::ImageType::kColor3D) {
			continue; // left to the shader's feedback, as before
		}
		images++;
		resolved += ResolveBindlessKey(heap, key) ? 1u : 0u;
	}
	// One line per shader and heap layout (the table moves every run).
	const auto line_key = shader_hash ^ (uint64_t {heap.table_offset} << 48u) ^ (uint64_t {heap.binding} << 56u);
	static std::unordered_map<uint64_t, uint32_t> lines;
	if (EagerDiagnostics && lines[line_key]++ < 4u) {
		std::printf("\x1b[91mbindless-eager: cs %016llx heap 0x%llx (+%u, binding %u) resolved on first "
		            "sight: %u of %u 3D T#s, %u other records skipped, %u keys\x1b[0m\n",
		            static_cast<unsigned long long>(shader_hash), static_cast<unsigned long long>(heap.base),
		            heap.table_offset, heap.binding, resolved, images, other, keys);
		std::fflush(stdout);
	}
}

void RenderExecutor::ResolveBindlessRequests() {
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	if (frame == m_bindless_frame) {
		return;
	}
	auto& table     = m_context.GetBindlessTable();
	auto& scheduler = m_context.GetCommandScheduler();
	// The flags come from a snapshot recorded in an earlier frame. Until the GPU has executed
	// it, check again at the next bindless draw instead of waiting.
	if (table.SnapshotRecorded() && !table.SnapshotReady(scheduler)) {
		return;
	}
	m_bindless_frame = frame;
	m_context.GetBindlessTable().DrainWatchdogReports(frame);
	if (!table.SnapshotRecorded()) {
		table.RecordFeedbackSnapshot(scheduler);
		return;
	}
	table.ConsumeSnapshot();
	// Each texture may upload and detile; spread first sight of a scene over a few frames.
	constexpr uint32_t Budget   = 128;
	uint32_t           resolved = 0;
	for (auto& heap: m_context.GetBindlessTable().Heaps()) {
		m_bindless_requests.clear();
		m_context.GetBindlessTable().TakeRequests(heap, m_bindless_requests);
		for (const auto key: m_bindless_requests) {
			if (heap.settled[key] != 0) {
				continue;
			}
			if (resolved >= Budget) {
				break; // still pending: the next frame flags it again
			}
			g_bindless_fail_context.requested = true;
			resolved += ResolveBindlessKey(heap, key) ? 1u : 0u;
			g_bindless_fail_context.requested = false;
		}
	}
	table.RecordFeedbackSnapshot(scheduler);
}

// Bindless samplers: the S# records of each sampler heap a draw indexes are mirrored into the
// sampler array of set 1, once per frame per heap, and the shader gets the region base and record
// count. Keys outside it, and heaps that cannot be read, use the default sampler in slot 0.
void RenderExecutor::PrepareBindlessSamplers(const ShaderStageRuntime& runtime,
                                             PreparedBindings&         prepared) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	if (snapshot.bindless_sampler_heaps.empty()) {
		return;
	}
	auto& table = m_context.GetBindlessTable();
	auto& cache = m_context.GetSamplerCache();
	{
		ShaderSamplerResource default_sampler;
		std::ranges::copy(ShaderRecompiler::IR::BindlessDefaultSampler, default_sampler.fields);
		table.WriteDefaultSampler(cache.GetSampler(default_sampler, false));
	}
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	constexpr uint64_t MaxRecords = 1024;
	for (const auto& use: snapshot.bindless_sampler_heaps) {
		uint32_t region  = 0;
		uint32_t entries = 0;
		if (table.SamplersEnabled() && use.sampler < program.info.samplers.size()) {
			const auto&    sampler = program.info.samplers[use.sampler];
			const uint32_t flags =
			    (sampler.depth_compare ? BindlessTable::SamplerDepthCompare : 0u) |
			    (sampler.force_point_filtering ? BindlessTable::SamplerPointFiltering : 0u) |
			    (sampler.integer_border ? BindlessTable::SamplerIntegerBorder : 0u);
			auto* heap = table.FindOrCreateSamplerHeap(use.base, use.table_offset, flags);
			if (heap->checked_frame != frame) {
				heap->checked_frame = frame;
				const uint64_t available =
				    use.size > use.table_offset ? (use.size - use.table_offset) / 16u : 0u;
				std::vector<std::array<uint32_t, 4>> records(std::min(available, MaxRecords));
				const uint64_t address = use.base + use.table_offset;
				const uint64_t bytes   = records.size() * sizeof(records[0]);
				if (!records.empty() &&
				    !m_context.GetBufferCache().HasGpuDirtyBytes(address, bytes) &&
				    Libs::LibKernel::Memory::TryReadBacking(address, records.data(), bytes)) {
					(void)table.MirrorSamplerHeap(*heap, records, cache);
				}
			}
			region  = heap->region;
			entries = region != 0 ? static_cast<uint32_t>(heap->records.size()) : 0u;
		}
		prepared.bindless_patches.push_back({use.mapping_offset, region, entries});
	}
}

void RenderExecutor::PrepareBindlessHeaps(const ShaderStageRuntime& runtime,
                                          PreparedBindings& prepared) {
	prepared.bindless_patches.clear();
	prepared.bindless_images.clear();
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	auto&       table    = m_context.GetBindlessTable();
	if ((snapshot.bindless_heaps.empty() && snapshot.bindless_sampler_heaps.empty()) ||
	    !table.Enabled()) {
		return;
	}
	PrepareBindlessSamplers(runtime, prepared);
	if (snapshot.bindless_heaps.empty()) {
		return;
	}
	ResolveBindlessRequests();
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	static const bool stale_log = KYTY_GETENV("KYTY_DEBUG_BINDLESS_STALE") != nullptr;
	if (stale_log) [[unlikely]] {
		static uint64_t reported = 0;
		if (frame >= reported + 300u) {
			reported = frame;
			std::printf("bindless-stale: frame=%llu heaps=%zu rechecked=%llu changed=%llu\n",
			            static_cast<unsigned long long>(frame), table.Heaps().size(),
			            static_cast<unsigned long long>(g_bindless_rechecked),
			            static_cast<unsigned long long>(g_bindless_changed));
			std::fflush(stdout);
			g_bindless_rechecked = 0;
			g_bindless_changed   = 0;
		}
	}
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.image >= program.info.images.size()) {
			continue;
		}
		const auto& resource = program.info.images[use.image];
		const uint64_t span  = use.size > use.table_offset ? use.size - use.table_offset : 0;
		// Keys whose whole 32-byte descriptor (or 8-byte pointer) lies inside the table.
		const uint64_t keys =
		    use.pointer_offset != UINT32_MAX
		        ? (use.size >= uint64_t {use.pointer_offset} + 8u
		               ? (use.size - use.pointer_offset - 8u) / use.record_stride + 1u
		               : 0u)
		    : use.record_stride == 32u ? span / 32u
		    : span >= 32u              ? (span - 32u) / use.record_stride + 1u
		                               : 0u;
		const auto layered   = use.pointer_offset != UINT32_MAX && use.sub_keys != 0u
		                           ? keys * use.sub_keys
		                           : keys;
		const auto entries   = static_cast<uint32_t>(std::min<uint64_t>(layered, 1u << 18u));
		auto* heap = table.FindOrCreateHeap(use.base, use.table_offset, use.record_stride,
		                                    BindlessBindingFor(resource), entries, resource,
		                                    use.pointer_offset, use.sub_stride, use.sub_keys);
		if (heap == nullptr) {
			continue; // region 0, count 0: every key samples the placeholder
		}
		heap->shader_hash = program.shader_hash;
		heap->valid_keys  = std::max(heap->valid_keys, std::min(entries, heap->entries));
		// A heap seen in an earlier frame: a record the guest rewrote since (a per-frame table
		// whose address comes round again, a streamed texture replaced) is resolved again, or
		// the key would keep showing the earlier frame's texture.
		// KYTY_BINDLESS_RECHECK=0 keeps every key as first resolved (A/B for the device losses);
		// checked_frame still marks the heap as used, which CompactHeaps relies on.
		static const bool recheck = [] {
			const char* text = KYTY_GETENV("KYTY_BINDLESS_RECHECK");
			return text == nullptr || std::strcmp(text, "0") != 0;
		}();
		if (heap->checked_frame != frame) {
			heap->checked_frame = frame;
			for (const auto key: heap->settled_keys) {
				if (!recheck) {
					break;
				}
				if (key >= heap->entries || heap->settled[key] == 0) {
					continue;
				}
				g_bindless_rechecked++;
				if (BindlessRecordHash(*heap, key) != heap->record_hash[key]) {
					g_bindless_changed++;
					heap->settled[key] = 0;
					(void)ResolveBindlessKey(*heap, key);
				}
			}
		}
		// A compute shader that runs rarely and gets a T# table at a new address each time (Ghost
		// of Yotei's moon volume pass ef5181198aadd55d, 8192 keys) finds every key still pending:
		// the feedback that would name them arrives after its heap is gone, so it sampled only
		// placeholders and wrote zeros. A volume heap of plain T# records nothing has resolved yet
		// is resolved right away at a compute dispatch, every key that holds a 3D T#. Only volumes:
		// resolving every image of the per-frame 2D tables cost seconds per 10 s and reached a
		// depth T# no shader samples (an EXIT).
		constexpr bool eager = true;
		if (eager && program.stage == ShaderType::Compute && use.record_stride == 32u &&
		    use.pointer_offset == UINT32_MAX && heap->binding == ShaderRecompiler::IR::BindlessImages3D &&
		    heap->settled_keys.empty()) {
			ResolveBindlessHeapEagerly(*heap, std::min(entries, heap->entries), program.shader_hash);
		}
		// A table of material records is rewritten at a new address every frame, so its heap
		// is new each frame and the shader's feedback would name its keys only after the heap
		// is gone. A small one is resolved completely right away.
		if (use.record_stride != 32u && entries <= 512u) {
			for (uint32_t key = 0; key < entries && key < heap->settled.size(); key++) {
				if (heap->settled[key] == 0) {
					(void)ResolveBindlessKey(*heap, key);
				}
			}
		}
		prepared.bindless_patches.push_back(
		    {use.mapping_offset, heap->region, std::min(entries, heap->entries)});
		prepared.bindless_images.insert(prepared.bindless_images.end(), heap->resolved.begin(),
		                                heap->resolved.end());
	}
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	if (g_draw_phases.Active()) [[unlikely]] {
		NoteBindingRepeat(prepared, runtime);
	}
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	// buffer_sources keeps the previous draw's entries of this program's slots for FindBuffers,
	// which replaces them all.
	prepared.buffer_sources.resize(program.info.buffers.size());
	prepared.buffers.clear();
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.image_sources.resize(program.info.images.size());
	prepared.samplers.clear();
	prepared.shader_data.clear();
	auto& texture_cache = m_context.GetTextureCache();
	const bool bind_stats = g_draw_phases.Active() && BindStatsEnabled();
	uint64_t   stats_start = bind_stats ? DrawPhaseTimer::Now() : 0;
	// A draw repeating the stage's program and image descriptors binds its last images while they
	// stand (see ImageGroupStands); RebindImages checks them again.
	const auto& group = prepared.image_group;
	prepared.image_group_kept =
	    ImageGroupsEnabled() && group.program == &program &&
	    std::ranges::equal(snapshot.images, group.images) &&
	    ImageGroupStandsOrRevalidates(texture_cache, prepared);
	if (bind_stats) [[unlikely]] {
		const auto now = DrawPhaseTimer::Now();
		NoteImageGroup(texture_cache, prepared, &program, snapshot.images,
		               prepared.image_group_kept);
		stats_start += DrawPhaseTimer::Now() - now; // Leave out the reason search.
	}
	if (prepared.image_group_kept) {
		for (const auto counter: group.mip_counters) {
			m_context.MarkMipStatsCounter(counter & 0xffu, (counter & 0x100u) != 0);
		}
		for (const auto& binding: prepared.images) {
			BindImage(binding.image_id, false);
		}
		if (VerifyImageGroups()) [[unlikely]] {
			for (uint32_t i = 0; i < program.info.images.size(); i++) {
				auto           source = prepared.image_sources[i];
				TextureBinding check;
				ResolveTextureInto(program.info.images[i], snapshot.images[i], &source, check);
				const auto& kept = prepared.images[i];
				ReportImageGroupCheck(check.image_id == kept.image_id &&
				                          check.desc.view_info == kept.desc.view_info,
				                      kept.desc.info.data.address);
			}
		}
	}
	for (uint32_t i = 0; !prepared.image_group_kept && i < program.info.images.size(); i++) {
		MarkMipStats(m_context, program.info.images[i], snapshot.images[i]);
		// Consecutive draws mostly bind the same textures.
		auto& previous = prepared.images[i];
		if (ReuseTexture(texture_cache, prepared.image_sources[i], previous,
		                 program.info.images[i], snapshot.images[i])) {
			// As ResolveTexture returns it; RebindImages acquires the views.
			previous.image_view = nullptr;
			previous.layout     = vk::ImageLayout::eUndefined;
			previous.mip_views.clear();
			BindImage(previous.image_id,
			          previous.desc.type == TextureCache::BindingType::Storage);
			continue;
		}
		ResolveTextureInto(program.info.images[i], snapshot.images[i], &prepared.image_sources[i],
		                   previous);
		BindImage(previous.image_id, previous.desc.type == TextureCache::BindingType::Storage);
	}
	if (bind_stats && !program.info.images.empty()) [[unlikely]] {
		const size_t kind = prepared.image_group_kept ? 0 : 1;
		g_bind_stats.prepare_ticks[kind] += DrawPhaseTimer::Now() - stats_start;
		g_bind_stats.prepare_count[kind]++;
	}
	g_draw_phases.Mark(DrawPhaseTimer::StageTextures);
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	g_draw_phases.Mark(DrawPhaseTimer::StageSamplers);
	PrepareBindlessHeaps(runtime, prepared);
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.bindings.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

// KYTY_DEBUG_DEVICE_FAULT=1: each compute shader's buffer V#s, once per shader, slot and V# words
// (256 lines at most). For Ghost of Yotei's deferred lighting shader d108b276bb7c5f0f, which writes
// zeros (blocky roofs), the material buffer (slot 1) also gets the guest bytes the PS5 reads for
// material 0 (+0x0) and material 6 (+0x480) and the buffer cache's state of that range at bind
// time: registered in a cache buffer, CPU-written since its upload, GPU-written; a read-only range
// of at most one cache page may be served from the stream ring instead (BufferCache::ObtainBuffer).
static void NoteComputeBufferVsharp(BufferCache& cache, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                    uint32_t slot, const ShaderBufferResource& descriptor,
                                    uint64_t address, uint64_t size) {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_DEVICE_FAULT") != nullptr;
	if (!enabled || program.stage != ShaderType::Compute) {
		return;
	}
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> logged;
	const uint64_t key = program.shader_hash ^ (uint64_t {slot} << 59u) ^ address ^
	                     (uint64_t {descriptor.fields[1]} << 7u) ^ (uint64_t {descriptor.fields[2]} << 19u) ^
	                     (uint64_t {descriptor.fields[3]} << 31u);
	std::scoped_lock lock {mutex};
	if (logged.size() >= 256u || !logged.insert(key).second) {
		return;
	}
	const auto& resource = program.info.buffers[slot];
	std::printf("cs-buffer-vsharp: shader=%016" PRIx64 " slot=%u written=%d address=0x%016" PRIx64
	            " stride=%u num_records=%u size=0x%" PRIx64 " bound=0x%" PRIx64
	            " fields3=0x%08x oob=%u swizzle=%d\n",
	            program.shader_hash, slot, resource.written ? 1 : 0, address,
	            static_cast<uint32_t>(descriptor.Stride()), descriptor.NumRecords(), descriptor.GetSize(),
	            size, descriptor.fields[3], (descriptor.fields[3] >> 28u) & 3u,
	            descriptor.SwizzleEnabled() ? 1 : 0);
	if (program.shader_hash == 0xd108b276bb7c5f0full && slot == 1u) {
		std::printf("cs-buffer-vsharp:   host: registered=%d cpu_modified=%d gpu_modified=%d "
		            "stream_eligible=%d (read-only, <= 0x%" PRIx64 " bytes)\n",
		            cache.IsRegionRegistered(address, size) ? 1 : 0,
		            cache.IsRegionCpuModified(address, size) ? 1 : 0,
		            cache.IsRegionGpuModified(address, size) ? 1 : 0,
		            !resource.written && size <= BufferCache::CACHING_PAGESIZE ? 1 : 0,
		            BufferCache::CACHING_PAGESIZE);
		const auto dump = [&](uint64_t offset, uint64_t bytes_wanted) {
			std::array<uint8_t, 192> bytes {};
			if (!Libs::LibKernel::Memory::TryReadBacking(address + offset, bytes.data(), bytes_wanted)) {
				std::printf("cs-buffer-vsharp:   +0x%03" PRIx64 " unreadable\n", offset);
				return;
			}
			for (uint64_t row = 0; row < bytes_wanted; row += 32u) {
				std::printf("cs-buffer-vsharp:   +0x%03" PRIx64 ":", offset + row);
				for (uint64_t i = row; i < row + 32u && i < bytes_wanted; i += 4u) {
					uint32_t word = 0;
					std::memcpy(&word, bytes.data() + i, sizeof(word));
					std::printf(" %08x", word);
				}
				std::printf("\n");
			}
		};
		dump(0, 64);
		dump(1152, 192);
	}
	std::fflush(stdout);
}

// KYTY_BUFFER_KEEP=1: a buffer binding whose descriptor equals the previous draw's (same stage
// record) keeps what FindBuffers and RebindBuffers made of it while nothing they read changed:
// the memory map (ClampRangeSize), the set of cached buffers (ids still cover their ranges), the
// image set (a texel buffer still has no image over it), and the range is still CPU-clean (what
// ObtainBuffer's clean path checks). Only read-only bindings the cached buffer served itself keep
// their rebind. Premises: a live BufferId never changes its handle or
// bounds in place (joins make a new buffer), and compiled programs are never freed, so an equal
// program pointer means equal resource flags. The title binds ~9 buffers per draw (~40k a frame),
// 46 % of them kept. KYTY_DEBUG_AB=bufkeep
// alternates; KYTY_VERIFY_BUFFER_KEEP=1 does the work anyway and reports differences;
// KYTY_DEBUG_BUFFER_KEEP=1 prints counts every 5 s.
static bool BufferKeepEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BUFFER_KEEP");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("bufkeep");
	return enabled && !(ab && AbFeatureOff());
}

static bool VerifyBufferKeep() {
	static const bool enabled = KYTY_GETENV("KYTY_VERIFY_BUFFER_KEEP") != nullptr;
	return enabled;
}

namespace {

struct BufferKeepStats {
	uint64_t sources       = 0; // FindBuffers entries.
	uint64_t found_kept    = 0;
	uint64_t bindings      = 0; // RebindBuffers non-table entries.
	uint64_t rebind_kept   = 0;
	uint64_t not_found     = 0; // Rebind misses: the source was found anew.
	uint64_t no_record     = 0; // ... no reusable rebind recorded (written, stream, image, null).
	uint64_t image_set     = 0; // ... the image set changed.
	uint64_t not_clean     = 0; // ... the buffer moved or the range is CPU-dirty.
	uint64_t verify_diff   = 0;
	std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();

	void MaybePrint() {
		static const bool enabled = KYTY_GETENV("KYTY_DEBUG_BUFFER_KEEP") != nullptr;
		if (!enabled) {
			return;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - window < std::chrono::seconds(5)) {
			return;
		}
		window = now;
		std::printf("buffer-keep: sources=%" PRIu64 " found-kept=%" PRIu64 " bindings=%" PRIu64
		            " rebind-kept=%" PRIu64 " miss: not-found=%" PRIu64 " no-record=%" PRIu64
		            " image-set=%" PRIu64 " not-clean=%" PRIu64 " verify-diff=%" PRIu64 "\n",
		            sources, found_kept, bindings, rebind_kept, not_found, no_record, image_set,
		            not_clean, verify_diff);
		std::fflush(stdout);
		*this = {};
		window = now;
	}
};

BufferKeepStats g_buffer_keep_stats;

void ReportBufferKeepDiff(const char* what, uint64_t address, uint64_t size) {
	g_buffer_keep_stats.verify_diff++;
	static uint32_t printed = 0;
	if (printed++ < 32) {
		std::printf("buffer-keep verify: %s differs (address=0x%016" PRIx64 " size=0x%" PRIx64 ")\n",
		            what, address, size);
		std::fflush(stdout);
	}
}

} // namespace

void RenderExecutor::FindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       cache    = m_context.GetBufferCache();
	const bool  keep     = BufferKeepEnabled();
	const bool  verify   = keep && VerifyBufferKeep();
	// Map generations are stored plus one: 0 marks a source that is not kept.
	const auto  map_generation = keep ? Libs::LibKernel::Memory::RangeMapGeneration() + 1u : 0u;

	// The previous draw's sources of this stage: consecutive draws mostly bind buffers the same
	// cached buffer still covers, which RefindBuffer confirms without the page table.
	auto& sources = prepared.buffer_sources;
	const auto previous_count = sources.size();
	sources.resize(program.info.buffers.size());
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		auto& source = sources[i];
		const auto& value = snapshot.buffers[i];
		source.found_kept = false;
		if (keep) {
			g_buffer_keep_stats.sources++;
			if (i < previous_count && source.map_generation == map_generation &&
			    source.buffer_set == cache.BufferSetGeneration() && source.descriptor == value) {
				g_buffer_keep_stats.found_kept++;
				if (!verify) {
					source.found_kept = true;
					continue;
				}
				const auto kept = source;
				auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(value);
				const auto address = descriptor.Base48();
				const auto requested_size = descriptor.GetSize();
				const auto size = address == 0 || requested_size == 0
				                      ? 0
				                      : Libs::LibKernel::Memory::ClampRangeSize(address, requested_size);
				const auto id = size == 0 ? BufferId {}
				                : kept.id ? cache.RefindBuffer(kept.id, address, size)
				                          : cache.FindBuffer(address, size);
				if ((size == 0 ? 0 : address) != kept.address || size != kept.size || id != kept.id) {
					ReportBufferKeepDiff("source", address, size);
				}
				source.found_kept = true;
				continue;
			}
		}
		auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(value);
		const auto address = descriptor.Base48();
		const auto requested_size = descriptor.GetSize();
		// A base in no mapped range is bound as a null buffer (reads zero) instead of ending the
		// emulator: the title in Pro mode, an indirect dispatch at the first LBVH build read a V# at
		// 0x12ffff9964 (2026-10-30, one run in several).
		const bool unmapped = address != 0 && requested_size != 0 &&
		                      Libs::LibKernel::Memory::MappedRangeSize(address, requested_size) == 0;
		if (unmapped) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 8) {
				std::printf("FindBuffers: buffer slot %u V# base 0x%012" PRIx64 " size 0x%" PRIx64
				            " is not mapped: null buffer\n",
				            i, address, requested_size);
				std::fflush(stdout);
			}
		}
		if (address == 0 || requested_size == 0 || unmapped) {
			source = {};
		} else {
			const auto size = Libs::LibKernel::Memory::ClampRangeSize(address, requested_size);
			NoteComputeBufferVsharp(cache, program, i, descriptor, address, size);
			const auto id   = i < previous_count && source.id ? source.id : BufferId {};
			source          = {address, size,
			                   id ? cache.RefindBuffer(id, address, size) : cache.FindBuffer(address, size)};
		}
		if (keep) {
			// After FindBuffer, which may have registered a buffer.
			source.descriptor     = value;
			source.map_generation = map_generation;
			source.buffer_set     = cache.BufferSetGeneration();
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != program.info.buffers.size());

	prepared.buffers.clear();
	prepared.buffers.reserve(program.info.buffers.size());
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	// Table mode: buffer entries and the flattened SRT go into the draw's block instead
	// (TableDword: its address).
	auto& block = m_table_block;
	if (layout.table) {
		block.assign(layout.table_srt_dword, 0u);
	}
	for (uint32_t i = 0; i < program.info.buffers.size() && layout.table; i++) {
		uint32_t          buffer_offset = 0;
		vk::DeviceAddress address       = 0;
		const auto&       resource      = program.info.buffers[i];
		auto              runtime_resource = resource;
		if (resource.descriptor_format == ShaderRecompiler::IR::GenericBufferFormat) {
			runtime_resource.descriptor_format = RuntimeBufferFormat(snapshot, i);
		}
		const auto info = NativeStorageBuffer(m_context, prepared.buffer_sources[i],
		                                      runtime_resource, buffer_offset, &address);
		const uint64_t bytes = info.buffer ? info.range : 0u;
		EXIT_IF(bytes > UINT32_MAX || address == 0);
		auto* entry = &block[i * ShaderRecompiler::IR::TableBufferEntryDwords];
		entry[0]    = static_cast<uint32_t>(address);
		entry[1]    = static_cast<uint32_t>(address >> 32u);
		entry[2]    = static_cast<uint32_t>(bytes);
		entry[3]    = buffer_offset;
	}
	m_track_entries.clear();
	const bool keep   = BufferKeepEnabled();
	const bool verify = keep && VerifyBufferKeep();
	auto&      cache  = m_context.GetBufferCache();
	const auto image_set =
	    keep ? m_context.GetTextureCache().m_image_set_generation.load(std::memory_order_acquire) : 0;
	for (uint32_t i = 0; i < program.info.buffers.size() && !layout.table; i++) {
		uint32_t buffer_offset = 0;
		const auto& resource = program.info.buffers[i];
		auto&       source   = prepared.buffer_sources[i];
		bool        kept     = false;
		if (keep) {
			auto& stats = g_buffer_keep_stats;
			stats.bindings++;
			if (!source.found_kept) {
				stats.not_found++;
			} else if (source.image_set == 0 || source.program != &program) {
				stats.no_record++;
			} else if (source.image_set != image_set ||
			           source.buffer_set != cache.BufferSetGeneration()) {
				stats.image_set++;
			} else if (!cache.KeepReadBinding(source.id, source.address, source.size)) {
				stats.not_clean++;
			} else {
				// What NativeStorageBuffer returned for this source before (KeepReadBinding touched
				// the buffer as ObtainBuffer's clean path does).
				stats.rebind_kept++;
				kept = true;
			}
			// A later rebind without FindBuffers (draw memo) must not take this draw's state.
			source.found_kept = false;
		}
		if (kept && !verify) {
			prepared.buffers.push_back(source.info);
			pack_memory_offset(i, source.buffer_offset);
			continue;
		}
		const bool texel = resource.formatted;
		// KYTY_GPU_DIRTY_TRACK: a written buffer of a tracked program may take its image's bitmap.
		TextureCache::TrackWrite  track_write {};
		TextureCache::TrackWrite* track =
		    layout.track && resource.written ? &track_write : nullptr;
		if (resource.descriptor_format == ShaderRecompiler::IR::GenericBufferFormat) {
			// The element size the binding offset may follow is this dispatch's format.
			auto runtime_resource              = resource;
			runtime_resource.descriptor_format = RuntimeBufferFormat(snapshot, i);
			prepared.buffers.push_back(NativeStorageBuffer(m_context, source, runtime_resource,
			                                               buffer_offset, nullptr, track));
		} else {
			prepared.buffers.push_back(
			    NativeStorageBuffer(m_context, source, resource, buffer_offset, nullptr, track));
		}
		pack_memory_offset(i, buffer_offset);
		if (track_write.bitmap != 0) {
			// The shader's byte offsets count from the host binding's start, `buffer_offset` below
			// the V#'s base.
			if (m_track_entries.empty()) {
				m_track_entries.assign(program.info.buffers.size() *
				                           ShaderRecompiler::IR::TrackEntryDwords,
				                       0u);
			}
			auto* entry = &m_track_entries[i * ShaderRecompiler::IR::TrackEntryDwords];
			entry[0]    = static_cast<uint32_t>(track_write.bitmap);
			entry[1]    = static_cast<uint32_t>(track_write.bitmap >> 32u);
			entry[2]    = static_cast<uint32_t>(source.address - buffer_offset -
			                                    track_write.image_address);
			entry[3]    = static_cast<uint32_t>(track_write.image_size);
		}
		if (kept) {
			const auto& made = prepared.buffers.back();
			if (made.buffer != source.info.buffer || made.offset != source.info.offset ||
			    made.range != source.info.range || buffer_offset != source.buffer_offset) {
				ReportBufferKeepDiff("rebind", source.address, source.size);
			}
		}
		if (keep) {
			// Reusable: read-only, served by the cached buffer itself (not a stream copy or the
			// null buffer), and for texel buffers no image over the range now or while the image
			// set stands.
			const auto& made = prepared.buffers.back();
			const bool  reusable = !resource.written && source.size != 0 && made.buffer &&
			                       cache.IsBufferHandle(source.id, made.buffer) &&
			                       (!texel || cache.LastNoImageStable());
			source.image_set     = reusable ? image_set : 0;
			source.program       = &program;
			source.info          = made;
			source.buffer_offset = buffer_offset;
		}
	}
	if (keep) {
		g_buffer_keep_stats.MaybePrint();
	}
	if (!m_track_entries.empty()) {
		auto& ring = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		EXIT_IF(!ring.HasDeviceAddress());
		const auto bytes = m_track_entries.size() * sizeof(uint32_t);
		const auto [data, offset] =
		    ring.Map(bytes, ShaderRecompiler::IR::TrackEntryDwords * sizeof(uint32_t));
		EXIT_IF(data == nullptr);
		std::memcpy(data, m_track_entries.data(), bytes);
		ring.Commit();
		const auto address = ring.BufferDeviceAddress() + offset;
		prepared.shader_data[layout.TrackDword()]      = static_cast<uint32_t>(address);
		prepared.shader_data[layout.TrackDword() + 1u] = static_cast<uint32_t>(address >> 32u);
	}
	// The strides of this dispatch's V#s, two per DWORD, for the buffers whose module reads them.
	for (uint32_t k = 0; k < layout.buffer_strides.size(); k++) {
		const auto index = layout.buffer_strides[k];
		EXIT_IF(index >= snapshot.buffers.size());
		prepared.shader_data[layout.BufferStrideDword() + k / 2u] |=
		    (ShaderRecompiler::IR::CanonicalPackedStride(snapshot.buffers[index]) & 0x3fffu)
		    << ((k % 2u) * 16u);
	}
	// Stride, format and dst_sel of the buffers whose formatted loads decode them.
	for (uint32_t k = 0; k < layout.buffer_formats.size(); k++) {
		const auto index = layout.buffer_formats[k];
		EXIT_IF(index >= snapshot.buffers.size());
		prepared.shader_data[layout.BufferFormatDword() + k] =
		    ShaderRecompiler::IR::RuntimeBufferSlotWord(snapshot.buffers[index]);
	}
	g_draw_phases.Mark(DrawPhaseTimer::BufferViews);
	if (layout.table) {
		// The header's first DWORD is the flattened SRT's size; at least one (zero) DWORD
		// follows, which a shader index past the SRT reads.
		// Written straight into the ring: the entries, then the SRT with its bindless patches.
		const auto& srt = snapshot.flattened_srt;
		block[layout.table_srt_dword - ShaderRecompiler::IR::TableSrtHeaderDwords] =
		    static_cast<uint32_t>(srt.size());
		const size_t srt_dwords = std::max<size_t>(srt.size(), 1u);
		auto& ring = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		EXIT_IF(!ring.HasDeviceAddress());
		const auto [data, offset] =
		    ring.Map((block.size() + srt_dwords) * sizeof(uint32_t),
		             ShaderRecompiler::IR::TableBufferEntryDwords * sizeof(uint32_t));
		EXIT_IF(data == nullptr);
		auto* words = reinterpret_cast<uint32_t*>(data);
		std::memcpy(words, block.data(), block.size() * sizeof(uint32_t));
		words += block.size();
		if (srt.empty()) {
			words[0] = 0u;
		} else {
			std::memcpy(words, srt.data(), srt.size() * sizeof(uint32_t));
		}
		for (const auto& [patch, region, entries]: prepared.bindless_patches) {
			if (static_cast<size_t>(patch) + 1u < srt.size()) {
				words[patch]      = region;
				words[patch + 1u] = entries;
			}
		}
		ring.Commit();
		const auto address = ring.BufferDeviceAddress() + offset;
		prepared.shader_data[layout.TableDword()]      = static_cast<uint32_t>(address);
		prepared.shader_data[layout.TableDword() + 1u] = static_cast<uint32_t>(address >> 32u);
	}
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		if (prepared.bindless_patches.empty()) {
			prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
		} else {
			m_bindless_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
			for (const auto& [offset, region, entries]: prepared.bindless_patches) {
				if (static_cast<size_t>(offset) + 1u < m_bindless_srt.size()) {
					m_bindless_srt[offset]      = region;
					m_bindless_srt[offset + 1u] = entries;
				}
			}
			prepared.flattened_srt = NativeUpload(m_context, m_bindless_srt);
		}
	}
	if (ShaderRecompiler::IR::FindBinding(
	        program.bindings, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

// KYTY_DEBUG_STORAGE_REUSE=0 acquires storage images every draw; KYTY_DEBUG_AB=storagereuse
// alternates.
static bool StorageReuseEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_STORAGE_REUSE");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("storagereuse");
	return enabled && !(ab && AbFeatureOff());
}

// KYTY_VERIFY_STORAGE_REUSE=1: a storage binding that keeps its view acquires it anyway and
// reports what FindTexture returned differently or changed (see TextureCache::IsStorageCurrent).
static bool VerifyStorageReuse() {
	static const bool enabled = KYTY_GETENV("KYTY_VERIFY_STORAGE_REUSE") != nullptr;
	return enabled;
}

// KYTY_VERIFY_VIEW_REUSE=1: the same for sampled textures, including views that another slot's
// binding of the description acquired (the view memo, see ResolveTextureInto).
static bool VerifyViewReuse() {
	static const bool enabled = KYTY_GETENV("KYTY_VERIFY_VIEW_REUSE") != nullptr;
	return enabled;
}

static void CheckViewReuse(TextureCache& cache, const TextureBinding& binding) {
	const bool  storage         = binding.desc.type == TextureCache::BindingType::Storage;
	const auto& image           = cache.GetImage(binding.image_id);
	const bool  gpu_modified    = image.IsGpuModified();
	const bool  buffer_modified = image.IsBufferModified();
	const bool  cpu_dirty       = image.IsCpuDirty();
	const auto  view            = cache.FindTexture(binding.image_id, binding.desc);
	const auto& after           = cache.GetImage(binding.image_id);
	const bool  same            = view == binding.image_view &&
	                  after.IsGpuModified() == gpu_modified &&
	                  after.IsBufferModified() == buffer_modified && after.IsCpuDirty() == cpu_dirty;
	static std::atomic<uint64_t> checked_counts[2] {};
	static std::atomic<uint64_t> missed_counts[2] {};
	auto&       checked = checked_counts[storage ? 1 : 0];
	auto&       missed  = missed_counts[storage ? 1 : 0];
	const char* kind    = storage ? "storage" : "texture";
	const auto  count   = checked.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!same && missed.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::printf("%s-reuse verify: 0x%016" PRIx64
		            " kept view would have missed a change (view %s, gpu %d->%d, buffer %d->%d, "
		            "cpu %d->%d)\n",
		            kind, image.info.data.address, view == binding.image_view ? "same" : "differs",
		            gpu_modified, after.IsGpuModified(), buffer_modified, after.IsBufferModified(),
		            cpu_dirty, after.IsCpuDirty());
	}
	if (count % 100000 == 0) {
		std::printf("%s-reuse verify: reuses=%" PRIu64 " missed=%" PRIu64 "\n", kind, count,
		            missed.load(std::memory_order_relaxed));
		std::fflush(stdout);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	// Bindings assembled without PrepareBindings have no recorded sources.
	prepared.image_sources.resize(images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// PrepareBindings kept the group; BDA synchronization since may have dirtied an image.
	const bool group_kept     = prepared.image_group_kept;
	prepared.image_group_kept = false;
	// KYTY_DEBUG_BIND_STATS: this call's time, for kept groups and the rest.
	struct StatsScope {
		bool     on;
		size_t   kind;
		uint64_t start;
		~StatsScope() {
			if (on) [[unlikely]] {
				g_bind_stats.rebind_ticks[kind] += DrawPhaseTimer::Now() - start;
				g_bind_stats.rebind_count[kind]++;
			}
		}
	};
	const bool bind_stats =
	    g_draw_phases.Active() && BindStatsEnabled() && !program.info.images.empty();
	StatsScope stats_scope {bind_stats, group_kept ? 0u : 1u,
	                        bind_stats ? DrawPhaseTimer::Now() : 0};
	const bool stands = group_kept && ImageGroupStandsOrRevalidates(texture_cache, prepared);
	if (bind_stats && group_kept && !stands) [[unlikely]] {
		g_bind_stats.kept_lost++;
	}
	if (stands) {
		if (VerifyImageGroups()) [[unlikely]] {
			for (const auto& binding: images) {
				CheckViewReuse(texture_cache, binding);
			}
		}
		return;
	}
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			ResolveTextureInto(program.info.images[i], snapshot.images[i],
			                   &prepared.image_sources[i], images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else if (auto& source = prepared.image_sources[i];
		           TextureReuseEnabled() && source.view_image == binding.image_id &&
		           source.view_info == binding.desc.view_info &&
		           (binding.desc.type == TextureCache::BindingType::Texture
		                ? (source.null_image && source.view_generation != 0 && NullReuseEnabled()) ||
		                      texture_cache.IsTextureCurrent(binding.image_id, source.view_generation)
		                : binding.desc.type == TextureCache::BindingType::Storage &&
		                      StorageReuseEnabled() &&
		                      texture_cache.IsStorageCurrent(binding.image_id,
		                                                     source.view_generation))) {
			// A clean image keeps the view FindTexture returned for this binding before.
			binding.image_view = source.view;
			if (binding.desc.type == TextureCache::BindingType::Storage ? VerifyStorageReuse()
			                                                              : VerifyViewReuse())
			    [[unlikely]] {
				CheckViewReuse(texture_cache, binding);
			}
		} else {
			const auto generation = texture_cache.ImageGeneration(binding.image_id);
			binding.image_view    = texture_cache.FindTexture(binding.image_id, binding.desc);
			source.view_generation =
			    binding.desc.type == TextureCache::BindingType::Texture ||
			            binding.desc.type == TextureCache::BindingType::Storage
			        ? generation
			        : 0;
			source.view       = binding.image_view;
			source.view_image = binding.image_id;
			source.view_info  = binding.desc.view_info;
			if (source.view_memo != nullptr && source.view_generation != 0) {
				*source.view_memo = {source.view, source.view_image, source.view_info,
				                     source.view_generation};
			}
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
	}
	RecordImageGroup(m_context, prepared, program, snapshot.images);
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool uses_dma = false;
	for (auto* stage: stages) {
		FindBuffers(*stage);
		uses_dma |= stage->runtime->program->info.uses_dma;
	}
	if (uses_dma) {
		DrawPhaseTimer::ProbeScope probe(g_draw_phases, DrawPhaseTimer::Bda);
		m_context.PrepareBda();
		for (auto* stage: stages) {
			if (stage->runtime->program->info.uses_dma) {
				NoteBdaShader(stage->runtime->program->shader_hash);
			}
		}
	}
	g_draw_phases.Mark(DrawPhaseTimer::FindBuffers);
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	g_draw_phases.Mark(DrawPhaseTimer::RebindImages);
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

// KYTY_DRAW_MEMO: a recorded stage's images stand for this draw as a kept image group stands for
// PrepareBindings and RebindImages (the key guarantees the same program and image descriptors).
bool RenderExecutor::ReuseMemoImages(DrawMemoStage& stage, const ShaderStageRuntime& runtime) {
	auto& prepared = stage.bindings;
	return prepared.images.empty() ||
	       (prepared.image_group.program == runtime.program &&
	        ImageGroupStandsOrRevalidates(m_context.GetTextureCache(), prepared));
}

// KYTY_DRAW_MEMO: what PrepareBindings and PrepareGraphicsBindings do for a stage whose images
// stand (ReuseMemoImages): binds the images, takes the samplers and shader data as recorded (the
// same snapshot gives the same ones), keeps each buffer binding the cached buffer still serves
// unchanged and obtains the others anew (NativeStorageBuffer, as RebindBuffers), and uploads the
// flattened SRT and shader data for this draw.
RenderExecutor::DrawMemoBufferCounts RenderExecutor::ApplyMemoStage(DrawMemoStage&            stage,
                                                                    const ShaderStageRuntime& runtime) {
	auto&       prepared = stage.bindings;
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	const auto& layout   = program.bindings;
	prepared.runtime     = &runtime;
	for (const auto counter: prepared.image_group.mip_counters) {
		m_context.MarkMipStatsCounter(counter & 0xffu, (counter & 0x100u) != 0);
	}
	for (const auto& binding: prepared.images) {
		BindImage(binding.image_id, false);
	}
	g_draw_phases.Mark(DrawPhaseTimer::StageTextures);
	DrawMemoBufferCounts counts;
	auto&                cache = m_context.GetBufferCache();
	prepared.buffers.assign(stage.buffers.begin(), stage.buffers.end());
	prepared.shader_data.assign(stage.shader_data.begin(), stage.shader_data.end());
	// This draw's user data words, as PrepareBindings writes them before RebindBuffers fills in
	// the words from memory_offset_dword on (the memo's key leaves user data out).
	const auto& user_registers = layout.user_data_registers;
	const auto  user_words =
	    std::min<size_t>(user_registers.size(), layout.memory_offset_dword);
	for (size_t r = 0; r < user_words; r++) {
		prepared.shader_data[r] = snapshot.user_data[user_registers[r] - program.user_data_base];
	}
	auto&      textures       = m_context.GetTextureCache();
	// A changed memory map can clamp a range differently (FindBuffers): each range is clamped
	// again, and one that changed is obtained anew.
	const auto map_generation = Libs::LibKernel::Memory::RangeMapGeneration();
	const bool remapped       = map_generation != stage.map_generation;
	stage.map_generation      = map_generation;
	for (uint32_t i = 0; i < stage.buffer_kinds.size(); i++) {
		auto&      source = prepared.buffer_sources[i];
		const auto kind   = stage.buffer_kinds[i];
		const bool moved  = !(stage.buffer_values[i] == snapshot.buffers[i]) ||
		                   (remapped && source.size != 0 &&
		                    Libs::LibKernel::Memory::ClampRangeSize(
		                        source.address,
		                        DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i])
		                            .GetSize()) != source.size);
		if (!moved) {
			const bool same_images = kind != 3 || textures.RangeGeneration({source.address, source.size}) ==
			                                          stage.buffer_generations[i];
			if (kind == 2 || (kind != 0 && same_images &&
			                  cache.KeepReadBinding(source.id, source.address, source.size))) {
				continue;
			}
			counts.fresh += kind == 0 ? 1u : 0u;
			counts.images += kind != 0 && !same_images ? 1u : 0u;
			counts.dirty += kind != 0 && same_images ? 1u : 0u;
		}
		const auto& resource         = program.info.buffers[i];
		auto        runtime_resource = resource;
		if (resource.descriptor_format == ShaderRecompiler::IR::GenericBufferFormat) {
			runtime_resource.descriptor_format = RuntimeBufferFormat(snapshot, i);
		}
		// As FindBuffers, from this draw's descriptor: it may have moved, and the mapping may have
		// changed since (an unmap leaves the range CPU-dirty).
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
		const auto address    = descriptor.Base48();
		const auto requested  = descriptor.GetSize();
		if (address == 0 || requested == 0) {
			source = {};
		} else {
			source.address = address;
			source.size    = Libs::LibKernel::Memory::ClampRangeSize(address, requested);
			source.id      = cache.RefindBuffer(source.id, source.address, source.size);
		}
		uint32_t   offset      = 0;
		const auto info        = NativeStorageBuffer(m_context, source, runtime_resource, offset);
		const auto dword       = layout.memory_offset_dword + i / 4u;
		const auto shift       = (i % 4u) * 8u;
		const auto byte_mask   = ~(0xffu << shift);
		prepared.buffers[i]    = info;
		prepared.shader_data[dword] = (prepared.shader_data[dword] & byte_mask) | (offset << shift);
		counts.obtained++;
		counts.texel += resource.formatted ? 1u : 0u;
		counts.moved += moved ? 1u : 0u;
		// The binding as obtained now is what the next draw of this entry starts from.
		stage.buffer_values[i]   = snapshot.buffers[i];
		stage.buffer_kinds[i]    = MemoBufferKind(resource, source, info, stage.buffer_generations[i]);
		stage.buffers[i]         = info;
		stage.shader_data[dword] = (stage.shader_data[dword] & byte_mask) | (offset << shift);
	}
	g_draw_phases.Mark(DrawPhaseTimer::BufferViews);
	prepared.flattened_srt      = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory      = {};
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::FlattenedSrt) != nullptr) {
		prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
	}
	if (ShaderRecompiler::IR::FindBinding(
	        layout, ShaderRecompiler::IR::DescriptorBindingKind::ShaderData) != nullptr) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
	return counts;
}

// KYTY_DRAW_MEMO: keeps a stage's bindings as the normal path left them, unless the stage needs
// what the memo does not repeat (table blocks, GDS, bindless heaps, images outside a kept group,
// per-mip views, written buffers).
bool RenderExecutor::RecordMemoStage(DrawMemoStage& stage, const PreparedBindings& prepared) {
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program = *prepared.runtime->program;
	const auto  count   = program.info.buffers.size();
	if (program.bindings.table || prepared.gds.buffer != nullptr ||
	    !prepared.bindless_patches.empty() || !prepared.bindless_images.empty() ||
	    (!prepared.images.empty() && prepared.image_group.program != &program) ||
	    prepared.buffers.size() != count || prepared.buffer_sources.size() != count ||
	    prepared.shader_data.size() != program.bindings.ShaderDataDwords()) {
		return false;
	}
	for (const auto& image: prepared.images) {
		if (!image.mip_views.empty()) {
			return false;
		}
	}
	const auto& snapshot = *prepared.runtime->resources;
	if (snapshot.buffers.size() != count) {
		return false;
	}
	stage.buffer_kinds.assign(count, 0);
	stage.buffer_generations.assign(count, 0);
	for (size_t i = 0; i < count; i++) {
		const auto& resource = program.info.buffers[i];
		if (resource.written) {
			return false;
		}
		stage.buffer_kinds[i] = MemoBufferKind(resource, prepared.buffer_sources[i],
		                                       prepared.buffers[i], stage.buffer_generations[i]);
	}
	stage.bindings         = prepared;
	stage.bindings.runtime = nullptr;
	stage.buffers          = prepared.buffers;
	stage.shader_data      = prepared.shader_data;
	stage.buffer_values    = snapshot.buffers;
	// The map may have changed since FindBuffers clamped the ranges: the first draw that takes
	// the entry clamps them again (no generation is 0).
	stage.map_generation   = 0;
	return true;
}

// KYTY_DRAW_MEMO: how a read binding just obtained can be kept (see DrawMemoStage::buffer_kinds).
uint8_t RenderExecutor::MemoBufferKind(const ShaderRecompiler::IR::BufferResource& resource,
                                       const PreparedBindings::BufferSource&       source,
                                       const vk::DescriptorBufferInfo& info, uint64_t& generation) {
	if (source.address == 0 || source.size == 0 || info.buffer == nullptr) {
		return 2;
	}
	if (!m_context.GetBufferCache().IsBufferHandle(source.id, info.buffer)) {
		return 0; // A stream copy, or the null buffer for an unsupported offset.
	}
	if (!resource.formatted) {
		return 1;
	}
	// A texel buffer: ObtainBuffer also copies an image's data into it when an image starts at
	// its address (SynchronizeBufferFromImage), so it is kept only where none does.
	auto& textures = m_context.GetTextureCache();
	generation     = textures.RangeGeneration({source.address, source.size});
	return textures.HasImageStartingAt(source.address, source.size) ? 0 : 3;
}

namespace {

// A draw's push constants and push descriptor sets as CommitBindings hands them to the thread
// recording its command buffer (see ReserveRecordedCall): this header, then the writes of set 0
// and set 1, then each write's image or buffer infos in turn.
struct RecordedBindings {
	VkDevice                       device      = VK_NULL_HANDLE;
	VkPipelineLayout               layout      = VK_NULL_HANDLE;
	VkPipelineBindPoint            bind_point  = VK_PIPELINE_BIND_POINT_GRAPHICS;
	VkShaderStageFlags             push_stages = 0; // 0: no push constants.
	std::array<uint32_t, 2>        set_writes {};
	// A set that does not push: allocated by CommitBindings, updated and bound by the recording
	// thread. VK_NULL_HANDLE: the set pushes.
	std::array<VkDescriptorSet, 2> sets {};
	ShaderRecompiler::IR::PushData push_data;
	// HandleRegistry::CheckEnabled: one stamp per info follows the infos (0 for image infos), and
	// the recording thread skips the call when a buffer was destroyed since (a dead VkBuffer in a
	// pushed descriptor loses the device).
	uint64_t                       stamped = 0;
};

struct RecordedWrite {
	uint32_t         binding = 0;
	uint32_t         count   = 0;
	VkDescriptorType type    = VK_DESCRIPTOR_TYPE_SAMPLER;
	uint32_t         images  = 0; // Whether the infos are image infos rather than buffer infos.
};

static_assert(sizeof(vk::DescriptorImageInfo) == sizeof(vk::DescriptorBufferInfo));
static_assert(sizeof(RecordedBindings) % alignof(vk::DescriptorImageInfo) == 0 &&
              sizeof(RecordedWrite) % alignof(vk::DescriptorImageInfo) == 0);
constexpr size_t RecordedInfoBytes = sizeof(vk::DescriptorImageInfo);

// The stamps of a payload's buffer infos (see RecordedBindings::stamped), one per info after them.
void StampRecordedInfos(uint64_t* stamps, const RecordedWrite* writes, size_t write_count,
                        const uint8_t* infos) {
	for (size_t i = 0; i < write_count; i++) {
		for (uint32_t e = 0; e < writes[i].count; e++, infos += RecordedInfoBytes) {
			uint64_t stamp = 0;
			if (writes[i].images == 0) {
				vk::DescriptorBufferInfo info;
				std::memcpy(&info, infos, sizeof(info));
				stamp = HandleRegistry::Stamp(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(info.buffer)));
			}
			*stamps++ = stamp;
		}
	}
}

// Whether every stamped buffer is the one the payload was filled with (logs the stale ones).
bool RecordedInfosCurrent(const uint64_t* stamps, const RecordedWrite* writes, size_t write_count,
                          const uint8_t* infos) {
	bool current = true;
	for (size_t i = 0; i < write_count; i++) {
		for (uint32_t e = 0; e < writes[i].count; e++, infos += RecordedInfoBytes, stamps++) {
			if (writes[i].images != 0 || *stamps == 0) {
				continue;
			}
			vk::DescriptorBufferInfo info;
			std::memcpy(&info, infos, sizeof(info));
			current = HandleRegistry::Same(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(info.buffer)), *stamps,
			                               "pushed bindings", "storage buffer") &&
			          current;
		}
	}
	return current;
}

// On the recording thread: the calls CommitBindings would have made.
void RunRecordedBindings(VkCommandBuffer command_buffer, const uint8_t* payload) {
	const auto& header = *reinterpret_cast<const RecordedBindings*>(payload);
	const auto* writes = reinterpret_cast<const RecordedWrite*>(payload + sizeof(RecordedBindings));
	const auto  count  = header.set_writes[0] + header.set_writes[1];
	const auto* infos  = payload + sizeof(RecordedBindings) + count * sizeof(RecordedWrite);
	if (header.stamped != 0) {
		size_t info_count = 0;
		for (uint32_t i = 0; i < count; i++) {
			info_count += writes[i].count;
		}
		const auto* stamps = reinterpret_cast<const uint64_t*>(infos + info_count * RecordedInfoBytes);
		if (!RecordedInfosCurrent(stamps, writes, count, infos)) {
			// A dead buffer would lose the device: skip the draw's bindings (it draws with the
			// previous ones) and keep running, so the stale report is the result.
			return;
		}
	}
	thread_local std::vector<vk::WriteDescriptorSet> vk_writes;
	vk_writes.resize(count);
	for (uint32_t i = 0; i < count; i++) {
		const auto& write     = writes[i];
		auto&       vk_write  = vk_writes[i];
		vk_write              = vk::WriteDescriptorSet {};
		vk_write.dstBinding      = write.binding;
		vk_write.descriptorCount = write.count;
		vk_write.descriptorType  = static_cast<vk::DescriptorType>(write.type);
		if (write.images != 0) {
			vk_write.pImageInfo = reinterpret_cast<const vk::DescriptorImageInfo*>(infos);
		} else {
			vk_write.pBufferInfo = reinterpret_cast<const vk::DescriptorBufferInfo*>(infos);
		}
		infos += write.count * RecordedInfoBytes;
	}
	const vk::CommandBuffer  buffer(command_buffer);
	const vk::PipelineLayout layout(header.layout);
	const auto               bind_point = static_cast<vk::PipelineBindPoint>(header.bind_point);
	if (header.push_stages != 0) {
		buffer.pushConstants(layout, static_cast<vk::ShaderStageFlags>(header.push_stages), 0,
		                     sizeof(header.push_data), header.push_data.dwords.data());
	}
	auto* set_writes = vk_writes.data();
	for (uint32_t set_index = 0; set_index < 2; set_index++) {
		const auto count_in_set = header.set_writes[set_index];
		if (count_in_set != 0) {
			const vk::DescriptorSet set(header.sets[set_index]);
			if (!set) {
				buffer.pushDescriptorSetKHR(bind_point, layout, set_index, count_in_set, set_writes);
			} else {
				// A freshly allocated set, bound nowhere yet: updating it here, before its bind,
				// is what CommitBindings did on its own thread.
				for (uint32_t i = 0; i < count_in_set; i++) {
					set_writes[i].dstSet = set;
				}
				vk::Device(header.device).updateDescriptorSets(count_in_set, set_writes, 0, nullptr);
				buffer.bindDescriptorSets(bind_point, layout, set_index, 1, &set, 0, nullptr);
			}
		}
		set_writes += count_in_set;
	}
}

// Hands the push constants (with nonempty stages) and both descriptor sets to the thread that
// records `buffer`, as one payload call instead of one deep-copied call each; false when its
// recording is not deferred, and the caller makes the calls. A set that does not push
// (push[set] false) gets its set from allocate(set) here, once the call is certain, and the
// recording thread updates and binds it. KYTY_DEBUG_AB=pushpayload has the caller make the
// calls in every other window.
template <typename Allocate>
bool RecordPushedBindings(vk::CommandBuffer buffer, vk::Device device,
                          vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                          vk::ShaderStageFlags push_stages,
                          const ShaderRecompiler::IR::PushData&    push_data,
                          std::span<const vk::WriteDescriptorSet> set0,
                          std::span<const vk::WriteDescriptorSet> set1, std::array<bool, 2> push,
                          Allocate&& allocate) {
	static const bool ab = AbSelected("pushpayload");
	if (ab && AbFeatureOff()) {
		return false;
	}
	size_t info_count = 0;
	for (const auto* set: {&set0, &set1}) {
		for (const auto& write: *set) {
			// Each write carries its infos in one of the two arrays; leave anything else to the
			// calls as they were.
			if ((write.pImageInfo == nullptr) == (write.pBufferInfo == nullptr)) {
				return false;
			}
			info_count += write.descriptorCount;
		}
	}
	const auto write_count = set0.size() + set1.size();
	if (write_count == 0 && !push_stages) {
		return true;
	}
	auto* payload =
	    ReserveRecordedCall(static_cast<VkCommandBuffer>(buffer),
	                        sizeof(RecordedBindings) + write_count * sizeof(RecordedWrite) +
	                            info_count * RecordedInfoBytes +
	                            (HandleRegistry::CheckEnabled() ? info_count * sizeof(uint64_t) : 0));
	if (payload == nullptr) {
		return false;
	}
	auto& header       = *::new (payload) RecordedBindings;
	header.device      = static_cast<VkDevice>(device);
	header.layout      = static_cast<VkPipelineLayout>(layout);
	header.bind_point  = static_cast<VkPipelineBindPoint>(bind_point);
	header.push_stages = static_cast<VkShaderStageFlags>(push_stages);
	header.set_writes  = {static_cast<uint32_t>(set0.size()), static_cast<uint32_t>(set1.size())};
	for (uint32_t set_index = 0; set_index < 2; set_index++) {
		if (header.set_writes[set_index] != 0 && !push[set_index]) {
			header.sets[set_index] = static_cast<VkDescriptorSet>(allocate(set_index));
		}
	}
	header.push_data   = push_data;
	header.stamped     = HandleRegistry::CheckEnabled() ? 1u : 0u;
	auto* writes = reinterpret_cast<RecordedWrite*>(payload + sizeof(RecordedBindings));
	auto* infos  = payload + sizeof(RecordedBindings) + write_count * sizeof(RecordedWrite);
	for (const auto* set: {&set0, &set1}) {
		for (const auto& write: *set) {
			const bool images = write.pImageInfo != nullptr;
			*writes++         = {write.dstBinding, write.descriptorCount,
			                     static_cast<VkDescriptorType>(write.descriptorType), images ? 1u : 0u};
			const size_t bytes = write.descriptorCount * RecordedInfoBytes;
			std::memcpy(infos,
			            images ? static_cast<const void*>(write.pImageInfo)
			                   : static_cast<const void*>(write.pBufferInfo),
			            bytes);
			infos += bytes;
		}
	}
	if (header.stamped != 0) {
		const auto* all_writes = reinterpret_cast<const RecordedWrite*>(payload + sizeof(RecordedBindings));
		const auto* all_infos  = payload + sizeof(RecordedBindings) + write_count * sizeof(RecordedWrite);
		StampRecordedInfos(reinterpret_cast<uint64_t*>(const_cast<uint8_t*>(all_infos) + info_count * RecordedInfoBytes),
		                   all_writes, write_count, all_infos);
	}
	CommitRecordedCall(RunRecordedBindings);
	return true;
}

// KYTY_BINDING_PLAN=0 commits without plans; KYTY_DEBUG_AB=bindplan alternates.
bool BindingPlanEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_BINDING_PLAN");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	static const bool ab = AbSelected("bindplan");
	return enabled && !(ab && AbFeatureOff());
}

// What CommitBindings derives from a program's binding layout alone, made once per program: the
// payload's writes as RecordPushedBindings stores them, and where each of their infos comes from.
// A draw then only fills in the infos (RecordPlannedBindings).
struct BindingPlan {
	enum class Source : uint8_t {
		Image,        // MakeImageInfo(images[resource], element).
		Sampler,      // samplers[resource].
		Buffer,       // buffers[resource].
		BdaPagetable, // The buffer cache's BDA page table.
		FaultBuffer,  // The buffer cache's fault buffer.
		FlattenedSrt,
		ShaderData,
		Gds,
	};
	struct Info {
		Source   source   = Source::Image;
		uint32_t resource = 0;
		uint32_t element  = 0;
	};
	size_t                     descriptor_bindings = 0; // The layout it was made for.
	// False: a write the payload cannot carry (no infos); commit as before.
	bool                       usable              = true;
	uint32_t                   set                 = 0;
	vk::ShaderStageFlags       shader_stage;
	std::vector<RecordedWrite> writes;
	std::vector<Info>          infos;
};

const BindingPlan& MakeBindingPlan(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	auto  plan  = std::make_shared<BindingPlan>();
	auto& made  = *plan;
	made.descriptor_bindings = program.bindings.descriptors.size();
	made.set                 = ShaderRecompiler::IR::NativeDescriptorSet(program.stage);
	made.shader_stage        = NativeShaderStage(program.stage);
	std::vector<uint32_t> occurrences(program.info.images.size(), 0);
	for (const auto& binding: program.bindings.descriptors) {
		const auto type  = NativeDescriptorType(binding.kind);
		const auto count = NativeDescriptorCount(binding);
		const auto first = made.infos.size();
		bool       image_infos = true;
		using Source           = BindingPlan::Source;
		if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
		    ShaderRecompiler::IR::ImageResourceClass::None) {
			for (const auto resource: binding.resources) {
				EXIT_IF(resource >= occurrences.size());
				made.infos.push_back({Source::Image, resource, occurrences[resource]++});
			}
		} else {
			image_infos = binding.kind == BindingKind::Samplers;
			switch (binding.kind) {
				case BindingKind::Buffers:
					for (const auto resource: binding.resources) {
						EXIT_IF(resource >= program.info.buffers.size());
						made.infos.push_back({Source::Buffer, resource, 0});
					}
					break;
				case BindingKind::Samplers:
					for (const auto resource: binding.resources) {
						EXIT_IF(resource >= program.info.samplers.size());
						made.infos.push_back({Source::Sampler, resource, 0});
					}
					break;
				case BindingKind::BdaPagetable: made.infos.push_back({Source::BdaPagetable}); break;
				case BindingKind::FaultBuffer: made.infos.push_back({Source::FaultBuffer}); break;
				case BindingKind::FlattenedSrt: made.infos.push_back({Source::FlattenedSrt}); break;
				case BindingKind::ShaderData: made.infos.push_back({Source::ShaderData}); break;
				case BindingKind::Gds: made.infos.push_back({Source::Gds}); break;
				case BindingKind::Count: EXIT("invalid descriptor binding kind");
			}
		}
		// CommitBindings' writes carry exactly `count` infos, or none (then RecordPushedBindings
		// leaves them to direct calls).
		made.usable = made.usable && made.infos.size() - first == count;
		made.writes.push_back({ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind),
		                       count, static_cast<VkDescriptorType>(type), image_infos ? 1u : 0u});
	}
	// Each image is read once, or once per mip for dynamic storage mips (see RebindImages).
	for (uint32_t i = 0; i < occurrences.size(); i++) {
		const auto& resource = program.info.images[i];
		const auto  expected =
		    resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic
		         ? resource.mip_count
		         : 1u;
		made.usable = made.usable && occurrences[i] == expected;
	}
	program.binding_plan = std::move(plan);
	return made;
}

const BindingPlan& GetBindingPlan(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	const auto* plan = static_cast<const BindingPlan*>(program.binding_plan.get());
	if (plan != nullptr && plan->descriptor_bindings == program.bindings.descriptors.size())
	    [[likely]] {
		return *plan;
	}
	return MakeBindingPlan(program);
}

// RecordPushedBindings for planned stages: the payload's writes are copied from the plans, and
// the infos are written into it directly from the stages' prepared bindings. False as there.
template <typename Allocate>
bool RecordPlannedBindings(vk::CommandBuffer buffer, vk::Device device,
                           vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                           vk::ShaderStageFlags                       push_stages,
                           const ShaderRecompiler::IR::PushData&      push_data,
                           std::span<const BindingPlan* const>        plans,
                           std::span<PreparedBindings* const>         stages,
                           BufferCache& cache, bool hardware_bounds, std::array<bool, 2> push,
                           Allocate&& allocate) {
	static const bool ab = AbSelected("pushpayload");
	if (ab && AbFeatureOff()) {
		return false;
	}
	std::array<uint32_t, 2> set_writes {};
	size_t                  info_count = 0;
	for (const auto* plan: plans) {
		set_writes[plan->set] += static_cast<uint32_t>(plan->writes.size());
		info_count += plan->infos.size();
	}
	const size_t write_count = set_writes[0] + set_writes[1];
	if (write_count == 0 && !push_stages) {
		return true;
	}
	auto* payload =
	    ReserveRecordedCall(static_cast<VkCommandBuffer>(buffer),
	                        sizeof(RecordedBindings) + write_count * sizeof(RecordedWrite) +
	                            info_count * RecordedInfoBytes +
	                            (HandleRegistry::CheckEnabled() ? info_count * sizeof(uint64_t) : 0));
	if (payload == nullptr) {
		return false;
	}
	auto& header       = *::new (payload) RecordedBindings;
	header.device      = static_cast<VkDevice>(device);
	header.layout      = static_cast<VkPipelineLayout>(layout);
	header.bind_point  = static_cast<VkPipelineBindPoint>(bind_point);
	header.push_stages = static_cast<VkShaderStageFlags>(push_stages);
	header.set_writes  = set_writes;
	for (uint32_t set_index = 0; set_index < 2; set_index++) {
		if (header.set_writes[set_index] != 0 && !push[set_index]) {
			header.sets[set_index] = static_cast<VkDescriptorSet>(allocate(set_index));
		}
	}
	header.push_data = push_data;
	header.stamped     = HandleRegistry::CheckEnabled() ? 1u : 0u;
	auto* writes = payload + sizeof(RecordedBindings);
	auto* infos  = writes + write_count * sizeof(RecordedWrite);
	for (size_t s = 0; s < plans.size(); s++) {
		const auto& plan  = *plans[s];
		const auto& stage = *stages[s];
		const auto  bytes = plan.writes.size() * sizeof(RecordedWrite);
		std::memcpy(writes, plan.writes.data(), bytes);
		writes += bytes;
		for (const auto& info: plan.infos) {
			using Source = BindingPlan::Source;
			switch (info.source) {
				case Source::Image: {
					const auto image_info = MakeImageInfo(stage.images[info.resource], info.element);
					std::memcpy(infos, &image_info, sizeof(image_info));
					break;
				}
				case Source::Sampler: {
					const auto sampler = stage.samplers[info.resource];
					EXIT_IF(sampler == nullptr);
					const vk::DescriptorImageInfo image_info {sampler, nullptr,
					                                          vk::ImageLayout::eUndefined};
					std::memcpy(infos, &image_info, sizeof(image_info));
					break;
				}
				case Source::Buffer: {
					const auto& view = stage.buffers[info.resource];
					// With hardware bounds, a range without a whole dword is a null descriptor.
					EXIT_IF(view.buffer == nullptr && !hardware_bounds);
					std::memcpy(infos, &view, sizeof(view));
					break;
				}
				case Source::BdaPagetable:
				case Source::FaultBuffer: {
					const auto* bda_buffer = info.source == Source::BdaPagetable
					                             ? cache.GetBdaPageTableBuffer()
					                             : cache.GetFaultBuffer();
					const vk::DescriptorBufferInfo view {bda_buffer->Handle(), 0, bda_buffer->Size()};
					std::memcpy(infos, &view, sizeof(view));
					break;
				}
				case Source::FlattenedSrt:
				case Source::ShaderData:
				case Source::Gds: {
					const auto& view = info.source == Source::FlattenedSrt ? stage.flattened_srt
					                   : info.source == Source::ShaderData ? stage.shader_data_buffer
					                                                       : stage.gds;
					EXIT_IF(view.buffer == nullptr);
					std::memcpy(infos, &view, sizeof(view));
					break;
				}
			}
			infos += RecordedInfoBytes;
		}
	}
	if (g_draw_phases.Active() && BindStatsEnabled()) [[unlikely]] {
		// Each set's writes and infos, compared with the previous payload's for the set.
		thread_local std::array<std::vector<uint8_t>, 2> last_bytes;
		thread_local std::array<VkPipelineLayout, 2>      last_layout {};
		const auto* set_writes_start = reinterpret_cast<const RecordedWrite*>(
		    payload + sizeof(RecordedBindings));
		const auto* set_infos_start =
		    payload + sizeof(RecordedBindings) + write_count * sizeof(RecordedWrite);
		for (uint32_t set_index = 0; set_index < 2; set_index++) {
			const auto count     = header.set_writes[set_index];
			uint32_t   set_infos = 0;
			for (uint32_t i = 0; i < count; i++) {
				set_infos += set_writes_start[i].count;
			}
			if (count != 0) {
				std::vector<uint8_t> bytes(count * sizeof(RecordedWrite) +
				                           set_infos * RecordedInfoBytes);
				std::memcpy(bytes.data(), set_writes_start, count * sizeof(RecordedWrite));
				std::memcpy(bytes.data() + count * sizeof(RecordedWrite), set_infos_start,
				            set_infos * RecordedInfoBytes);
				g_bind_stats.set_payloads[set_index]++;
				g_bind_stats.set_repeats[set_index] +=
				    last_layout[set_index] == header.layout && last_bytes[set_index] == bytes ? 1
				                                                                                : 0;
				last_bytes[set_index]  = std::move(bytes);
				last_layout[set_index] = header.layout;
			}
			set_writes_start += count;
			set_infos_start += set_infos * RecordedInfoBytes;
		}
	}
	if (header.stamped != 0) {
		const auto* all_writes = reinterpret_cast<const RecordedWrite*>(payload + sizeof(RecordedBindings));
		const auto* all_infos  = payload + sizeof(RecordedBindings) + write_count * sizeof(RecordedWrite);
		StampRecordedInfos(reinterpret_cast<uint64_t*>(const_cast<uint8_t*>(all_infos) + info_count * RecordedInfoBytes),
		                   all_writes, write_count, all_infos);
	}
	CommitRecordedCall(RunRecordedBindings);
	return true;
}

} // namespace

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings,
                                    bool                               compute_chain) {
	KYTY_PROFILER_FUNCTION();
	uint64_t lap              = g_draw_phases.Lap();
	// Binding commands do not touch memory, so they leave a pending compute chain barrier pending.
	auto     vk_buffer        = compute_chain ? buffer.ChainHandle() : buffer.Handle();
	size_t descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	// With every stage planned, the payload is filled from the plans (RecordPlannedBindings).
	std::array<const BindingPlan*, 4> plans {};
	bool planned = BindingPlanEnabled() && prepared_bindings.size() <= plans.size();
	for (size_t s = 0; s < prepared_bindings.size(); s++) {
		const auto* prepared = prepared_bindings[s];
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		vk::ShaderStageFlags shader_stage;
		if (planned) {
			const auto& plan = GetBindingPlan(program);
			plans[s]         = &plan;
			shader_stage     = plan.shader_stage;
			// Pixel last (set 1 is the payload's tail), and the arrays the infos index sized for
			// the program.
			planned = plan.usable && (plan.set == 0 || s + 1 == prepared_bindings.size()) &&
			          prepared->images.size() == program.info.images.size() &&
			          prepared->samplers.size() == program.info.samplers.size() &&
			          (program.bindings.table ||
			           prepared->buffers.size() == program.info.buffers.size());
		} else {
			shader_stage = NativeShaderStage(program.stage);
		}
		write_count += program.bindings.descriptors.size();
		if (planned) {
			descriptor_count += plans[s]->infos.size();
		} else {
			for (const auto& binding: program.bindings.descriptors) {
				descriptor_count += NativeDescriptorCount(binding);
			}
		}
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	for (const auto* reader: prepared_bindings) {
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty()) continue;
		for (const auto* writer: prepared_bindings) {
			if (writer->runtime->program->has_address_writes) {
				// Stores through GPU-selected descriptors (BDA) are not tracked; their targets are
				// assumed not to hold this stage's descriptors.
				static std::atomic<uint32_t> reported {0};
				if (reported.fetch_add(1, std::memory_order_relaxed) < 4u) {
					std::printf("warning: scalar resource reads not proven disjoint from shader "
					            "address writes (shader %016llx)\n",
					            static_cast<unsigned long long>(writer->runtime->program->shader_hash));
				}
			}
		}
		for (const auto [address, size]: reads) {
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0 && ImageRangeOverlaps(address, size,
					                                          written.address, written.size)) {
						EXIT("scalar resource reads overlap an image or attachment write\n");
					}
				}
			}
			for (const auto* writer: prepared_bindings) {
				const auto& program = *writer->runtime->program;
				for (uint32_t i = 0; i < writer->buffer_sources.size(); ++i) {
					const auto resource = program.bindings.descriptors.front().resources[i];
					if (!program.info.buffers[resource].written) continue;
					const auto& written = writer->buffer_sources[i];
					if (written.size != 0 && ImageRangeOverlaps(address, size,
					                                          written.address, written.size)) {
						EXIT("scalar resource reads overlap a shader buffer write\n");
					}
				}
			}
		}
	}
	m_descriptor_buffers.clear();
	m_descriptor_images.clear();
	m_descriptor_writes.clear();
	m_descriptor_buffers.reserve(descriptor_count);
	m_descriptor_images.reserve(descriptor_count);
	m_descriptor_writes.reserve(write_count);
	size_t pixel_write_start = write_count;
	g_draw_phases.Lap(DrawPhaseTimer::CommitScan, lap);

	// A stage's descriptor writes and their infos, for the payload or the calls.
	const auto append_writes = [&](PreparedBindings& descriptors) {
		const auto& program = *descriptors.runtime->program;
		// The pixel stage comes last, so its writes form the tail committed to set 1.
		EXIT_IF(pixel_write_start != write_count);
		if (ShaderRecompiler::IR::NativeDescriptorSet(program.stage) != 0) {
			pixel_write_start = m_descriptor_writes.size();
		}
		m_image_occurrences.assign(descriptors.images.size(), 0);
		for (const auto& binding: program.bindings.descriptors) {
			vk::WriteDescriptorSet write {};
			write.dstBinding     = ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind);
			write.descriptorType = NativeDescriptorType(binding.kind);
			write.descriptorCount   = NativeDescriptorCount(binding);
			const auto buffer_start = m_descriptor_buffers.size();
			const auto image_start  = m_descriptor_images.size();
			if (ShaderRecompiler::IR::ImageBindingResourceClass(binding.kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				for (const auto resource: binding.resources) {
					m_descriptor_images.push_back(MakeImageInfo(
					    descriptors.images.at(resource), m_image_occurrences.at(resource)++));
				}
			} else {
				switch (binding.kind) {
					case BindingKind::Buffers:
						for (const auto resource: binding.resources) {
							const auto& view = descriptors.buffers.at(resource);
							// With hardware bounds, a range without a whole dword is a null
							// descriptor (see NativeStorageBuffer).
							EXIT_IF(view.buffer == nullptr &&
							        !m_context.GetGraphics().hardware_storage_buffer_bounds);
							m_descriptor_buffers.push_back(view);
						}
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto&       cache      = m_context.GetBufferCache();
						const auto* bda_buffer = binding.kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						m_descriptor_buffers.emplace_back(bda_buffer->Handle(), 0,
						                                  bda_buffer->Size());
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (binding.kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (binding.kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (binding.kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						m_descriptor_buffers.push_back(*view);
						break;
					}
					case BindingKind::Samplers:
						for (const auto resource: binding.resources) {
							const auto sampler = descriptors.samplers.at(resource);
							EXIT_IF(sampler == nullptr);
							m_descriptor_images.emplace_back(sampler, nullptr,
							                                 vk::ImageLayout::eUndefined);
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (m_descriptor_buffers.size() != buffer_start) {
				write.pBufferInfo = m_descriptor_buffers.data() + buffer_start;
			}
			if (m_descriptor_images.size() != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < descriptors.images.size(); i++) {
			const auto expected =
			    descriptors.images[i].mip_views.empty()
			        ? 1u
			        : static_cast<uint32_t>(descriptors.images[i].mip_views.size());
			EXIT_IF(m_image_occurrences[i] != expected);
		}
	};

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		// Bindless textures are sampled through set 1 with a read-only layout.
		for (const auto id: descriptors.bindless_images) {
			auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
			if (image == nullptr || !image->registered || image->binding.is_target ||
			    image->info.data.Empty()) {
				continue;
			}
			const auto layout = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
			                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
			if (image->backing.state.layout != layout) {
				image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
			}
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& image   = m_context.GetTextureCache().GetImage(descriptors.images[i].image_id);
			auto& binding = descriptors.images[i];
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
				              vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}
		g_draw_phases.Lap(DrawPhaseTimer::CommitTransit, lap);

		if (!planned) {
			append_writes(descriptors);
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
		g_draw_phases.Lap(DrawPhaseTimer::CommitWrites, lap);
	}
	// Charges the rest (the payload or the calls) when CommitBindings returns.
	struct PayloadLap {
		uint64_t& lap;
		~PayloadLap() { g_draw_phases.Lap(DrawPhaseTimer::CommitPayload, lap); }
	} payload_lap {lap};

	if (has_push_data && pipeline.push_constant_stages) {
		// The stages must match the layout's push constant range exactly.
		push_stages = pipeline.push_constant_stages;
	}
	if (pipeline.uses_bindless) {
		// The shared bindless table; the pipeline's own sets 0 and 1 follow below.
		const auto set = m_context.GetGraphics().bindless_set;
		EXIT_IF(set == nullptr);
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout,
		                             ShaderRecompiler::IR::BindlessDescriptorSet, 1, &set, 0, nullptr);
	}
	if (planned) {
		std::array<bool, 2> sets_used {};
		for (size_t s = 0; s < prepared_bindings.size(); s++) {
			sets_used[plans[s]->set] = sets_used[plans[s]->set] || !plans[s]->writes.empty();
		}
		EXIT_IF((sets_used[0] && pipeline.descriptor_set_layout == nullptr) ||
		        (sets_used[1] && pipeline.pixel_set_layout == nullptr));
		if (RecordPlannedBindings(
		        vk_buffer, m_context.GetGraphics().device, pipeline_bind_point,
		        pipeline.pipeline_layout, has_push_data ? push_stages : vk::ShaderStageFlags {},
		        push_data, std::span(plans.data(), prepared_bindings.size()), prepared_bindings,
		        m_context.GetBufferCache(), m_context.GetGraphics().hardware_storage_buffer_bounds,
		        {pipeline.uses_push_descriptors, pipeline.pixel_uses_push}, [&](uint32_t set_index) {
			        return m_context.GetDescriptorHeap().Commit(
			            set_index == 0 ? pipeline.descriptor_set_layout : pipeline.pixel_set_layout);
		        })) {
			return;
		}
		// Not recorded through a payload: make the writes for the calls below.
		for (auto* prepared: prepared_bindings) {
			append_writes(*prepared);
		}
	}
	// Set 0 holds compute or vertex-side descriptors; the pixel shader's tail goes to set 1.
	const auto set0 = std::span(m_descriptor_writes).first(pixel_write_start);
	const auto set1 = std::span(m_descriptor_writes).subspan(pixel_write_start);
	EXIT_IF((!set0.empty() && pipeline.descriptor_set_layout == nullptr) ||
	        (!set1.empty() && pipeline.pixel_set_layout == nullptr));
	// A graphics layout pushes at most one of its two sets, so the payload carries the other as
	// an allocated set (see RecordPushedBindings).
	if (RecordPushedBindings(
	        vk_buffer, m_context.GetGraphics().device, pipeline_bind_point, pipeline.pipeline_layout,
	        has_push_data ? push_stages : vk::ShaderStageFlags {}, push_data, set0, set1,
	        {pipeline.uses_push_descriptors, pipeline.pixel_uses_push}, [&](uint32_t set_index) {
		        return m_context.GetDescriptorHeap().Commit(
		            set_index == 0 ? pipeline.descriptor_set_layout : pipeline.pixel_set_layout);
	        })) {
		return;
	}
	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}

	const auto commit_set = [&](uint32_t set_index, vk::DescriptorSetLayout layout, bool push,
	                            std::span<vk::WriteDescriptorSet> writes) {
		if (writes.empty()) {
			return;
		}
		EXIT_IF(layout == nullptr);
		if (push) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, set_index,
			                               static_cast<uint32_t>(writes.size()), writes.data());
			return;
		}
		const auto set = m_context.GetDescriptorHeap().Commit(layout);
		for (auto& write: writes) {
			write.dstSet = set;
		}
		m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()),
		                                                    writes.data(), 0, nullptr);
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, set_index, 1,
		                             &set, 0, nullptr);
	};
	commit_set(0, pipeline.descriptor_set_layout, pipeline.uses_push_descriptors, set0);
	commit_set(1, pipeline.pixel_set_layout, pipeline.pixel_uses_push, set1);
}

} // namespace Libs::Graphics
