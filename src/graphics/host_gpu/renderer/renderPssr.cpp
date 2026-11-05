#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/performanceOverlay.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace Libs::Graphics {

// After upstream PR 1553 (339ff29db, PSSR1 for Astro Bot); here also PSSR2 (the title, 2026-10-29).

namespace {

enum class PssrRole : uint8_t { Unknown, Color, Depth, Motion, Output };

struct PssrSlot {
	bool                  valid = false;
	ShaderTextureResource descriptor;
	uint32_t              width  = 0;
	uint32_t              height = 0;
	vk::Format            format = vk::Format::eUndefined;
	PssrRole              role   = PssrRole::Unknown;
};

uint32_t ComponentCount(vk::Format format) {
	switch (format) {
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16Snorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR16G16Sint:
		case vk::Format::eR8G8Snorm:
		case vk::Format::eR8G8Unorm: return 2;
		case vk::Format::eR32Sfloat:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eD32Sfloat:
		case vk::Format::eD16Unorm:
		case vk::Format::eR8Unorm:
		case vk::Format::eR32Uint: return 1;
		default: return 4;
	}
}

} // namespace

// PS5 Pro upscaling request (graphics/guest_gpu/pssr.h). Which descriptor slot holds which image
// is worked out from sizes and formats: the largest image is the output; at the render size, a
// two-channel image is the motion vectors, a one-channel one the depth, the rest colour.
// The colour image is scaled into the output (linear filter).
void RenderExecutor::ProcessPssr(uint64_t /*submit_id*/, CommandBuffer& buffer,
                                 const Pssr::Request& request) {
	if (request.magic != Pssr::Magic || (request.version != 1 && request.version != Pssr::Version)) {
		LOGF("PSSR: ignoring request with magic 0x%08x version %u\n", request.magic, request.version);
		return;
	}
	std::array<PssrSlot, Pssr::TextureSlots> slots {};
	for (uint32_t i = 0; i < Pssr::TextureSlots; i++) {
		if ((request.texture_mask & (1u << i)) == 0) {
			continue;
		}
		auto& slot = slots[i];
		std::memcpy(slot.descriptor.fields, request.textures[i], sizeof(slot.descriptor.fields));
		if (slot.descriptor.IsNull()) {
			continue;
		}
		slot.valid  = true;
		slot.width  = static_cast<uint32_t>(slot.descriptor.Width5()) + 1u;
		slot.height = static_cast<uint32_t>(slot.descriptor.Height5()) + 1u;
		slot.format = TextureGetSurfaceFormatInfo(slot.descriptor.Format()).vk_format;
	}

	int output = -1;
	for (uint32_t i = 0; i < slots.size(); i++) {
		if (slots[i].valid && ComponentCount(slots[i].format) >= 3 &&
		    (output < 0 || uint64_t {slots[i].width} * slots[i].height >
		                       uint64_t {slots[output].width} * slots[output].height)) {
			output = static_cast<int>(i);
		}
	}
	int color = -1;
	for (uint32_t i = 0; i < slots.size(); i++) {
		auto& slot = slots[i];
		if (!slot.valid) {
			continue;
		}
		if (static_cast<int>(i) == output) {
			slot.role = PssrRole::Output;
			continue;
		}
		const auto components = ComponentCount(slot.format);
		slot.role = components == 2   ? PssrRole::Motion
		            : components == 1 ? PssrRole::Depth
		                              : PssrRole::Color;
		if (slot.role == PssrRole::Color && color < 0 &&
		    (output < 0 || slot.width < slots[output].width || slot.height < slots[output].height)) {
			color = static_cast<int>(i);
		}
	}

	if (output < 0 || color < 0 || slots[output].format == vk::Format::eUndefined) {
		return;
	}

	const auto resolve = [&](const PssrSlot& slot) {
		ShaderRecompiler::IR::ImageResource resource {};
		resource.resource_class = ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class  = Prospero::SampledTextureNumericClass(slot.descriptor.Format());
		resource.dimension      = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read           = true;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::memcpy(value.dwords.data(), slot.descriptor.fields, sizeof(slot.descriptor.fields));
		value.dword_count = 8;
		return ResolveTexture(resource, value);
	};
	auto&      cache          = m_context.GetTextureCache();
	const auto source_binding = resolve(slots[color]);
	const auto target_binding = resolve(slots[output]);
	auto&      source         = cache.GetImage(source_binding.image_id);
	auto&      target         = cache.GetImage(target_binding.image_id);
	if (&source == &target) {
		return;
	}

	// The first two requests are named: this line marks the first real game frame.
	static std::atomic<uint32_t> named {0};
	if (named.fetch_add(1, std::memory_order_relaxed) < 2) {
		std::printf("PSSR%u: colour slot %d %ux%u -> output slot %d %ux%u\n", request.version, color,
		            slots[color].width, slots[color].height, output, slots[output].width, slots[output].height);
		std::fflush(stdout);
	}

	// Drawn through views in the T# formats: the host image can have another base format (the title:
	// the colour input is written as 11_11_10 float into an image created as 10_10_10_2 unorm,
	// which vkCmdBlitImage decoded as unorm - false colours, 2026-10-29).
	const auto source_view = cache.FindTexture(source_binding.image_id, source_binding.desc);
	buffer.EndRendering();
	cache.GetBlitHelper().ScaleColor(buffer.Handle(), source, source_view,
	                                 {slots[color].width, slots[color].height}, target,
	                                 slots[output].format, {slots[output].width, slots[output].height});
	PerformanceOverlayRecordPssr("linear",
	                             slots[color].width,
	                             slots[color].height, slots[output].width, slots[output].height);
	buffer.InvalidateGraphicsState();
	cache.MarkGpuWritten(target_binding.image_id);
}

} // namespace Libs::Graphics
