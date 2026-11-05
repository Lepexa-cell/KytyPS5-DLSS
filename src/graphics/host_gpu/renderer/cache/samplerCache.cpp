#include "graphics/host_gpu/renderer/cache/samplerCache.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "common/switches.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace Libs::Graphics {

SamplerCache::~SamplerCache() {
	for (const auto& [key, sampler]: m_made) {
		(void)key;
		m_graphics.device.destroySampler(sampler, nullptr);
	}
}

static uint32_t FloatBits(float value) {
	uint32_t bits = 0;
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

// Words 0-12: what must match exactly; 13-15: LOD bias, min and max LOD, which a stand-in may
// differ in.
static constexpr size_t kExactWords = 13;

vk::Sampler SamplerCache::NearestMade(const CreateKey& key) const {
	vk::Sampler best;
	float       best_distance = 0.0f;
	for (const auto& [made, sampler]: m_made) {
		if (!std::equal(made.begin(), made.begin() + kExactWords, key.begin())) {
			continue;
		}
		float distance = 0.0f;
		for (size_t i = kExactWords; i < key.size(); i++) {
			float a = 0.0f;
			float b = 0.0f;
			std::memcpy(&a, &made[i], sizeof(a));
			std::memcpy(&b, &key[i], sizeof(b));
			distance += std::abs(a - b);
		}
		if (!best || distance < best_distance) {
			best          = sampler;
			best_distance = distance;
		}
	}
	return best;
}

vk::Sampler SamplerCache::CreateShared(const vk::SamplerCreateInfo& info,
                                       vk::SamplerReductionMode     reduction) {
	const CreateKey key {static_cast<uint32_t>(info.magFilter),
	                     static_cast<uint32_t>(info.minFilter),
	                     static_cast<uint32_t>(info.mipmapMode),
	                     static_cast<uint32_t>(info.addressModeU),
	                     static_cast<uint32_t>(info.addressModeV),
	                     static_cast<uint32_t>(info.addressModeW),
	                     info.anisotropyEnable,
	                     FloatBits(info.maxAnisotropy),
	                     info.compareEnable,
	                     static_cast<uint32_t>(info.compareOp),
	                     static_cast<uint32_t>(info.borderColor),
	                     info.unnormalizedCoordinates,
	                     static_cast<uint32_t>(reduction),
	                     FloatBits(info.mipLodBias),
	                     FloatBits(info.minLod),
	                     FloatBits(info.maxLod)};
	if (auto iter = m_made.find(key); iter != m_made.end()) {
		return iter->second;
	}
	const auto limit = m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerAllocationCount;
	vk::Sampler sampler = nullptr;
	if (m_made.size() + 256 < limit) {
		if (m_graphics.device.createSampler(&info, nullptr, &sampler) != vk::Result::eSuccess) {
			sampler = nullptr;
		}
	}
	if (sampler == nullptr) {
		// The S# map keeps the stand-in for this S#: it is looked up here once.
		sampler = NearestMade(key);
		EXIT_NOT_IMPLEMENTED(sampler == nullptr);
		if (m_stand_ins++ < 16) {
			LOGF("Samplers: %zu made (device limit %u); a new LOD bias/range takes the nearest made "
			     "sampler (%u stand-ins)\n",
			     m_made.size(), limit, m_stand_ins);
		}
		return sampler;
	}
	m_made.emplace(key, sampler);
	if (const auto count = m_made.size(); count >= 256 && (count & (count - 1)) == 0) {
		LOGF("Samplers: %zu made for %zu S#s\n", count, m_samplers.size());
	}
	return sampler;
}

vk::Sampler SamplerCache::GetSampler(const ShaderSamplerResource& r, bool integer_border) {
	const SamplerKey key {r.fields[0], r.fields[1], r.fields[2], r.fields[3],
	                      static_cast<uint32_t>(integer_border)};
	// Every draw looks up each of its samplers. Samplers live as long as the cache, so each thread
	// keeps its recent lookups and repeats them without the lock and the map.
	struct Recent {
		uint64_t    cache = 0;
		SamplerKey  key {};
		vk::Sampler sampler;
	};
	thread_local std::array<Recent, 64> recent {};
	auto& slot = recent[SamplerKeyHash {}(key) % recent.size()];
	if (slot.cache == m_id && slot.key == key) {
		return slot.sampler;
	}
	const auto sampler = FindOrCreateSampler(r, key);
	slot               = {m_id, key, sampler};
	return sampler;
}

vk::Sampler SamplerCache::FindOrCreateSampler(const ShaderSamplerResource& r,
                                              const SamplerKey&            key) {
	Common::LockGuard lock(m_mutex);
	const bool        integer_border = key[4] != 0;

	if (auto iter = m_samplers.find(key); iter != m_samplers.end()) {
		return iter->second;
	}

	float      aniso_ratio = 1.0f;
	const auto mag_filter  = r.XyMagFilter();
	const auto min_filter  = r.XyMinFilter();

	auto is_aniso_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kAnisoPoint:
			case Prospero::SamplerFilter::kAnisoLinear: return true;
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kBilinear: return false;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return false;
	};

	auto to_vk_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kAnisoPoint: return vk::Filter::eNearest;
			case Prospero::SamplerFilter::kBilinear:
			case Prospero::SamplerFilter::kAnisoLinear: return vk::Filter::eLinear;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return vk::Filter::eNearest;
	};

	const bool aniso = is_aniso_filter(mag_filter) || is_aniso_filter(min_filter);
	if (aniso) {
		switch (static_cast<Prospero::SamplerAnisoRatio>(r.MaxAnisoRatio())) {
			case Prospero::SamplerAnisoRatio::kOne: aniso_ratio = 1.0f; break;
			case Prospero::SamplerAnisoRatio::kTwo: aniso_ratio = 2.0f; break;
			case Prospero::SamplerAnisoRatio::kFour: aniso_ratio = 4.0f; break;
			case Prospero::SamplerAnisoRatio::kEight: aniso_ratio = 8.0f; break;
			case Prospero::SamplerAnisoRatio::kSixteen:
			// Reserved encodings 5-7 fall back to the maximum ratio.
			default: aniso_ratio = 16.0f; break;
		}
	}

	const auto mip_filter = r.MipFilter();
	float      min_lod    = 0.0f;
	float      max_lod    = 0.0f;
	if (static_cast<Prospero::SamplerMipFilter>(mip_filter) != Prospero::SamplerMipFilter::kNone) {
		min_lod = static_cast<float>(r.MinLod()) / 256.0f;
		max_lod = static_cast<float>(r.MaxLod()) / 256.0f;
	}

	vk::SamplerCreateInfo sampler_info {};

	auto to_vk_address_mode = [](uint8_t clamp) {
		switch (static_cast<Prospero::SamplerClampMode>(clamp)) {
			case Prospero::SamplerClampMode::kWrap: return vk::SamplerAddressMode::eRepeat;
			case Prospero::SamplerClampMode::kMirror:
				return vk::SamplerAddressMode::eMirroredRepeat;
			case Prospero::SamplerClampMode::kClampLastTexel:
				return vk::SamplerAddressMode::eClampToEdge;
			case Prospero::SamplerClampMode::kMirrorOnceLastTexel:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampHalfBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceHalfBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			default: EXIT("unknown clamp: %u\n", clamp);
		}
		return vk::SamplerAddressMode::eClampToBorder;
	};

	vk::BorderColor border = integer_border ? vk::BorderColor::eIntTransparentBlack
	                                       : vk::BorderColor::eFloatTransparentBlack;
	switch (static_cast<Prospero::SamplerBorderColor>(r.BorderColorType())) {
		case Prospero::SamplerBorderColor::kTransBlack: break;
		case Prospero::SamplerBorderColor::kOpaqueBlack:
			border = integer_border ? vk::BorderColor::eIntOpaqueBlack
			                        : vk::BorderColor::eFloatOpaqueBlack;
			break;
		case Prospero::SamplerBorderColor::kOpaqueWhite:
			border = integer_border ? vk::BorderColor::eIntOpaqueWhite
			                        : vk::BorderColor::eFloatOpaqueWhite;
			break;
		case Prospero::SamplerBorderColor::kFromTable:
			LOGF(
			    "temporary: approximating table border color as transparent black, index = %" PRIu16
			    "\n",
			    r.BorderColorPtr());
			break;
		default: EXIT("unknown border color: %d", static_cast<int>(r.BorderColorType()));
	}

	sampler_info.magFilter = to_vk_filter(mag_filter);
	sampler_info.minFilter = to_vk_filter(min_filter);
	sampler_info.mipmapMode =
	    (static_cast<Prospero::SamplerMipFilter>(mip_filter) == Prospero::SamplerMipFilter::kLinear
	         ? vk::SamplerMipmapMode::eLinear
	         : vk::SamplerMipmapMode::eNearest);
	sampler_info.addressModeU = to_vk_address_mode(r.ClampX());
	sampler_info.addressModeV = to_vk_address_mode(r.ClampY());
	sampler_info.addressModeW = to_vk_address_mode(r.ClampZ());
    const auto max_lod_bias = m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerLodBias;
    sampler_info.mipLodBias = std::clamp(
        static_cast<float>(static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u)) / 256.0f,
        -max_lod_bias, max_lod_bias
    );
	sampler_info.anisotropyEnable        = (aniso ? VK_TRUE : VK_FALSE);
	sampler_info.maxAnisotropy           = aniso_ratio;
	sampler_info.compareEnable           = (r.DepthCompareFunc() != 0 ? VK_TRUE : VK_FALSE);
	sampler_info.compareOp               = static_cast<vk::CompareOp>(r.DepthCompareFunc());
	sampler_info.minLod                  = min_lod;
	sampler_info.maxLod                  = max_lod;
	sampler_info.borderColor             = border;
	sampler_info.unnormalizedCoordinates = (r.ForceUnormCoords() ? VK_TRUE : VK_FALSE);

	if (r.ForceUnormCoords()) {
		sampler_info.addressModeU     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeV     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeW     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.mipmapMode       = vk::SamplerMipmapMode::eNearest;
		sampler_info.minLod           = 0.0f;
		sampler_info.maxLod           = 0.0f;
		sampler_info.anisotropyEnable = VK_FALSE;
		sampler_info.maxAnisotropy    = 1.0f;
		sampler_info.compareEnable    = VK_FALSE;
		sampler_info.mipLodBias       = 0.0f;
	}

	// S# FILTER_MODE: 0 blends the filter footprint, 1 takes its minimum, 2 its maximum (RDNA2).
	// The minimum/maximum filters were applied as a blend before.
	vk::SamplerReductionModeCreateInfo reduction {};
	if (const auto mode = r.FilterMode(); mode == 1u || mode == 2u) {
		reduction.reductionMode =
		    mode == 1u ? vk::SamplerReductionMode::eMin : vk::SamplerReductionMode::eMax;
		sampler_info.pNext = &reduction;
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::printf("sampler %s reduction: fields=%08x,%08x,%08x,%08x\n", mode == 1u ? "min" : "max",
			            r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
			std::fflush(stdout);
		}
	}

	const auto vk_sampler = CreateShared(
	    sampler_info, sampler_info.pNext != nullptr ? reduction.reductionMode
	                                                : vk::SamplerReductionMode::eWeightedAverage);
	m_samplers.emplace(key, vk_sampler);
	return vk_sampler;
}

} // namespace Libs::Graphics
