#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;

class SamplerCache {
public:
	explicit SamplerCache(GraphicContext& graphics): m_graphics(graphics) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	}
	~SamplerCache();
	KYTY_CLASS_NO_COPY(SamplerCache);

	vk::Sampler GetSampler(const ShaderSamplerResource& r, bool integer_border);

private:
	using SamplerKey = std::array<uint32_t, 5>;
	// The Vulkan create info a S# comes out as (filters, mip mode, address modes, LOD bias,
	// anisotropy, compare, LOD range, border, unnormalized coordinates, reduction), as words.
	using CreateKey = std::array<uint32_t, 16>;

	vk::Sampler FindOrCreateSampler(const ShaderSamplerResource& r, const SamplerKey& key);
	// Idea from usa-ratt's GT7 fork (round 128): S#s that make the same sampler share one, and
	// past the device's maxSamplerAllocationCount (less 256 for the helpers' own samplers; 4000 on
	// the driver) a new one is the nearest made sampler that differs only in LOD bias or range,
	// instead of a failed vkCreateSampler ending the emulator. m_mutex must be held.
	vk::Sampler CreateShared(const vk::SamplerCreateInfo& info, vk::SamplerReductionMode reduction);
	vk::Sampler NearestMade(const CreateKey& key) const;

	// Tells caches apart in the per-thread lookups (see GetSampler); never reused.
	static uint64_t NextId() {
		static std::atomic<uint64_t> next {0};
		return next.fetch_add(1, std::memory_order_relaxed) + 1;
	}

	struct SamplerKeyHash {
		std::size_t operator()(const SamplerKey& key) const {
			std::size_t hash = 0;
			for (auto value: key) {
				hash ^= static_cast<std::size_t>(value) +
				        static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
				        (hash >> 2u);
			}
			return hash;
		}
	};

	GraphicContext&                                             m_graphics;
	const uint64_t                                              m_id = NextId();
	Common::Mutex                                               m_mutex;
	std::unordered_map<SamplerKey, vk::Sampler, SamplerKeyHash> m_samplers;
	// Every sampler made, by create info; destroyed with the cache.
	std::map<CreateKey, vk::Sampler>                            m_made;
	uint32_t                                                    m_stand_ins = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
