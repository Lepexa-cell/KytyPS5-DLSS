#ifndef KYTY_GRAPHICS_DRIVER_PIPELINE_CACHE_H_
#define KYTY_GRAPHICS_DRIVER_PIPELINE_CACHE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

// The driver pipeline cache, split into shards. The driver stops adding pipelines to a
// VkPipelineCache once it holds about 2.8 GB (Ghost of Yotei filled one: the same pipelines
// compiled for a minute each on every run, and the saved size never changed). New pipelines go
// into the last shard; the earlier, full shards are only looked up, with
// VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT, so a miss there costs no compile.
struct DriverPipelineCache {
	static constexpr uint32_t kMaxShards = 4;

	std::array<vk::PipelineCache, kMaxShards> shards {};
	uint32_t                                  count = 0;

	explicit operator bool() const { return count != 0; }
	// The shard new pipelines are added to.
	[[nodiscard]] vk::PipelineCache Growing() const { return count == 0 ? nullptr : shards[count - 1]; }

	vk::Result CreateGraphics(vk::Device device, vk::GraphicsPipelineCreateInfo info,
	                          vk::Pipeline* pipeline) const {
		return Create(info, pipeline, [device](vk::PipelineCache cache, const auto& create_info,
		                                       vk::Pipeline* out) {
			return device.createGraphicsPipelines(cache, 1, &create_info, nullptr, out);
		});
	}
	vk::Result CreateCompute(vk::Device device, vk::ComputePipelineCreateInfo info,
	                         vk::Pipeline* pipeline) const {
		return Create(info, pipeline, [device](vk::PipelineCache cache, const auto& create_info,
		                                       vk::Pipeline* out) {
			return device.createComputePipelines(cache, 1, &create_info, nullptr, out);
		});
	}

private:
	template <typename Info, typename Call>
	vk::Result Create(Info info, vk::Pipeline* pipeline, Call&& call) const {
		if (count > 1 && !(info.flags & vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired)) {
			auto lookup = info;
			lookup.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
			for (uint32_t i = 0; i + 1 < count; i++) {
				*pipeline         = nullptr;
				const auto result = call(shards[i], lookup, pipeline);
				if (result == vk::Result::eSuccess && *pipeline != nullptr) {
					return result;
				}
			}
		}
		*pipeline = nullptr;
		return call(Growing(), info, pipeline);
	}
};

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_DRIVER_PIPELINE_CACHE_H_ */
