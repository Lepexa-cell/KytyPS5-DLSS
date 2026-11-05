#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	// KYTY_DEBUG_BDA_STATS: this thread's protection changes since the last call.
	struct ProtectStats {
		uint64_t calls = 0;
		uint64_t ns    = 0;
	};
	[[nodiscard]] static ProtectStats TakeProtectStats() noexcept;

	// While set, this thread's protection changes are queued instead of made. Only for clearing
	// CPU-dirty pages before copying them: whoever copies them must RunDeferredProtects first, so
	// a write that lands before the protection is in the copy, and every later write faults.
	static void DeferProtects(bool defer) noexcept;
	static void RunDeferredProtects() noexcept;
	// How many changes this thread has queued so far.
	[[nodiscard]] static uint64_t DeferredPushes() noexcept;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
