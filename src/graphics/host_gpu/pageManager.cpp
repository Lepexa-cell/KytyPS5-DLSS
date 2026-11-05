#include "graphics/host_gpu/pageManager.h"

#include "common/alignment.h"
#include "common/virtualMemory.h"
#include "common/switches.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <utility>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		while (m_lock.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

bool ProtectStatsEnabled() {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_BDA_STATS") != nullptr;
	return enabled;
}

thread_local PageManager::ProtectStats t_protect_stats;

void ProtectNow(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
	const bool stats = ProtectStatsEnabled();
	const auto start = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const bool done  = Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode);
	if (stats) {
		t_protect_stats.calls++;
		t_protect_stats.ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
		        .count());
	}
	if (!done) {
		Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
		      static_cast<uint32_t>(mode));
	}
}

// Deferred protection changes (see PageManager::DeferProtects), run in order. Every change of a
// page's protection happens under its region's lock, and an immediate change first runs the
// queued ones, so a page's changes take effect in the order they were made.
struct DeferredProtect {
	uint64_t                    vaddr;
	uint64_t                    size;
	Common::VirtualMemory::Mode mode;
};
std::mutex                   g_deferred_mutex;
std::vector<DeferredProtect> g_deferred;
std::atomic<uint64_t>        g_deferred_count {0};
// Split mode: g_deferred_mutex only guards the queue and is never held during a protection
// change, so the GPU thread queueing a change no longer waits for the upload helper's
// VirtualProtect calls. Whoever runs queued changes holds g_run_mutex (taken before
// g_deferred_mutex) from taking them off the queue until they are made, which keeps a page's
// changes in order; g_deferred_count then counts the queued and the running changes.
// KYTY_DEFER_PROTECTS_SPLIT=0 runs them under g_deferred_mutex as before.
std::mutex g_run_mutex;

bool SplitEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEFER_PROTECTS_SPLIT");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	return enabled;
}
thread_local bool            t_defer_protects = false;
thread_local uint64_t        t_deferred_pushes = 0;

// Caller holds g_deferred_mutex.
void RunDeferredLocked() noexcept {
	for (const auto& item: g_deferred) {
		ProtectNow(item.vaddr, item.size, item.mode);
	}
	g_deferred.clear();
	g_deferred_count.store(0, std::memory_order_release);
}

// Caller holds g_deferred_mutex: runs, in order, the queued changes overlapping the range; the
// others keep their place.
void RunDeferredOverlappingLocked(uint64_t vaddr, uint64_t size) noexcept {
	size_t kept = 0;
	for (size_t i = 0; i < g_deferred.size(); i++) {
		const auto& item = g_deferred[i];
		if (item.vaddr < vaddr + size && vaddr < item.vaddr + item.size) {
			ProtectNow(item.vaddr, item.size, item.mode);
		} else {
			g_deferred[kept++] = item;
		}
	}
	g_deferred.resize(kept);
	g_deferred_count.store(kept, std::memory_order_release);
}

// Split mode: runs the queued changes (all, or those overlapping the range) outside
// g_deferred_mutex.
void RunDeferredSplit(bool all, uint64_t vaddr = 0, uint64_t size = 0) noexcept {
	thread_local std::vector<DeferredProtect> batch;
	std::scoped_lock                          run(g_run_mutex);
	{
		std::scoped_lock lock(g_deferred_mutex);
		if (all) {
			batch.swap(g_deferred);
		} else {
			size_t kept = 0;
			for (size_t i = 0; i < g_deferred.size(); i++) {
				const auto& item = g_deferred[i];
				if (item.vaddr < vaddr + size && vaddr < item.vaddr + item.size) {
					batch.push_back(item);
				} else {
					g_deferred[kept++] = item;
				}
			}
			g_deferred.resize(kept);
		}
	}
	for (const auto& item: batch) {
		ProtectNow(item.vaddr, item.size, item.mode);
	}
	g_deferred_count.fetch_sub(batch.size(), std::memory_order_release);
	batch.clear();
}

} // namespace

uint64_t PageManager::DeferredPushes() noexcept {
	return t_deferred_pushes;
}

void PageManager::DeferProtects(bool defer) noexcept {
	t_defer_protects = defer;
}

void PageManager::RunDeferredProtects() noexcept {
	if (g_deferred_count.load(std::memory_order_acquire) == 0) {
		return;
	}
	if (SplitEnabled()) {
		RunDeferredSplit(true);
		return;
	}
	std::scoped_lock lock(g_deferred_mutex);
	RunDeferredLocked();
}

PageManager::ProtectStats PageManager::TakeProtectStats() noexcept {
	return std::exchange(t_protect_stats, {});
}

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7f) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 1);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		if (t_defer_protects) {
			std::scoped_lock lock(g_deferred_mutex);
			g_deferred.push_back({vaddr, size, mode});
			g_deferred_count.fetch_add(1, std::memory_order_release);
			t_deferred_pushes++;
			return;
		}
		if (g_deferred_count.load(std::memory_order_acquire) != 0) {
			if (SplitEnabled()) {
				// Changes queued later for these pages wait for this region's lock, held here.
				RunDeferredSplit(false, vaddr, size);
			} else {
				std::scoped_lock lock(g_deferred_mutex);
				RunDeferredOverlappingLocked(vaddr, size);
			}
		}
		ProtectNow(vaddr, size, mode);
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		SpinGuard lock(region.lock);
		auto      perms                 = region.pages[first].Perms();
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = page.Perms();
			const auto new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = page.Perms();

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
