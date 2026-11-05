#include "graphics/host_gpu/handleRegistry.h"

#include "common/emulatorConfig.h"
#include "common/switches.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#endif

namespace Libs::Graphics::HandleRegistry {

namespace {

constexpr size_t StackDepth = 10;

struct Entry {
	uint64_t                          generation = 0;
	Kind                              kind       = Kind::Buffer;
	bool                              dead       = false;
	std::string                       origin;
	std::array<uint64_t, StackDepth> destroyed_by {};
};

std::mutex                          g_mutex;
std::unordered_map<uint64_t, Entry> g_entries;
uint64_t                            g_next_generation = 1;
std::atomic<uint64_t>               g_stale {0};
std::atomic<uint32_t>               g_logged {0};

const char* KindName(Kind kind) {
	return kind == Kind::Image ? "image" : kind == Kind::ImageView ? "image view" : "buffer";
}

// The caller's return addresses as offsets into the executable (0 for other modules), so they
// can be looked up in the link map.
std::array<uint64_t, StackDepth> CaptureStack() {
	std::array<uint64_t, StackDepth> offsets {};
#ifdef _WIN32
	std::array<void*, StackDepth + 2> frames {};
	const auto count = RtlCaptureStackBackTrace(2, static_cast<DWORD>(frames.size()), frames.data(), nullptr);
	static const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	size_t            n    = 0;
	for (USHORT i = 0; i < count && n < offsets.size(); i++) {
		const auto address = reinterpret_cast<uint64_t>(frames[i]);
		offsets[n++]       = address >= base && address - base < 0x10000000 ? address - base : 0;
	}
#endif
	return offsets;
}

} // namespace

bool CheckEnabled() {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEFERRED_HANDLE_CHECK");
		if (text != nullptr) {
			return std::strcmp(text, "0") != 0;
		}
		return Config::RenderDocEnabled() || KYTY_GETENV("KYTY_DEBUG_DEVICE_FAULT") != nullptr;
	}();
	return enabled;
}

void Register(uint64_t handle, Kind kind, const char* origin) {
	if (handle == 0 || !CheckEnabled()) {
		return;
	}
	std::scoped_lock lock {g_mutex};
	auto&            entry = g_entries[handle];
	entry.generation       = g_next_generation++;
	entry.kind             = kind;
	entry.dead             = false;
	entry.origin           = origin != nullptr ? origin : "";
	entry.destroyed_by     = {};
}

void Unregister(uint64_t handle) {
	if (handle == 0 || !CheckEnabled()) {
		return;
	}
	const auto       stack = CaptureStack();
	std::scoped_lock lock {g_mutex};
	if (const auto found = g_entries.find(handle); found != g_entries.end()) {
		found->second.dead         = true;
		found->second.destroyed_by = stack;
	}
}

uint64_t Stamp(uint64_t handle) {
	if (!CheckEnabled() || handle == 0) {
		return 0;
	}
	std::scoped_lock lock {g_mutex};
	const auto       found = g_entries.find(handle);
	return found != g_entries.end() && !found->second.dead ? found->second.generation : 0;
}

bool Same(uint64_t handle, uint64_t stamp, const char* call, const char* role) {
	if (stamp == 0) {
		return true;
	}
	Entry now {};
	{
		std::scoped_lock lock {g_mutex};
		if (const auto found = g_entries.find(handle); found != g_entries.end()) {
			now = found->second;
		}
	}
	if (now.generation == stamp && !now.dead) {
		return true;
	}
	g_stale.fetch_add(1, std::memory_order_relaxed);
	if (g_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::string stack;
		for (const auto offset: now.destroyed_by) {
			if (offset != 0) {
				char text[24] = {};
				std::snprintf(text, sizeof(text), " +0x%llx", static_cast<unsigned long long>(offset));
				stack += text;
			}
		}
		std::printf("deferred-call: stale %s %s handle=0x%016llx recorded generation %llu, now %llu%s; "
		            "%s skipped; origin: %s; destroyed by kyty_emulator%s\n",
		            KindName(now.kind), role, static_cast<unsigned long long>(handle),
		            static_cast<unsigned long long>(stamp),
		            static_cast<unsigned long long>(now.generation),
		            now.dead ? " (destroyed)" : " (recreated)", call,
		            now.origin.empty() ? "?" : now.origin.c_str(),
		            stack.empty() ? " (no stack)" : stack.c_str());
		std::fflush(stdout);
	}
	return false;
}

uint64_t StaleCount() {
	return g_stale.load(std::memory_order_relaxed);
}

bool IsDead(uint64_t handle) {
	if (!CheckEnabled() || handle == 0) {
		return false;
	}
	std::scoped_lock lock {g_mutex};
	const auto       found = g_entries.find(handle);
	return found != g_entries.end() && found->second.dead;
}

void Describe(uint64_t handle, const char* prefix) {
	Entry now {};
	bool  known = false;
	{
		std::scoped_lock lock {g_mutex};
		if (const auto found = g_entries.find(handle); found != g_entries.end()) {
			now   = found->second;
			known = true;
		}
	}
	std::string stack;
	for (const auto offset: now.destroyed_by) {
		if (offset != 0) {
			char text[24] = {};
			std::snprintf(text, sizeof(text), " +0x%llx", static_cast<unsigned long long>(offset));
			stack += text;
		}
	}
	std::printf("%s handle=0x%016llx %s %s origin: %s; destroyed by kyty_emulator%s\n", prefix,
	            static_cast<unsigned long long>(handle), known ? KindName(now.kind) : "unknown",
	            !known ? "(not registered)" : now.dead ? "(destroyed)" : "(alive)",
	            now.origin.empty() ? "?" : now.origin.c_str(), stack.empty() ? " (no stack)" : stack.c_str());
	std::fflush(stdout);
}

} // namespace Libs::Graphics::HandleRegistry
