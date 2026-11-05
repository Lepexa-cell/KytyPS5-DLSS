#include "graphics/presentation/renderDoc.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <renderdoc_app.h>
#include <string>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <dlfcn.h>
#endif

namespace Libs::Graphics {

enum class RenderDocState : uint32_t {
	Idle,
	Requested,
	Capturing,
};

static RENDERDOC_API_1_6_0*        g_api             = nullptr;
static std::atomic<RenderDocState> g_state           = RenderDocState::Idle;
static uint32_t                    g_captured_flips  = 0;
static std::atomic_bool            g_unavailable_log = false;

// RenderDoc messages go to the console as well as the log: the log is silent with
// --printf-direction Silent, and the user has to see that F1 took and where the file went.
template <typename... Args>
static void Say(const char* format, Args... args) {
	LOGF(format, args...);
	std::printf(format, args...);
	std::fflush(stdout);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

static bool BindRenderDocApi(HMODULE module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(module, "RENDERDOC_GetAPI"));
#else
static bool BindRenderDocApi(void* module) {
	auto* get_api = reinterpret_cast<pRENDERDOC_GetAPI>(::dlsym(module, "RENDERDOC_GetAPI"));
#endif
	if (get_api == nullptr) {
		return false;
	}

	void* api = nullptr;
	if (get_api(eRENDERDOC_API_Version_1_6_0, &api) != 1 || api == nullptr) {
		return false;
	}

	g_api = static_cast<RENDERDOC_API_1_6_0*>(api);
	g_api->SetCaptureKeys(nullptr, 0);
	g_api->UnloadCrashHandler();
	Say("RenderDoc: API 1.6.0 bound\n");
	return true;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}

	auto* module = GetModuleHandleA("renderdoc.dll");
	if (module == nullptr) {
		HKEY key = nullptr;
		if (RegOpenKeyExW(HKEY_LOCAL_MACHINE,
		                  L"SOFTWARE\\Classes\\RenderDoc.RDCCapture.1\\DefaultIcon\\", 0, KEY_READ,
		                  &key) != ERROR_SUCCESS) {
			return;
		}

		std::array<wchar_t, MAX_PATH> path_buffer {};
		DWORD      path_size = static_cast<DWORD>(path_buffer.size() * sizeof(wchar_t));
		const auto result    = RegQueryValueExW(
		    key, L"", nullptr, nullptr, reinterpret_cast<LPBYTE>(path_buffer.data()), &path_size);
		RegCloseKey(key);
		if (result != ERROR_SUCCESS) {
			return;
		}

		auto path = std::filesystem::path(path_buffer.data()).parent_path() / "renderdoc.dll";
		module    = LoadLibraryW(path.c_str());
		if (module == nullptr) {
			return;
		}
	}

	if (!BindRenderDocApi(module)) {
		Say("RenderDoc: API 1.6.0 is unavailable\n");
	}
}

#else

void RenderDocInit() {
	if (g_api != nullptr) {
		return;
	}

	auto* module = ::dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
	if (module == nullptr) {
		module = ::dlopen("librenderdoc.so", RTLD_NOW);
	}
	if (module == nullptr) {
		return;
	}

	if (!BindRenderDocApi(module)) {
		Say("RenderDoc: API 1.6.0 is unavailable\n");
		::dlclose(module);
	}
}

#endif

void RenderDocRequestCapture() {
	if (g_api == nullptr) {
		if (!g_unavailable_log.exchange(true)) {
			Say("RenderDoc: capture requested, but RenderDoc is unavailable\n");
		}
		return;
	}

	RenderDocState expected = RenderDocState::Idle;
	if (g_state.compare_exchange_strong(expected, RenderDocState::Requested)) {
		Say("RenderDoc: capture requested\n");
	}
}

static void StartCapture() {
	if (g_api->IsFrameCapturing() != 0) {
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		Say("RenderDoc: capture request ignored because a capture is already active\n");
		return;
	}

	const auto capture_id   = std::chrono::duration_cast<std::chrono::microseconds>(
	                              std::chrono::system_clock::now().time_since_epoch())
	                              .count();
	// An absolute path in the working directory's _RenderDoc folder: what RenderDoc does with a
	// relative template is not specified, and the file has to be where the user looks.
	std::error_code error;
	std::filesystem::create_directories("_RenderDoc", error);
	const auto capture_path =
	    (std::filesystem::absolute("_RenderDoc", error) / ("kyty_" + std::to_string(capture_id)))
	        .string();
	g_api->SetCaptureFilePathTemplate(capture_path.c_str());
	g_api->StartFrameCapture(nullptr, nullptr);
	if (g_api->IsFrameCapturing() == 0) {
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		Say("RenderDoc: capture failed to start\n");
		return;
	}
	g_captured_flips = 0;
	g_state.store(RenderDocState::Capturing, std::memory_order_release);
	Say("RenderDoc: capture started\n");
}

void RenderDocOnGuestFlip(RenderContext& renderer) {
	// KYTY_RENDERDOC_CAPTURE_FILE=<path>: a capture is requested (as with F1) when that file
	// appears; it is removed then. Unattended test runs create it at a chosen second.
	static const char* trigger = std::getenv("KYTY_RENDERDOC_CAPTURE_FILE");
	if (trigger != nullptr && g_api != nullptr) [[unlikely]] {
		static uint32_t flips = 0;
		if (++flips % 30u == 0u) {
			std::error_code error;
			if (std::filesystem::exists(trigger, error)) {
				std::filesystem::remove(trigger, error);
				RenderDocRequestCapture();
			}
		}
	}
	// KYTY_RENDERDOC_ON_SLOW_FLIPS=<ms>,<after>,<streak>: one capture is requested when `streak` guest
	// flips in a row each come more than `ms` after the previous one, once `after` flips have passed
	// (the title in Pro mode: the green-noise picture comes with a drop to 13-20 fps, seconds long, rare).
	static const auto slow = [] {
		struct {
			uint32_t ms = 0, after = 0, streak = 0;
		} s;
		if (const char* text = std::getenv("KYTY_RENDERDOC_ON_SLOW_FLIPS"); text != nullptr) {
			(void)std::sscanf(text, "%u,%u,%u", &s.ms, &s.after, &s.streak);
		}
		return s;
	}();
	// Without --rd (RenderDoc's cost hid the picture) it creates the F9 trigger files instead:
	// KYTY_DEBUG_FRAME_PASSES and KYTY_DEBUG_DUMP_IMAGES_FILE (with KYTY_DEBUG_DUMP_VIDEO_OUT=<n>, a film).
	if (slow.ms != 0) [[unlikely]] {
		static uint32_t flips = 0, run = 0;
		static bool     fired = false;
		static auto     last  = std::chrono::steady_clock::now();
		const auto      now   = std::chrono::steady_clock::now();
		const bool      long_flip = now - last > std::chrono::milliseconds(slow.ms);
		last                      = now;
		run                       = long_flip ? run + 1 : 0;
		if (!fired && ++flips > slow.after && run >= std::max(slow.streak, 1u)) {
			fired = true;
			Say("RenderDoc: %u slow flips in a row at flip %u\n", run, flips);
			if (g_api != nullptr) {
				RenderDocRequestCapture();
			}
			for (const char* name: {"KYTY_DEBUG_FRAME_PASSES", "KYTY_DEBUG_DUMP_IMAGES_FILE"}) {
				if (const char* path = std::getenv(name); path != nullptr && *path != '\0') {
					if (std::FILE* file = std::fopen(path, "wb"); file != nullptr) {
						std::fclose(file);
					}
				}
			}
		}
	}
	// KYTY_RENDERDOC_REPEAT=<flips>,<count>: after a capture, `count` more captures follow, each
	// requested `flips` guest flips after the previous one ended (a scene that breaks a few seconds
	// after the trigger is caught in one run).
	static int      repeat_left  = -1;
	static uint32_t repeat_every = 0;
	static uint32_t repeat_wait  = 0;
	if (repeat_left < 0) {
		repeat_left = 0;
		if (const char* text = std::getenv("KYTY_RENDERDOC_REPEAT"); text != nullptr) {
			char* end    = nullptr;
			repeat_every = static_cast<uint32_t>(std::strtoul(text, &end, 10));
			repeat_left  = *end == ',' ? std::atoi(end + 1) : 0;
		}
	}
	if (repeat_wait != 0 && --repeat_wait == 0 && g_api != nullptr) {
		RenderDocRequestCapture();
	}
	const auto state = g_state.load(std::memory_order_acquire);
	if (g_api == nullptr || state == RenderDocState::Idle) {
		return;
	}
	if (state == RenderDocState::Capturing) {
		// KYTY_RENDERDOC_FLIPS=<n>: guest flips per capture (default 2).
		static const uint32_t flips_per_capture = [] {
			const char* text = std::getenv("KYTY_RENDERDOC_FLIPS");
			const int   n    = text != nullptr ? std::atoi(text) : 2;
			return static_cast<uint32_t>(n > 0 ? n : 2);
		}();
		Say("RenderDoc: captured guest flip %u/%u\n", ++g_captured_flips, flips_per_capture);
		if (g_captured_flips < flips_per_capture) {
			return;
		}
	}

	// Capture boundaries follow presentation and exclude concurrent queue access.
	Common::LockGuard render_lock(renderer.GetMutex());
	Common::LockGuard queue_lock(renderer.GetGraphics().queue_mutex);
	if (state == RenderDocState::Requested) {
		StartCapture();
	} else {
		const auto ok = g_api->EndFrameCapture(nullptr, nullptr);
		g_state.store(RenderDocState::Idle, std::memory_order_release);
		if (repeat_left > 0 && repeat_every != 0) {
			repeat_left--;
			repeat_wait = repeat_every;
		}
		if (ok == 0) {
			Say("RenderDoc: capture failed\n");
			return;
		}
		// EndFrameCapture has written the file: name it with its full path.
		const auto count  = g_api->GetNumCaptures();
		uint32_t   length = 0;
		if (count == 0 || g_api->GetCapture(count - 1, nullptr, &length, nullptr) == 0 || length == 0) {
			Say("RenderDoc: capture finished, but RenderDoc lists no capture file\n");
			return;
		}
		std::string path(length, ' ');
		uint64_t    timestamp = 0;
		if (g_api->GetCapture(count - 1, path.data(), &length, &timestamp) != 0) {
			path.resize(std::strlen(path.c_str()));
			Say("RenderDoc: capture %u written to %s\n", count, path.c_str());
		}
	}
}

} // namespace Libs::Graphics
