#include "common/archive.h"
#include "common/common.h"
#include "common/consoleColors.h"
#include "common/dateTime.h"
#include "common/debug.h"
#include "common/file.h"
#include "common/lastWords.h"
#include "common/settingsFile.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "common/switches.h"
#include "emulator.h"
#include "graphics/host_gpu/renderer/pipeline/shaderBench.h"
#include "kytyGitVersion.h"
#include "loader/gamePatch.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

using namespace Common;
using namespace Emulator;

static std::string GetBuildString() {
	Date date = Date::FromMacros(std::string(__DATE__));

#if KYTY_BUILD == KYTY_BUILD_DEBUG
	std::string type = "Debug";
#elif KYTY_BUILD == KYTY_BUILD_RELEASE
	std::string type = "Release";
#else
	std::string type = "????";
#endif

	std::string compiler = Debug::GetCompiler() + "-" + Debug::GetLinker();

	std::string str =
	    fmt::format("{}, {}, ver = {}, git = {}, date = {}", type.c_str(), compiler.c_str(),
	                KYTY_VERSION, KYTY_GIT_VERSION, date.ToString().c_str());

	return str;
}

static void PrintUsage() {
	::printf("%s\n", GetBuildString().c_str());
	::printf("kyty_emulator --game <dir|elf|zar> [options]\n\n");
	::printf("Options can also be set in kyty_settings.ini in the working directory, one per\n"
	         "line without \"--\" (e.g. gpu-timestamp-scale = 115). The command line overrides it.\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf|zar>                 Game directory, ELF, or ZArchive to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
	::printf("  --controller-color <#RRGGBB>        Override the controller lightbar color.\n");
	::printf("  --controller-volume <0-100>         DualSense speaker volume. Default: 50.\n");
	::printf("  --controller-vibration <0-100>      DualSense vibration intensity. Default: 100.\n");
	::printf("  --audio-master-volume <0-200>       Volume of everything on the main output. Default: 100.\n");
	::printf("  --audio-main-volume <0-200>         Game sound (main ports). Default: 100.\n");
	::printf("  --audio-music-volume <0-200>        Music on BGM ports. Default: 100.\n");
	::printf("  --audio-pad-speaker-volume <0-200>  Controller-speaker sounds played on the main output\n"
	         "                                      when no DualSense takes them. Default: %u.\n",
	         Config::DEFAULT_AUDIO_PAD_SPEAKER_MAIN_VOLUME);
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
	::printf("  --bda-sync <value>                   Selective, Legacy, or SelectiveChecked. "
	         "Default: Selective.\n");
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf("  --hdr                                Enable HDR10 output when supported.\n");
	::printf(
	    "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
	::printf("  --no-splash                          Disable the startup splash screen.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --amd-cpu                            Apply AMD CPU instruction patches.\n");
	::printf("  --dualsense-fix                      Vibration, adaptive triggers and controller speaker\n"
	         "                                       on the DualSense. Without it: none of them;\n"
	         "                                       controller speaker sounds play on the PC output.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
	::printf("  --network-connected <true|false>     Use the host local network (default: false).\n");
	::printf("  --gpu-assisted-validation <t|f>      Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --shader-validation <true|false>     Enable shader validation.\n");
	::printf("  --shader-precompile <true|false>     Replay recorded shaders before drawing. "
	         "Default: true.\n");
	::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
	::printf("  --bindless                          Sample heap-indexed images through bindless arrays.\n");
	::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump <true|false>   Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump <true|false>   Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profile                            Enable the Tracy profiler.\n");
	::printf("  --spirv-debug-printf <true|false>    Enable SPIR-V debug printf.\n");
	::printf(
	    "  --readback-linear-images <true|false> Read back writable linear images on submit.\n");
	::printf(
	    "  --sync-raw-image-buffers <true|false> Synchronize raw reads of GPU images. Default: false.\n");
	::printf(
	    "  --trophy-notifications <true|false>   Show trophy unlock toasts and play their sound.\n");
	::printf("  --perf-overlay <true|false>           Show the performance panel (F2 or Cmd+P).\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
	::printf("  --drain-stats <seconds>              Report GPU waits by cause every N seconds.\n");
	::printf("  --dcc-gpu-clear <true|false>         Apply GPU-written DCC clears on the GPU. "
	         "Default: true.\n");
	::printf("  --async-submit <true|false>          Submit GPU work from a dedicated queue thread. "
	         "Default: true.\n");
	::printf("  --gpu-mesh-indirect <true|false>     Build mesh-emulated indirect draws on the GPU. "
	         "Default: true.\n");
	::printf("  --gpu-frames-ahead <0-3>             Frames the game may build ahead of the GPU "
	         "thread. 0 waits for idle. Default: 0.\n");
	::printf("  --label-flush-interval-us <us>       Minimum time between RELEASE_MEM submits. "
	         "Default: 2000.\n");
	::printf("  --gpu-timestamp-scale <25-1000>      Stretch GPU time the game measures, in "
	         "percent, so dynamic resolution keeps headroom (below 100: aims higher). Default: 100 (off).\n");
	::printf("  --pipeline-libraries <true|false>    Build new graphics pipelines from cached, "
	         "separately compiled parts where the driver supports it. Default: true.\n");
	::printf("  --async-pipelines <true|false>       Skip draws whose new pipeline is still "
	         "compiling instead of stalling; they appear a few frames late. Default: false.\n");
	::printf("  --pipeline-compile-threads <n>       Threads that build compute pipelines in the "
	         "background. Default: 0 (all hardware threads but two, at least two).\n");
	::printf("  --pipeline-compile-threads-replay <n>  Of those, how many may build the shader "
	         "journal's pipelines at startup, at the lowest priority. Default: 0 (half, at least one).\n");
	::printf("  --dispatch-lookahead <true|false>    Walk each submission ahead for new compute "
	         "shaders and start compiling their pipelines early. Default: true.\n");
	::printf("  --defer-dispatches <true|false>      Suspend a queue whose dispatch waits for a "
	         "compiling compute pipeline; other queues go on. Default: false.\n");
	::printf("  --relaxed-readback <true|false>      Let game threads read memory the GPU is still "
	         "writing without waiting; a value can be a frame old. Default: false.\n");
	::printf("  --fxaa <true|false>                  Smooth jagged edges of the presented frame "
	         "with FXAA. Default: false.\n");
	::printf("  --speculative-draws <true|false>     Prepare draws' shader resources on a second "
	         "thread ahead of the GPU thread. Default: true.\n");
	::printf("  --record-thread <true|false>         Record the GPU thread's Vulkan commands on the "
	         "submit thread. Default: true.\n");
	::printf("  --hardware-buffer-bounds <true|false> Let the GPU check shader buffer ranges where "
	         "it supports that, instead of checks compiled into each shader. Default: true.\n");
	::printf(
	    "  --skip-notice-screen <true|false>    Skip startup logos and notices in supported games.\n"
	    "                                      Default: false.\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
	::printf("  --red-zone-protection <true|false>   Same as --redzone; the settings panel (F2) "
	         "saves it. Default: false.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

static bool ParseBool(const std::string& value, bool& out) {
	if (Common::EqualNoCase(value, "true") || value == "1" || Common::EqualNoCase(value, "yes") ||
	    Common::EqualNoCase(value, "on")) {
		out = true;
		return true;
	}

	if (Common::EqualNoCase(value, "false") || value == "0" || Common::EqualNoCase(value, "no") ||
	    Common::EqualNoCase(value, "off")) {
		out = false;
		return true;
	}

	return false;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value.c_str());
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUint32(const std::string& value, uint32_t& out) {
	uint32_t number   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseControllerColor(const std::string& value, Config::ControllerColor& out) {
	if (value.size() != 7 || value[0] != '#') {
		return false;
	}
	uint32_t rgb = 0;
	auto [end, error] = std::from_chars(value.data() + 1, value.data() + value.size(), rgb, 16);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8),
	       static_cast<uint8_t>(rgb)};
	return true;
}

static bool ParseInt32(const std::string& value, int32_t& out) {
	int32_t number    = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

static bool g_dualsense_fix         = false;
static bool g_timestamp_scale_given = false;

// The game's title ID: from sce_sys/param.json of a game folder, else the first PPSA/CUSA ID in
// the game path.
static std::string DetectTitleId(const RunOptions& options, const std::string& game_arg) {
	if (!options.app0_dir.empty()) {
		const auto json = options.app0_dir / "sce_sys" / "param.json";
		if (std::FILE* file = Common::File::IsFileExisting(json) ? std::fopen(Common::PathToString(json).c_str(), "rb")
		                                                         : nullptr;
		    file != nullptr) {
			std::string text(65536, '\0');
			text.resize(std::fread(text.data(), 1, text.size(), file));
			std::fclose(file);
			if (const auto key = text.find("\"titleId\""); key != std::string::npos) {
				const auto open  = text.find('"', text.find(':', key) + 1);
				const auto close = open != std::string::npos ? text.find('"', open + 1) : open;
				if (close != std::string::npos) {
					return text.substr(open + 1, close - open - 1);
				}
			}
		}
	}
	for (const char* prefix: {"PPSA", "CUSA"}) {
		for (auto at = game_arg.find(prefix); at != std::string::npos; at = game_arg.find(prefix, at + 1)) {
			if (at + 9 <= game_arg.size() &&
			    std::all_of(game_arg.begin() + static_cast<std::ptrdiff_t>(at + 4),
			                game_arg.begin() + static_cast<std::ptrdiff_t>(at + 9),
			                [](char c) { return c >= '0' && c <= '9'; })) {
				return game_arg.substr(at, 9);
			}
		}
	}
	return {};
}

// Per-title defaults, so a game starts the same way from the launcher, a script or a double click:
// what the tested launch scripts set. Switches already in the environment, and a timestamp scale on
// the command line, are kept.
static void ApplyTitleProfile(RunOptions& options, const std::string& title_id) {
	if (title_id == "PPSA21564" || title_id == "PPSA01325") {
		// ASTRO BOT, ASTRO's PLAYROOM.
		KYTY_SET_DEFAULT_SWITCH("KYTY_COMPUTE_CHAIN", "1");
		// The game advances 1/60 s per frame: cap at 60 fps so light scenes do not run fast.
		KYTY_SET_DEFAULT_SWITCH("KYTY_DEBUG_FRAME_LIMIT_MS", "16.667");
		KYTY_SET_DEFAULT_SWITCH("KYTY_PRESENT_WHEN_READY", "1");
		// The GPU timestamp scale stays at its configured default: the
		// game's own dynamic resolution sees unstretched GPU times, as on the PS5.
		KYTY_INTERNAL_PRINTF("Title profile %s: compute chain, 60 fps limit, present when ready, "
		                     "gpu-timestamp-scale %u\n",
		                     title_id.c_str(), options.config.gpu_timestamp_scale_percent);
	}
	if (title_id == "PPSA01317") {
		// Gran Turismo 7 (2026-10-23). Same as the title: a job-thread loop keeps its array pointer in the
		// red zone ([rsp-0x28]); it read 0 after 11 rounds -> null read at 0x90168a7b8 in the World Map
		// after ~160 s without protection.
		options.config.red_zone_protection_enabled = true;
		KYTY_INTERNAL_PRINTF("Title profile %s: red zone protection\n", title_id.c_str());
	}
	if (title_id == "PPSA23566") {
		// The title (2026-10-18). Job threads keep a pointer in the red zone that the emulator's fault
		// handling overwrites: null write at 0x900e7170d after the first cutscene without protection.
		options.config.red_zone_protection_enabled = true;
		// The timestamp scale stays at its configured value (100 by default).
		// The interim 200 of 2026-10-18 kept the dynamic resolution on one step.
		// A frame of ~29 ms waited for the second vblank (fixed 30 fps): show it when ready
		// (fight: p50 33 -> 29 ms, 31 -> 33.5 fps).
		KYTY_SET_DEFAULT_SWITCH("KYTY_PRESENT_WHEN_READY", "1");
		// Fight with an unchanged picture: 31.4 -> 33.8 fps, GPU 18.2-19.3 -> 15.6-15.8 ms per frame
		// (2026-10-18, runs base3/chain2 with the same pad route).
		KYTY_SET_DEFAULT_SWITCH("KYTY_COMPUTE_CHAIN", "1");
		// The blue haze that took over some fights after 1-2 minutes (fighters dark, fog, blocks; 3
		// of 6 runs) looks like an LDS race between lanes of one wave (KytyPS5 issue 1458 found one in
		// The title with GPU-assisted validation): the first run with the barrier stayed clear, at
		// unchanged fps (lds1, 2026-10-18).
		KYTY_SET_DEFAULT_SWITCH("KYTY_LDS_WAIT_BARRIER", "1");
		// ~380 BDA uploads a frame behind one barrier pair instead of one pair each: in
		// the 40-fps arena buffer copies 6.5 -> 3.0 ms GPU, frame 24.5 -> 20.5 ms (2026-10-20, upb1).
		KYTY_SET_DEFAULT_SWITCH("KYTY_UPLOAD_BATCH", "1");
		// End-of-pipe labels without GPU drains: the fences sit next to compute
		// output and in old texture memory; written around the GPU bytes, the full drains went from
		// 3.5-5.5 ms to ~0.1 ms a frame (clean pictures in every run since, 2026-10-18).
		KYTY_SET_DEFAULT_SWITCH("KYTY_EOP_GPU_LABELS", "2");
		KYTY_SET_DEFAULT_SWITCH("KYTY_EOP_LABEL_MARKERS", "0");
		KYTY_SET_DEFAULT_SWITCH("KYTY_INDIRECT_REGS_BACKING", "1");
		KYTY_SET_DEFAULT_SWITCH("KYTY_EOP_IMAGE_LABELS", "1");
		// Draw preparation: the draw and pipeline chains also hold across context
		// registers written again with the same values (target chain 21 -> 88 % of draws, verified
		// 0 differences), and texel buffers check the whole image set first: 3.31 -> 3.15 us/draw.
		KYTY_SET_DEFAULT_SWITCH("KYTY_CHAIN_CONTENT_HASH", "1");
		KYTY_SET_DEFAULT_SWITCH("KYTY_TEXEL_WHOLE_FIRST", "1");
		// Queue switches (graphics <-> async compute) at waits no longer submit: 61 ->
		// 53 submits and 29.1 -> 28.3 ms a frame in an A/B run (code-reviewed).
		KYTY_SET_DEFAULT_SWITCH("KYTY_SLICE_FLUSH_DEFER", "1");
		// KYTY_ALIAS_SYNC (texture cache alias sync, code-reviewed) is NOT set: hazed in 3 of 12 runs
		// with it on 2026-10-19, ~3 of 11 without it on 2026-10-18. In the career menu it
		// removes the black frames at dynamic-resolution steps (29/420 -> 5/300); a fight
		// measurement (fps, drains) is still missing.
		// Career-menu flicker part 1: a stencil read at a dynamic-resolution step resolves
		// to the depth image's stencil plane instead of stale guest memory (code-reviewed).
		KYTY_SET_DEFAULT_SWITCH("KYTY_STENCIL_READ_ADOPT", "1");
		// Career-menu flicker, the cause of the remaining black frames: retiring the
		// old-extent depth image after a dynamic-resolution step erased the HTile entry it shares with
		// the current depth image; the current one was then re-acquired as cleared and its depth wiped
		// mid-frame (scene black except hair). The entry stays while another image uses it: 13-17 -> 0
		// black pictures (flick2e, ~550 career-menu pictures).
		KYTY_SET_DEFAULT_SWITCH("KYTY_HTILE_META_KEEP", "1");
		// Upstream PR 1743 (fighter select black blocks): HTile clear state owned by the depth surface,
		// kept while any HTile depth image of the surface remains (covers the D32/D32S8 pair with
		// separate HTiles, which META_KEEP's same-HTile rule does not).
		KYTY_SET_DEFAULT_SWITCH("KYTY_HTILE_SURFACE_KEEP", "1");
		// DB_ALPHA_TO_MASK as alpha-to-coverage (upstream 13b291f99, accuracy). The title did not enable it
		// in the title, nightclub fight and post-fight scenes (a2m2: no "first enable" line).
		KYTY_SET_DEFAULT_SWITCH("KYTY_ALPHA_TO_MASK", "1");
		// ~1100 full memory barriers a frame from the game's partial flushes and cache writebacks
		// (upstream 70668fbd9 drops them): the flush points only end the render pass now, which
		// records the deferred draw-write barriers. Bell Centre arena: frame 21.9 -> 19.1 ms, GPU
		// 17.5 -> 15.3 ms.
		KYTY_SET_DEFAULT_SWITCH("KYTY_GLOBAL_BARRIER_LITE", "1");
		// The draw speculator also prepares dispatches' compute stages (DISPATCH_DIRECT 3.84 -> 3.41
		// us in the arena, 0 differences in 865k verified adoptions), and
		// small uploads read the backing first (exact, faster loading).
		KYTY_SET_DEFAULT_SWITCH("KYTY_SPECULATE_DISPATCHES", "1");
		KYTY_SET_DEFAULT_SWITCH("KYTY_BDA_BACKING_FIRST", "1");
		// KYTY_RELEASE_UNBOUND (upstream PR 1809 part) stays OFF: review 10-23 found buffers used
		// only through the BDA page table look unbound and their BDA-written data is lost on
		// release; the refault counter could not count.
		// Thread_Gpu (the command processor, the arena's only limit) and the draw speculator each get
		// a physical core to themselves, sibling idle (host scheduling only): arena -1.3..1.6
		// ms/frame for Thread_Gpu in two pairs, pictures clean. Core numbers fit
		// a one-CCD CPU; core 0 stays with the OS.
		KYTY_SET_DEFAULT_SWITCH("KYTY_GPU_CORE", "1");
		KYTY_SET_DEFAULT_SWITCH("KYTY_SPEC_CORE", "2");
		// The opponent's shorts (cloth): the copy cs f00685dcc53ec93e read its record count through
		// the BDA and got 0 in some frames; validated with the native-CFG address the read is a
		// flattened slot.
		KYTY_SET_DEFAULT_SWITCH("KYTY_SRT_NATIVE_READS", "1");
		// Not scoped: it flattens reads of 6 shaders (f00685dc, 5c88a697, 68c11c51,
		// 959dc755, 99c47b82, cb8ae54f); limited to f00685dc the shorts explode again (scoped1), so
		// the others are part of the cloth chain and need it too.
		// Frame overlap: the frame's end-of-pipe interrupt fires when the CP reaches it, so the game
		// builds the next frame while the GPU finishes this one (fight 36-38.5 -> 44.6-48.8
		// fps). The blue haze is not more frequent with it any more (2026-10-19: 3 of 7 runs hazed
		// with it, 3 of 4 without; the haze is the old race, independent of this switch).
		// KYTY_EOP_IRQ_AT_CP=0 turns it off.
		KYTY_SET_DEFAULT_SWITCH("KYTY_EOP_IRQ_AT_CP", "1");
		// PS5 Pro is on by default in this fork (KYTY_TRINITY_MODE=0: base PS5), with the switches it is
		// tested with. Everything here can be overridden from the environment.
		KYTY_SET_DEFAULT_SWITCH("KYTY_TRINITY_MODE", "1");
		const char* trinity = KYTY_GETENV("KYTY_TRINITY_MODE");
		if (trinity != nullptr && trinity[0] == '1') {
			KYTY_SET_DEFAULT_SWITCH("KYTY_HTILE_LATE_CLEAR", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_SKIP_UNTRANSLATABLE", "1");
			// The ray-traced shadow shader calls hit functions from a table: linked here, skipped at
			// the other call site.
			KYTY_SET_DEFAULT_SWITCH("KYTY_PRO_SWAPPC_SKIP", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_PRO_SWAPPC_LINK", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_DEBUG_LOOP_GUARD", "1000000");
			// GPU-driven ray-tracing chain: indirect sizes on the GPU, command-processor writes
			// without waits, own loop limit for the shaders with data-bounded loops.
			KYTY_SET_DEFAULT_SWITCH("KYTY_INDIRECT_THREADS_GPU", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_WRITE_DATA_BACKED", "1");
			KYTY_SET_DEFAULT_SWITCH(
			    "KYTY_GPU_DRIVEN",
			    "8369f070c43af7cf,32fe50a1e918498b,ac6f8f78cc7d117d,c448f587831bdfe4,4fa7c6c2f0a013f2,"
			    "7a18c7034cd3dc83,52760f45fd9cf53e,8886acba5af0571c,ba78160c39d50f77,5899096c09994d70,"
			    "c393c7631d5779aa,7845ca4c02ae22ad,c86bedeacb4837be,04f19f80a909b1d5,15f2973b84e41c6a,"
			    "17a4b810f2137f7d,fd114615c62d2f26,329f5331d8c2c733,d0f76103e045d1fd,a43eef93bbb8230a,"
			    "1df15270c9e3607b,91fa31099c1e67f7");
			KYTY_SET_DEFAULT_SWITCH("KYTY_DEBUG_LOOP_GUARD_TIGHT",
			                        "65536:17a4b810f2137f7d,d0f76103e045d1fd,a43eef93bbb8230a,"
			                        "1df15270c9e3607b,91fa31099c1e67f7");
			// GPU-result reads deferred while graphics is recorded, partial image reloads after GPU
			// writes, barriers without indirect-argument reads, staging ring in device memory, chained
			// indirect dispatches, labels joined with the upload helper.
			KYTY_SET_DEFAULT_SWITCH("KYTY_DEFER_DIRTY_READS", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_GPU_WRITE_RANGES", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_LATER_ACCESS", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_STAGING_DEVICE", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_INDIRECT_IN_CHAIN", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_LABEL_JOIN_UPLOADS", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_BDA_SINGLE_CROSS_SHADERS", "ba78160c39d50f77");
			KYTY_SET_DEFAULT_SWITCH("KYTY_HALF_UNIFORM", "1");
			KYTY_SET_DEFAULT_SWITCH("KYTY_GPU_DIRTY_TRACK", "1");
			KYTY_INTERNAL_PRINTF("Title profile %s: PS5 Pro switches\n", title_id.c_str());
		}
		KYTY_INTERNAL_PRINTF("Title profile %s: red zone protection, present when ready, compute chain, "
		                     "LDS wait barrier, labels without drains, content chains, native SRT reads, "
		                     "interrupt at CP, gpu-timestamp-scale %u\n",
		                     title_id.c_str(), options.config.gpu_timestamp_scale_percent);
	}
	if (title_id == "PPSA26344") {
		// Ghost of Yotei: heap-indexed textures need the bindless arrays. No shader is skipped (user
		// 2026-10-09): the former skip list (RENDERWORLD clear pair, terrain splat pair) runs.
		options.config.bindless_images_enabled = true;
		// Measured 8.2-8.4 instead of 5.3-5.6 fps, no new image errors (2026-10-07).
		KYTY_SET_DEFAULT_SWITCH("KYTY_COMPUTE_CHAIN", "1");
		KYTY_INTERNAL_PRINTF("Title profile %s: bindless, compute chain\n",
		                     title_id.c_str());
	}
	// Without --dualsense-fix nothing goes to the DualSense: no vibration, no adaptive trigger
	// resistance, and controller speaker sounds play on the PC output.
	if (g_dualsense_fix) {
		::printf("DualSense fix: vibration, adaptive triggers and controller speaker on the DualSense\n");
	} else {
		KYTY_SET_DEFAULT_SWITCH("KYTY_PAD_VIBRATION", "0");
		KYTY_SET_DEFAULT_SWITCH("KYTY_PAD_TRIGGERS", "0");
		KYTY_SET_DEFAULT_SWITCH("KYTY_PAD_SPEAKER", "0");
		::printf("DualSense fix off: no vibration or trigger effects, controller speaker sounds on "
		         "the PC output\n");
	}
}

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (arg == "--rd") {
			options.config.renderdoc_enabled = true;
			continue;
		}

		if (arg == "--fullscreen") {
			options.config.fullscreen_enabled = true;
			continue;
		}

		if (arg == "--hdr") {
			options.config.hdr_enabled = true;
			continue;
		}

		if (arg == "--hide-cursor") {
			options.config.hide_cursor_enabled = true;
			continue;
		}

		if (arg == "--no-splash") {
			options.config.splash_enabled = false;
			continue;
		}

		if (arg == "--vr") {
			options.config.vr_enabled = true;
			continue;
		}

		if (arg == "--amd-cpu") {
			options.config.amd_cpu_enabled = true;
			continue;
		}

		if (arg == "--dualsense-fix") {
			g_dualsense_fix = true;
			continue;
		}

		if (arg == "--playgo-hack") {
			options.config.playgo_hack_enabled = true;
			continue;
		}

		if (arg == "--tessellation") {
			options.config.tessellation_enabled = true;
			continue;
		}

		if (arg == "--bindless") {
			options.config.bindless_images_enabled = true;
			continue;
		}

		if (arg == "--profile") {
			options.config.profiler_enabled = true;
			continue;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (arg == "--redzone") {
			options.config.red_zone_protection_enabled = true;
			continue;
		}
#endif

		if (!arg.starts_with("--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value           = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (Common::File::IsDirectoryExisting(path)) {
				options.app0_dir = path;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::IsSupportedArchive(path) && Common::File::IsFileExisting(path)) {
				const auto root = Common::MakeArchivePath(path);
				if (!Common::File::IsFileExisting(root / "eboot.bin")) {
					::printf("Archive does not contain eboot.bin: %s\n", value.c_str());
					return false;
				}
				options.app0_dir = root;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(path)) {
				options.app0_dir = path.parent_path();

				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}

				options.elf = std::filesystem::path("/app0") / path.filename();
			} else {
				::printf("--game must point to an existing directory, ELF, or archive: %s\n",
				         value.c_str());
				return false;
			}
		} else if (arg == "--guest-exec") {
			options.elf = Common::PathFromUtf8(Common::FixFilenameSlash(value));
			options.config.guest_arguments.assign(argv + i + 1, argv + argc);
			break;
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (!Common::File::IsFileExisting(path)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = path;
		} else if (arg == "--screen-width") {
			if (!ParseUint32(value, options.config.screen_width) ||
			    options.config.screen_width == 0) {
				::printf("invalid screen width: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--screen-height") {
			if (!ParseUint32(value, options.config.screen_height) ||
			    options.config.screen_height == 0) {
				::printf("invalid screen height: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mic") {
			options.config.audio_input_device = value;
		} else if (arg == "--controller-color") {
			Config::ControllerColor color {};
			if (!ParseControllerColor(value, color)) {
				::printf("invalid controller color (expected #RRGGBB): %s\n", value.c_str());
				return false;
			}
			options.config.controller_color = color;
		} else if (arg == "--controller-volume") {
			if (!ParseUint32(value, options.config.controller_speaker_volume) ||
			    options.config.controller_speaker_volume > 100) {
				::printf("invalid controller volume: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--controller-vibration") {
			if (!ParseUint32(value, options.config.controller_vibration_intensity) ||
			    options.config.controller_vibration_intensity > 100) {
				::printf("invalid controller vibration intensity: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--audio-master-volume" || arg == "--audio-main-volume" ||
		           arg == "--audio-music-volume" || arg == "--audio-pad-speaker-volume") {
			auto& target = arg == "--audio-master-volume" ? options.config.audio_master_volume
			               : arg == "--audio-main-volume" ? options.config.audio_main_volume
			               : arg == "--audio-music-volume"
			                   ? options.config.audio_music_volume
			                   : options.config.audio_pad_speaker_main_volume;
			if (!ParseUint32(value, target) || target > Config::MAX_AUDIO_VOLUME) {
				::printf("invalid %s (expected 0-%u): %s\n", arg.c_str(), Config::MAX_AUDIO_VOLUME,
				         value.c_str());
				return false;
			}
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--bda-sync") {
			if (!ParseEnum(value, options.config.bda_sync_mode)) {
				::printf("invalid BDA synchronization mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--gpu") {
			if (!ParseInt32(value, options.config.gpu_index)) {
				::printf("invalid gpu index: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vblank-frequency") {
			if (!ParseUint32(value, options.config.vblank_frequency)) {
				::printf("invalid vblank frequency: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--network-connected") {
			if (!ParseBool(value, options.config.network_connected)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--vulkan-validation") {
			if (!ParseBool(value, options.config.vulkan_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-assisted-validation") {
			if (!ParseBool(value, options.config.gpu_assisted_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-validation") {
			if (!ParseBool(value, options.config.shader_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-precompile") {
			if (!ParseBool(value, options.config.shader_precompile_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-optimization-type") {
			if (!ParseEnum(value, options.config.shader_optimization_type)) {
				::printf("invalid shader optimization type: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = Common::PathFromUtf8(value);
		} else if (arg == "--command-buffer-dump") {
			if (!ParseBool(value, options.config.command_buffer_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
		} else if (arg == "--graphics-debug-dump") {
			if (!ParseBool(value, options.config.graphics_debug_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--spirv-debug-printf") {
			if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--drain-stats") {
			uint32_t interval = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), interval);
			if (error != std::errc {} || end != value.data() + value.size() || interval > 3600) {
				::printf("invalid drain-stats interval: %s\n", value.c_str());
				return false;
			}
			options.config.drain_stats_interval = interval;
		} else if (arg == "--label-flush-interval-us") {
			uint32_t interval = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), interval);
			if (error != std::errc {} || end != value.data() + value.size() || interval > 100000) {
				::printf("invalid label-flush-interval-us: %s\n", value.c_str());
				return false;
			}
			options.config.label_flush_interval_us = interval;
		} else if (arg == "--gpu-frames-ahead") {
			uint32_t frames = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), frames);
			if (error != std::errc {} || end != value.data() + value.size() || frames > 3) {
				::printf("invalid gpu-frames-ahead (0-3): %s\n", value.c_str());
				return false;
			}
			options.config.gpu_frames_ahead = frames;
		} else if (arg == "--dcc-gpu-clear") {
			if (!ParseBool(value, options.config.dcc_gpu_clear_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-mesh-indirect") {
			if (!ParseBool(value, options.config.gpu_mesh_indirect_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-timestamp-scale") {
			uint32_t percent = 0;
			auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), percent);
			// Up to 1000 on the command line (test 2026-10-23: the title's dynamic resolution then
			// sits at its lowest step); the F2 slider and the settings file keep 25-200.
			if (error != std::errc {} || end != value.data() + value.size() || percent < 25 ||
			    percent > 1000) {
				::printf("invalid gpu-timestamp-scale (25-1000): %s\n", value.c_str());
				return false;
			}
			options.config.gpu_timestamp_scale_percent = percent;
			g_timestamp_scale_given = true;
		} else if (arg == "--pipeline-libraries") {
			if (!ParseBool(value, options.config.pipeline_libraries_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--async-pipelines") {
			if (!ParseBool(value, options.config.async_pipelines_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--pipeline-compile-threads") {
			if (!ParseUint32(value, options.config.pipeline_compile_threads) ||
			    options.config.pipeline_compile_threads > 256) {
				::printf("invalid pipeline-compile-threads (0-256): %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--pipeline-compile-threads-replay") {
			if (!ParseUint32(value, options.config.pipeline_compile_threads_replay) ||
			    options.config.pipeline_compile_threads_replay > 256) {
				::printf("invalid pipeline-compile-threads-replay (0-256): %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--defer-dispatches") {
			if (!ParseBool(value, options.config.defer_dispatches_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--dispatch-lookahead") {
			if (!ParseBool(value, options.config.dispatch_lookahead_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--speculative-draws") {
			if (!ParseBool(value, options.config.speculative_draws_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--red-zone-protection") {
			if (!ParseBool(value, options.config.red_zone_protection_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--record-thread") {
			if (!ParseBool(value, options.config.record_thread_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--hardware-buffer-bounds") {
			if (!ParseBool(value, options.config.hardware_buffer_bounds)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--relaxed-readback") {
			if (!ParseBool(value, options.config.relaxed_readback_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--fxaa") {
			if (!ParseBool(value, options.config.fxaa_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--async-submit") {
			if (!ParseBool(value, options.config.async_submit_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-linear-images") {
			if (!ParseBool(value, options.config.readback_linear_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--sync-raw-image-buffers") {
			if (!ParseBool(value, options.config.sync_raw_image_buffers)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--trophy-notifications") {
			if (!ParseBool(value, options.config.trophy_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--skip-notice-screen") {
			if (!ParseBool(value, options.config.skip_notice_screen)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--perf-overlay") {
			if (!ParseBool(value, options.config.perf_overlay_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

#if defined(KYTY_PGO_GENERATE)
// A profile-generating build (-fprofile-generate) writes its counters every 20 s: the emulator
// usually ends without running atexit handlers, which is when the profile runtime would write.
extern "C" int __llvm_profile_write_file(void);

static void StartProfileWriter() {
	std::thread([] {
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(20));
			(void)__llvm_profile_write_file();
		}
	}).detach();
}
#endif

static int Main(int argc, char* argv[]) {
	ConsoleColors::Install();
#if defined(KYTY_PGO_GENERATE)
	StartProfileWriter();
#endif
	// KYTY_DEBUG_UNBUFFERED=1: a redirected console log is complete even when the emulator is
	// killed (stdout to a file is block-buffered otherwise).
	if (KYTY_GETENV("KYTY_DEBUG_UNBUFFERED") != nullptr) {
		std::setvbuf(stdout, nullptr, _IONBF, 0);
	}
	// Offline shader bench (no game, no window): see shaderBench.h.
	if (argc >= 2 && std::string_view(argv[1]) == "--shader-bench") {
		return Libs::Graphics::ShaderBench::Run(argc - 1, argv + 1);
	}
	VirtualMemory::Init();
	InitializeThreads();
	// After the address-space reservations, as its thread's stack must not land in the guest
	// range: the report of an end no fault handler sees (usa-ratt's GT7 fork, round 125).
	Common::LastWords::Install();

	RunOptions options;
	bool       show_help = false;

	std::vector<std::string> settings;
	if (!SettingsFile::LoadArguments(settings)) {
		return 1;
	}
	if (!settings.empty()) {
		::printf("Settings: %s\n", SettingsFile::FileName);
	}

	if (argc < 2 && settings.empty()) {
		PrintUsage();
		return 0;
	}

	// Only a timestamp scale on the command line overrides a title profile's (kyty_settings.ini keeps
	// one for builds without profiles).
	bool command_line_scale = false;
	for (int i = 1; i < argc; i++) {
		command_line_scale = command_line_scale || std::string_view(argv[i]) == "--gpu-timestamp-scale";
	}

	// The settings file's options go first, so the same option on the command line overrides it.
	std::vector<char*> arguments {argv[0]};
	for (auto& setting: settings) {
		arguments.push_back(setting.data());
	}
	arguments.insert(arguments.end(), argv + 1, argv + argc);
	argc = static_cast<int>(arguments.size());
	argv = arguments.data();

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}
	g_timestamp_scale_given = command_line_scale;

	if (show_help) {
		PrintUsage();
		return 0;
	}

	std::string game_arg;
	for (int i = 1; i + 1 < argc; i++) {
		if (std::string_view(argv[i]) == "--game") {
			game_arg = argv[i + 1];
		}
	}
	ApplyTitleProfile(options, DetectTitleId(options, game_arg));
	// An enabled cheat that names a timestamp scale (the title's fixed 1080p) wins over the
	// title profile; the command line still wins over both.
	if (const auto scale = Loader::GamePatch::TimestampScale(options.game_patch);
	    scale != 0 && !g_timestamp_scale_given) {
		options.config.gpu_timestamp_scale_percent = scale;
		KYTY_INTERNAL_PRINTF("Cheat file sets gpu-timestamp-scale %u\n", scale);
	}
	std::fflush(stdout);

	Run(options);

	return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

int wmain(int argc, wchar_t* argv[]) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
