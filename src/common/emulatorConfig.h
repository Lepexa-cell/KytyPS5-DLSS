#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <array>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class ShaderOptimizationType { None, Size, Performance };

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

enum class BdaSyncMode { Selective, Legacy, SelectiveChecked };

using Keymap = std::vector<std::string>;
using ControllerColor = std::array<uint8_t, 3>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;

constexpr int     MAX_PLAYERS         = 4;
constexpr int32_t PLAYER_USER_ID_STEP = 0x100;
constexpr int32_t MAX_USER_ID =
    std::numeric_limits<int32_t>::max() - (MAX_PLAYERS - 1) * PLAYER_USER_ID_STEP;

constexpr uint32_t MAX_AUDIO_VOLUME         = 200;
// Astro Bot sends a lot of its sound to the DualSense speaker (PADSPK); played on the main output
// it measured up to ~19 dB above the main mix (2026-10-06). The console speaker is small, so it
// is turned down (30% = -10.5 dB) when no DualSense plays it.
constexpr uint32_t DEFAULT_AUDIO_PAD_SPEAKER_MAIN_VOLUME = 30;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
	constexpr int32_t USER_ID_EVERYONE = 0xfe;
	constexpr int32_t USER_ID_SYSTEM   = 0xff;
	return user_id >= 0 && user_id != USER_ID_EVERYONE && user_id != USER_ID_SYSTEM;
}

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	std::string            user_name                   = "Kyty";
	int32_t                user_id                     = DEFAULT_USER_ID;
	std::string            audio_input_device;
	std::optional<ControllerColor> controller_color;
	uint32_t               controller_speaker_volume      = 50;
	uint32_t               controller_vibration_intensity = 100;
	// Host mix levels in percent (0-200, 100 = unity); see libs/audioMix.h.
	uint32_t               audio_master_volume            = 100;
	uint32_t               audio_main_volume              = 100;
	uint32_t               audio_music_volume             = 100;
	uint32_t               audio_pad_speaker_main_volume  = DEFAULT_AUDIO_PAD_SPEAKER_MAIN_VOLUME;
	PresentMode            present_mode                = PresentMode::Mailbox;
	BdaSyncMode            bda_sync_mode                   = BdaSyncMode::Selective;
	int32_t                gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   hdr_enabled                 = false;
	bool                   hide_cursor_enabled         = false;
	bool                   splash_enabled              = true;
	bool                   vr_enabled                  = false;
	bool                   amd_cpu_enabled             = false;
	bool                   network_connected           = false;
	uint32_t               vblank_frequency            = 60;
	uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	bool                   shader_precompile_enabled       = true;
	ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::None;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	LogDirection           printf_direction            = LogDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	bool                   profiler_enabled            = false;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   renderdoc_enabled           = false;
	bool                   readback_linear_images      = false;
	bool                   sync_raw_image_buffers      = false;
	bool                   tessellation_enabled        = false;
	bool                   bindless_images_enabled     = false;
	bool                   trophy_enabled              = true;
	bool                   perf_overlay_enabled        = false;
	bool                   playgo_hack_enabled         = false;
	uint32_t               drain_stats_interval        = 0;
	bool                   dcc_gpu_clear_enabled       = true;
	bool                   async_submit_enabled        = true;
	bool                   gpu_mesh_indirect_enabled   = true;
	uint32_t               gpu_frames_ahead            = 0;
	uint32_t               label_flush_interval_us     = 2000;
	uint32_t               gpu_timestamp_scale_percent = 100;
	bool                   pipeline_libraries_enabled  = true;
	bool                   async_pipelines_enabled     = false;
	uint32_t               pipeline_compile_threads    = 0;
	uint32_t               pipeline_compile_threads_replay = 0;
	bool                   dispatch_lookahead_enabled  = true;
	bool                   defer_dispatches_enabled    = false;
	bool                   relaxed_readback_enabled    = false;
	bool                   fxaa_enabled                = false;
	bool                   speculative_draws_enabled   = true;
	bool                   record_thread_enabled       = true;
	bool                   hardware_buffer_bounds      = true;
	bool                   skip_notice_screen          = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool red_zone_protection_enabled = false;
#endif
	Keymap keymap;
	// Ordinary boots need a non-null argv[0]; --guest-exec supplies the caller's arguments.
	std::vector<std::string> guest_arguments {""};
};

void Load(const ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
const std::string& GetAudioInputDevice();
const std::optional<ControllerColor>& GetControllerColor();
uint32_t GetControllerSpeakerVolume();
uint32_t GetControllerVibrationIntensity();
uint32_t GetAudioMasterVolume();
uint32_t GetAudioMainVolume();
uint32_t GetAudioMusicVolume();
uint32_t GetAudioPadSpeakerOnMainVolume();
PresentMode GetPresentMode();
BdaSyncMode        GetBdaSyncMode();
int32_t GetGpuIndex();
bool     FullscreenEnabled();
bool     HdrEnabled();
bool     HideCursorEnabled();
bool     VrEnabled();
bool     AmdCpuEnabled();
bool     IsNetworkConnected();
uint32_t GetVblankFrequency();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
bool                   ShaderPrecompileEnabled();
ShaderOptimizationType GetShaderOptimizationType();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

bool GraphicsDebugDumpEnabled();

LogDirection          GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool SyncRawImageBuffersEnabled();
bool TessellationEnabled();
bool BindlessImagesEnabled();
bool TrophyEnabled();
bool PerfOverlayEnabled();
bool PlayGoHackEnabled();
// Seconds between GPU wait reports; 0 disables the accounting.
uint32_t GetDrainStatsInterval();
// Apply GPU-written DCC fast clears on the GPU instead of reading the keys back.
bool DccGpuClearEnabled();
// Submit the GPU thread's command buffers from a dedicated queue thread.
bool AsyncSubmitEnabled();
// Build mesh-emulated indirect draws with GPU-written arguments on the GPU.
bool GpuMeshIndirectEnabled();
// Frames the game may build ahead of Thread_Gpu at sceAgcSuspendPoint; 0 waits for idle.
uint32_t GetGpuFramesAhead();
// Minimum time between submits made at plain RELEASE_MEM labels; 0 submits at every idle label.
uint32_t GetLabelFlushIntervalUs();
// Stretch time measured between guest GPU timestamps within a frame, in percent (100 = off).
// Games that size their dynamic resolution from GPU timestamps then leave more GPU headroom;
// below 100 they see less GPU time and pick a higher resolution.
uint32_t GetGpuTimestampScalePercent();
// Changes it while running (the settings panel); takes effect at the next timestamp.
void     SetGpuTimestampScalePercent(uint32_t percent);
// Build new graphics pipelines from separately compiled, cached parts (pipeline libraries) where
// the GPU driver supports it, so a new pipeline stalls for less time.
bool PipelineLibrariesEnabled();
// Changes it while running (the settings panel); applies to pipelines created afterwards.
void SetPipelineLibrariesEnabled(bool enabled);
// Draws whose graphics pipeline is still compiling are skipped instead of waiting for it, so a
// new pipeline never stalls the game; what it draws appears a few frames late. Needs pipeline
// libraries.
bool AsyncPipelinesEnabled();
// Changes it while running (the settings panel).
void SetAsyncPipelinesEnabled(bool enabled);
// Worker threads that build compute pipelines (the journal replay's, look-ahead prefetches and
// dispatches'): the configured count, or by default all hardware threads but two, at least two.
uint32_t PipelineCompileThreads();
// How many of those may build the shader journal's replayed pipelines at a time, at the lowest
// thread priority, so a dispatch's compile always finds a free worker: the configured count, or by
// default half of PipelineCompileThreads, at least one.
uint32_t PipelineCompileReplayThreads();
// Each submission is walked ahead for dispatches of shaders not dispatched yet, so their compute
// pipelines start compiling before the dispatch needs them.
bool DispatchLookaheadEnabled();
// A dispatch whose compute pipeline is still compiling suspends its queue instead of stalling the
// GPU thread; the queue runs the dispatch again once the pipeline is ready, and the other queues
// go on meanwhile.
bool DeferDispatchesEnabled();
// A game thread reading memory the GPU is still writing gets the previous bytes at once instead
// of waiting, as on hardware when the CPU reads before the GPU has written; the new bytes follow
// with the download already under way. Off by default: a value can be a frame old.
bool RelaxedReadbackEnabled();
// Changes it while running (the settings panel).
void SetRelaxedReadbackEnabled(bool enabled);
// FXAA over the presented frame (--fxaa, the F2 menu).
bool FxaaEnabled();
void SetFxaaEnabled(bool enabled);
// A second thread prepares draws' shader resources ahead of the GPU thread, which takes them
// when they are still what it would prepare itself. Faster in scenes with many draws; uses one
// more CPU core.
bool SpeculativeDrawsEnabled();
// Changes it while running (the settings panel).
void SetSpeculativeDrawsEnabled(bool enabled);
// The render thread queues its Vulkan commands for the submit thread, which records and submits
// them. Faster in scenes with many draws; uses one more CPU core. Read at start.
bool RecordThreadEnabled();
// Storage buffer range checks are left to the device where it defines out-of-range dword
// accesses (robustBufferAccess2), instead of being compiled into every shader. Read at start.
bool HardwareBufferBoundsEnabled();
bool SkipNoticeScreen();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();
const std::vector<std::string>& GetGuestArguments();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
