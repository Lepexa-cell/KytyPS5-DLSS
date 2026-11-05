#include "graphics/host_gpu/aftermath.h"

#include "common/switches.h"
#include "graphics/host_gpu/renderer/gpuZones.h"
#include "kytyGitVersion.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#ifdef KYTY_AFTERMATH
#include <vulkan/vulkan.h>

#include "GFSDK_Aftermath.h"
#include "GFSDK_Aftermath_GpuCrashDump.h"
#include "GFSDK_Aftermath_GpuCrashDumpDecoding.h"
#endif

namespace Libs::Graphics::Aftermath {

namespace {

constexpr const char* DumpDirectory   = "_crashdumps";
constexpr const char* ShaderDirectory = "_crashdumps/shaders";

std::mutex g_mutex;
bool       g_enabled = false;
uint32_t   g_dumps   = 0;

std::string TimeStamp() {
	const auto now  = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::tm    local {};
#ifdef _WIN32
	localtime_s(&local, &now);
#else
	localtime_r(&now, &local);
#endif
	char text[32] = {};
	std::strftime(text, sizeof(text), "%Y%m%d-%H%M%S", &local);
	return text;
}

bool WriteWhole(const std::filesystem::path& path, const void* data, size_t size) {
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	if (!file) {
		return false;
	}
	file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
	return static_cast<bool>(file);
}

#ifdef KYTY_AFTERMATH

std::filesystem::path ShaderPath(uint64_t hash) {
	char name[32] = {};
	std::snprintf(name, sizeof(name), "%016llx.spv", static_cast<unsigned long long>(hash));
	return std::filesystem::path(ShaderDirectory) / name;
}

// The decoder asks for the SPIR-V of a shader named in the dump: the copy NoteShaderModule kept.
void GFSDK_AFTERMATH_CALL ShaderLookup(const GFSDK_Aftermath_ShaderBinaryHash* hash,
                                       PFN_GFSDK_Aftermath_SetData set_shader_binary, void*) {
	std::ifstream file(ShaderPath(hash->hash), std::ios::binary);
	if (!file) {
		return;
	}
	std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (!data.empty()) {
		set_shader_binary(data.data(), static_cast<uint32_t>(data.size()));
	}
}

void GFSDK_AFTERMATH_CALL OnCrashDump(const void* dump, uint32_t size, void*) {
	std::scoped_lock lock {g_mutex};
	const auto       base = std::string("kyty-") + TimeStamp() + "-" + std::to_string(++g_dumps);
	const auto       path = std::filesystem::path(DumpDirectory) / (base + ".nv-gpudmp");
	std::error_code  error;
	std::filesystem::create_directories(DumpDirectory, error);
	const bool written = WriteWhole(path, dump, size);
	std::printf("aftermath: GPU crash dump %s (%u bytes) -> %s\n", written ? "written" : "NOT written",
	            size, path.string().c_str());
	// The same decoded to JSON, readable without the GPU debugger: faulting shader, page fault address,
	// resource tracking, checkpoints.
	GFSDK_Aftermath_GpuCrashDump_Decoder decoder = nullptr;
	if (GFSDK_Aftermath_SUCCEED(GFSDK_Aftermath_GpuCrashDump_CreateDecoder(
	        GFSDK_Aftermath_Version_API, dump, size, &decoder))) {
		uint32_t json_size = 0;
		if (GFSDK_Aftermath_SUCCEED(GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
		        decoder, GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO,
		        GFSDK_Aftermath_GpuCrashDumpFormatterFlags_UTF8_OUTPUT, nullptr, ShaderLookup, nullptr,
		        nullptr, &json_size)) &&
		    json_size != 0) {
			std::vector<char> json(json_size);
			if (GFSDK_Aftermath_SUCCEED(
			        GFSDK_Aftermath_GpuCrashDump_GetJSON(decoder, json_size, json.data()))) {
				const auto json_path = std::filesystem::path(DumpDirectory) / (base + ".json");
				// The JSON is NUL-terminated.
				WriteWhole(json_path, json.data(), json_size > 0 ? json_size - 1 : 0);
				std::printf("aftermath: decoded -> %s\n", json_path.string().c_str());
			}
		}
		GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder);
	}
	std::fflush(stdout);
}

// Shader debug information (line tables) the driver generates for each shader: the vendor's GPU debugger
// uses it to map the faulting instruction to the SPIR-V.
void GFSDK_AFTERMATH_CALL OnShaderDebugInfo(const void* info, uint32_t size, void*) {
	GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier {};
	if (!GFSDK_Aftermath_SUCCEED(GFSDK_Aftermath_GetShaderDebugInfoIdentifier(
	        GFSDK_Aftermath_Version_API, info, size, &identifier))) {
		return;
	}
	char name[64] = {};
	std::snprintf(name, sizeof(name), "shader-%016llx-%016llx.nvdbg",
	              static_cast<unsigned long long>(identifier.id[0]),
	              static_cast<unsigned long long>(identifier.id[1]));
	std::scoped_lock lock {g_mutex};
	std::error_code  error;
	std::filesystem::create_directories(DumpDirectory, error);
	WriteWhole(std::filesystem::path(DumpDirectory) / name, info, size);
}

void GFSDK_AFTERMATH_CALL OnDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription add, void*) {
	add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "KytyPS5");
	add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, KYTY_GIT_VERSION);
	add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_UserDefined,
	    "shader cache key " KYTY_SHADER_CACHE_KEY);
}

// The emulator's checkpoints carry the index of a GpuZones::g_checkpoints entry as their marker
// (CommandScheduler::CheckpointThunk): the dump gets the zone, key (shader hash) and details.
void GFSDK_AFTERMATH_CALL OnResolveMarker(const void* marker, uint32_t marker_size, void*,
                                          PFN_GFSDK_Aftermath_ResolveMarker resolve) {
	uint64_t index = 0;
	if (marker_size == 0) {
		index = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(marker));
	} else if (marker_size >= sizeof(uint64_t)) {
		std::memcpy(&index, marker, sizeof(index));
	} else {
		return;
	}
	const auto& mark = GpuZones::g_checkpoints[index % GpuZones::CheckpointRing];
	char        text[192] = {};
	std::snprintf(text, sizeof(text),
	              "kyty checkpoint %llu zone=%u key=%016llx detail=%016llx,%016llx,%016llx",
	              static_cast<unsigned long long>(index), static_cast<uint32_t>(mark.zone),
	              static_cast<unsigned long long>(mark.key),
	              static_cast<unsigned long long>(mark.detail[0]),
	              static_cast<unsigned long long>(mark.detail[1]),
	              static_cast<unsigned long long>(mark.detail[2]));
	resolve(text, static_cast<uint32_t>(std::strlen(text) + 1));
}

#endif

} // namespace

bool Requested() {
	static const bool requested = [] {
		const char* text = KYTY_GETENV("KYTY_AFTERMATH");
		return text != nullptr && std::strcmp(text, "0") != 0;
	}();
	return requested;
}

bool Enabled() {
	return g_enabled;
}

bool EnableCrashDumps() {
	if (!Requested() || g_enabled) {
		return g_enabled;
	}
#ifdef KYTY_AFTERMATH
	std::error_code error;
	std::filesystem::create_directories(ShaderDirectory, error);
	const auto result = GFSDK_Aftermath_EnableGpuCrashDumps(
	    GFSDK_Aftermath_Version_API, GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
	    GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks, OnCrashDump,
	    OnShaderDebugInfo, OnDescription, OnResolveMarker, nullptr);
	g_enabled = GFSDK_Aftermath_SUCCEED(result);
	if (g_enabled) {
		std::printf("aftermath: GPU crash dumps on, written to %s (SPIR-V copies in %s)\n",
		            DumpDirectory, ShaderDirectory);
	} else {
		std::printf("aftermath: GFSDK_Aftermath_EnableGpuCrashDumps failed: 0x%x\n",
		            static_cast<uint32_t>(result));
	}
#else
	std::printf("aftermath: KYTY_AFTERMATH is set, but this build has no Aftermath "
	            "(configure with -DKYTY_AFTERMATH=ON)\n");
#endif
	std::fflush(stdout);
	return g_enabled;
}

void NoteShaderModule(std::span<const uint32_t> spirv) {
	if (!g_enabled || spirv.empty()) {
		return;
	}
#ifdef KYTY_AFTERMATH
	GFSDK_Aftermath_SpirvCode code {};
	code.pData = spirv.data();
	code.size  = static_cast<uint32_t>(spirv.size_bytes());
	GFSDK_Aftermath_ShaderBinaryHash hash {};
	if (!GFSDK_Aftermath_SUCCEED(
	        GFSDK_Aftermath_GetShaderHashSpirv(GFSDK_Aftermath_Version_API, &code, &hash))) {
		return;
	}
	const auto       path = ShaderPath(hash.hash);
	std::scoped_lock lock {g_mutex};
	std::error_code  error;
	if (!std::filesystem::exists(path, error)) {
		WriteWhole(path, spirv.data(), spirv.size_bytes());
	}
#endif
}

void NoteGuestShader(std::span<const uint32_t> spirv, uint64_t guest_hash, const char* stage) {
	if (!g_enabled || spirv.empty()) {
		return;
	}
#ifdef KYTY_AFTERMATH
	GFSDK_Aftermath_SpirvCode code {};
	code.pData = spirv.data();
	code.size  = static_cast<uint32_t>(spirv.size_bytes());
	GFSDK_Aftermath_ShaderBinaryHash hash {};
	if (!GFSDK_Aftermath_SUCCEED(
	        GFSDK_Aftermath_GetShaderHashSpirv(GFSDK_Aftermath_Version_API, &code, &hash))) {
		return;
	}
	std::scoped_lock lock {g_mutex};
	std::error_code  error;
	std::filesystem::create_directories(ShaderDirectory, error);
	std::ofstream file(std::filesystem::path(ShaderDirectory) / "guest.txt", std::ios::app);
	char          line[96] = {};
	std::snprintf(line, sizeof(line), "%016llx %016llx %s\n",
	              static_cast<unsigned long long>(hash.hash),
	              static_cast<unsigned long long>(guest_hash), stage);
	file << line;
#else
	(void)guest_hash;
	(void)stage;
#endif
}

void Disable() {
#ifdef KYTY_AFTERMATH
	if (g_enabled) {
		GFSDK_Aftermath_DisableGpuCrashDumps();
		g_enabled = false;
	}
#endif
}

} // namespace Libs::Graphics::Aftermath
