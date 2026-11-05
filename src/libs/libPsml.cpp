#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/pssr.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // VirtualQuery (Psml_mfsr2 logging)
#endif

namespace Libs {

namespace LibPsml {

LIB_VERSION("Psml", 1, "Psml", 1, 1);

namespace Psml {

constexpr int PSML_ERROR_NOT_INITIALIZED = static_cast<int>(0x8a810001u);
constexpr int PSML_ERROR_INVALID_OBJECT  = static_cast<int>(0x8a810005u);
constexpr int PSML_ERROR_INVALID_POINTER = static_cast<int>(0x8a810009u);
constexpr int PSML_ERROR_INVALID_VALUE   = static_cast<int>(0x8a81000du);
constexpr int PSML_ERROR_NULL_OBJECT     = static_cast<int>(0x8a810014u);

constexpr uint32_t SHARED_RESOURCES_MAGIC = 0xa9c4;
constexpr uint32_t CONTEXT_MAGIC          = 0x9231;

constexpr uint64_t MAIN_MEMORY_BLOCK_SIZE = 0x200000;
constexpr uint64_t MAIN_MEMORY_ALIGNMENT  = 0x200000;
constexpr uint64_t EXTRA_VA_BYTES         = 0x600000;

struct MainMemoryRequirements {
	uint64_t block_size;
	uint64_t alignment;
	uint64_t block_count;
};

struct MainMemoryParameters {
	uint32_t type;
	uint32_t reserved;
};

struct DirectMemoryBlock {
	uint64_t address;
	uint64_t size;
};

struct SharedResourcesInitParameters {
	uint32_t                 type;
	uint32_t                 reserved;
	const DirectMemoryBlock* blocks;
	uint64_t                 block_count;
	uint64_t                 virtual_address_start;
};

static_assert(sizeof(MainMemoryRequirements) == 24);
static_assert(sizeof(DirectMemoryBlock) == 16);

static bool                 g_initialized      = false;
static uint32_t             g_supported_modes  = 0x3;
static std::atomic<int32_t> g_shared_ref_count = 0;

static uint64_t BaseVaBytes(uint32_t type) {
	// The native implementation indexes this table before adding one 6 MiB range.
	switch (type) {
		case 0: return 0x06200000;
		case 1: return 0x18200000;
		case 2: return 0x12200000;
		default: return 0;
	}
}

static bool IsSupportedType(uint32_t type) {
	switch (type) {
		case 0: return true;
		case 1: return (g_supported_modes & 0x1u) != 0;
		case 2: return (g_supported_modes & 0x2u) != 0;
		default: return false;
	}
}

static uint64_t RequiredBlockCount(uint32_t type) {
	const auto base = BaseVaBytes(type);
	if (base == 0) {
		return 0;
	}
	return (base + EXTRA_VA_BYTES) / MAIN_MEMORY_BLOCK_SIZE;
}

static int CheckInitialized() {
	return (g_initialized ? OK : PSML_ERROR_NOT_INITIALIZED);
}

static void MarkSharedResources(void* resources, const SharedResourcesInitParameters& params) {
	auto* bytes = static_cast<uint8_t*>(resources);

	*reinterpret_cast<uint32_t*>(bytes)                        = SHARED_RESOURCES_MAGIC;
	*reinterpret_cast<const DirectMemoryBlock**>(bytes + 0x08) = params.blocks;
	*reinterpret_cast<uint64_t*>(bytes + 0x18)                 = params.block_count;
	*reinterpret_cast<uint32_t*>(bytes + 0x20)                 = params.type;
	*reinterpret_cast<uint64_t*>(bytes + 0x28)                 = params.virtual_address_start;
}

static bool HasKnownObjectMagic(const void* object) {
	if (object == nullptr) {
		return false;
	}
	const auto magic = *static_cast<const uint32_t*>(object);
	return magic == SHARED_RESOURCES_MAGIC || magic == CONTEXT_MAGIC;
}

static int KYTY_SYSV_ABI PsmlInitialize() {
	PRINT_NAME();

	g_initialized      = true;
	g_supported_modes  = 0x3;
	g_shared_ref_count = 0;

	return OK;
}

static int KYTY_SYSV_ABI PsmlGetMainMemoryRequirements(MainMemoryRequirements*     out,
                                                       const MainMemoryParameters* params) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (out == nullptr || params == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}
	if (!IsSupportedType(params->type)) {
		return PSML_ERROR_INVALID_VALUE;
	}

	out->block_size  = MAIN_MEMORY_BLOCK_SIZE;
	out->alignment   = MAIN_MEMORY_ALIGNMENT;
	out->block_count = RequiredBlockCount(params->type);

	return OK;
}

static int KYTY_SYSV_ABI
PsmlSharedResourcesInitialize(void* resources, const SharedResourcesInitParameters* params) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (resources == nullptr || params == nullptr || params->blocks == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}
	if (params->reserved != 0 || !IsSupportedType(params->type) ||
	    params->block_count < RequiredBlockCount(params->type)) {
		return PSML_ERROR_INVALID_VALUE;
	}

	MarkSharedResources(resources, *params);
	g_shared_ref_count.fetch_add(1, std::memory_order_relaxed);

	return OK;
}

static int KYTY_SYSV_ABI PsmlSharedResourcesFinalize(void* resources) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (resources == nullptr) {
		return PSML_ERROR_NULL_OBJECT;
	}

	g_shared_ref_count.fetch_sub(1, std::memory_order_relaxed);

	return OK;
}

static int KYTY_SYSV_ABI PsmlGetContextMemoryRequirements(MainMemoryRequirements* out,
                                                          uint64_t /*params*/) {
	PRINT_NAME();
	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (out == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}

	out->block_size  = MAIN_MEMORY_BLOCK_SIZE;
	out->alignment   = MAIN_MEMORY_ALIGNMENT;
	out->block_count = 1;

	return OK;
}

static int KYTY_SYSV_ABI PsmlContextInitialize(void* context, const void* params) {
	PRINT_NAME();
	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (context == nullptr || params == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}

	auto* bytes            = static_cast<uint8_t*>(context);
	auto* params_bytes     = static_cast<const uint8_t*>(params);
	auto* shared_resources = *reinterpret_cast<void* const*>(params_bytes + 0x08);
	if (shared_resources == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}

	*reinterpret_cast<uint32_t*>(bytes)        = CONTEXT_MAGIC;
	*reinterpret_cast<void**>(bytes + 0x360)   = shared_resources;
	*reinterpret_cast<uint8_t*>(bytes + 0x368) = 0;

	return OK;
}

static int KYTY_SYSV_ABI PsmlContextFinalize(void* context) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (!HasKnownObjectMagic(context)) {
		return PSML_ERROR_INVALID_OBJECT;
	}

	*static_cast<uint32_t*>(context) = 0;

	return OK;
}

static int KYTY_SYSV_ABI PsmlGetWorkAreaSize(const void* object, uint32_t* out_size) {
	PRINT_NAME();
	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (!HasKnownObjectMagic(object)) {
		return PSML_ERROR_INVALID_OBJECT;
	}
	if (out_size == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}

	*out_size = 0x600;

	return OK;
}

static int KYTY_SYSV_ABI PsmlDispatch(void* context, uint64_t command, uint64_t params) {
	PRINT_NAME();
	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (!HasKnownObjectMagic(context)) {
		return PSML_ERROR_INVALID_OBJECT;
	}
	if (command == 0 || params == 0) {
		return PSML_ERROR_INVALID_POINTER;
	}

	return OK;
}

static int KYTY_SYSV_ABI PsmlGetProgress(const void* object, float* out_progress,
                                         uint32_t /*frame_index*/) {
	PRINT_NAME();
	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (!HasKnownObjectMagic(object)) {
		return PSML_ERROR_INVALID_OBJECT;
	}
	if (out_progress == nullptr) {
		return PSML_ERROR_INVALID_POINTER;
	}

	*out_progress = 0.0f;

	return OK;
}

static int KYTY_SYSV_ABI PsmlRequestCapture(const void* object) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (!HasKnownObjectMagic(object)) {
		return PSML_ERROR_INVALID_OBJECT;
	}

	return OK;
}

static int KYTY_SYSV_ABI PsmlValidateObject(const void* object) {
	PRINT_NAME();

	const auto init_error = CheckInitialized();
	if (init_error != OK) {
		return init_error;
	}
	if (!HasKnownObjectMagic(object)) {
		return PSML_ERROR_INVALID_OBJECT;
	}

	return OK;
}

} // namespace Psml

LIB_DEFINE(InitPsml_1) {
	LIB_FUNC("3WVD91e12ZQ", Psml::PsmlInitialize);
	LIB_FUNC("+2KpvixvL6E", Psml::PsmlGetMainMemoryRequirements);
	LIB_FUNC("eWoKNeB6V-k", Psml::PsmlSharedResourcesInitialize);
	LIB_FUNC("jEevBXmagOQ", Psml::PsmlSharedResourcesFinalize);
	LIB_FUNC("EO9YQXmJEN8", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("IMlj247LdTo", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("sjTyYheKTrU", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("ArakEpzsZo0", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("VGjrQa-WqdU", Psml::PsmlGetContextMemoryRequirements);
	LIB_FUNC("2ecEbQaf9VU", Psml::PsmlContextInitialize);
	LIB_FUNC("vUk2pWMx3KQ", Psml::PsmlContextInitialize);
	LIB_FUNC("gxv3i+MTEzU", Psml::PsmlContextInitialize);
	LIB_FUNC("JaLBe0P3jSU", Psml::PsmlContextFinalize);
	LIB_FUNC("AHalTX9wFZY", Psml::PsmlGetWorkAreaSize);
	LIB_FUNC("RUNLFro+qok", Psml::PsmlDispatch);
	LIB_FUNC("GHna9-DvnUk", Psml::PsmlGetProgress);
	LIB_FUNC("GJY0MvuTcs8", Psml::PsmlRequestCapture);
	LIB_FUNC("LXq+6mIxpCw", Psml::PsmlValidateObject);
	LIB_FUNC("FSGaTQze0UY", Psml::PsmlValidateObject);
}

} // namespace LibPsml

// PSSR2 ("Multi-Frame Super Resolution 2", library Psml_mfsr2 in module Psml). The title calls three of
// its functions in PS5 Pro mode (KYTY_TRINITY_MODE=1). Their names and parameters are unknown: these
// placeholders log the first six arguments of the first calls and return 0, like the unresolved
// import stubs did (PS5 Pro experiment 2026-10-29).
namespace LibPsmlMfsr2 {

LIB_VERSION("Psml_mfsr2", 1, "Psml", 1, 1);

// Whether [address, address + size) is committed readable memory (pointer slots of a request).
static bool IsReadable(uint64_t address, uint64_t size) {
#ifdef _WIN32
	MEMORY_BASIC_INFORMATION info {};
	return address >= 0x10000 &&
	       VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) != 0 &&
	       info.State == MEM_COMMIT &&
	       (info.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)) != 0 &&
	       address + size <= reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
#else
	return address >= 0x10000 && size != 0;
#endif
}

template <int N>
static int KYTY_SYSV_ABI Mfsr2Unknown(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                                      uint64_t a5) {
	static constexpr const char* NAMES[] = {"1ic5q-kdOsc", "5z2gBlqxJ+0", "o+NM86gEwFE", "gMduXCLYrNg",
	                                        "0GAw7SmkwII", "m9JLPc3wOQw", "ZLL31lzzxr4", "lrJwpLjKXRc"};
	static std::atomic_uint32_t  calls {0};
	if (calls.fetch_add(1, std::memory_order_relaxed) < 4) {
		std::printf("Psml_mfsr2 %s(0x%llx, 0x%llx, 0x%llx, ...)\n", NAMES[N], static_cast<unsigned long long>(a0),
		            static_cast<unsigned long long>(a1), static_cast<unsigned long long>(a2));
		std::fflush(stdout);
	}
	// Read from the title's code (Ghidra, FUN_903dbad10 / FUN_903dc9b10 / FUN_903dca010): the imports follow
	// the PSSR1 pattern: o+NM86gEwFE() = initialize; 1ic5q-kdOsc(out, params) = main memory requirements;
	// gMduXCLYrNg(resources, blocks) = shared resources initialize; 5z2gBlqxJ+0(out, params) = context
	// memory requirements; 0GAw7SmkwII(context, blocks) = context initialize. Both requirement calls
	// fill out = {block size, alignment, block count} (the game allocates size * count with that
	// alignment); params + 0x18 = output width | height << 16 (16 bit each). The block counts are
	// guesses (64 / 32 MiB in 2 MiB blocks); the memory stays unused.
	// lrJwpLjKXRc(?, writer, params, cursor, ...) = dispatch: the game reserved writer[1] dwords at
	// writer[0] (= cursor) in its command buffer, as for PSSR1 (ZLL31lzzxr4 before it = dispatch
	// size). Until PSSR2 is emulated, the space becomes one PM4 NOP: the upscale is left out, and the
	// command processor no longer reads stale words there (pro5/pro6: "unknown op 0x228").
	// ZLL31lzzxr4(?, size, params, ...): the dispatch size. The game reads *size (a dword; 7 before
	// any write) and reserves that many dwords for lrJwpLjKXRc; the R_PSSR packet needs
	// 1 + Pssr::RequestDwords.
	constexpr uint32_t pssr_packet_dwords = 1 + Graphics::Pssr::RequestDwords;
	if constexpr (N == 6) {
		if (a1 != 0) {
			*reinterpret_cast<uint32_t*>(a1) = pssr_packet_dwords;
		}
	}
	if constexpr (N == 7) {
		if (a1 != 0) {
			const auto* writer = reinterpret_cast<const uint64_t*>(a1);
			auto*       cursor = reinterpret_cast<uint32_t*>(writer[0]);
			const auto  dwords = static_cast<uint32_t>(writer[1] & 0xffffffffu);
			if (cursor != nullptr && dwords >= pssr_packet_dwords && dwords <= 0x4000 && a2 != 0) {
				// One IT_NOP/R_PSSR packet over the whole space: a copy of the request and the
				// T# behind its first 5 pointer slots (graphics/guest_gpu/pssr.h).
				Graphics::Pssr::Request request {};
				request.magic          = Graphics::Pssr::Magic;
				request.version        = Graphics::Pssr::Version;
				request.params_address = a2;
				std::memcpy(request.params, reinterpret_cast<const void*>(a2), sizeof(request.params));
				for (uint32_t slot = 0; slot < 5; slot++) {
					const auto descriptor = reinterpret_cast<const uint64_t*>(a2)[slot];
					if (IsReadable(descriptor, 32)) {
						std::memcpy(request.textures[slot], reinterpret_cast<const void*>(descriptor), 32);
						request.texture_mask |= 1u << slot;
					}
				}
				namespace Pm4 = Graphics::Pm4; // used by KYTY_PM4
				cursor[0] = KYTY_PM4(dwords, Pm4::IT_NOP, Pm4::R_PSSR);
				std::memcpy(cursor + 1, &request, sizeof(request));
				std::memset(cursor + pssr_packet_dwords, 0, (dwords - pssr_packet_dwords) * sizeof(uint32_t));
				static std::atomic_uint32_t logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
					std::printf("Psml_mfsr2: PSSR2 request 0x%llx slots=0x%02x into %u dwords\n",
					            static_cast<unsigned long long>(a2), request.texture_mask, dwords);
					std::fflush(stdout);
				}
			} else if (cursor != nullptr && dwords >= 2 && dwords <= 0x4000) {
				cursor[0] = 0xC0001000u | ((dwords - 2u) << 16u); // type 3 NOP, dwords - 1 body words
			}
		}
	}
	if constexpr (N == 0 || N == 1) {
		if (a0 != 0) {
			auto* out = reinterpret_cast<uint64_t*>(a0);
			out[0]    = 0x200000;
			out[1]    = 0x200000;
			out[2]    = N == 0 ? 32 : 16;
		}
	}
	return 0;
}

LIB_DEFINE(InitPsmlMfsr2_1) {
	LIB_FUNC("1ic5q-kdOsc", Mfsr2Unknown<0>);
	LIB_FUNC("5z2gBlqxJ+0", Mfsr2Unknown<1>);
	LIB_FUNC("o+NM86gEwFE", Mfsr2Unknown<2>);
	LIB_FUNC("gMduXCLYrNg", Mfsr2Unknown<3>);
	LIB_FUNC("0GAw7SmkwII", Mfsr2Unknown<4>);
	// Called once the contexts exist (per frame?): still unknown, logged.
	LIB_FUNC("m9JLPc3wOQw", Mfsr2Unknown<5>);
	LIB_FUNC("ZLL31lzzxr4", Mfsr2Unknown<6>);
	LIB_FUNC("lrJwpLjKXRc", Mfsr2Unknown<7>);
}

} // namespace LibPsmlMfsr2

} // namespace Libs
