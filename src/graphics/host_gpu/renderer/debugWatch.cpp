#include "graphics/host_gpu/renderer/debug.h"
#include "common/switches.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace Libs::Graphics {

// One write per line, so other threads' output (stdout is unbuffered for logs) does not split it.
[[gnu::format(printf, 1, 2)]] static void PrintLine(const char* format, ...) {
	char    text[2048];
	va_list args;
	va_start(args, format);
	const int length = std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	if (length > 0) {
		std::fwrite(text, 1, std::min<size_t>(static_cast<size_t>(length), sizeof(text) - 1), stdout);
	}
}

// KYTY_DEBUG_WATCH=<hex address>[,<hex address>...] (up to four, 8-byte aligned): hardware
// watchpoints on those guest addresses in every thread. Each code address outside the emulator
// that reads or writes one is printed once ("watch-hit"), with the code bytes around it, to find
// the game code that uses a value (e.g. the GPU timestamps its dynamic resolution measures).
#if defined(_WIN32)

namespace {

struct Hit {
	std::atomic<uint64_t> rip {0};
	uint32_t              slot = 0;
	uint8_t               bytes[64] {};
	uint64_t              regs[16] {};
	uint64_t              values[4] {};
};

std::array<uint64_t, 4> g_watch_addresses {};
// An "x" before an address makes it an execution breakpoint (registers on entry).
std::array<bool, 4>     g_watch_exec {};
uint32_t                g_watch_count = 0;
std::array<Hit, 256>    g_hits;
std::atomic<uint32_t>   g_hit_count {0};
uint64_t                g_module_begin = 0;
uint64_t                g_module_end   = 0;
// KYTY_DEBUG_WATCH_REPEAT=<ms>: record a code address again once that long passed since the last
// record, to follow values over time.
uint64_t                g_repeat_ms = 0;
std::atomic<uint64_t>   g_last_record_ms {0};
// "x<address>@<register>+<hex offset>": the execution breakpoint arms a data watchpoint (in the slot
// after it) at that register plus offset whenever it hits, for objects whose address varies.
std::array<int, 4>      g_arm_register {-1, -1, -1, -1};
std::array<uint64_t, 4> g_arm_offset {};
constexpr const char*   RegisterNames[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                                             "r8",  "r9",  "r10", "r11", "r12", "r13", "r14", "r15"};

uint64_t ContextRegister(const CONTEXT& ctx, int index) {
	const uint64_t regs[16] = {ctx.Rax, ctx.Rcx, ctx.Rdx, ctx.Rbx, ctx.Rsp, ctx.Rbp, ctx.Rsi, ctx.Rdi,
	                           ctx.R8,  ctx.R9,  ctx.R10, ctx.R11, ctx.R12, ctx.R13, ctx.R14, ctx.R15};
	return regs[index];
}

LONG CALLBACK WatchHandler(PEXCEPTION_POINTERS info) {
	if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	auto*      ctx  = info->ContextRecord;
	const auto dr6  = ctx->Dr6;
	if ((dr6 & 0xfu) == 0) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	ctx->Dr6       = 0;
	const auto rip = ctx->Rip;
	// The handler's own reads of the watched values trap again: ignore those.
	thread_local bool in_handler = false;
	if (in_handler) {
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	struct Guard {
		Guard() { in_handler = true; }
		~Guard() { in_handler = false; }
	} guard;
	for (uint32_t i = 0; i < g_watch_count; i++) {
		if (g_watch_exec[i] && (dr6 & (1u << i)) != 0) {
			ctx->EFlags |= 0x10000; // RF: resume past the execution breakpoint.
			if (g_arm_register[i] >= 0 && i + 1 < g_watch_count) {
				// KYTY_DEBUG_WATCH_ARM_PREVIOUS=1: arm the address the previous hit computed, for
				// objects used in turn (the next frame writes the one this frame does not use).
				static const bool previous = KYTY_GETENV("KYTY_DEBUG_WATCH_ARM_PREVIOUS") != nullptr;
				static std::array<uint64_t, 4> last {};
				const auto computed =
				    (ContextRegister(*ctx, g_arm_register[i]) + g_arm_offset[i]) & ~uint64_t {7};
				const auto address = previous ? last[i] : computed;
				last[i]            = computed;
				if (address == 0 || (previous && address == computed)) {
					continue;
				}
				const auto slot = i + 1;
				if (g_watch_addresses[slot] != address) {
					g_watch_addresses[slot] = address;
					PrintLine("watch-arm: slot %u at 0x%" PRIx64 "\n", slot, address);
				}
				// This thread at once; the others at the next ApplyToThreads.
				DWORD64* regs[4] = {&ctx->Dr0, &ctx->Dr1, &ctx->Dr2, &ctx->Dr3};
				*regs[slot] = address;
				ctx->Dr7 |= (uint64_t {1} << (slot * 2)) | (uint64_t {0x3} << (16 + slot * 4)) |
				            (uint64_t {0x2} << (18 + slot * 4));
			}
		}
	}
	auto rip_or_caller = rip;
	if (rip >= g_module_begin && rip < g_module_end) {
		// The emulator's own access (the timestamp write, or a libc copy the game called):
		// with KYTY_DEBUG_WATCH_HOST=1, the first game code address on the stack stands for it.
		static const bool host = KYTY_GETENV("KYTY_DEBUG_WATCH_HOST") != nullptr;
		if (!host) {
			return EXCEPTION_CONTINUE_EXECUTION;
		}
		rip_or_caller = 0;
		const auto* stack = reinterpret_cast<const uint64_t*>(ctx->Rsp);
		for (int i = 0; i < 512 && rip_or_caller == 0; i++) {
			if (stack[i] >= 0x900000000ull && stack[i] < 0x920000000ull) {
				rip_or_caller = stack[i];
			}
		}
		if (rip_or_caller == 0) {
			return EXCEPTION_CONTINUE_EXECUTION;
		}
	}
	const auto rip_key = rip_or_caller;
	const auto now_ms = GetTickCount64();
	if (g_repeat_ms == 0 || now_ms - g_last_record_ms.load(std::memory_order_relaxed) < g_repeat_ms) {
		for (uint32_t i = 0; i < std::min<uint32_t>(g_hit_count.load(), g_hits.size()); i++) {
			if (g_hits[i].rip.load(std::memory_order_relaxed) == rip_key) {
				return EXCEPTION_CONTINUE_EXECUTION;
			}
		}
	}
	g_last_record_ms.store(now_ms, std::memory_order_relaxed);
	const auto index = g_hit_count.fetch_add(1);
	if (index < g_hits.size()) {
		auto& hit = g_hits[index];
		// Bit 4: an emulator access, recorded at the game's return address.
		hit.slot  = static_cast<uint32_t>(dr6 & 0xfu) | (rip_key != rip ? 0x10u : 0u);
		// The trap fires after the access: the instruction ends at rip.
		std::memcpy(hit.bytes, reinterpret_cast<const void*>(rip_key - 48), sizeof(hit.bytes));
		const uint64_t regs[16] = {ctx->Rax, ctx->Rcx, ctx->Rdx, ctx->Rbx, ctx->Rsp, ctx->Rbp,
		                           ctx->Rsi, ctx->Rdi, ctx->R8,  ctx->R9,  ctx->R10, ctx->R11,
		                           ctx->R12, ctx->R13, ctx->R14, ctx->R15};
		std::memcpy(hit.regs, regs, sizeof(regs));
		for (uint32_t i = 0; i < g_watch_count; i++) {
			// An execution breakpoint hits on a function's first instruction: [rsp] is its caller.
			hit.values[i] = g_watch_exec[i]               ? *reinterpret_cast<const uint64_t*>(ctx->Rsp)
			                : g_watch_addresses[i] == 0 ? 0
			                                            : *reinterpret_cast<const uint64_t*>(g_watch_addresses[i]);
		}
		hit.rip.store(rip_key, std::memory_order_release);
	}
	return EXCEPTION_CONTINUE_EXECUTION;
}

void ApplyToThreads() {
	const auto own      = GetCurrentThreadId();
	HANDLE     snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		return;
	}
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == own) {
			continue;
		}
		HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME,
		                           FALSE, entry.th32ThreadID);
		if (thread == nullptr) {
			continue;
		}
		if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
			CONTEXT ctx {};
			ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
			if (GetThreadContext(thread, &ctx)) {
				DWORD64* regs[4] = {&ctx.Dr0, &ctx.Dr1, &ctx.Dr2, &ctx.Dr3};
				uint64_t dr7     = 0;
				for (uint32_t i = 0; i < g_watch_count; i++) {
					*regs[i] = g_watch_addresses[i];
					// Local enable; data: break on read or write (RW=11), 8 bytes (LEN=10);
					// execution: RW=00, LEN=00.
					dr7 |= uint64_t {1} << (i * 2);
					if (!g_watch_exec[i]) {
						dr7 |= (uint64_t {0x3} << (16 + i * 4)) | (uint64_t {0x2} << (18 + i * 4));
					}
				}
				ctx.Dr7 = dr7;
				SetThreadContext(thread, &ctx);
			}
			ResumeThread(thread);
		}
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
}

void PrintHits(uint32_t& printed) {
	const auto count = std::min<uint32_t>(g_hit_count.load(), g_hits.size());
	for (; printed < count; printed++) {
		auto&    hit = g_hits[printed];
		uint64_t rip = 0;
		for (int spin = 0; spin < 1000 && (rip = hit.rip.load(std::memory_order_acquire)) == 0; spin++) {
			std::this_thread::yield();
		}
		std::string bytes;
		for (const auto byte: hit.bytes) {
			char text[4];
			std::snprintf(text, sizeof(text), "%02x", byte);
			bytes += text;
		}
		PrintLine("watch-hit: rip=0x%016" PRIx64 " dr6=0x%x bytes(rip-48..rip+16)=%s\n", rip,
		            hit.slot, bytes.c_str());
		static constexpr const char* Names[16] = {"rax", "rcx", "rdx", "rbx", "rsp", "rbp",
		                                          "rsi", "rdi", "r8",  "r9",  "r10", "r11",
		                                          "r12", "r13", "r14", "r15"};
		std::string regs;
		for (uint32_t i = 0; i < 16; i++) {
			char text[48];
			std::snprintf(text, sizeof(text), " %s=%" PRIx64, Names[i], hit.regs[i]);
			regs += text;
		}
		PrintLine("watch-regs:%s\n", regs.c_str());
		std::string values;
		for (uint32_t i = 0; i < g_watch_count; i++) {
			char text[64];
			if (!g_watch_exec[i]) {
				std::snprintf(text, sizeof(text), " [%" PRIx64 "]=%" PRIx64, g_watch_addresses[i],
				              hit.values[i]);
				values += text;
			} else if ((hit.slot & (1u << i)) != 0) {
				std::snprintf(text, sizeof(text), " caller=%" PRIx64, hit.values[i]);
				values += text;
			}
		}
		if (!values.empty()) {
			PrintLine("watch-values:%s\n", values.c_str());
		}
	}
	std::fflush(stdout);
}

} // namespace

// KYTY_DEBUG_DUMP_GUEST=<hex address>:<hex size>[,...] writes those guest ranges every 3 s to
// _guestdump_<n>_<range>.bin in the working directory and prints "guest-dump <n>", to diff a
// game's state between moments (e.g. two render resolutions, see KYTY_DEBUG_RESOLUTION).
static void DebugDumpGuestTick() {
	static const std::vector<std::pair<uint64_t, uint64_t>> ranges = [] {
		std::vector<std::pair<uint64_t, uint64_t>> list;
		const char* text = KYTY_GETENV("KYTY_DEBUG_DUMP_GUEST");
		for (const char* p = text; p != nullptr && *p != '\0';) {
			char*      end     = nullptr;
			const auto address = std::strtoull(p, &end, 16);
			if (end == p || *end != ':') {
				break;
			}
			const char* size_text = end + 1;
			const auto  size      = std::strtoull(size_text, &end, 16);
			list.emplace_back(address, size);
			p = *end == ',' ? end + 1 : end;
		}
		return list;
	}();
	if (ranges.empty()) [[likely]] {
		return;
	}
	static auto     last  = std::chrono::steady_clock::now();
	static uint32_t count = 0;
	const auto      now   = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(3) || count >= 200) {
		return;
	}
	last = now;
	for (size_t i = 0; i < ranges.size(); i++) {
		char name[64];
		std::snprintf(name, sizeof(name), "_guestdump_%03u_%zu.bin", count, i);
		if (std::FILE* file = std::fopen(name, "wb"); file != nullptr) {
			std::fwrite(reinterpret_cast<const void*>(ranges[i].first), 1, ranges[i].second, file);
			std::fclose(file);
		}
	}
	PrintLine("guest-dump %u\n", count++);
	std::fflush(stdout);
}

// KYTY_DEBUG_FIND_GUEST=<hex bytes>[,<hex bytes>...] searches all committed memory between 4 GB and
// 2 TB (the guest's mappings; the emulator's own image is skipped) for those byte patterns at 4-byte
// aligned addresses, KYTY_DEBUG_FIND_GUEST_AT=<s>[,<s>...] seconds after the first frame (default
// 20), and prints each hit ("guest-find"), to locate where a game keeps a value (e.g. its render
// size) before watching it with KYTY_DEBUG_WATCH.
static void DebugFindGuestTick() {
	static const std::vector<std::vector<uint8_t>> patterns = [] {
		std::vector<std::vector<uint8_t>> list;
		const char*                       text = KYTY_GETENV("KYTY_DEBUG_FIND_GUEST");
		for (const char* p = text; p != nullptr && *p != '\0';) {
			std::vector<uint8_t> bytes;
			while (std::isxdigit(static_cast<unsigned char>(p[0])) != 0 &&
			       std::isxdigit(static_cast<unsigned char>(p[1])) != 0) {
				const char hex[3] = {p[0], p[1], '\0'};
				bytes.push_back(static_cast<uint8_t>(std::strtoul(hex, nullptr, 16)));
				p += 2;
			}
			if (bytes.size() >= 4) {
				list.push_back(std::move(bytes));
			}
			if (*p != ',') {
				break;
			}
			p++;
		}
		return list;
	}();
	if (patterns.empty()) [[likely]] {
		return;
	}
	static std::once_flag started;
	std::call_once(started, [] {
		std::vector<uint32_t> times;
		const char*           text = KYTY_GETENV("KYTY_DEBUG_FIND_GUEST_AT");
		for (const char* p = text; p != nullptr && *p != '\0';) {
			char* end = nullptr;
			times.push_back(static_cast<uint32_t>(std::strtoul(p, &end, 10)));
			p = (end != p && *end == ',') ? end + 1 : "";
		}
		if (times.empty()) {
			times.push_back(20);
		}
		std::thread([times] {
			const auto start  = std::chrono::steady_clock::now();
			auto*      module = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
			for (const auto at: times) {
				std::this_thread::sleep_until(start + std::chrono::seconds(at));
				std::vector<uint8_t> chunk(uint64_t {1} << 20u);
				uint32_t             hits = 0;
				for (uint64_t address = uint64_t {1} << 32u; address < (uint64_t {1} << 41u);) {
					MEMORY_BASIC_INFORMATION info {};
					if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) {
						break;
					}
					const auto base = reinterpret_cast<uint64_t>(info.BaseAddress);
					const auto next = base + info.RegionSize;
					const bool readable =
					    info.State == MEM_COMMIT && info.Type != MEM_IMAGE &&
					    (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0 &&
					    (info.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ |
					                     PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY)) != 0 &&
					    info.AllocationBase != module;
					for (uint64_t at_address = base; readable && at_address < next && hits < 400;
					     at_address += chunk.size()) {
						const auto size  = std::min<uint64_t>(chunk.size(), next - at_address);
						SIZE_T     moved = 0;
						if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at_address),
						                      chunk.data(), size, &moved) == 0) {
							continue;
						}
						for (size_t pi = 0; pi < patterns.size(); pi++) {
							const auto& pattern = patterns[pi];
							for (size_t o = 0; o + pattern.size() <= moved; o += 4) {
								if (std::memcmp(chunk.data() + o, pattern.data(), pattern.size()) == 0 &&
								    hits++ < 400) {
									PrintLine("guest-find t=%us pattern=%zu 0x%016" PRIx64
									            " region=0x%" PRIx64 "+0x%" PRIx64 " prot=0x%lx\n",
									            at, pi, at_address + o, base, next - base, info.Protect);
								}
							}
						}
					}
					address = next;
				}
				PrintLine("guest-find t=%us done, %u hits\n", at, hits);
				std::fflush(stdout);
			}
		}).detach();
	});
}

void DebugWatchTick() {
	DebugDumpGuestTick();
	DebugFindGuestTick();
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_WATCH");
		if (text == nullptr) {
			return false;
		}
		for (const char* p = text; *p != '\0' && g_watch_count < g_watch_addresses.size();) {
			const bool exec = *p == 'x';
			p += exec ? 1 : 0;
			char*      end   = nullptr;
			const auto value = std::strtoull(p, &end, 16);
			if (end == p) {
				break;
			}
			g_watch_exec[g_watch_count]        = exec;
			g_watch_addresses[g_watch_count++] = exec ? value : value & ~uint64_t {7};
			if (exec && *end == '@' && g_watch_count < g_watch_addresses.size()) {
				const char* name = end + 1;
				for (int r = 0; r < 16; r++) {
					const auto length = std::strlen(RegisterNames[r]);
					if (std::strncmp(name, RegisterNames[r], length) == 0 && name[length] == '+') {
						g_arm_register[g_watch_count - 1] = r;
						g_arm_offset[g_watch_count - 1] = std::strtoull(name + length + 1, &end, 16);
						// The armed data slot, set when the breakpoint first hits.
						g_watch_exec[g_watch_count]        = false;
						g_watch_addresses[g_watch_count++] = 0;
						break;
					}
				}
			}
			p = *end == ',' ? end + 1 : end;
		}
		if (const char* repeat = KYTY_GETENV("KYTY_DEBUG_WATCH_REPEAT"); repeat != nullptr) {
			g_repeat_ms = std::strtoull(repeat, nullptr, 10);
		}
		return g_watch_count != 0;
	}();
	if (!enabled) [[likely]] {
		return;
	}
	static std::once_flag started;
	std::call_once(started, [] {
		auto* module   = reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr));
		const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(
		    module + reinterpret_cast<const IMAGE_DOS_HEADER*>(module)->e_lfanew);
		g_module_begin = reinterpret_cast<uint64_t>(module);
		g_module_end   = g_module_begin + nt->OptionalHeader.SizeOfImage;
		AddVectoredExceptionHandler(1, WatchHandler);
		for (uint32_t i = 0; i < g_watch_count; i++) {
			PrintLine("watch: 0x%016" PRIx64 "\n", g_watch_addresses[i]);
		}
		std::fflush(stdout);
		std::thread([] {
			uint32_t printed = 0;
			for (;;) {
				ApplyToThreads();
				PrintHits(printed);
				std::this_thread::sleep_for(std::chrono::seconds(2));
			}
		}).detach();
	});
}

#else

void DebugWatchTick() {}

#endif

} // namespace Libs::Graphics
