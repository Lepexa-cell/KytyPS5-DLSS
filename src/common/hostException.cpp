#include "common/hostException.h"

#include "common/lastWords.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <exception>
#include <string_view>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep

#include <cstdint>
#include <cstdlib>
#include <tlhelp32.h>

#include <algorithm>
#include <string>
#include <vector>
#include <fmt/format.h>
#else
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <initializer_list>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/ucontext.h>
#else
#include <ucontext.h> // IWYU pragma: keep
#endif
#endif

// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <excpt.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <wtypes.h>

namespace Common::HostException {

#if !defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX

// macOS uses the same POSIX platform setting and needs a signal stack too.
class ThreadSignalStack {
public:
	ThreadSignalStack() {
		const auto page_size = static_cast<size_t>(::getpagesize());
		const auto stack_size =
		    (std::max<size_t>(64 * 1024, MINSIGSTKSZ) + page_size - 1) & ~(page_size - 1);
		if (::posix_memalign(&m_memory, page_size, stack_size) != 0) {
			return;
		}

		stack_t stack {};
		stack.ss_sp   = m_memory;
		stack.ss_size = stack_size;
		if (::sigaltstack(&stack, &m_previous) != 0) {
			std::free(m_memory);
			m_memory = nullptr;
		}
	}

	~ThreadSignalStack() {
		if (m_memory != nullptr && ::sigaltstack(&m_previous, nullptr) == 0) {
			std::free(m_memory);
		}
	}

	[[nodiscard]] bool IsInitialized() const { return m_memory != nullptr; }

	KYTY_CLASS_NO_COPY(ThreadSignalStack)

private:
	void*   m_memory = nullptr;
	stack_t m_previous {};
};

bool InitializeThreadSignalStack() {
	// Keep fault handling off guest stacks, which GPU tracking can make read-only.
	thread_local ThreadSignalStack signal_stack;
	return signal_stack.IsInitialized();
}

#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// KYTY_DEBUG_HW_WATCH=<address>[,<address>...] (hex, up to four): hardware write breakpoints on
// those four-byte words in every thread. Each CPU write (guest code or the emulator itself, also
// one that stores the value already there) prints the code address after the writing instruction.
static uint64_t g_hw_watch[4] {};
static int      g_hw_watch_count = 0;
// KYTY_DEBUG_HW_EXEC=<address>[,<address>...] (hex): execution breakpoints instead, after the
// write watches (four slots in all). Each hit prints and dumps like a write hit, before the
// instruction runs; the first KYTY_DEBUG_HW_EXEC_LIMIT (default 40) hits are reported.
static bool     g_hw_watch_exec[4] {};
// Bumped when a watch is added at run time (KYTY_DEBUG_HW_WATCH_FOLLOW): re-arms every thread.
static std::atomic<uint32_t> g_hw_watch_generation {0};

// KYTY_DEBUG_HW_WATCH_PTR=<pointer address>:<offset>[:<offset>] (hex): watches the words at
// *pointer + offset once the pointer is set, and again whenever it changes (an object that is
// allocated at a different address in every run).
static uint64_t g_hw_watch_ptr          = 0;
static uint64_t g_hw_watch_ptr_value    = 0;
static uint64_t g_hw_watch_ptr_offsets[2] {};
static int      g_hw_watch_ptr_count    = 0;
static int      g_hw_watch_ptr_first    = 0;

static void PollHwWatchPointer() {
	if (g_hw_watch_ptr == 0) {
		return;
	}
	uint64_t value = 0;
	SIZE_T   got   = 0;
	if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(g_hw_watch_ptr), &value,
	                      sizeof(value), &got) == 0 ||
	    got != sizeof(value) || value == 0 || value == g_hw_watch_ptr_value) {
		return;
	}
	g_hw_watch_ptr_value = value;
	for (int i = 0; i < g_hw_watch_ptr_count; i++) {
		g_hw_watch[g_hw_watch_ptr_first + i] = value + g_hw_watch_ptr_offsets[i];
	}
	g_hw_watch_count = g_hw_watch_ptr_first + g_hw_watch_ptr_count;
	std::printf("hw-watch: pointer %llx = %llx, watching +%llx%s\n",
	            static_cast<unsigned long long>(g_hw_watch_ptr), static_cast<unsigned long long>(value),
	            static_cast<unsigned long long>(g_hw_watch_ptr_offsets[0]),
	            g_hw_watch_ptr_count > 1 ? " and the second offset" : "");
	std::fflush(stdout);
	g_hw_watch_generation++;
}

static DWORD WINAPI HwWatchThread(void* /*unused*/) {
	static DWORD armed[4096];
	int          armed_count = 0;
	const auto   self        = GetCurrentThreadId();
	const auto   process     = GetCurrentProcessId();
	uint32_t     generation  = 0;
	for (;;) {
		PollHwWatchPointer();
		if (const auto now = g_hw_watch_generation.load(); now != generation) {
			generation  = now;
			armed_count = 0;
		}
		const auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
		if (snapshot != INVALID_HANDLE_VALUE) {
			THREADENTRY32 entry {};
			entry.dwSize = sizeof(entry);
			for (BOOL more = Thread32First(snapshot, &entry); more != 0;
			     more      = Thread32Next(snapshot, &entry)) {
				if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) {
					continue;
				}
				bool known = false;
				for (int i = 0; i < armed_count; i++) {
					known = known || armed[i] == entry.th32ThreadID;
				}
				if (known || armed_count >= 4096) {
					continue;
				}
				const auto thread =
				    OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE,
				               entry.th32ThreadID);
				if (thread == nullptr) {
					continue;
				}
				if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
					CONTEXT context {};
					context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
					if (GetThreadContext(thread, &context) != 0) {
						DWORD64 control = 0;
						for (int i = 0; i < g_hw_watch_count; i++) {
							(&context.Dr0)[i] = g_hw_watch[i];
							// Local enable, break on write, four bytes (or on execution).
							control |= (DWORD64 {1} << (i * 2)) |
							           (DWORD64 {g_hw_watch_exec[i] ? 0x0u : 0xdu} << (16 + i * 4));
						}
						context.Dr7 = control;
						context.Dr6 = 0;
						SetThreadContext(thread, &context);
					}
					ResumeThread(thread);
					armed[armed_count++] = entry.th32ThreadID;
				}
				CloseHandle(thread);
			}
			CloseHandle(snapshot);
		}
		Sleep(100);
	}
}

static void StartHwWatch() {
	const char* text = std::getenv("KYTY_DEBUG_HW_WATCH");
	while (text != nullptr && *text != '\0' && g_hw_watch_count < 4) {
		char* end                      = nullptr;
		g_hw_watch[g_hw_watch_count++] = std::strtoull(text, &end, 16);
		text                           = *end == ',' ? end + 1 : nullptr;
	}
	if (const char* ptr = std::getenv("KYTY_DEBUG_HW_WATCH_PTR"); ptr != nullptr) {
		char* end      = nullptr;
		g_hw_watch_ptr = std::strtoull(ptr, &end, 16);
		while (*end == ':' && g_hw_watch_ptr_count < 2 && g_hw_watch_count + g_hw_watch_ptr_count < 4) {
			g_hw_watch_ptr_offsets[g_hw_watch_ptr_count++] = std::strtoull(end + 1, &end, 16);
		}
		g_hw_watch_ptr_first = g_hw_watch_count;
	}
	for (const char* exec = std::getenv("KYTY_DEBUG_HW_EXEC");
	     exec != nullptr && *exec != '\0' && g_hw_watch_count < 4 && g_hw_watch_ptr == 0;) {
		char* end                         = nullptr;
		g_hw_watch_exec[g_hw_watch_count] = true;
		g_hw_watch[g_hw_watch_count++]    = std::strtoull(exec, &end, 16);
		exec                              = *end == ',' ? end + 1 : nullptr;
	}
	if (g_hw_watch_count != 0 || g_hw_watch_ptr != 0) {
		CloseHandle(CreateThread(nullptr, 0, HwWatchThread, nullptr, 0, nullptr));
	}
}

static LONG WINAPI ExceptionFilter(PEXCEPTION_POINTERS exception) noexcept {
	auto* exception_record = exception->ExceptionRecord;

	if (exception_record->ExceptionCode == EXCEPTION_SINGLE_STEP && g_hw_watch_count != 0 &&
	    (exception->ContextRecord->Dr6 & 0xfu) != 0) {
		static std::atomic<uint32_t> count {0};
		static std::atomic<uint32_t> exec_count {0};
		static const uint32_t        exec_limit = [] {
			const char* text = std::getenv("KYTY_DEBUG_HW_EXEC_LIMIT");
			return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 0)) : 40u;
		}();
		for (int i = 0; i < g_hw_watch_count; i++) {
			if ((exception->ContextRecord->Dr6 & (DWORD64 {1} << i)) == 0) {
				continue;
			}
			if (g_hw_watch_exec[i]) {
				// Resume flag: the instruction runs once without breaking again.
				exception->ContextRecord->EFlags |= 0x10000u;
				// KYTY_DEBUG_HW_EXEC_SKIP=<n>: the first n hits are not reported.
				static const uint32_t exec_skip = [] {
					const char* text = std::getenv("KYTY_DEBUG_HW_EXEC_SKIP");
					return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 0)) : 0u;
				}();
				const auto hit = exec_count.fetch_add(1, std::memory_order_relaxed);
				if (hit < exec_skip || hit >= exec_skip + exec_limit) {
					continue;
				}
				// KYTY_DEBUG_HW_EXEC_ARM=<r9|r15|rbx|rdi|rsi>:<hex offset>: the first reported hit
				// adds a write watch on that register + offset (who writes the data read here).
				if (const char* arm = std::getenv("KYTY_DEBUG_HW_EXEC_ARM");
				    arm != nullptr && g_hw_watch_count < 4) {
					static std::atomic<bool> armed {false};
					if (!armed.exchange(true)) {
						const auto* c      = exception->ContextRecord;
						const char* colon  = std::strchr(arm, ':');
						const auto  offset = colon != nullptr ? std::strtoull(colon + 1, nullptr, 16) : 0;
						const auto  name   = std::string_view(arm, colon != nullptr ? colon - arm : 0);
						const uint64_t reg = name == "r9"    ? c->R9
						                     : name == "r15" ? c->R15
						                     : name == "rbx" ? c->Rbx
						                     : name == "rdi" ? c->Rdi
						                                     : c->Rsi;
						g_hw_watch_exec[g_hw_watch_count] = false;
						g_hw_watch[g_hw_watch_count]      = reg + offset;
						std::printf("hw-watch: armed write watch %llx\n",
						            static_cast<unsigned long long>(reg + offset));
						g_hw_watch_count++;
						g_hw_watch_generation++;
					}
				}
			} else if (count.fetch_add(1, std::memory_order_relaxed) >= 4000) {
				continue;
			}
			uint32_t value = 0;
			SIZE_T   read  = 0;
			ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(g_hw_watch[i]),
			                  &value, sizeof(value), &read);
			std::printf("hw-watch: %llx = %08x%s pc=%llx thread=%lu rsp=%llx\n",
			            static_cast<unsigned long long>(g_hw_watch[i]), value,
			            read == sizeof(value) ? "" : " (unreadable)",
			            static_cast<unsigned long long>(exception->ContextRecord->Rip),
			            GetCurrentThreadId(),
			            static_cast<unsigned long long>(exception->ContextRecord->Rsp));
			// KYTY_DEBUG_HW_WATCH_FOLLOW=<hex offset>: the first hit adds a watch on r15 + offset.
			if (const char* follow = std::getenv("KYTY_DEBUG_HW_WATCH_FOLLOW");
			    follow != nullptr && g_hw_watch_count < 4) {
				static std::atomic<bool> followed {false};
				if (!followed.exchange(true)) {
					g_hw_watch[g_hw_watch_count] =
					    exception->ContextRecord->R15 + std::strtoull(follow, nullptr, 16);
					std::printf("hw-watch: follow %llx\n",
					            static_cast<unsigned long long>(g_hw_watch[g_hw_watch_count]));
					g_hw_watch_count++;
					g_hw_watch_generation++;
				}
			}
			// KYTY_DEBUG_HW_WATCH_DUMP=<hex offset>:<hex bytes>,...: also the general registers
			// and those bytes at r15 + offset (the writer's object, read safely).
			if (const char* dump = std::getenv("KYTY_DEBUG_HW_WATCH_DUMP"); dump != nullptr) {
				const auto* c = exception->ContextRecord;
				std::printf("hw-watch:   rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx "
				            "r11=%llx r12=%llx r13=%llx r15=%llx\n",
				            c->Rax, c->Rbx, c->Rcx, c->Rdx, c->Rsi, c->Rdi, c->R11, c->R12, c->R13,
				            c->R15);
				std::printf("hw-watch:   r8=%llx r9=%llx r10=%llx r14=%llx rbp=%llx xmm:", c->R8, c->R9,
				            c->R10, c->R14, c->Rbp);
				for (int x = 0; x < 8; x++) {
					float f[4];
					std::memcpy(f, &(&c->Xmm0)[x], sizeof(f));
					std::printf(" (%g %g %g %g)", f[0], f[1], f[2], f[3]);
				}
				// Return addresses along the frame-pointer chain (guest code keeps rbp frames).
				std::printf("\nhw-watch:   bt:");
				uint64_t frame = c->Rbp;
				for (int depth = 0; depth < 10 && frame != 0; depth++) {
					uint64_t pair[2] {};
					SIZE_T   got = 0;
					if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(frame), pair,
					                       sizeof(pair), &got) ||
					    got != sizeof(pair)) {
						break;
					}
					std::printf(" %llx", static_cast<unsigned long long>(pair[1]));
					if (pair[0] <= frame) {
						break;
					}
					frame = pair[0];
				}
				std::printf("\n");
				while (*dump != '\0') {
					// '@<address>' absolute, 'p<offset>' through the pointer at r15 + offset,
					// 's<offset>' at rsp + offset.
					// 'b<offset>' at rbx + offset, 'q<offset>' through the pointer at rbx + offset.
					const char     mode   = (*dump == '@' || *dump == 'p' || *dump == 's' ||
					                         *dump == 'b' || *dump == 'q' || *dump == 'd' ||
					                         *dump == 'e' || *dump == 'i' || *dump == 'x' ||
					                         *dump == 'a')
					                            ? *dump++
					                            : 'r';
					char*          end    = nullptr;
					const uint64_t offset = std::strtoull(dump, &end, 16);
					const uint64_t size   = *end == ':' ? std::strtoull(end + 1, &end, 16) : 16;
					// 'd<offset>' at rdi + offset, 'e<offset>' through the pointer at rdi + offset,
					// 'i<offset>' at rsi + offset.
					// 'x<offset>' at rdx + offset (also printed as text, e.g. a path argument).
					// 'a<offset>' at rax + offset.
					const uint64_t base   = mode == 'a'                 ? c->Rax
					                        : mode == 'x'                 ? c->Rdx
					                        : mode == 'i'                 ? c->Rsi
					                        : mode == 's'                 ? c->Rsp
					                        : mode == 'b' || mode == 'q' ? c->Rbx
					                        : mode == 'd' || mode == 'e' ? c->Rdi
					                                                     : c->R15;
					uint64_t       source = mode == '@' ? offset : base + offset;
					SIZE_T         got    = 0;
					if (mode == 'p' || mode == 'q' || mode == 'e') {
						uint64_t pointer = 0;
						ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(source),
						                  &pointer, sizeof(pointer), &got);
						source = pointer;
					}
					static uint8_t bytes[0x8000];
					ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(source),
					                  bytes, (size < sizeof(bytes) ? size : sizeof(bytes)), &got);
					std::printf("hw-watch:   [%c%llx @%llx]", mode, static_cast<unsigned long long>(offset), static_cast<unsigned long long>(source));
					for (SIZE_T b = 0; b + 4 <= got; b += 4) {
						uint32_t word = 0;
						std::memcpy(&word, bytes + b, 4);
						std::printf(" %08x", word);
					}
					if (mode == 'x') {
						std::printf(" \"");
						for (SIZE_T b = 0; b < got && bytes[b] != 0; b++) {
							std::printf("%c", bytes[b] >= 0x20 && bytes[b] < 0x7f ? bytes[b] : '.');
						}
						std::printf("\"");
					}
					std::printf("\n");
					if (*end != ',') {
						break;
					}
					dump = end + 1;
				}
			}
			std::fflush(stdout);
		}
		exception->ContextRecord->Dr6 = 0;
		return EXCEPTION_CONTINUE_EXECUTION;
	}

	if (exception_record->ExceptionCode == DBG_PRINTEXCEPTION_C ||
	    exception_record->ExceptionCode == DBG_PRINTEXCEPTION_WIDE_C) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	if (exception_record->ExceptionCode == 0x406D1388) {
		// Set a thread name.
		return EXCEPTION_CONTINUE_EXECUTION;
	}

	ExceptionInfo info {};
	info.exception_address = reinterpret_cast<uint64_t>(exception_record->ExceptionAddress);
	info.native_code       = exception_record->ExceptionCode;
	info.native_context    = exception->ContextRecord;

	if (exception_record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
		info.type = ExceptionType::AccessViolation;
		switch (exception_record->ExceptionInformation[0]) {
			case 0: info.access_violation_type = AccessViolationType::Read; break;
			case 1: info.access_violation_type = AccessViolationType::Write; break;
			case 8: info.access_violation_type = AccessViolationType::Execute; break;
			default: info.access_violation_type = AccessViolationType::Unknown; break;
		}
		info.access_violation_vaddr = exception_record->ExceptionInformation[1];
	} else if (exception_record->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		// One that ends the process where no handler can be found is reported first (usa-ratt's
		// GT7 fork, round 125).
		uint64_t   parameters[4] {};
		const auto count =
		    exception_record->NumberParameters < 4 ? exception_record->NumberParameters : DWORD {4};
		for (DWORD i = 0; i < count; i++) {
			parameters[i] = exception_record->ExceptionInformation[i];
		}
		LastWords::ReportFirstChance(exception_record->ExceptionCode, info.exception_address,
		                             parameters, count, exception->ContextRecord);
		return EXCEPTION_CONTINUE_SEARCH;
	}

	info.rax = exception->ContextRecord->Rax;
	info.rbx = exception->ContextRecord->Rbx;
	info.rcx = exception->ContextRecord->Rcx;
	info.rdx = exception->ContextRecord->Rdx;
	info.rsi = exception->ContextRecord->Rsi;
	info.rdi = exception->ContextRecord->Rdi;
	info.rbp = exception->ContextRecord->Rbp;
	info.rsp = exception->ContextRecord->Rsp;
	info.r8  = exception->ContextRecord->R8;
	info.r9  = exception->ContextRecord->R9;
	info.r10 = exception->ContextRecord->R10;
	info.r11 = exception->ContextRecord->R11;
	info.r12 = exception->ContextRecord->R12;
	info.r13 = exception->ContextRecord->R13;
	info.r14 = exception->ContextRecord->R14;
	info.r15 = exception->ContextRecord->R15;

	const auto handler = g_handler.load(std::memory_order_acquire);
	if (handler != nullptr && handler(info)) {
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	return EXCEPTION_CONTINUE_SEARCH;
}

#elif defined(__APPLE__)

static std::atomic<Handler> g_handler {nullptr};
static std::atomic_uint32_t g_install_state {0};

static_assert(decltype(g_handler)::is_always_lock_free);
static_assert(decltype(g_install_state)::is_always_lock_free);

// Translate the x86-64 page-fault error code (mcontext __es.__err) into an access type.
// bit 1 (0x2) = write, bit 4 (0x10) = instruction fetch, otherwise a read.
static AccessViolationType DecodeAccess(uint64_t err) {
	if ((err & 0x10u) != 0) {
		return AccessViolationType::Execute;
	}
	if ((err & 0x2u) != 0) {
		return AccessViolationType::Write;
	}
	return AccessViolationType::Read;
}

// POSIX signal handler that mirrors the Windows vectored handler: build an ExceptionInfo
// from the mcontext and dispatch. A resolved fault (handler returns true) simply returns,
// re-executing the faulting instruction against the now-fixed protection. An unresolved
// fault restores the default disposition so the retry terminates the process.
static void SignalHandler(int sig, siginfo_t* si, void* uctx) {
	auto*       uc = static_cast<ucontext_t*>(uctx);
	const auto* mc = uc->uc_mcontext;
	const auto& ss = mc->__ss;

	ExceptionInfo info {};
	info.exception_address = ss.__rip;
	info.native_code       = static_cast<uint32_t>(si->si_code);
	info.native_context    = uctx;

	if (sig == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		info.type                   = ExceptionType::AccessViolation;
		info.access_violation_type  = DecodeAccess(mc->__es.__err);
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(si->si_addr);
	}

	info.rax = ss.__rax;
	info.rbx = ss.__rbx;
	info.rcx = ss.__rcx;
	info.rdx = ss.__rdx;
	info.rsi = ss.__rsi;
	info.rdi = ss.__rdi;
	info.rbp = ss.__rbp;
	info.rsp = ss.__rsp;
	info.r8  = ss.__r8;
	info.r9  = ss.__r9;
	info.r10 = ss.__r10;
	info.r11 = ss.__r11;
	info.r12 = ss.__r12;
	info.r13 = ss.__r13;
	info.r14 = ss.__r14;
	info.r15 = ss.__r15;

	const auto handler = g_handler.load(std::memory_order_acquire);
	if (handler != nullptr && handler(info)) {
		return; // retry the faulting instruction against the fixed mapping
	}

	// Unresolved: restore the default action so the re-executed instruction terminates.
	struct sigaction dfl {};
	dfl.sa_handler = SIG_DFL;
	sigemptyset(&dfl.sa_mask);
	sigaction(sig, &dfl, nullptr);
}

#else

// x86-64 page-fault error bits.
constexpr uint64_t PAGE_FAULT_ERROR_WRITE       = 0x02;
constexpr uint64_t PAGE_FAULT_ERROR_INSTRUCTION = 0x10;

// Let the kernel handle an unresolved fault on retry.
static void ChainToDefault(int signal_number) noexcept {
	struct sigaction restore {};
	restore.sa_handler = SIG_DFL;
	sigemptyset(&restore.sa_mask);
	restore.sa_flags = 0;
	::sigaction(signal_number, &restore, nullptr);
}

static void SignalHandler(int signal_number, siginfo_t* signal_info, void* native_context) {
	auto* context = static_cast<ucontext_t*>(native_context);
	auto* gregs   = context->uc_mcontext.gregs;

	ExceptionInfo info {};
	info.exception_address = static_cast<uint64_t>(gregs[REG_RIP]);
	info.native_code       = static_cast<uint32_t>(signal_number);
	info.native_context    = context;

	if (signal_number == SIGSEGV || signal_number == SIGBUS) {
		info.type             = ExceptionType::AccessViolation;
		const auto error_code = static_cast<uint64_t>(gregs[REG_ERR]);
		if ((error_code & PAGE_FAULT_ERROR_INSTRUCTION) != 0) {
			info.access_violation_type = AccessViolationType::Execute;
		} else if ((error_code & PAGE_FAULT_ERROR_WRITE) != 0) {
			info.access_violation_type = AccessViolationType::Write;
		} else {
			info.access_violation_type = AccessViolationType::Read;
		}
		info.access_violation_vaddr = reinterpret_cast<uint64_t>(signal_info->si_addr);
	} else if (signal_number == SIGILL) {
		info.type = ExceptionType::IllegalInstruction;
	} else {
		ChainToDefault(signal_number);
		return;
	}

	info.rax = static_cast<uint64_t>(gregs[REG_RAX]);
	info.rbx = static_cast<uint64_t>(gregs[REG_RBX]);
	info.rcx = static_cast<uint64_t>(gregs[REG_RCX]);
	info.rdx = static_cast<uint64_t>(gregs[REG_RDX]);
	info.rsi = static_cast<uint64_t>(gregs[REG_RSI]);
	info.rdi = static_cast<uint64_t>(gregs[REG_RDI]);
	info.rbp = static_cast<uint64_t>(gregs[REG_RBP]);
	info.rsp = static_cast<uint64_t>(gregs[REG_RSP]);
	info.r8  = static_cast<uint64_t>(gregs[REG_R8]);
	info.r9  = static_cast<uint64_t>(gregs[REG_R9]);
	info.r10 = static_cast<uint64_t>(gregs[REG_R10]);
	info.r11 = static_cast<uint64_t>(gregs[REG_R11]);
	info.r12 = static_cast<uint64_t>(gregs[REG_R12]);
	info.r13 = static_cast<uint64_t>(gregs[REG_R13]);
	info.r14 = static_cast<uint64_t>(gregs[REG_R14]);
	info.r15 = static_cast<uint64_t>(gregs[REG_R15]);

	const auto handler = g_handler.load(std::memory_order_acquire);
	if (handler != nullptr && handler(info)) {
		return;
	}

	ChainToDefault(signal_number);
}

#endif

bool InstallHandler(Handler handler) {
	if (handler == nullptr) {
		return false;
	}

	uint32_t expected_state = 0;
	if (!g_install_state.compare_exchange_strong(expected_state, 1, std::memory_order_acq_rel)) {
		return expected_state == 2 && g_handler.load(std::memory_order_acquire) == handler;
	}

	g_handler.store(handler, std::memory_order_release);

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (AddVectoredExceptionHandler(0, ExceptionFilter) == nullptr) {
		g_handler.store(nullptr, std::memory_order_release);
		g_install_state.store(0, std::memory_order_release);
		printf("AddVectoredExceptionHandler() failed\n");
		return false;
	}
	StartHwWatch();
	// abort() and std::terminate end the process through __fastfail (0xc0000409, code 7), which
	// no vectored handler sees: Common::LastWords (installed by main) reports them, with the
	// thread's symbolized stack, in the log. Only without it: the stack on stderr.
	if (!LastWords::Installed()) {
		std::signal(SIGABRT, [](int) {
			void*      frames[48] {};
			const auto count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
			std::fprintf(stderr, "--- Error ---\nabort() called\nstack:");
			for (USHORT i = 0; i < count; i++) {
				std::fprintf(stderr, " 0x%llx", reinterpret_cast<unsigned long long>(frames[i]));
			}
			std::fprintf(stderr, "\n");
			std::fflush(stderr);
		});
		std::set_terminate([] {
			std::fprintf(stderr, "--- Error ---\nstd::terminate called\n");
			std::fflush(stderr);
			std::abort();
		});
	}
#elif defined(__APPLE__)
	struct sigaction sa {};
	sa.sa_sigaction = SignalHandler;
	sa.sa_flags     = SA_SIGINFO | SA_ONSTACK;
	sigemptyset(&sa.sa_mask);
	// The guest signal-dispatch path (KernelRaiseException) interrupts threads with
	// SIGUSR1; block it while a fault is being resolved so a stop-the-world request
	// cannot preempt the handler between the protection fix and the retry.
	sigaddset(&sa.sa_mask, SIGUSR1);

	// macOS raises SIGBUS for protection faults on some paths and SIGSEGV on others;
	// SIGILL covers instructions the host cannot execute (routed to the x64 emulator).
	bool ok = sigaction(SIGSEGV, &sa, nullptr) == 0 && sigaction(SIGBUS, &sa, nullptr) == 0 &&
	          sigaction(SIGILL, &sa, nullptr) == 0;
	if (!ok) {
		g_handler.store(nullptr, std::memory_order_release);
		g_install_state.store(0, std::memory_order_release);
		printf("sigaction() failed to install the host fault handler\n");
		return false;
	}
#else
	struct sigaction action {};
	action.sa_sigaction = SignalHandler;
	sigemptyset(&action.sa_mask);
	action.sa_flags = SA_SIGINFO | SA_RESTART | SA_ONSTACK;

	for (const int signal_number: {SIGSEGV, SIGBUS, SIGILL}) {
		if (::sigaction(signal_number, &action, nullptr) != 0) {
			g_handler.store(nullptr, std::memory_order_release);
			g_install_state.store(0, std::memory_order_release);
			printf("sigaction(%d) failed\n", signal_number);
			return false;
		}
	}
#endif

	g_install_state.store(2, std::memory_order_release);
	return true;
}

// KYTY_DEBUG_MEM_SAMPLE=<pointer>:<offset>:<floats>[:<every n frames>][,...] (hex, counts
// decimal): prints the floats at *pointer + offset every n presented frames (default 10), e.g. a
// game object's position followed through a scene. Pointer 0 samples <offset> directly.
void SampleDebugMemory(uint64_t frame) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	struct Sample {
		uint64_t pointer = 0;
		uint64_t offset  = 0;
		uint32_t floats  = 1;
		uint32_t every   = 10;
	};
	static const std::vector<Sample> samples = [] {
		std::vector<Sample> list;
		const char*         text = std::getenv("KYTY_DEBUG_MEM_SAMPLE");
		while (text != nullptr && *text != '\0') {
			Sample sample;
			char*  end     = nullptr;
			sample.pointer = std::strtoull(text, &end, 16);
			if (*end == ':') sample.offset = std::strtoull(end + 1, &end, 16);
			if (*end == ':') sample.floats = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
			if (*end == ':') sample.every = static_cast<uint32_t>(std::strtoul(end + 1, &end, 10));
			sample.floats = (std::min)((std::max)(sample.floats, 1u), 16u);
			sample.every  = (std::max)(sample.every, 1u);
			list.push_back(sample);
			if (*end != ',') break;
			text = end + 1;
		}
		return list;
	}();
	for (size_t index = 0; index < samples.size(); index++) {
		const auto& sample = samples[index];
		if (frame % sample.every != 0) {
			continue;
		}
		uint64_t base = 0;
		SIZE_T   got  = 0;
		if (sample.pointer != 0 &&
		    (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(sample.pointer),
		                       &base, sizeof(base), &got) == 0 ||
		     got != sizeof(base) || base == 0)) {
			continue;
		}
		float values[16] {};
		if (ReadProcessMemory(GetCurrentProcess(),
		                      reinterpret_cast<const void*>(base + sample.offset), values,
		                      sample.floats * sizeof(float), &got) == 0 ||
		    got != sample.floats * sizeof(float)) {
			continue;
		}
		std::string line = fmt::format("mem-sample[{}]: frame={} at={:x}", index, frame,
		                               base + sample.offset);
		for (uint32_t i = 0; i < sample.floats; i++) {
			line += fmt::format(" {:.3f}", values[i]);
		}
		std::printf("%s\n", line.c_str());
	}
	if (!samples.empty()) {
		std::fflush(stdout);
	}
#else
	(void)frame;
#endif
}

} // namespace Common::HostException
