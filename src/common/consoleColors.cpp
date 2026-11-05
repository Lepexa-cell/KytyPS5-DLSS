#include "common/consoleColors.h"

#include "common/common.h"
#include "common/switches.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <io.h>
#include <string>
#include <windows.h>

namespace Common::ConsoleColors {

void Install() {
	if (const char* value = KYTY_GETENV("KYTY_CONSOLE_COLORS"); value != nullptr && std::strcmp(value, "0") == 0) {
		return;
	}

	HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
	DWORD  mode    = 0;
	if (console == nullptr || console == INVALID_HANDLE_VALUE || GetConsoleMode(console, &mode) == FALSE) {
		return;
	}

	wchar_t exe[MAX_PATH] = {};
	const DWORD length    = GetModuleFileNameW(nullptr, exe, MAX_PATH);
	if (length == 0 || length >= MAX_PATH) {
		return;
	}
	std::wstring helper(exe, length);
	helper.erase(helper.find_last_of(L"\\/") + 1);
	helper += L"kyty_console.exe";
	if (GetFileAttributesW(helper.c_str()) == INVALID_FILE_ATTRIBUTES) {
		return;
	}

	SECURITY_ATTRIBUTES inherit {sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
	HANDLE              read  = nullptr;
	HANDLE              write = nullptr;
	if (CreatePipe(&read, &write, &inherit, 1u << 20u) == FALSE) {
		return;
	}
	SetHandleInformation(write, HANDLE_FLAG_INHERIT, 0);

	// The helper inherits only the pipe's read end and writes to the shared console itself.
	SIZE_T list_size = 0;
	InitializeProcThreadAttributeList(nullptr, 1, 0, &list_size);
	std::string list_storage(list_size, '\0');
	auto* list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list_storage.data());
	bool  ok   = InitializeProcThreadAttributeList(list, 1, 0, &list_size) != FALSE &&
	          UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &read, sizeof(read),
	                                    nullptr, nullptr) != FALSE;

	STARTUPINFOEXW startup {};
	startup.StartupInfo.cb         = sizeof(startup);
	startup.StartupInfo.dwFlags    = STARTF_USESTDHANDLES;
	startup.StartupInfo.hStdInput  = read;
	startup.StartupInfo.hStdOutput = console;
	startup.StartupInfo.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
	startup.lpAttributeList        = list;

	std::wstring        command = L"\"" + helper + L"\"";
	PROCESS_INFORMATION process {};
	ok = ok && CreateProcessW(helper.c_str(), command.data(), nullptr, nullptr, TRUE,
	                          EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr, &startup.StartupInfo,
	                          &process) != FALSE;
	if (list_size != 0) {
		DeleteProcThreadAttributeList(list);
	}
	CloseHandle(read);
	if (!ok) {
		CloseHandle(write);
		return;
	}
	CloseHandle(process.hThread);
	CloseHandle(process.hProcess);

	// stdout and stderr (CRT and Win32 handles) go to the pipe, unbuffered as on the console.
	std::fflush(stdout);
	std::fflush(stderr);
	const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(write), _O_WRONLY | _O_BINARY);
	if (fd < 0) {
		CloseHandle(write);
		return;
	}
	_dup2(fd, _fileno(stdout));
	_dup2(fd, _fileno(stderr));
	_close(fd);
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	std::setvbuf(stderr, nullptr, _IONBF, 0);
	SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stdout))));
	SetStdHandle(STD_ERROR_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stderr))));
}

} // namespace Common::ConsoleColors

#else

namespace Common::ConsoleColors {

void Install() {}

} // namespace Common::ConsoleColors

#endif
