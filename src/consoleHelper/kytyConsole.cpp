// kyty_console.exe: started by kyty_emulator.exe when its stdout is a console window (see
// common/consoleColors.h). Reads the emulator's output from stdin and writes it line by line in
// pastel colors (the palette of the console colorizer script) to the shared console. Lines that
// already carry escape codes pass through unchanged. Ends when the emulator's pipe closes.
// KYTY_LOG_FILE=<path> (inherited from the emulator): every line also goes to that file, without
// escape codes; "%d" in the path becomes the date and time of the start (one file per session). This
// process outlives the emulator, so the file keeps the last lines of a crash too.

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <string_view>
#include <windows.h>

namespace {

struct Rule {
	const char*                    color;
	std::array<std::string_view, 7> words; // lower case; a leading '^' anchors at the line start
};

// Checked in order, the first match wins (case-insensitive, like the script).
constexpr std::array<Rule, 6> RULES = {{
    // errors: pastel red
    {"\x1b[38;2;255;145;145m",
     {"fatal error", "--- error ---", "failed", "device-fault", "devicelost", "device lost", "not implemented"}},
    // fixed: pastel green
    {"\x1b[38;2;150;230;160m", {"works now", "fixed"}},
    // warnings: pastel yellow
    {"\x1b[38;2;240;220;130m", {"warning", "skipping", "placeholder", "unreadable", "stale"}},
    // audio: pastel blue
    {"\x1b[38;2;145;185;255m", {"^audio", "audioout", "audiodiag", "dualsense"}},
    // shaders: pastel purple
    {"\x1b[38;2;205;165;255m", {"pipeline", "shader", "spir-v", "spirv", "compil"}},
    // memory and textures: pastel turquoise
    {"\x1b[38;2;130;220;210m", {"gpu-zones", "bindless", "buffer", "memory", "cache", "readback"}},
}};

constexpr std::string_view GREY  = "\x1b[38;2;200;200;205m";
constexpr std::string_view RESET = "\x1b[0m";

std::string_view ColorOf(std::string_view line) {
	std::string lower(line);
	for (auto& c: lower) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	for (const auto& rule: RULES) {
		for (auto word: rule.words) {
			if (word.empty()) {
				continue;
			}
			if (word.front() == '^' ? lower.starts_with(word.substr(1)) : lower.find(word) != std::string::npos) {
				return rule.color;
			}
		}
	}
	return GREY;
}

// The KYTY_LOG_FILE copy: plain text, written through at once.
class LogFile {
public:
	LogFile() {
		const char* path = std::getenv("KYTY_LOG_FILE");
		if (path == nullptr || *path == '\0') {
			return;
		}
		std::string name(path);
		if (const auto at = name.find("%d"); at != std::string::npos) {
			char        stamp[32] = {};
			std::time_t now       = std::time(nullptr);
			std::tm     local {};
			localtime_s(&local, &now);
			std::strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H-%M-%S", &local);
			name.replace(at, 2, stamp);
		}
		m_file = std::fopen(name.c_str(), "ab");
	}
	~LogFile() {
		if (m_file != nullptr) {
			std::fclose(m_file);
		}
	}
	LogFile(const LogFile&)            = delete;
	LogFile& operator=(const LogFile&) = delete;

	void Write(std::string_view text) {
		if (m_file == nullptr) {
			return;
		}
		// Escape sequences (ESC '[' ... final letter) are left out.
		for (size_t i = 0; i < text.size();) {
			if (text[i] == '\x1b' && i + 1 < text.size() && text[i + 1] == '[') {
				i += 2;
				while (i < text.size() && !std::isalpha(static_cast<unsigned char>(text[i]))) {
					i++;
				}
				i++;
				continue;
			}
			std::fputc(text[i], m_file);
			i++;
		}
	}
	void Flush() {
		if (m_file != nullptr) {
			std::fflush(m_file);
		}
	}

private:
	std::FILE* m_file = nullptr;
};

class Writer {
public:
	Writer(HANDLE console, LogFile& log): m_console(console), m_log(log) {}

	// A complete line (without '\n').
	void Line(std::string_view line) {
		if (!line.empty() && line.back() == '\r') {
			line.remove_suffix(1);
		}
		m_log.Write(line);
		m_log.Write("\n");
		if (m_open_color.empty()) {
			Colored(line, line.find('\x1b') != std::string_view::npos ? std::string_view {} : ColorOf(line));
		} else {
			Colored(line, m_open_color);
		}
		Put("\n");
		m_open_color = {};
	}

	// The start of a line that has no '\n' yet (a prompt or a slow writer): shown now, and the rest
	// of the line keeps its color.
	void Partial(std::string_view text) {
		m_log.Write(text);
		if (m_open_color.empty()) {
			m_open_color = text.find('\x1b') != std::string_view::npos ? RESET : ColorOf(text);
		}
		Colored(text, m_open_color);
	}

	void Flush() {
		const char* data = m_buffer.data();
		auto        size = static_cast<DWORD>(m_buffer.size());
		while (size > 0) {
			DWORD written = 0;
			if (WriteFile(m_console, data, size, &written, nullptr) == FALSE || written == 0) {
				break;
			}
			data += written;
			size -= written;
		}
		m_buffer.clear();
		m_log.Flush();
	}

private:
	void Colored(std::string_view text, std::string_view color) {
		if (color.empty()) {
			Put(text);
			return;
		}
		Put(color);
		Put(text);
		Put(RESET);
	}

	void Put(std::string_view text) { m_buffer.append(text); }

	HANDLE           m_console;
	LogFile&         m_log;
	std::string      m_buffer;
	std::string_view m_open_color;
};

} // namespace

int main() {
	// Ctrl+C and Ctrl+Break go to the emulator; this process keeps printing until its pipe closes.
	SetConsoleCtrlHandler(nullptr, TRUE);

	HANDLE input   = GetStdHandle(STD_INPUT_HANDLE);
	HANDLE console = CreateFileW(L"CONOUT$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
	                             nullptr, OPEN_EXISTING, 0, nullptr);
	if (console == INVALID_HANDLE_VALUE) {
		console = GetStdHandle(STD_OUTPUT_HANDLE);
	}
	DWORD mode = 0;
	if (GetConsoleMode(console, &mode) != FALSE) {
		SetConsoleMode(console, mode | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
	}

	LogFile          log;
	Writer           writer(console, log);
	std::string      pending;
	std::array<char, 1u << 16u> chunk {};
	for (;;) {
		DWORD read = 0;
		if (ReadFile(input, chunk.data(), static_cast<DWORD>(chunk.size()), &read, nullptr) == FALSE || read == 0) {
			break;
		}
		pending.append(chunk.data(), read);

		size_t start = 0;
		for (size_t end = pending.find('\n'); end != std::string::npos; end = pending.find('\n', start)) {
			writer.Line(std::string_view(pending).substr(start, end - start));
			start = end + 1;
		}
		pending.erase(0, start);

		DWORD available = 0;
		if (!pending.empty() && PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr) != FALSE &&
		    available == 0) {
			writer.Partial(pending);
			pending.clear();
		}
		writer.Flush();
	}
	if (!pending.empty()) {
		writer.Partial(pending);
	}
	writer.Flush();
	return 0;
}
