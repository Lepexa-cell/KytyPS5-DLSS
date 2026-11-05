#ifndef KYTY_COMMON_SWITCHES_H_
#define KYTY_COMMON_SWITCHES_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <vector>

// Environment switches (KYTY_*), read with KYTY_GETENV("KYTY_NAME") and given a default with
// KYTY_SET_DEFAULT_SWITCH. A share build (KYTY_SHARE_BUILD=1, the build given to other people)
// stores no switch names: a name is only hashed at compile time, and the lookup hashes the names
// in the environment instead. There the diagnostic switches (KYTY_DEBUG_*, KYTY_VERIFY_* and the
// other names IsDiagnostic lists) are always unset, so their code and texts drop out of the build.
namespace Common::Switches {

constexpr uint64_t Hash(std::string_view name) {
	uint64_t hash = 0xcbf29ce484222325ull;
	for (const char c: name) {
		hash = (hash ^ static_cast<uint8_t>(c)) * 0x100000001b3ull;
	}
	return hash;
}

constexpr bool IsDiagnostic(std::string_view name) {
	// Debug-named switches that change what a game does, not what is measured or printed.
	if (name == "KYTY_DEBUG_FRAME_LIMIT_MS" || name == "KYTY_DEBUG_FRAME_LIMIT_FILE" ||
	    name == "KYTY_DEBUG_VK_ERRORS_NONFATAL" || name == "KYTY_DEBUG_LOOP_GUARD" ||
	    name == "KYTY_DEBUG_LOOP_GUARD_TIGHT") {
		return false;
	}
	return name.starts_with("KYTY_DEBUG_") || name.starts_with("KYTY_VERIFY_") ||
	       name.ends_with("_STATS") || name.ends_with("_LOG") || name.starts_with("KYTY_TRACE_");
}

// Share build: the value of the switch with this name hash (environment first, then a default
// set with SetDefault), or nullptr.
const char* Get(uint64_t hash);
void        SetDefault(uint64_t hash, const char* value);

// A switch's list of hex hashes (pass KYTY_GETENV("KYTY_...")): hashes separated by commas, zeros
// skipped; the list ends at the first text that is not a hash. "*" alone sets `all` (what it
// covers is the caller's choice) and lists no hash.
struct HashList {
	std::vector<uint64_t> hashes;
	bool                  all = false;

	explicit HashList(const char* text) {
		if (text != nullptr && text[0] == '*' && text[1] == '\0') {
			all = true;
			return;
		}
		while (text != nullptr && *text != '\0') {
			char* end = nullptr;
			if (const auto hash = std::strtoull(text, &end, 16); hash != 0) hashes.push_back(hash);
			if (end == text) break;
			text = (*end == ',') ? end + 1 : end;
		}
	}
	[[nodiscard]] bool Contains(uint64_t hash) const {
		for (const auto listed: hashes) {
			if (listed == hash) return true;
		}
		return false;
	}
};

} // namespace Common::Switches

#if defined(KYTY_SHARE_BUILD) && KYTY_SHARE_BUILD
// Messages about the emulator's own techniques (title profiles, built-in patches, caches) print in
// own builds only.
#define KYTY_INTERNAL_PRINTF(...) ((void)0)
#define KYTY_GETENV(name)                                                                          \
	(std::integral_constant<bool, ::Common::Switches::IsDiagnostic(name)>::value                   \
	     ? static_cast<const char*>(nullptr)                                                       \
	     : ::Common::Switches::Get(                                                                \
	           std::integral_constant<uint64_t, ::Common::Switches::Hash(name)>::value))
// (The second parameter must not be called "value": it would replace the "::value" member below.)
#define KYTY_SET_DEFAULT_SWITCH(name, text)                                                        \
	::Common::Switches::SetDefault(                                                                \
	    std::integral_constant<uint64_t, ::Common::Switches::Hash(name)>::value, (text))
#else
#define KYTY_INTERNAL_PRINTF(...) ((void)::printf(__VA_ARGS__))
#define KYTY_GETENV(name) (std::getenv(name))
#define KYTY_SET_DEFAULT_SWITCH(name, value) ::Common::Switches::SetDefaultByName((name), (value))
namespace Common::Switches {
// Sets the environment variable unless it is set already.
void SetDefaultByName(const char* name, const char* value);
} // namespace Common::Switches
#endif

#endif // KYTY_COMMON_SWITCHES_H_
