#ifndef KYTY_COMMON_ASSERT_H_
#define KYTY_COMMON_ASSERT_H_

#include "common/common.h"
#include "common/logging/log.h"

#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <type_traits>

namespace Common {

#ifdef __clang__
int DbgExitHandler(char const* file, int line, std::string_view text)
    __attribute__((analyzer_noreturn));
int DbgExitHandler(char const* file, int line, fmt::text_style style, std::string_view text)
    __attribute__((analyzer_noreturn));
int DbgExitIfHandler(char const* expr, char const* file, int line)
    __attribute__((analyzer_noreturn));
int DbgNotImplementedHandler(char const* expr, char const* file, int line)
    __attribute__((analyzer_noreturn));
void DbgExit(int status) __attribute__((analyzer_noreturn));
#else
int  DbgExitHandler(char const* file, int line, std::string_view text);
int  DbgExitHandler(char const* file, int line, fmt::text_style style, std::string_view text);
int  DbgExitIfHandler(char const* expr, char const* file, int line);
int  DbgNotImplementedHandler(char const* expr, char const* file, int line);
void DbgExit(int status);
#endif

} // namespace Common

#define EXIT_HALT() (Common::DbgExit(321), 1)

#if defined(KYTY_SHARE_BUILD) && KYTY_SHARE_BUILD
// A share build (the build given to other people) keeps no source paths, conditions or messages
// of fatal errors: the report names a site code, a hash of the source file name and the line
// (the hash identifies the line). Message arguments are not evaluated.
namespace Common {
constexpr uint32_t ExitSiteCode(std::string_view file, uint32_t line) {
	const auto slash = file.find_last_of("/\\");
	const auto name  = slash == std::string_view::npos ? file : file.substr(slash + 1);
	uint32_t   hash  = 2166136261u;
	for (const char c: name) {
		hash = (hash ^ static_cast<uint8_t>(c)) * 16777619u;
	}
	return hash ^ (line * 0x9e3779b1u);
}
int DbgExitSiteHandler(uint32_t code);
} // namespace Common
#define KYTY_EXIT_SITE                                                                             \
	(std::integral_constant<uint32_t, ::Common::ExitSiteCode(__FILE__, __LINE__)>::value)
#define EXIT_IF(x)                                                                                 \
	((void)((x) && Common::DbgExitSiteHandler(KYTY_EXIT_SITE) != 0 && (EXIT_HALT(), 1) != 0))
#define EXIT(...)                                                                                  \
	do {                                                                                           \
		((void)(Common::DbgExitSiteHandler(KYTY_EXIT_SITE) && (EXIT_HALT(), 1)));                  \
	} while (0)
#define EXIT_COLOR(style, ...) EXIT()
#define EXIT_NOT_IMPLEMENTED(x)                                                                    \
	((void)((x) && Common::DbgExitSiteHandler(KYTY_EXIT_SITE) != 0 && (EXIT_HALT(), 1) != 0))
#else
#ifndef KYTY_FINAL
#define EXIT_IF(x)                                                                                 \
	((void)((x) && Common::DbgExitIfHandler(#x, __FILE__, __LINE__) != 0 && (EXIT_HALT(), 1) != 0))
#else
#define EXIT_IF(x)                                                                                 \
	do {                                                                                           \
		constexpr bool kyty_exit_if_disabled = false && (x);                                       \
		(void)kyty_exit_if_disabled;                                                               \
	} while (0)
#endif

#define EXIT(...)                                                                                  \
	do {                                                                                           \
		((void)(Common::DbgExitHandler(__FILE__, __LINE__, ::fmt::sprintf(__VA_ARGS__)) &&         \
		        (EXIT_HALT(), 1)));                                                                \
	} while (0)

#define EXIT_COLOR(style, ...)                                                                     \
	do {                                                                                           \
		((void)(Common::DbgExitHandler(__FILE__, __LINE__, (style),                                \
		                               ::fmt::sprintf(__VA_ARGS__)) &&                             \
		        (EXIT_HALT(), 1)));                                                                \
	} while (0)

#define EXIT_NOT_IMPLEMENTED(x)                                                                    \
	((void)((x) && Common::DbgNotImplementedHandler(#x, __FILE__, __LINE__) != 0 &&                \
	        (EXIT_HALT(), 1) != 0))
#endif
#define KYTY_NOT_IMPLEMENTED EXIT_NOT_IMPLEMENTED(true)

#endif /* KYTY_COMMON_ASSERT_H_ */
