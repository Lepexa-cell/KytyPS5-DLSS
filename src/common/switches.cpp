#include "common/switches.h"

#include "common/common.h"

#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
extern char** environ; // NOLINT(readability-redundant-declaration)
#endif

namespace Common::Switches {

#if defined(KYTY_SHARE_BUILD) && KYTY_SHARE_BUILD

namespace {

struct Entry {
	uint64_t    hash;
	std::string value;
};

// Entries never move (deque), so the values handed out stay valid.
struct Table {
	std::deque<Entry> environment;
	std::deque<Entry> defaults;
	std::mutex        mutex;
};

void AddFromEnvironment(Table& table, std::string_view text) {
	const auto equals = text.find('=');
	if (equals == std::string_view::npos || !text.starts_with("KYTY_")) {
		return;
	}
	table.environment.push_back({Hash(text.substr(0, equals)), std::string(text.substr(equals + 1))});
}

Table& GetTable() {
	// The environment as the process started; switches only change through SetDefault after.
	static Table* table = [] {
		auto* result = new Table;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (char* block = GetEnvironmentStringsA(); block != nullptr) {
			for (const char* at = block; *at != '\0'; at += std::strlen(at) + 1) {
				AddFromEnvironment(*result, at);
			}
			FreeEnvironmentStringsA(block);
		}
#else
		for (char** at = environ; at != nullptr && *at != nullptr; at++) {
			AddFromEnvironment(*result, *at);
		}
#endif
		return result;
	}();
	return *table;
}

} // namespace

const char* Get(uint64_t hash) {
	auto& table = GetTable();
	for (const auto& entry: table.environment) {
		if (entry.hash == hash) {
			return entry.value.c_str();
		}
	}
	std::lock_guard lock(table.mutex);
	for (const auto& entry: table.defaults) {
		if (entry.hash == hash) {
			return entry.value.c_str();
		}
	}
	return nullptr;
}

void SetDefault(uint64_t hash, const char* value) {
	auto& table = GetTable();
	std::lock_guard lock(table.mutex);
	for (const auto* entries: {&table.environment, &table.defaults}) {
		for (const auto& entry: *entries) {
			if (entry.hash == hash) {
				return;
			}
		}
	}
	table.defaults.push_back({hash, value});
}

#else

const char* Get(uint64_t /*hash*/) {
	return nullptr;
}

void SetDefault(uint64_t /*hash*/, const char* /*value*/) {}

void SetDefaultByName(const char* name, const char* value) {
	if (std::getenv(name) == nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		_putenv_s(name, value);
#else
		setenv(name, value, 0);
#endif
	}
}

#endif

} // namespace Common::Switches
