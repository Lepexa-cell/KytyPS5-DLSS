#include "common/switches.h"
#include "graphics/host_gpu/renderer/debug.h"

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

// KYTY_DEBUG_FLIGHT_RECORDER (see debug.h). Events are a few dozen per frame (no per-draw lines
// except depth clears and watched shaders), formatted into strings under one mutex; a frame's
// events are dropped once they are <frames> flips old.

namespace Libs::Graphics {

namespace {

struct FlightEvent {
	uint64_t    flip = 0;
	std::string text;
};

struct FlightState {
	std::mutex              lock;
	std::deque<FlightEvent> events;
	uint64_t                flip         = 0;
	uint64_t                dump_at      = 0; // flip to dump at (0: none)
	std::string             dump_why;
	uint64_t                last_anomaly = 0;
	uint32_t                dumps        = 0;
	std::vector<uint32_t>   depth_bound; // depth images bound this frame
};

FlightState& State() {
	static FlightState state;
	return state;
}

uint64_t FrameCount() {
	static const uint64_t frames = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_FLIGHT_RECORDER");
		return text != nullptr ? std::max<uint64_t>(1, std::strtoull(text, nullptr, 10)) : 0;
	}();
	return frames;
}

const std::vector<uint64_t>& WatchedShaders() {
	static const std::vector<uint64_t> shaders = [] {
		std::vector<uint64_t> list;
		const char*           text = KYTY_GETENV("KYTY_DEBUG_FLIGHT_RECORDER_SHADERS");
		while (text != nullptr && *text != '\0') {
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) {
				break;
			}
			// Short prefixes as the logs print them (1959d762) match the full hash's top bits.
			const auto digits = static_cast<uint32_t>(end - text);
			list.push_back(digits < 16 ? value << (4u * (16u - digits)) : value);
			list.push_back(digits < 16 ? 16u - digits : 0u);
			text = *end == ',' ? end + 1 : end;
		}
		return list;
	}();
	return shaders;
}

void Push(FlightState& state, std::string text) {
	state.events.push_back({state.flip, std::move(text)});
}

void Dump(FlightState& state) {
	if (state.dumps >= 64) {
		return;
	}
	state.dumps++;
	const auto name = "_flight_" + std::to_string(state.flip) + "_" + state.dump_why + ".txt";
	if (FILE* file = std::fopen(name.c_str(), "w")) {
		std::fprintf(file, "flight-recorder dump at flip %" PRIu64 " (%s), %zu events, frames %" PRIu64 "\n",
		             state.flip, state.dump_why.c_str(), state.events.size(), FrameCount());
		uint64_t flip = UINT64_MAX;
		for (const auto& event: state.events) {
			if (event.flip != flip) {
				flip = event.flip;
				std::fprintf(file, "== flip %" PRIu64 "\n", flip);
			}
			std::fprintf(file, "%s\n", event.text.c_str());
		}
		std::fclose(file);
	}
	std::printf("flight-recorder: %s written (%zu events)\n", name.c_str(), state.events.size());
	std::fflush(stdout);
}

} // namespace

bool FlightRecorderEnabled() noexcept {
	return FrameCount() != 0;
}

bool FlightRecorderWatches(uint64_t pixel_hash, uint64_t vertex_hash) noexcept {
	const auto& shaders = WatchedShaders();
	for (size_t i = 0; i + 1 < shaders.size(); i += 2) {
		const auto shift = static_cast<uint32_t>(shaders[i + 1]) * 4u;
		const auto match = [&](uint64_t hash) {
			return hash != 0 && (shift == 0 ? hash == shaders[i] : (hash >> shift) == (shaders[i] >> shift));
		};
		if (match(pixel_hash) || match(vertex_hash)) {
			return true;
		}
	}
	return false;
}

void FlightRecord(const char* format, ...) {
	if (!FlightRecorderEnabled()) {
		return;
	}
	char    buffer[1024];
	va_list args;
	va_start(args, format);
	std::vsnprintf(buffer, sizeof(buffer), format, args);
	va_end(args);
	auto&            state = State();
	std::scoped_lock lock {state.lock};
	Push(state, buffer);
}

void FlightRecorderDepthBind(uint32_t image, uint64_t address, uint32_t width, uint32_t height,
                             bool game_clear, bool meta_clear, bool late_clear) {
	if (!FlightRecorderEnabled()) {
		return;
	}
	auto&            state = State();
	std::scoped_lock lock {state.lock};
	const bool       drawn = std::ranges::find(state.depth_bound, image) != state.depth_bound.end();
	if (drawn && !game_clear && !meta_clear && !late_clear) {
		return;
	}
	if (!drawn) {
		state.depth_bound.push_back(image);
	}
	char buffer[256];
	std::snprintf(buffer, sizeof(buffer), "depth-bind image %u 0x%" PRIx64 " %ux%u %s clr=%d meta_clear=%d late=%d",
	              image, address, width, height, drawn ? "again" : "first", game_clear ? 1 : 0,
	              meta_clear ? 1 : 0, late_clear ? 1 : 0);
	Push(state, buffer);
	// The anomaly: a clear the game did not ask for (no DEPTH_CLEAR_ENABLE, no HTile fill taken now)
	// of a depth target already drawn to this frame. Dumped two flips later, so the dump also holds
	// the frame after it; at most one per 10 flips.
	if (drawn && meta_clear && !game_clear && !late_clear && state.dump_at == 0 &&
	    (state.last_anomaly == 0 || state.flip >= state.last_anomaly + 10)) {
		state.last_anomaly = state.flip;
		state.dump_at      = state.flip + 2;
		state.dump_why     = "midframe_clear";
		Push(state, "ANOMALY midframe depth clear");
	}
}

void FlightRecorderRequestDump(const char* why) {
	if (!FlightRecorderEnabled()) {
		return;
	}
	auto&            state = State();
	std::scoped_lock lock {state.lock};
	state.dump_at  = state.flip + 1;
	state.dump_why = why;
	Push(state, std::string("REQUEST ") + why);
}

void FlightRecorderFlip() {
	if (!FlightRecorderEnabled()) {
		return;
	}
	static const char* trigger = KYTY_GETENV("KYTY_DEBUG_FLIGHT_RECORDER_FILE");
	auto&              state   = State();
	std::scoped_lock   lock {state.lock};
	state.flip++;
	state.depth_bound.clear();
	if (trigger != nullptr && state.flip % 15 == 0 && state.dump_at == 0) {
		std::error_code error;
		if (std::filesystem::exists(trigger, error)) {
			std::filesystem::remove(trigger, error);
			state.dump_at  = state.flip + 1;
			state.dump_why = "file";
		}
	}
	if (state.dump_at != 0 && state.flip >= state.dump_at) {
		Dump(state);
		state.dump_at = 0;
	}
	while (!state.events.empty() && state.events.front().flip + FrameCount() < state.flip) {
		state.events.pop_front();
	}
}

} // namespace Libs::Graphics
