#include "graphics/presentation/performanceOverlay.h"

#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/hostStats.h"
#include "common/switches.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/dirtyRefresh.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "imgui.h"
#include "kernel/memory.h"
#include "libs/agc.h"

#include <cstdlib>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

using Clock   = std::chrono::steady_clock;
using Counter = Common::DebugCounters::Counter;
using Gauge   = Common::DebugCounters::Gauge;

constexpr auto   REFRESH_INTERVAL  = std::chrono::seconds(1);
constexpr size_t FRAME_HISTORY     = 120;
constexpr float  FRAME_GRAPH_MS    = 50.0f;
constexpr float  FRAME_TIME_GOOD   = 17.5f;
constexpr float  FRAME_TIME_MEDIUM = 34.0f;
constexpr size_t COUNTER_COUNT     = static_cast<size_t>(Counter::Count);

// Tables as wide as their columns; the default fills the window, which an auto-resizing
// window then never shrinks. Faint stripes on every other row.
constexpr ImGuiTableFlags TABLE_FLAGS =
    ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoHostExtendX | ImGuiTableFlags_RowBg;

constexpr ImU32 GOOD    = IM_COL32(96, 208, 128, 255);
constexpr ImU32 MEDIUM  = IM_COL32(246, 190, 64, 255);
constexpr ImU32 BAD     = IM_COL32(238, 84, 84, 255);
constexpr ImU32 NEUTRAL = IM_COL32(150, 150, 154, 255);
constexpr ImU32 LABEL   = IM_COL32(146, 146, 152, 255);
constexpr ImU32 HINT    = IM_COL32(146, 146, 152, 130);
constexpr ImU32 ACCENT  = IM_COL32(255, 168, 38, 255);
constexpr ImU32 VALUE   = IM_COL32(242, 242, 242, 255);
constexpr ImU32 ROW_ALT = IM_COL32(255, 255, 255, 6);

// Panel background: a dark blue glass gradient, more opaque in the detailed view.
constexpr ImU32 PANEL         = IM_COL32(14, 14, 17, 214);
constexpr ImU32 PANEL_DETAILS = IM_COL32(14, 14, 17, 236);
constexpr ImU32 PANEL_EDGE    = IM_COL32(255, 255, 255, 28);

struct FrameWindow {
	std::array<float, FRAME_HISTORY> intervals_ms {};
	size_t                           next         = 0;
	size_t                           count        = 0;
	uint32_t                         flips        = 0;
	float                            max_ms       = 0.0f;
	uint32_t                         guest_width  = 0;
	uint32_t                         guest_height = 0;
	Clock::time_point                last_flip    = {};
	Clock::time_point                window_start = Clock::now();
};

// Shown values, recomputed once per refresh interval.
struct Snapshot {
	double                                   fps          = 0.0;
	double                                   frame_ms     = 0.0;
	double                                   max_frame_ms = 0.0;
	double                                   low_fps      = 0.0; // From the 99th percentile.
	double                                   stalled_s    = 0.0;
	uint32_t                                 guest_width  = 0;
	uint32_t                                 guest_height = 0;
	std::array<double, COUNTER_COUNT>        rates {};
	std::array<uint64_t, COUNTER_COUNT>      totals {};
	Common::HostStats                        host;
	std::optional<uint64_t>                  device_memory;
	uint64_t                                 device_budget = 0;
	std::vector<GraphicContext::HeapUsage>   heaps;
	LibKernel::Memory::DebugStats            guest_memory;
	uint64_t                                 render_size = 0; // Width << 32 | height.
	uint64_t                                 pssr_sizes  = 0; // 0: no PS5 Pro upscaling lately.
	const char*                              pssr_upscaler       = "linear";
	uint64_t                                 texture_cache_bytes = 0;
	uint64_t                                 texture_images      = 0;
	uint64_t                                 buffer_cache_bytes  = 0;
};

std::mutex            g_frame_mutex;
FrameWindow           g_frames;
std::once_flag        g_enabled_once;
std::atomic<bool>     g_enabled {false};
std::atomic<bool>     g_details {false};
std::atomic<uint64_t> g_revision {0};

std::mutex        g_host_mutex;
Common::HostStats g_host_stats;
std::jthread      g_sampler;

// The last PS5 Pro upscaling request: render w << 48 | h << 32 | output w << 16 | h, when (steady
// clock nanoseconds) and the upscaler's name, see PerformanceOverlayRecordPssr.
std::atomic<uint64_t>    g_pssr_sizes {0};
std::atomic<int64_t>     g_pssr_time_ns {0};
std::atomic<const char*> g_pssr_upscaler {"linear"};

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// Widest label and value of the two-column tables, measured while drawing and used on the next
// frame, so that the tables of one panel column line up: 0 = top of the panel, 1-3 = the
// detail columns.
struct ColumnWidths {
	float label = 0.0f;
	float value = 0.0f;
};
std::array<ColumnWidths, 4> g_widths;
std::array<ColumnWidths, 4> g_widths_next;
size_t                      g_group = 0;

// Segoe UI and Segoe UI Semibold (PerformanceOverlayLoadFonts), or nullptr: ImGui's default font.
ImFont* g_font_regular = nullptr;
ImFont* g_font_bold    = nullptr;

void EnsureEnabledLoaded() {
	std::call_once(g_enabled_once, [] {
		g_enabled.store(Config::PerfOverlayEnabled(), std::memory_order_release);
		// KYTY_PERF_OVERLAY_DETAILS=1: start with the detailed view (screenshots of test runs).
		if (std::getenv("KYTY_PERF_OVERLAY_DETAILS") != nullptr) g_details.store(true, std::memory_order_release);
	});
}

bool GuestStalled(Clock::time_point now) {
	std::scoped_lock lock(g_frame_mutex);
	return now - g_frames.last_flip >= REFRESH_INTERVAL;
}

// Host statistics can block for milliseconds (IOKit, thread enumeration), so sample them off
// the present path. While the guest stops flipping (loading screens), ask for one repaint per
// interval.
void SamplerThread(const std::stop_token& stop) {
	Common::HostStatsSampler    sampler;
	std::mutex                  wait_mutex;
	std::condition_variable_any wait;
	while (!stop.stop_requested()) {
		auto stats = sampler.Sample();
		{
			std::scoped_lock lock(g_host_mutex);
			g_host_stats = std::move(stats);
		}
		if (g_enabled.load(std::memory_order_acquire) && GuestStalled(Clock::now())) {
			g_revision.fetch_add(1, std::memory_order_acq_rel);
		}
		std::unique_lock lock(wait_mutex);
		wait.wait_for(lock, stop, REFRESH_INTERVAL, [] { return false; });
	}
}

void EnsureSampler() {
	static std::once_flag once;
	std::call_once(once, [] { g_sampler = std::jthread(SamplerThread); });
}

ImU32 FrameTimeColor(double ms) {
	return ms <= FRAME_TIME_GOOD ? GOOD : ms <= FRAME_TIME_MEDIUM ? MEDIUM : BAD;
}

// Green below `warn`, yellow below `bad`, red above.
ImU32 LevelColor(double value, double warn, double bad) {
	return value < warn ? GOOD : value < bad ? MEDIUM : BAD;
}

ImVec4 Color(ImU32 color) {
	return ImGui::ColorConvertU32ToFloat4(color);
}

ImU32 WithAlpha(ImU32 color, uint32_t alpha) {
	return (color & ~IM_COL32_A_MASK) | (alpha << IM_COL32_A_SHIFT);
}

std::string Bytes(uint64_t bytes) {
	constexpr double GIB = 1024.0 * 1024.0 * 1024.0;
	constexpr double MIB = 1024.0 * 1024.0;
	char             buffer[32];
	if (static_cast<double>(bytes) >= GIB) {
		std::snprintf(buffer, sizeof(buffer), "%.2f GB", static_cast<double>(bytes) / GIB);
	} else {
		std::snprintf(buffer, sizeof(buffer), "%.0f MB", static_cast<double>(bytes) / MIB);
	}
	return buffer;
}

std::string Bytes(std::optional<uint64_t> bytes) {
	return bytes ? Bytes(*bytes) : "n/a";
}

std::string Throughput(double bytes_per_second) {
	char buffer[32];
	std::snprintf(buffer, sizeof(buffer), "%.1f MB/s", bytes_per_second / (1024.0 * 1024.0));
	return buffer;
}

std::string Percent(std::optional<double> value) {
	if (!value) {
		return "n/a";
	}
	char buffer[16];
	std::snprintf(buffer, sizeof(buffer), "%.0f%%", *value);
	return buffer;
}

// Values sit at the right edge of their column. Labels and values are padded to the widest ones
// of the panel column (g_widths), so that its tables share one column layout.
void Row(const char* label, const std::string& value, std::optional<ImU32> color = std::nullopt) {
	const auto& widths      = g_widths[g_group];
	auto&       next        = g_widths_next[g_group];
	const float label_width = ImGui::CalcTextSize(label).x;
	const float value_width = ImGui::CalcTextSize(value.c_str()).x;
	next.label              = std::max(next.label, label_width);
	next.value              = std::max(next.value, value_width);
	ImGui::TableNextRow();
	ImGui::TableNextColumn();
	ImGui::TextDisabled("%s", label);
	if (widths.label > label_width) {
		ImGui::SameLine(0.0f, 0.0f);
		ImGui::Dummy({widths.label - label_width, 1.0f});
	}
	ImGui::TableNextColumn();
	if (widths.value > value_width) {
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() + widths.value - value_width);
	}
	if (color) {
		ImGui::TextColored(Color(*color), "%s", value.c_str());
	} else {
		ImGui::TextUnformatted(value.c_str());
	}
	// A little room after the value: the column's clip edge otherwise cuts its last pixels.
	ImGui::SameLine(0.0f, 0.0f);
	ImGui::Dummy({2.0f, 1.0f});
}

[[gnu::format(printf, 1, 2)]] std::string Fmt(const char* format, ...) {
	char    buffer[128];
	va_list args;
	va_start(args, format);
	std::vsnprintf(buffer, sizeof(buffer), format, args);
	va_end(args);
	return buffer;
}

// Small upper-case title with a thin line running to the right edge of the column.
void SectionTitle(const char* title) {
	ImGui::Spacing();
	ImGui::PushFont(g_font_bold, ImGui::GetFontSize() * 0.86f);
	ImGui::TextColored(Color(ACCENT), "%s", title);
	ImGui::PopFont();
}

bool BeginSection(const char* title) {
	SectionTitle(title);
	return ImGui::BeginTable(title, 2, TABLE_FLAGS);
}

// Frame times as a filled line graph (newest on the right), dashed 60 and 30 FPS marks. Drawn
// across the whole panel, but reserves only `width`, so the panel can still shrink.
void DrawFrameGraph(const std::array<float, FRAME_HISTORY>& history, size_t count, float reserved,
                    float height) {
	const float  width  = std::max(reserved, ImGui::GetContentRegionAvail().x);
	const ImVec2 origin = ImGui::GetCursorScreenPos();
	const ImVec2 end {origin.x + width, origin.y + height};
	auto*        draw = ImGui::GetWindowDrawList();
	draw->AddLine({origin.x, end.y}, end, IM_COL32(255, 255, 255, 36));
	const auto y_of = [&](float ms) {
		return end.y - std::min(ms, FRAME_GRAPH_MS) / FRAME_GRAPH_MS * height;
	};
	const float y60 = y_of(1000.0f / 60.0f);
	draw->AddLine({origin.x, y60}, {end.x, y60}, IM_COL32(255, 255, 255, 30));
	const float step = width / static_cast<float>(FRAME_HISTORY);
	const float bar  = std::max(step - 1.0f, 1.0f);
	for (size_t i = 0; i < count; i++) {
		const float x = end.x - static_cast<float>(count - i) * step;
		draw->AddRectFilled({x, y_of(history[i])}, {x + bar, end.y},
		                    WithAlpha(FrameTimeColor(history[i]), 200));
	}
	ImGui::Dummy({reserved, height});
}

// Gradient glass behind the panel, a stripe on the left in the frame-time colour and a light
// top edge. The window's own background is transparent.
void DrawPanelBackground(ImU32 accent, bool details, float scale) {
	auto*        draw = ImGui::GetWindowDrawList();
	const ImVec2 pos  = ImGui::GetWindowPos();
	const ImVec2 size = ImGui::GetWindowSize();
	const ImVec2 end {pos.x + size.x, pos.y + size.y};
	const float  round = 6.0f * scale;
	draw->PushClipRect(pos, end, false);
	draw->AddRectFilled(pos, end, details ? PANEL_DETAILS : PANEL, round);
	draw->AddRect(pos, end, PANEL_EDGE, round, 0, 1.0f);
	draw->AddLine({pos.x + round, end.y - 1.0f}, {end.x - round, end.y - 1.0f}, accent, 2.0f * scale);
	draw->PopClipRect();
}

// The panel's content width from the previous frame's tables and the graph, without the badge:
// pushed to the edge of the window itself, it would keep the panel from ever shrinking.
float ContentWidth(bool details, float graph_width) {
	const float pad   = ImGui::GetStyle().CellPadding.x;
	const auto  table = [&](const ColumnWidths& widths) { return widths.label + widths.value + 2.0f + 4.0f * pad; };
	float       width = std::max(graph_width, table(g_widths[0]));
	if (details) {
		float columns = 0.0f;
		for (size_t group = 1; group < g_widths.size(); group++) {
			columns += table(g_widths[group]) + 2.0f * pad;
		}
		width = std::max(width, columns);
	}
	return width;
}

// A small outlined "PRO" tag at the end of the header line, right-aligned to `right` (screen x) and
// centred on a line of `line_height`, when the Pro mode is on (KYTY_TRINITY_MODE=1); nothing when it
// is off.
void DrawProBadge(float line_y, float line_height, float right, float scale, bool on) {
	if (!on) {
		return;
	}
	constexpr const char* NAME = "PRO";
	ImGui::PushFont(g_font_bold, ImGui::GetFontSize() * 0.78f);
	const ImVec2 text = ImGui::CalcTextSize(NAME);
	const float  pad  = 4.0f * scale;
	const ImVec2 size {text.x + pad * 2.0f, text.y + pad * 0.6f};
	ImGui::SameLine(0.0f, 16.0f * scale);
	const float room = right - size.x - ImGui::GetCursorScreenPos().x;
	if (room > 0.0f) {
		ImGui::SetCursorPosX(ImGui::GetCursorPosX() + room);
	}
	ImGui::SetCursorPosY(line_y + std::max((line_height - size.y) * 0.5f, 0.0f));
	auto*        draw   = ImGui::GetWindowDrawList();
	const ImVec2 origin = ImGui::GetCursorScreenPos();
	draw->AddRect(origin, {origin.x + size.x, origin.y + size.y}, ACCENT, 3.0f * scale, 0, 1.0f);
	draw->AddText({origin.x + pad, origin.y + pad * 0.3f}, ACCENT, NAME);
	ImGui::PopFont();
	ImGui::Dummy(size);
}

// Large FPS number, then the frame time on the same baseline; the Pro badge ends at `right`.
void DrawHeader(const Snapshot& shown, float big_size, float right, float scale) {
	const float line_y      = ImGui::GetCursorPosY();
	float       line_height = ImGui::GetFontSize();
	if (shown.fps > 0.0) {
		ImGui::PushFont(g_font_bold, big_size);
		const float big_ascent = ImGui::GetFontBaked()->Ascent;
		line_height            = ImGui::GetFontSize();
		ImGui::TextColored(Color(VALUE), "%.0f", shown.fps);
		ImGui::PopFont();
		const float small_y = line_y + big_ascent - ImGui::GetFontBaked()->Ascent;
		ImGui::SameLine(0.0f, 4.0f * scale);
		ImGui::SetCursorPosY(small_y);
		ImGui::TextColored(Color(LABEL), "fps");
		ImGui::SameLine(0.0f, 12.0f * scale);
		ImGui::SetCursorPosY(small_y);
		ImGui::TextColored(Color(FrameTimeColor(shown.frame_ms)), "%.1f ms", shown.frame_ms);
	} else if (shown.stalled_s >= 1.0) {
		ImGui::TextColored(Color(BAD), "No frames for %.0f s", shown.stalled_s);
	} else {
		ImGui::TextDisabled("Waiting for frames");
	}
	DrawProBadge(line_y, line_height, right, scale, Gen5::TrinityModeTest());
}

// Render size and, on a PS5 Pro, the upscaler standing in for Sony's PSSR (renderPssr.cpp:
// a linear filter) with its input and output size.
void DrawRender(const Snapshot& shown) {
	SectionTitle("Render");
	if (!ImGui::BeginTable("##render", 2, TABLE_FLAGS)) {
		return;
	}
	const auto sizes = shown.pssr_sizes;
	if (Gen5::TrinityModeTest()) {
		if (sizes == 0) {
			Row("Upscaler", "idle (no PSSR request)");
		} else {
			Row("Upscaler", "Linear", MEDIUM);
		}
	}
	if (sizes != 0) {
		Row("Render > output",
		    Fmt("%ux%u > %ux%u", static_cast<uint32_t>(sizes >> 48u), static_cast<uint32_t>(sizes >> 32u) & 0xffffu,
		        static_cast<uint32_t>(sizes >> 16u) & 0xffffu, static_cast<uint32_t>(sizes) & 0xffffu));
	} else if (shown.render_size != 0) {
		Row("Render size", Fmt("%ux%u", static_cast<uint32_t>(shown.render_size >> 32u),
		                       static_cast<uint32_t>(shown.render_size)));
	}
	ImGui::EndTable();
}

const char* PressureName(Common::MemoryPressure pressure) {
	switch (pressure) {
		case Common::MemoryPressure::Normal: return "Normal";
		case Common::MemoryPressure::Warning: return "Warning";
		case Common::MemoryPressure::Critical: return "Critical";
		case Common::MemoryPressure::Unknown: break;
	}
	return "n/a";
}

ImU32 PressureColor(Common::MemoryPressure pressure) {
	switch (pressure) {
		case Common::MemoryPressure::Normal: return GOOD;
		case Common::MemoryPressure::Warning: return MEDIUM;
		case Common::MemoryPressure::Critical: return BAD;
		case Common::MemoryPressure::Unknown: break;
	}
	return NEUTRAL;
}

const char* ThermalName(Common::ThermalPressure thermal) {
	switch (thermal) {
		case Common::ThermalPressure::Nominal: return "Nominal";
		case Common::ThermalPressure::Moderate: return "Moderate";
		case Common::ThermalPressure::Heavy: return "Heavy (throttling)";
		case Common::ThermalPressure::Critical: return "Critical (throttling)";
		case Common::ThermalPressure::Unknown: break;
	}
	return "n/a";
}

ImU32 ThermalColor(Common::ThermalPressure thermal) {
	switch (thermal) {
		case Common::ThermalPressure::Nominal: return GOOD;
		case Common::ThermalPressure::Moderate: return MEDIUM;
		case Common::ThermalPressure::Heavy:
		case Common::ThermalPressure::Critical: return BAD;
		case Common::ThermalPressure::Unknown: break;
	}
	return NEUTRAL;
}

const char* PresentModeName(Config::PresentMode mode) {
	switch (mode) {
		case Config::PresentMode::Fifo: return "FIFO";
		case Config::PresentMode::Mailbox: return "Mailbox";
		case Config::PresentMode::Immediate: return "Immediate";
	}
	return "?";
}

double Rate(const Snapshot& shown, Counter counter) {
	return shown.rates[static_cast<size_t>(counter)];
}

uint64_t Total(const Snapshot& shown, Counter counter) {
	return shown.totals[static_cast<size_t>(counter)];
}

void Refresh(Snapshot& shown, const GraphicContext& graphics, Clock::time_point now) {
	static std::array<uint64_t, COUNTER_COUNT> previous {};
	static Clock::time_point                   previous_time;
	const double seconds = previous_time == Clock::time_point {}
	                           ? 0.0
	                           : std::chrono::duration<double>(now - previous_time).count();
	for (size_t i = 0; i < COUNTER_COUNT; i++) {
		const auto value = Common::DebugCounters::Get(static_cast<Counter>(i));
		shown.rates[i] =
		    seconds > 0.0 ? static_cast<double>(value - previous[i]) / seconds : 0.0;
		shown.totals[i] = value;
		previous[i]     = value;
	}
	previous_time = now;

	{
		std::scoped_lock lock(g_host_mutex);
		shown.host = g_host_stats;
	}
	shown.device_memory = graphics.CanReportMemoryUsage()
	                          ? std::optional(graphics.GetDeviceMemoryUsage())
	                          : std::nullopt;
	shown.device_budget = graphics.GetTotalMemoryBudget();
	if (g_details.load(std::memory_order_acquire)) {
		shown.heaps        = graphics.GetHeapUsage();
		shown.guest_memory = LibKernel::Memory::GetDebugStats();
	}
	// The viewport size drawn most often (not square, so no shadow map): the game's render
	// resolution, also with dynamic resolution.
	shown.render_size = g_hud.render_size.load(std::memory_order_relaxed);
	const bool pssr   = NowNs() - g_pssr_time_ns.load(std::memory_order_relaxed) < 2'000'000'000;
	shown.pssr_sizes  = pssr ? g_pssr_sizes.load(std::memory_order_relaxed) : 0;
	shown.pssr_upscaler       = g_pssr_upscaler.load(std::memory_order_relaxed);
	shown.texture_cache_bytes = Common::DebugCounters::Get(Gauge::TextureCacheBytes);
	shown.texture_images      = Common::DebugCounters::Get(Gauge::TextureImages);
	shown.buffer_cache_bytes  = Common::DebugCounters::Get(Gauge::BufferCacheBytes);
}

void DrawCompact(const Snapshot& shown) {
	SectionTitle("System");
	if (!ImGui::BeginTable("##performance", 2, TABLE_FLAGS)) {
		return;
	}
	const auto& host = shown.host;
	if (shown.max_frame_ms > 0.0) {
		Row("Worst frame", Fmt("%.1f ms", shown.max_frame_ms),
		    FrameTimeColor(shown.max_frame_ms));
	}
	Row("CPU", host.process_cpu_percent
	               ? Fmt("%.0f%%  of %u threads", *host.process_cpu_percent,
	                          host.logical_cores)
	               : std::string("n/a"));
	Row("GPU", Percent(host.gpu_percent));
	if (shown.device_memory) {
		Row("GPU memory", Bytes(*shown.device_memory) + " / " + Bytes(shown.device_budget));
	}
	Row("Emulator RAM", Bytes(host.process_memory));
	if (host.system_memory_used && host.system_memory_total) {
		Row("System RAM", Bytes(*host.system_memory_used) + " / " + Bytes(*host.system_memory_total));
	}
	if (host.swap_used) {
		Row("Swap", Bytes(*host.swap_used));
	}
	Row("Memory pressure", PressureName(host.memory_pressure), PressureColor(host.memory_pressure));
	ImGui::EndTable();
}

// Where the frame time goes, from counters the panel already keeps (no extra cost): the command
// thread (Thread_Gpu, one core), the GPU, and the time the command thread waits for the GPU. The
// verdict names the first limit that is saturated, so a slow scene points at its lever at once.
void DrawBottleneck(const Snapshot& shown) {
	if (shown.fps <= 0.0) {
		return;
	}
	SectionTitle("Bottleneck");
	if (!ImGui::BeginTable("##bottleneck", 2, TABLE_FLAGS)) {
		return;
	}
	const auto& host     = shown.host;
	const auto  frame_ms = shown.frame_ms;
	double      cp_percent = 0.0;
	std::string busiest;
	double      busiest_percent = 0.0;
	for (const auto& thread: host.busiest_threads) {
		if (thread.name.find("Thread_Gpu") != std::string::npos) {
			cp_percent = thread.cpu_percent;
		} else if (thread.name.find("Thread_Present") == std::string::npos &&
		           thread.cpu_percent > busiest_percent) {
			busiest         = thread.name;
			busiest_percent = thread.cpu_percent;
		}
	}
	const double gpu_percent = host.gpu_percent.value_or(0.0);
	const double gpu_wait_ms = Rate(shown, Counter::GpuWaitNs) / 1e6 / std::max(shown.fps, 1.0);
	Row("Frame", Fmt("%.1f ms  (worst %.1f)", frame_ms, shown.max_frame_ms),
	    FrameTimeColor(frame_ms));
	Row("Command thread", Fmt("%.0f%%  = %.1f ms", cp_percent, cp_percent / 100.0 * frame_ms),
	    LevelColor(cp_percent, 70, 90));
	Row("GPU busy", Fmt("%.0f%%  = %.1f ms", gpu_percent, gpu_percent / 100.0 * frame_ms),
	    LevelColor(gpu_percent, 70, 90));
	Row("Waits for GPU", Fmt("%.1f ms/frame", gpu_wait_ms),
	    LevelColor(gpu_wait_ms / std::max(frame_ms, 1.0) * 100.0, 20, 40));
	if (!busiest.empty()) {
		Row(Fmt("  %.20s", busiest.c_str()).c_str(), Fmt("%.0f%%", busiest_percent),
		    LevelColor(busiest_percent, 70, 90));
	}
	const char* verdict = "none saturated: pacing / sync";
	ImU32       color   = GOOD;
	if (gpu_percent >= 90.0) {
		verdict = "GPU (render size, shader cost)";
		color   = BAD;
	} else if (cp_percent >= 90.0) {
		verdict = "command thread (per-draw CPU work)";
		color   = BAD;
	} else if (busiest_percent >= 90.0) {
		verdict = "game thread (guest code)";
		color   = MEDIUM;
	} else if (gpu_wait_ms > 0.3 * frame_ms) {
		verdict = "CPU waits for GPU (overlap)";
		color   = MEDIUM;
	} else if (frame_ms > 17.0) {
		verdict = "spread: nothing at 90% (lanes / drains)";
		color   = MEDIUM;
	}
	Row("Limit", verdict, color);
	ImGui::EndTable();
}

void DrawFramesAndCpu(const Snapshot& shown, vk::Extent2D extent) {
	const auto& host = shown.host;
	if (BeginSection("Frames")) {
		if (shown.low_fps > 0.0) {
			Row("1% low", Fmt("%.1f FPS", shown.low_fps),
			    FrameTimeColor(1000.0 / shown.low_fps));
		}
		Row("Worst frame", Fmt("%.1f ms", shown.max_frame_ms),
		    FrameTimeColor(shown.max_frame_ms));
		Row("Guest output", shown.guest_width != 0
		                        ? Fmt("%ux%u", shown.guest_width, shown.guest_height)
		                        : std::string("n/a"));
		Row("Window", Fmt("%ux%u  %s  %u Hz", extent.width, extent.height,
		                       PresentModeName(Config::GetPresentMode()),
		                       Config::GetVblankFrequency()));
		Row("Flips", Fmt("%.0f/s requested", Rate(shown, Counter::FlipsRequested)));
		const auto rejected = Rate(shown, Counter::FlipsRejected);
		Row("Flip queue full", Fmt("%.0f/s", rejected), rejected > 0.0 ? BAD : GOOD);
		const auto late = Rate(shown, Counter::LateVblanks);
		Row("Late vblanks", Fmt("%.0f/s", late), LevelColor(late, 1.0, 10.0));
		Row("Swapchain rebuilds",
		    Fmt("%llu  (dropped %llu)",
		             static_cast<unsigned long long>(Total(shown, Counter::SwapchainRecreations)),
		             static_cast<unsigned long long>(Total(shown, Counter::DroppedPresents))));
		ImGui::EndTable();
	}
	if (BeginSection("CPU")) {
		Row("Emulator", host.process_cpu_percent
		                    ? Fmt("%.0f%% of %u threads  (%.1f cores)",
		                               *host.process_cpu_percent, host.logical_cores,
		                               *host.process_cpu_percent * host.logical_cores / 100.0)
		                    : std::string("n/a"));
		Row("Machine", Percent(host.system_cpu_percent),
		    host.system_cpu_percent ? std::optional(LevelColor(*host.system_cpu_percent, 70, 90))
		                            : std::nullopt);
		const auto created = Total(shown, Counter::GuestThreadsCreated);
		const auto exited  = Total(shown, Counter::GuestThreadsExited);
		Row("Threads", Fmt("%u host  %llu guest", host.thread_count.value_or(0),
		                        static_cast<unsigned long long>(created - std::min(created, exited))));
		for (const auto& thread: host.busiest_threads) {
			Row(Fmt("  %.24s", thread.name.c_str()).c_str(),
			    Fmt("%.0f%%", thread.cpu_percent), LevelColor(thread.cpu_percent, 70, 95));
		}
		if (host.context_switches_per_s) {
			Row("Context switches", Fmt("%.0f/s", *host.context_switches_per_s));
		}
		if (host.page_faults_per_s) {
			Row("Page faults", Fmt("%.0f/s  (%.0f from disk)", *host.page_faults_per_s,
			                            host.page_ins_per_s.value_or(0.0)));
		}
		Row("Thermal", ThermalName(host.thermal_pressure), ThermalColor(host.thermal_pressure));
		const auto emulated = Rate(shown, Counter::EmulatedInstructions);
		Row("Emulated instrs", Fmt("%.0f/s", emulated), LevelColor(emulated, 1000, 100000));
		const auto unresolved = Rate(shown, Counter::UnresolvedImportCalls);
		Row("Missing HLE calls",
		    Fmt("%.0f/s  (%llu total)", unresolved,
		             static_cast<unsigned long long>(Total(shown, Counter::UnresolvedImportCalls))),
		    unresolved > 0.0 ? MEDIUM : GOOD);
		ImGui::EndTable();
	}
}

void DrawGpu(const Snapshot& shown, const GraphicContext& graphics) {
	const auto& host = shown.host;
	if (BeginSection("GPU")) {
		Row("Device", host.gpu_model.empty()
		                  ? std::string(graphics.GetPhysicalDeviceProperties().deviceName.data())
		                  : host.gpu_cores ? Fmt("%s (%u cores)", host.gpu_model.c_str(),
		                                              *host.gpu_cores)
		                                   : host.gpu_model);
		Row("Utilization", Percent(host.gpu_percent),
		    host.gpu_percent ? std::optional(LevelColor(*host.gpu_percent, 80, 95)) : std::nullopt);
		if (host.gpu_renderer_percent || host.gpu_tiler_percent) {
			Row("Renderer / tiler", Percent(host.gpu_renderer_percent) + " / " +
			                            Percent(host.gpu_tiler_percent));
		}
		if (host.gpu_memory_in_use) {
			Row("Driver memory", Bytes(*host.gpu_memory_in_use) + " in use / " +
			                         Bytes(host.gpu_memory_allocated) + " allocated");
		}
		if (host.gpu_recoveries) {
			Row("GPU resets", Fmt("%llu", static_cast<unsigned long long>(*host.gpu_recoveries)),
			    *host.gpu_recoveries > 0 ? BAD : GOOD);
		}
		const double fps   = std::max(shown.fps, 1.0);
		const auto   draws = Rate(shown, Counter::Draws);
		Row("Draws", Fmt("%.0f/frame  %.0f/s", draws / fps, draws));
		Row("Instances", Fmt("%.0f/frame", Rate(shown, Counter::DrawInstances) / fps));
		Row("Indirect draws", Fmt("%.0f/s", Rate(shown, Counter::IndirectDraws)));
		Row("Dispatches", Fmt("%.0f/frame  %.0f/s", Rate(shown, Counter::Dispatches) / fps,
		                           Rate(shown, Counter::Dispatches)));
		Row("Queue submits", Fmt("%.0f/s", Rate(shown, Counter::QueueSubmits)));
		// Includes present pacing waits, so a high value alone is not a stall.
		Row("CPU waiting on GPU", Fmt("%.0f ms/s  (%.0f waits/s)", Rate(shown, Counter::GpuWaitNs) / 1e6,
		                              Rate(shown, Counter::GpuWaits)));
		const auto faults = Rate(shown, Counter::GpuFaults);
		Row("Write-tracking faults", Fmt("%.0f/s", faults), LevelColor(faults, 20000, 100000));
		const auto shaders = Total(shown, Counter::ShadersCompiled);
		Row("Shaders", Fmt("%llu  (avg %.1f ms, %.0f/s now)",
		                        static_cast<unsigned long long>(shaders),
		                        shaders != 0 ? static_cast<double>(Total(shown, Counter::ShaderCompileNs)) /
		                                           1e6 / static_cast<double>(shaders)
		                                     : 0.0,
		                        Rate(shown, Counter::ShadersCompiled)),
		    Rate(shown, Counter::ShadersCompiled) > 0.0 ? MEDIUM : GOOD);
		const auto pipelines = Total(shown, Counter::PipelinesCreated);
		Row("Pipelines", Fmt("%llu  (avg %.1f ms, %.0f/s now)",
		                          static_cast<unsigned long long>(pipelines),
		                          pipelines != 0
		                              ? static_cast<double>(Total(shown, Counter::PipelineCreateNs)) /
		                                    1e6 / static_cast<double>(pipelines)
		                              : 0.0,
		                          Rate(shown, Counter::PipelinesCreated)),
		    Rate(shown, Counter::PipelinesCreated) > 0.0 ? MEDIUM : GOOD);
		ImGui::EndTable();
	}
	if (BeginSection("GPU memory")) {
		if (shown.device_memory) {
			const double share = shown.device_budget != 0
			                         ? static_cast<double>(*shown.device_memory) /
			                               static_cast<double>(shown.device_budget) * 100.0
			                         : 0.0;
			Row("Vulkan budget", Bytes(*shown.device_memory) + " / " + Bytes(shown.device_budget),
			    LevelColor(share, 80, 95));
		}
		for (size_t heap = 0; heap < shown.heaps.size(); heap++) {
			const auto& usage = shown.heaps[heap];
			Row(Fmt("  Heap %zu%s", heap, usage.device_local ? " (device)" : "").c_str(),
			    Bytes(usage.usage) + " / " + Bytes(usage.budget) +
			        Fmt("  %u allocs", usage.allocations));
		}
		Row("Texture cache", Bytes(shown.texture_cache_bytes) +
		                         Fmt("  %llu images", static_cast<unsigned long long>(shown.texture_images)));
		Row("  Upload / download", Throughput(Rate(shown, Counter::TextureUploadBytes)) + " / " +
		                               Throughput(Rate(shown, Counter::TextureDownloadBytes)));
		Row("  Evictions", Fmt("%.0f/s", Rate(shown, Counter::TextureEvictions)));
		Row("Buffer cache", Bytes(shown.buffer_cache_bytes));
		Row("  Upload / download", Throughput(Rate(shown, Counter::BufferUploadBytes)) + " / " +
		                               Throughput(Rate(shown, Counter::BufferDownloadBytes)));
		Row("  Evictions", Fmt("%.0f/s", Rate(shown, Counter::BufferEvictions)));
		ImGui::EndTable();
	}
}

void DrawMemoryAndIo(const Snapshot& shown) {
	const auto& host  = shown.host;
	const auto& guest = shown.guest_memory;
	if (BeginSection("PS5 memory")) {
		const auto direct_used = guest.direct_allocated + guest.pooled_allocated;
		Row("Direct", Bytes(direct_used) + " / " + Bytes(guest.direct_total),
		    LevelColor(guest.direct_total != 0 ? static_cast<double>(direct_used) /
		                                             static_cast<double>(guest.direct_total) * 100.0
		                                       : 0.0,
		               85, 97));
		Row("  Mapped", Bytes(guest.direct_mapped));
		Row("  Pool", Bytes(guest.pooled_allocated) + " (" + Bytes(guest.pool_committed) +
		                  " committed)");
		Row("Flexible", Bytes(guest.flexible_used) + " / " + Bytes(guest.flexible_total));
		Row("Extended", Bytes(guest.automatic_allocated));
		// Past the pool size the guest sees no page-table entries available.
		const auto entries = std::max(guest.cpu_page_entries, guest.gpu_page_entries);
		Row("Page table (2 MB)",
		    Fmt("CPU %llu  GPU %llu  / %llu",
		             static_cast<unsigned long long>(guest.cpu_page_entries),
		             static_cast<unsigned long long>(guest.gpu_page_entries),
		             static_cast<unsigned long long>(guest.page_entries_total)),
		    entries > guest.page_entries_total ? BAD
		    : entries * 10 > guest.page_entries_total * 9 ? MEDIUM
		                                                  : GOOD);
		ImGui::EndTable();
	}
	if (BeginSection("Emulator process")) {
		Row("Footprint", Bytes(host.process_memory));
		Row("Resident", Bytes(host.process_resident));
		if (host.process_compressed) {
			Row("Compressed / swapped", Bytes(*host.process_compressed),
			    *host.process_compressed > (1ull << 30u) ? MEDIUM : GOOD);
		}
		Row("File-backed", Bytes(host.process_file_backed));
		Row("Address space", Bytes(host.process_virtual));
		ImGui::EndTable();
	}
	if (BeginSection("System memory")) {
		if (host.system_memory_used && host.system_memory_total) {
			Row("Used", Bytes(*host.system_memory_used) + " / " + Bytes(*host.system_memory_total));
		}
		if (host.system_wired) {
			Row("Wired", Bytes(*host.system_wired));
		}
		if (host.system_compressed) {
			Row("Compressed", Bytes(*host.system_compressed));
		}
		Row("Cached files", Bytes(host.system_cached));
		Row("Free", Bytes(host.system_free));
		if (host.swap_used) {
			Row("Swap", Bytes(*host.swap_used) + " / " + Bytes(host.swap_total),
			    *host.swap_used > (4ull << 30u) ? MEDIUM : GOOD);
		}
		Row("Pressure", PressureName(host.memory_pressure), PressureColor(host.memory_pressure));
		ImGui::EndTable();
	}
	if (BeginSection("I/O & audio")) {
		Row("Disk read / write", Throughput(host.disk_read_per_s.value_or(0.0)) + " / " +
		                             Throughput(host.disk_write_per_s.value_or(0.0)));
		Row("Guest file read / write", Throughput(Rate(shown, Counter::GuestFileReadBytes)) + " / " +
		                                   Throughput(Rate(shown, Counter::GuestFileWriteBytes)));
		const auto underruns = Rate(shown, Counter::AudioUnderruns);
		Row("Audio underruns",
		    Fmt("%.0f/s  (%llu total)", underruns,
		             static_cast<unsigned long long>(Total(shown, Counter::AudioUnderruns))),
		    underruns > 0.0 ? BAD : GOOD);
		ImGui::EndTable();
	}
}

} // namespace

namespace {

// The F9 menu's rows; a row shows only where it does something.
enum class MenuRow { ImageRefresh, Timestamp, Panel, Capture };

std::atomic<bool> g_menu_open {false};
std::atomic<int>  g_menu_row {0};

std::vector<MenuRow> MenuRows() {
	std::vector<MenuRow> rows;
	if (DirtyTrackAvailable()) {
		rows.push_back(MenuRow::ImageRefresh);
	}
	rows.push_back(MenuRow::Timestamp);
	rows.push_back(MenuRow::Panel);
	const char* passes = KYTY_GETENV("KYTY_DEBUG_FRAME_PASSES");
	const char* images = KYTY_GETENV("KYTY_DEBUG_DUMP_IMAGES_FILE");
	if ((passes != nullptr && *passes != '\0') || (images != nullptr && *images != '\0')) {
		rows.push_back(MenuRow::Capture);
	}
	return rows;
}

// The capture F9 started before it opened the menu: the trigger files of KYTY_DEBUG_FRAME_PASSES
// and KYTY_DEBUG_DUMP_IMAGES_FILE.
void TriggerCaptures() {
	for (const char* path: {KYTY_GETENV("KYTY_DEBUG_FRAME_PASSES"), KYTY_GETENV("KYTY_DEBUG_DUMP_IMAGES_FILE")}) {
		if (path != nullptr && *path != '\0') {
			if (std::FILE* file = std::fopen(path, "wb"); file != nullptr) {
				std::fclose(file);
				std::printf("F9: capture triggered (%s)\n", path);
				std::fflush(stdout);
			}
		}
	}
}

const char* PanelState() {
	return !g_enabled.load(std::memory_order_acquire)  ? "Off"
	       : g_details.load(std::memory_order_acquire) ? "Details"
	                                                    : "Compact";
}

void ChangeRow(MenuRow row, int delta) {
	switch (row) {
		case MenuRow::ImageRefresh:
			SetDirtyTrackPaused(!DirtyTrackPaused());
			std::printf("F9: exact image refresh %s\n", DirtyTrackPaused() ? "paused" : "on");
			std::fflush(stdout);
			break;
		case MenuRow::Timestamp: {
			// GPU timestamp scale in 5 % steps (25-200 %), for this run only: never saved, so values
			// under 100 cannot reach kyty_settings.ini (older exes refuse to start with them).
			const int percent = static_cast<int>(Config::GetGpuTimestampScalePercent()) + (delta < 0 ? -5 : 5);
			Config::SetGpuTimestampScalePercent(static_cast<uint32_t>(std::clamp(percent, 25, 200)));
			std::printf("F9: timestamp scale %u%%\n", Config::GetGpuTimestampScalePercent());
			std::fflush(stdout);
			break;
		}
		case MenuRow::Panel: {
			// Off -> Compact -> Details -> Off (backwards the other way).
			const int state = !g_enabled.load(std::memory_order_acquire) ? 0 : g_details.load(std::memory_order_acquire) ? 2 : 1;
			const int next  = (state + (delta < 0 ? 2 : 1)) % 3;
			g_enabled.store(next != 0, std::memory_order_release);
			g_details.store(next == 2, std::memory_order_release);
			break;
		}
		case MenuRow::Capture:
			if (delta == 0) {
				TriggerCaptures();
			}
			break;
	}
}

void DrawDebugMenu(vk::Extent2D extent) {
	if (!g_menu_open.load(std::memory_order_acquire)) {
		return;
	}
	const float scale = std::clamp(std::min(static_cast<float>(extent.width) / 1920.0f,
	                                        static_cast<float>(extent.height) / 1080.0f),
	                               0.75f, 1.5f);
	const auto rows     = MenuRows();
	const int  selected = std::clamp(g_menu_row.load(std::memory_order_acquire), 0, static_cast<int>(rows.size()) - 1);
	ImGui::SetNextWindowPos({static_cast<float>(extent.width) - 12.0f * scale, 12.0f * scale},
	                        ImGuiCond_Always, {1.0f, 0.0f});
	ImGui::SetNextWindowBgAlpha(0.0f);
	constexpr ImGuiWindowFlags FLAGS =
	    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
	    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
	    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_AlwaysAutoResize;
	ImGui::PushFont(g_font_regular, 16.0f * scale);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {12.0f * scale, 8.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8.0f * scale, 3.0f * scale});
	if (ImGui::Begin("##kyty_debug_menu", nullptr, FLAGS)) {
		DrawPanelBackground(ACCENT, true, scale);
		SectionTitle("Menu");
		if (ImGui::BeginTable("##menu", 2, TABLE_FLAGS)) {
			for (int i = 0; i < static_cast<int>(rows.size()); i++) {
				const char* label = "";
				std::string value;
				switch (rows[i]) {
					case MenuRow::ImageRefresh:
						label = "Exact image refresh";
						value = DirtyTrackPaused() ? "Paused" : "On";
						break;
					case MenuRow::Timestamp:
						label = "Timestamp";
						value = Fmt("%u%%", Config::GetGpuTimestampScalePercent());
						break;
					case MenuRow::Panel:
						label = "Panel";
						value = PanelState();
						break;
					case MenuRow::Capture:
						label = "Capture";
						value = "Enter";
						break;
				}
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::TextColored(Color(i == selected ? ACCENT : LABEL), "%s %s", i == selected ? ">" : " ", label);
				ImGui::TableNextColumn();
				ImGui::TextColored(Color(i == selected ? VALUE : NEUTRAL), "%s", value.c_str());
			}
			ImGui::EndTable();
		}
		ImGui::Spacing();
		ImGui::TextColored(Color(HINT), "Up/Down choose   Left/Right/Enter change   F9 close");
	}
	ImGui::End();
	ImGui::PopStyleVar(4);
	ImGui::PopFont();
}

} // namespace

bool DebugMenuOpen() noexcept {
	return g_menu_open.load(std::memory_order_acquire);
}

void ToggleDebugMenu() noexcept {
	EnsureEnabledLoaded();
	g_menu_open.store(!g_menu_open.load(std::memory_order_acquire), std::memory_order_release);
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

void PressDebugMenuKey(DebugMenuKey key) noexcept {
	const auto rows = MenuRows();
	const int  last = static_cast<int>(rows.size()) - 1;
	const int  row  = std::clamp(g_menu_row.load(std::memory_order_acquire), 0, last);
	switch (key) {
		case DebugMenuKey::Up: g_menu_row.store(row == 0 ? last : row - 1, std::memory_order_release); break;
		case DebugMenuKey::Down: g_menu_row.store(row == last ? 0 : row + 1, std::memory_order_release); break;
		case DebugMenuKey::Left: ChangeRow(rows[row], -1); break;
		case DebugMenuKey::Right: ChangeRow(rows[row], 1); break;
		case DebugMenuKey::Enter: ChangeRow(rows[row], rows[row] == MenuRow::Capture ? 0 : 1); break;
		case DebugMenuKey::Close: g_menu_open.store(false, std::memory_order_release); break;
	}
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

void PerformanceOverlayLoadFonts() {
#if defined(_WIN32)
	auto&       fonts   = *ImGui::GetIO().Fonts;
	const char* windows = std::getenv("WINDIR");
	const auto  folder  = std::string(windows != nullptr ? windows : "C:\\Windows") + "\\Fonts\\";
	const auto  load    = [&](const char* file) -> ImFont* {
        const auto path = folder + file;
        if (std::FILE* probe = std::fopen(path.c_str(), "rb"); probe != nullptr) {
            std::fclose(probe);
            return fonts.AddFontFromFileTTF(path.c_str(), 16.0f);
        }
        return nullptr;
	};
	// The first font is the default one, which the system dialogs keep.
	if (fonts.Fonts.empty()) {
		fonts.AddFontDefault();
	}
	g_font_regular = load("bahnschrift.ttf");
	g_font_bold    = g_font_regular;
	if (g_font_regular == nullptr) {
		g_font_regular = load("segoeui.ttf");
		g_font_bold    = load("seguisb.ttf");
		if (g_font_bold == nullptr) {
			g_font_bold = g_font_regular;
		}
	}
#endif
}

void PerformanceOverlayRecordPssr(const char* upscaler, uint32_t render_width, uint32_t render_height,
                                  uint32_t output_width, uint32_t output_height) noexcept {
	g_pssr_upscaler.store(upscaler, std::memory_order_relaxed);
	g_pssr_sizes.store(uint64_t {render_width & 0xffffu} << 48u | uint64_t {render_height & 0xffffu} << 32u |
	                       uint64_t {output_width & 0xffffu} << 16u | (output_height & 0xffffu),
	                   std::memory_order_relaxed);
	g_pssr_time_ns.store(NowNs(), std::memory_order_relaxed);
}

void PerformanceOverlayRecordFlip(uint32_t guest_width, uint32_t guest_height) noexcept {
	const auto       now = Clock::now();
	std::scoped_lock lock(g_frame_mutex);
	auto&            frames = g_frames;
	if (frames.last_flip != Clock::time_point {}) {
		const float ms = std::chrono::duration<float, std::milli>(now - frames.last_flip).count();
		frames.intervals_ms[frames.next] = ms;
		frames.next                      = (frames.next + 1) % FRAME_HISTORY;
		frames.count                     = std::min(frames.count + 1, FRAME_HISTORY);
		frames.max_ms                    = std::max(frames.max_ms, ms);
	}
	frames.last_flip    = now;
	frames.guest_width  = guest_width;
	frames.guest_height = guest_height;
	frames.flips++;
}

bool PerformanceOverlayEnabled() noexcept {
	EnsureEnabledLoaded();
	return g_enabled.load(std::memory_order_acquire);
}

void TogglePerformanceOverlay() noexcept {
	EnsureEnabledLoaded();
	g_enabled.store(!g_enabled.load(std::memory_order_acquire), std::memory_order_release);
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

void TogglePerformanceOverlayDetails() noexcept {
	EnsureEnabledLoaded();
	if (g_enabled.load(std::memory_order_acquire)) {
		g_details.store(!g_details.load(std::memory_order_acquire), std::memory_order_release);
	} else {
		g_details.store(true, std::memory_order_release);
		g_enabled.store(true, std::memory_order_release);
	}
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

uint64_t PerformanceOverlayRevision() noexcept {
	return g_revision.load(std::memory_order_acquire);
}

float DrawPerformanceOverlay(const GraphicContext& graphics, vk::Extent2D extent) {
	DrawDebugMenu(extent);
	// While the panel is shown, draws count their viewport sizes (HudStats) for the render size.
	g_hud.active.store(PerformanceOverlayEnabled(), std::memory_order_relaxed);
	if (!PerformanceOverlayEnabled()) {
		return 0.0f;
	}
	EnsureSampler();

	static Snapshot                         shown;
	static Clock::time_point                next_refresh;
	static bool                             shown_details = false;
	static std::array<float, FRAME_HISTORY> history {};
	static size_t                           history_count = 0;
	const bool                              details = g_details.load(std::memory_order_acquire);
	const auto                              now     = Clock::now();
	const bool refresh = now >= next_refresh || details != shown_details;
	{
		std::scoped_lock lock(g_frame_mutex);
		auto&            frames = g_frames;
		history_count           = frames.count;
		for (size_t i = 0; i < frames.count; i++) {
			history[i] = frames.intervals_ms[(frames.next + FRAME_HISTORY - frames.count + i) %
			                                 FRAME_HISTORY];
		}
		if (refresh) {
			const double seconds = std::chrono::duration<double>(now - frames.window_start).count();
			shown.fps            = seconds > 0.0 ? frames.flips / seconds : 0.0;
			shown.frame_ms       = frames.flips > 0 ? seconds * 1000.0 / frames.flips : 0.0;
			shown.max_frame_ms   = frames.max_ms;
			shown.stalled_s      = frames.last_flip == Clock::time_point {}
			                           ? 0.0
			                           : std::chrono::duration<double>(now - frames.last_flip).count();
			shown.guest_width    = frames.guest_width;
			shown.guest_height   = frames.guest_height;
			frames.flips         = 0;
			frames.max_ms        = 0.0f;
			frames.window_start  = now;
		}
	}
	if (refresh) {
		next_refresh  = now + REFRESH_INTERVAL;
		shown_details = details;
		if (history_count > 0) {
			auto sorted = std::vector<float>(history.begin(), history.begin() + history_count);
			const auto percentile = sorted.begin() + static_cast<ptrdiff_t>(sorted.size() * 99 / 100);
			std::nth_element(sorted.begin(), percentile, sorted.end());
			shown.low_fps = *percentile > 0.0f ? 1000.0 / *percentile : 0.0;
		}
		Refresh(shown, graphics, now);
	}

	const float scale = std::clamp(std::min(static_cast<float>(extent.width) / 1920.0f,
	                                        static_cast<float>(extent.height) / 1080.0f),
	                               0.75f, 1.5f);
	// Top left; the F9 menu takes the top right.
	ImGui::SetNextWindowPos({12.0f * scale, 12.0f * scale}, ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.0f);
	constexpr ImGuiWindowFlags FLAGS =
	    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
	    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
	    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoBringToFrontOnFocus |
	    ImGuiWindowFlags_AlwaysAutoResize;
	const float font_size = (details ? 14.0f : 16.0f) * scale;
	ImGui::PushFont(g_font_regular, font_size);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {12.0f * scale, 8.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8.0f * scale, 2.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {5.0f * scale, 1.0f * scale});
	ImGui::PushStyleColor(ImGuiCol_TextDisabled, LABEL);
	ImGui::PushStyleColor(ImGuiCol_TableRowBg, IM_COL32(0, 0, 0, 0));
	ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ROW_ALT);
	if (ImGui::Begin("##kyty_performance_overlay", nullptr, FLAGS)) {
		g_widths      = g_widths_next;
		g_widths_next = {};
		g_group       = 0;
		const ImU32 accent = shown.fps > 0.0         ? FrameTimeColor(shown.frame_ms)
		                     : shown.stalled_s >= 1.0 ? BAD
		                                              : NEUTRAL;
		DrawPanelBackground(accent, details, scale);
		const float graph_width = (details ? 330.0f : 260.0f) * scale;
		DrawHeader(shown, font_size * 2.0f,
		           ImGui::GetCursorScreenPos().x + ContentWidth(details, graph_width), scale);
		ImGui::Spacing();
		DrawFrameGraph(history, history_count, graph_width, 46.0f * scale);
		DrawRender(shown);
		if (details) {
			DrawBottleneck(shown);
			DrawCompact(shown);
			// Three columns keep the detailed panel within a 720p window.
			if (ImGui::BeginTable("##details", 3, TABLE_FLAGS)) {
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				g_group = 1;
				DrawFramesAndCpu(shown, extent);
				ImGui::TableNextColumn();
				g_group = 2;
				DrawGpu(shown, graphics);
				ImGui::TableNextColumn();
				g_group = 3;
				DrawMemoryAndIo(shown);
				ImGui::EndTable();
			}
			g_group = 0;
			ImGui::Spacing();
#if defined(__APPLE__)
			ImGui::TextColored(Color(HINT), "Cmd+Shift+P compact   Cmd+P hide   F9 menu");
#else
			ImGui::TextColored(Color(HINT), "Shift+F2 compact   F2 hide   F9 menu");
#endif
		}
	}
	const float bottom = ImGui::GetWindowPos().y + ImGui::GetWindowSize().y;
	ImGui::End();
	ImGui::PopStyleColor(3);
	ImGui::PopStyleVar(5);
	ImGui::PopFont();
	return bottom;
}

} // namespace Libs::Graphics
