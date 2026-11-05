#include "kernel/fileSystem.h"

#include "common/archive.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/dateTime.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/switches.h"
#include "graphics/host_gpu/hostMemory.h"
#include "kernel/memory.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "libs/network.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <memory>
#include <mutex>
#include <random>
#include <system_error>
#include <thread>
#include <vector>

namespace Libs::LibKernel::FileSystem {

LIB_NAME("libkernel", "libkernel");

constexpr int      DESCRIPTOR_MIN = 3;
constexpr uint64_t DIR_BLOCK_SIZE = 512;

enum class SpecialFile {
	None,
	Random,
};

class MountPoints {
public:
	struct MountPair {
		std::filesystem::path dir;
		std::string           point;
		std::shared_ptr<Common::ArchiveReader> archive;
	};

	MountPoints() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	virtual ~MountPoints() = default;

	KYTY_CLASS_NO_COPY(MountPoints);

	void Mount(const std::filesystem::path& folder, const std::string& point);
	void Umount(const std::string& folder_or_point);

	[[nodiscard]] std::filesystem::path ResolvePath(const std::string& mounted_name);

private:
	std::vector<MountPair> m_mount_pairs;
	Common::Mutex          m_mutex;
};

struct File {
	Common::File                        f;
	std::string                         name;
	std::filesystem::path               real_name;
	std::atomic_bool                    opened;
	std::atomic_bool                    directory;
	std::atomic_bool                    readable;
	std::atomic_bool                    writable;
	std::atomic_bool                    append;
	std::atomic_bool                    sync_writes;
	std::atomic_bool                    close_on_exec;
	SpecialFile                         special;
	Common::Mutex                       mutex;
	std::vector<uint8_t>                dirents;
	uint64_t                            dents_offset;
};

class FileDescriptors {
public:
	FileDescriptors() { EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread()); }
	virtual ~FileDescriptors() = default;

	KYTY_CLASS_NO_COPY(FileDescriptors);

	int                   CreateDescriptor();
	void                  DeleteDescriptor(int d);
	std::shared_ptr<File> GetFile(int d);
	std::shared_ptr<File> GetFile(const std::filesystem::path& real_name);
	void                  CloseAll();

private:
	std::vector<std::shared_ptr<File>> m_files;
	Common::Mutex                      m_mutex;
};

static MountPoints*     g_mount_points = nullptr;
static FileDescriptors* g_files        = nullptr;

static bool IsRandomDevice(const std::string& path) {
	return path == "/dev/random" || path == "/dev/urandom";
}

static void FillRandomBuffer(void* buf, size_t nbytes) {
	auto*              out = reinterpret_cast<uint8_t*>(buf);
	std::random_device random;

	for (size_t i = 0; i < nbytes; i++) {
		out[i] = static_cast<uint8_t>(random());
	}
}

static void SecToTimespec(KernelTimespec* ts, double sec) {
	ts->tv_sec  = static_cast<int64_t>(sec);
	ts->tv_nsec = static_cast<int64_t>((sec - static_cast<double>(ts->tv_sec)) * 1000000000.0);
}

} // namespace Libs::LibKernel::FileSystem

namespace Libs::Graphics {
void NoteStreamingThrash() noexcept;
} // namespace Libs::Graphics

namespace Libs::LibKernel::FileSystem {

// Like libAmpr's NoteRepeatedRead for APR reads: a texture streamer asked to keep more than it
// fits reloads the same textures in turn, and ASTRO BOT's warp to Sky Garden reads them through
// plain read(): the same large file read again and again starts the mip statistics relief
// (Graphics::NoteStreamingThrash), without which the warp could reload them forever.
// KYTY_DEBUG_FILE_THRASH=0 turns this off.
static void NoteRepeatedFileRead(const std::filesystem::path& path, int64_t offset, uint64_t size) {
	// The warp reloads each texture several times a second; the hub re-reads a few files now and
	// then, which with APR's 4 repeats started reliefs (and their brief coarse textures) for nothing.
	constexpr uint64_t MinSize = 1024 * 1024;
	constexpr uint32_t Repeats = 12;
	constexpr auto     Window  = std::chrono::seconds(10);
	static const bool  enabled = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_FILE_THRASH");
		return text == nullptr || std::strcmp(text, "0") != 0;
	}();
	if (!enabled || size < MinSize) {
		return;
	}
	struct Read {
		uint64_t                              key   = 0;
		uint32_t                              count = 0;
		std::chrono::steady_clock::time_point first;
	};
	static std::mutex            mutex;
	static std::array<Read, 256> reads;
	const uint64_t key = (std::hash<std::filesystem::path::string_type> {}(path.native()) *
	                      0x9E3779B97F4A7C15ull) ^
	                     (static_cast<uint64_t>(offset) * 0xC2B2AE3D27D4EB4Full) ^ size;
	const auto now  = std::chrono::steady_clock::now();
	bool       note = false;
	{
		std::scoped_lock lock(mutex);
		auto&            read = reads[(key ^ (key >> 29u)) % reads.size()];
		if (read.key != key || now - read.first > Window) {
			read = {key, 1, now};
		} else if (++read.count >= Repeats) {
			read = {key, 0, now};
			note = true;
		}
	}
	if (note) {
		static std::atomic<uint32_t> notes {0};
		if (notes.fetch_add(1) < 8) {
			std::printf("file-thrash: %s re-read (%" PRIu64 " bytes)\n", Common::PathToString(path).c_str(),
			            size);
			std::fflush(stdout);
		}
		Libs::Graphics::NoteStreamingThrash();
	}
}

// KYTY_TRACE_TEXTURE=<substring>[,<substring>...]: print the guest destination of every read
// from a matching file to stdout, next to KYTY_DEBUG_IMAGE_CHURN's image creation and release,
// to follow a streamed texture from its file read to its GPU image.
static void TraceTextureRead(const std::filesystem::path& path, const void* buf, uint32_t bytes,
                             int64_t offset) {
	static const std::vector<std::string> filters = [] {
		std::vector<std::string> list;
		if (const char* value = KYTY_GETENV("KYTY_TRACE_TEXTURE"); value != nullptr) {
			std::string all(value);
			for (size_t start = 0; start < all.size();) {
				const auto end = std::min(all.find(',', start), all.size());
				if (end > start) {
					list.push_back(all.substr(start, end - start));
				}
				start = end + 1;
			}
		}
		return list;
	}();
	if (filters.empty()) {
		return;
	}
	const auto name = Common::PathToString(path);
	for (const auto& filter: filters) {
		if (name.find(filter) != std::string::npos) {
			std::printf("tex-read: 0x%016" PRIx64 " size=0x%x pos=%" PRId64 " %s\n",
			            reinterpret_cast<uint64_t>(buf), bytes, offset,
			            name.substr(name.find_last_of("/\\") + 1).c_str());
			return;
		}
	}
}

static bool TraceReadable(uint64_t addr, uint64_t size) {
	uint64_t readable = 0;
	return addr >= 0x10000 && Libs::Graphics::HostMemoryQueryReadable(addr, size, readable) &&
	       readable >= size;
}

// Whether the object holds the name inline or one pointer away (the texture's file name).
static bool TraceObjectHasName(uint64_t obj, const char* name) {
	const auto len = std::strlen(name);
	if (!TraceReadable(obj, 0x200)) {
		return false;
	}
	const auto* bytes = reinterpret_cast<const char*>(obj);
	if (std::search(bytes, bytes + 0x200, name, name + len) != bytes + 0x200) {
		return true;
	}
	for (uint64_t offset = 0; offset < 0x200; offset += 8) {
		const auto ptr = *reinterpret_cast<const uint64_t*>(obj + offset);
		if ((ptr & 7u) != 0 || !TraceReadable(ptr, 0x100)) {
			continue;
		}
		const auto* str = reinterpret_cast<const char*>(ptr);
		if (std::search(str, str + 0x100, name, name + len) != str + 0x100) {
			return true;
		}
	}
	return false;
}

// KYTY_TRACE_LISTS=<name>: with a KYTY_TRACE_FRAME hit inside ASTRO BOT's texture streamer update
// (0x90743ec30), log both of its {key, texture} work lists ([rbp-0x58, rbp-0x38) streams in,
// [rbp-0x70, rbp-0x78) streams out), the entry of the texture whose name matches, and that
// texture's first fields, to see why one texture keeps moving between the lists.
static uint64_t TraceMs() {
	static const auto start = std::chrono::steady_clock::now();
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now() - start)
	                                 .count());
}

// The streamer manager keeps its texture records in a list at manager+0x10: nodes are
// {next, ?, value, texture} where the value's low float is the requested priority. Poll it about
// every millisecond and log each change of the named texture's record, to see whether the request
// really disappears between two streamer updates.
static void TraceStreamerRequests(uint64_t manager, const char* name) {
	std::thread([manager, name] {
		uint64_t texture   = 0;
		uint64_t last      = ~0ull;
		bool     last_seen = true;
		uint64_t last_size = 0;
		for (;;) {
			std::this_thread::sleep_for(std::chrono::microseconds(500));
			if (!TraceReadable(manager + 0x10, 8)) {
				continue;
			}
			const auto head = *reinterpret_cast<const uint64_t*>(manager + 0x10);
			if (!TraceReadable(head, 8)) {
				continue;
			}
			uint64_t node  = *reinterpret_cast<const uint64_t*>(head);
			uint64_t size  = 0;
			bool     seen  = false;
			uint64_t value = 0;
			for (; node != head && size < 20000; size++) {
				if ((node & 7u) != 0 || !TraceReadable(node, 0x20)) {
					break;
				}
				const auto node_value = *reinterpret_cast<const uint64_t*>(node + 0x10);
				const auto node_tex   = *reinterpret_cast<const uint64_t*>(node + 0x18);
				if (texture == 0 && TraceObjectHasName(node_tex, name)) {
					texture = node_tex;
					LOGF("STREAMREQ t=%" PRIu64 " found %s texture=%#" PRIx64 " node=%#" PRIx64 "\n",
					     TraceMs(), name, texture, node);
				}
				if (texture != 0 && node_tex == texture) {
					seen  = true;
					value = node_value;
				}
				node = *reinterpret_cast<const uint64_t*>(node);
			}
			if (texture == 0) {
				continue;
			}
			// The usage table the streamer asks (0x907070fd0): [0x90ec99a18] + 0x3158b0, a map at
			// +0x108 from texture id (record+0x28) to a usage slot. Absent means "not wanted".
			{
				static uint32_t last_table = ~0u;
				uint32_t        state      = 0; // 0 unreadable, 1 absent, 2 present
				uint64_t        map_size   = 0;
				uint16_t        slot       = 0xffff;
				if (TraceReadable(0x90ec99a18ull, 8) && TraceReadable(texture + 0x28, 4)) {
					const auto table = *reinterpret_cast<const uint64_t*>(0x90ec99a18ull) + 0x3158b0;
					const auto id    = *reinterpret_cast<const uint32_t*>(texture + 0x28);
					if (TraceReadable(table + 0x108, 16)) {
						const auto head = *reinterpret_cast<const uint64_t*>(table + 0x108);
						map_size        = *reinterpret_cast<const uint64_t*>(table + 0x110);
						if (TraceReadable(head, 0x20)) {
							state       = 1;
							uint64_t at = *reinterpret_cast<const uint64_t*>(head + 8);
							for (int depth = 0; depth < 64 && TraceReadable(at, 0x38) &&
							                    *reinterpret_cast<const uint8_t*>(at + 0x19) == 0;
							     depth++) {
								const auto key = *reinterpret_cast<const uint32_t*>(at + 0x20);
								if (key == id) {
									state           = 2;
									const auto slot_ptr = *reinterpret_cast<const uint64_t*>(at + 0x30);
									if (TraceReadable(slot_ptr, 2)) {
										slot = *reinterpret_cast<const uint16_t*>(slot_ptr);
									}
									break;
								}
								at = *reinterpret_cast<const uint64_t*>(at + (id < key ? 0 : 0x10));
							}
						}
					}
				}
				// The slot's usage record: a map at +0xb0 from slot to a pointer to {mip info, priority};
				// the streamer reads the priority as the low float of the second qword.
				uint64_t usage_ptr = 0;
				uint64_t usage_lo  = 0;
				uint64_t usage_hi  = 0;
				if (state == 2 && TraceReadable(0x90ec99a18ull, 8)) {
					const auto table = *reinterpret_cast<const uint64_t*>(0x90ec99a18ull) + 0x3158b0;
					if (TraceReadable(table + 0xb0, 8)) {
						const auto head = *reinterpret_cast<const uint64_t*>(table + 0xb0);
						if (TraceReadable(head, 0x20)) {
							uint64_t at = *reinterpret_cast<const uint64_t*>(head + 8);
							for (int depth = 0; depth < 64 && TraceReadable(at, 0x30) &&
							                    *reinterpret_cast<const uint8_t*>(at + 0x19) == 0;
							     depth++) {
								const auto key = *reinterpret_cast<const uint16_t*>(at + 0x20);
								if (key == slot) {
									usage_ptr = *reinterpret_cast<const uint64_t*>(at + 0x28);
									if (TraceReadable(usage_ptr, 16)) {
										usage_lo = *reinterpret_cast<const uint64_t*>(usage_ptr);
										usage_hi = *reinterpret_cast<const uint64_t*>(usage_ptr + 8);
									}
									break;
								}
								at = *reinterpret_cast<const uint64_t*>(at + (slot < key ? 0 : 0x10));
							}
						}
					}
				}
				static uint64_t last_lo = ~0ull;
				static uint64_t last_hi = ~0ull;
				const uint32_t packed = state | (static_cast<uint32_t>(slot) << 8);
				if (packed != last_table || usage_lo != last_lo || usage_hi != last_hi) {
					float prio = 0.0f;
					std::memcpy(&prio, &usage_hi, 4);
					LOGF("STREAMUSE t=%" PRIu64 " %s slot=%u map_size=%" PRIu64 " usage=%#" PRIx64
					     " {%#" PRIx64 ", %#" PRIx64 "} prio=%g\n",
					     TraceMs(), state == 2 ? "present" : (state == 1 ? "absent" : "unreadable"), slot,
					     map_size, usage_ptr, usage_lo, usage_hi, prio);
					last_table = packed;
					last_lo    = usage_lo;
					last_hi    = usage_hi;
				}
			}
			if (seen != last_seen || value != last || (size / 16) != (last_size / 16)) {
				float prio = 0.0f;
				std::memcpy(&prio, &value, 4);
				const float tex_prio =
				    TraceReadable(texture + 0x2c, 4) ? *reinterpret_cast<const float*>(texture + 0x2c) : -1.0f;
				LOGF("STREAMREQ t=%" PRIu64 " list=%" PRIu64 " %s value=%#" PRIx64 " prio=%g tex+0x2c=%g\n",
				     TraceMs(), size, seen ? "present" : "absent", value, prio, tex_prio);
				last_seen = seen;
				last      = value;
				last_size = size;
			}
		}
	}).detach();
}

static void TraceStreamerLists(const char* path, uint64_t rbp) {
	static const char* name = KYTY_GETENV("KYTY_TRACE_LISTS");
	if (name == nullptr || name[0] == '\0') {
		return;
	}
	auto local = [rbp](uint64_t offset) { return *reinterpret_cast<const uint64_t*>(rbp - offset); };
	static std::once_flag poller;
	std::call_once(poller, [&] {
		LOGF("STREAMREQ t=%" PRIu64 " streamer=%#" PRIx64 " manager=%#" PRIx64 "\n", TraceMs(),
		     local(0x88), local(0x80));
		TraceStreamerRequests(local(0x80), name);
	});
	struct List {
		const char* label;
		uint64_t    begin;
		uint64_t    end;
	};
	const List lists[] = {{"in", local(0x58), local(0x38)}, {"out", local(0x70), local(0x78)}};
	for (const auto& list: lists) {
		if (list.end < list.begin || list.end - list.begin > 0x100000 ||
		    (list.end != list.begin && !TraceReadable(list.begin, list.end - list.begin))) {
			LOGF("STREAMLIST path=%s list=%s begin=%#" PRIx64 " end=%#" PRIx64 " unreadable\n", path,
			     list.label, list.begin, list.end);
			continue;
		}
		// Entries are {key, record value}; the stream-in list's values are remembered so the next
		// stream-out list can show how many of the textures just streamed in it drops again.
		static std::vector<uint64_t> last_in;
		const auto  count = (list.end - list.begin) / 16;
		std::string line  = fmt::format("STREAMLIST t={} list={} count={}", TraceMs(), list.label, count);
		std::vector<uint64_t> values;
		uint64_t              dropped = 0;
		for (uint64_t i = 0; i < count; i++) {
			const auto key   = *reinterpret_cast<const uint64_t*>(list.begin + i * 16);
			const auto value = *reinterpret_cast<const uint64_t*>(list.begin + i * 16 + 8);
			values.push_back(value);
			if (std::strcmp(list.label, "in") == 0 && i < 40) {
				line += fmt::format(" {:x}:{:x}", key, value);
			}
			if (std::find(last_in.begin(), last_in.end(), value) != last_in.end()) {
				dropped++;
			}
		}
		if (std::strcmp(list.label, "in") == 0) {
			if (!values.empty()) {
				last_in = values;
			}
		} else {
			line += fmt::format(" of_last_in={}/{}", dropped, last_in.size());
		}
		LOGF("%s\n", line.c_str());
	}
}

// KYTY_TRACE_OPEN=<substring>: for every opened path containing the substring, log the guest
// return addresses found by walking the frame-pointer chain, and dump the code around each new
// one to _TraceCode/<address>.bin, to find the guest code that keeps reopening a file.
static void TraceOpenCaller(const char* path, uint64_t return_address, const void* frame) {
	static const char* filter = KYTY_GETENV("KYTY_TRACE_OPEN");
	if (filter == nullptr || filter[0] == '\0' || std::strstr(path, filter) == nullptr) {
		return;
	}
	static Common::Mutex         mutex;
	static std::vector<uint64_t> dumped;

	// The host/guest transition breaks the frame-pointer chain, so scan the stack above this
	// frame for return addresses into the main module's code.
	constexpr uint64_t kModuleBegin = 0x900000000ull;
	constexpr uint64_t kModuleEnd   = 0x910000000ull;
	constexpr int      kScanWords   = 0x800;
	constexpr int      kMaxCallers  = 16;

	// The stack also holds data pointers into the module (vtables, function tables); only a value
	// right behind a call instruction is a return address.
	auto follows_call = [](uint64_t ret) {
		const auto* code = reinterpret_cast<const uint8_t*>(ret);
		if (code[-5] == 0xe8) { // call rel32
			return true;
		}
		// call r/m64 (FF /2) with register, disp8, disp32 or SIB operands.
		for (int len = 2; len <= 7; len++) {
			if (code[-len] == 0xff && ((code[-len + 1] >> 3) & 7) == 2) {
				return true;
			}
		}
		return false;
	};

	std::vector<uint64_t> callers;
	callers.push_back(return_address);
	const auto* stack = static_cast<const uint64_t*>(frame);
	// Guest thread stacks can end a few KB above this frame; never scan past what is mapped.
	uint64_t readable = 0;
	if (!Libs::Graphics::HostMemoryQueryReadable(reinterpret_cast<uint64_t>(frame),
	                                            kScanWords * sizeof(uint64_t), readable)) {
		readable = 0;
	}
	const int scan_words = static_cast<int>(std::min<uint64_t>(kScanWords, readable / sizeof(uint64_t)));
	for (int i = 0; i < scan_words && static_cast<int>(callers.size()) < kMaxCallers; i++) {
		const auto value = stack[i];
		if (value >= kModuleBegin + 8 && value < kModuleEnd && follows_call(value) &&
		    std::find(callers.begin(), callers.end(), value) == callers.end()) {
			callers.push_back(value);
		}
	}

	// KYTY_TRACE_FRAME=<ret>,<ret>,...: for each listed return address found on the stack, the
	// callee saved the caller's rbp right below it; log the caller's locals [rbp-0xb0, rbp-0x28).
	static const std::vector<uint64_t> frame_returns = [] {
		std::vector<uint64_t> list;
		if (const char* value = KYTY_GETENV("KYTY_TRACE_FRAME"); value != nullptr) {
			for (const char* p = value; *p != '\0';) {
				char*      end    = nullptr;
				const auto parsed = std::strtoull(p, &end, 16);
				if (end == p) {
					break;
				}
				list.push_back(parsed);
				p = (*end == ',') ? end + 1 : end;
			}
		}
		return list;
	}();
	for (int i = 1; i < scan_words && !frame_returns.empty(); i++) {
		if (std::find(frame_returns.begin(), frame_returns.end(), stack[i]) == frame_returns.end()) {
			continue;
		}
		const auto slot = reinterpret_cast<uint64_t>(&stack[i]);
		const auto rbp  = stack[i - 1];
		const auto stack_end = reinterpret_cast<uint64_t>(frame) + readable;
		if (rbp <= slot || rbp > stack_end || rbp - 0xb0 < slot || (rbp & 7u) != 0) {
			continue;
		}
		std::string locals = fmt::format("FRAMETRACE path={} ret={:#x} rbp={:#x} locals(-0xb0..-0x28)=",
		                                 path, stack[i], rbp);
		for (uint64_t offset = 0xb0; offset > 0x28; offset -= 8) {
			locals += fmt::format("{:x} ", *reinterpret_cast<const uint64_t*>(rbp - offset));
		}
		LOGF("%s\n", locals.c_str());
		TraceStreamerLists(path, rbp);
	}

	std::string line = fmt::format("OPENTRACE path={} stack=", path);
	for (const auto ret: callers) {
		line += fmt::format("{:#x} ", ret);

		constexpr uint64_t kCodeDumpBefore = 0x2000;
		constexpr uint64_t kCodeDumpAfter  = 0x200;
		if (ret < kModuleBegin || ret >= kModuleEnd) {
			continue;
		}
		Common::LockGuard lock(mutex);
		if (std::find(dumped.begin(), dumped.end(), ret) == dumped.end()) {
			dumped.push_back(ret);
			std::filesystem::create_directories("_TraceCode");
			const auto name = fmt::format("_TraceCode/{:x}.bin", ret - kCodeDumpBefore);
			if (auto* out = std::fopen(name.c_str(), "wb"); out != nullptr) {
				std::fwrite(reinterpret_cast<const void*>(ret - kCodeDumpBefore), 1,
				            kCodeDumpBefore + kCodeDumpAfter, out);
				std::fclose(out);
			}
		}
	}
	LOGF("%s\n", line.c_str());
}

// Game content sits on a read-only file system on the console, so its timestamps never move.
// Fixed for the session: a "now" timestamp changes on every stat and looks like a modification.
static Common::DateTime SessionStartTime() {
	static const auto start = Common::DateTime::FromSystemUTC();
	return start;
}

// All times come from the write time. The host access time changes whenever a file is read
// (NTFS last-access updates), and engines that hot-reload assets on a changed st_ctim or
// st_atim then reload the same files forever.
static void FillStatTimes(FileStat* stat, const Common::DateTime& write_time) {
	SecToTimespec(&stat->st_mtim, write_time.ToUnix());
	stat->st_atim     = stat->st_mtim;
	stat->st_ctim     = stat->st_mtim;
	stat->st_birthtim = stat->st_mtim;
}

static uint64_t AlignDown(uint64_t value, uint64_t alignment) {
	return (alignment != 0 ? value & ~(alignment - 1) : value);
}

static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
	return (alignment != 0 ? (value + alignment - 1) & ~(alignment - 1) : value);
}

static std::vector<uint8_t> PackDirents(const std::vector<Common::File::DirEntry>& entries) {
	std::vector<uint8_t> dirents;
	uint64_t             offset             = 0;
	uint64_t             last_reclen_offset = 0;
	for (const auto& entry: entries) {
		const auto& name = entry.name;
		EXIT_NOT_IMPLEMENTED(name.size() > 255);
		const auto reclen = AlignUp(8 + name.size() + 1, 4);
		if (offset + reclen > dirents.size()) {
			if (!dirents.empty()) {
				auto* last_reclen =
				    reinterpret_cast<uint16_t*>(dirents.data() + last_reclen_offset);
				*last_reclen += static_cast<uint16_t>(dirents.size() - offset);
			}
			offset = dirents.size();
			dirents.resize(offset + DIR_BLOCK_SIZE);
		}
		*reinterpret_cast<uint32_t*>(dirents.data() + offset) =
		    Common::hash(name.data(), static_cast<uint32_t>(name.size()));
		*reinterpret_cast<uint16_t*>(dirents.data() + offset + 4) = static_cast<uint16_t>(reclen);
		dirents[offset + 6]                                       = entry.is_file ? 8 : 4;
		dirents[offset + 7] = static_cast<uint8_t>(name.size());
		memcpy(dirents.data() + offset + 8, name.c_str(), name.size() + 1);
		last_reclen_offset = offset + 4;
		offset += reclen;
	}
	if (!dirents.empty()) {
		auto* last_reclen = reinterpret_cast<uint16_t*>(dirents.data() + last_reclen_offset);
		*last_reclen += static_cast<uint16_t>(dirents.size() - offset);
	}
	return dirents;
}

static uint64_t ReadDirectory(File* file, void* buf, uint64_t count) {
	const auto remaining =
	    file->dents_offset < file->dirents.size() ? file->dirents.size() - file->dents_offset : 0;
	const auto bytes = std::min(count, remaining);
	if (bytes != 0) {
		Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buf), bytes);
		memcpy(buf, file->dirents.data() + file->dents_offset, bytes);
		file->dents_offset += bytes;
	}
	return bytes;
}

static int PosixToKernel(int posix_errno) {
	return KERNEL_ERROR_UNKNOWN + posix_errno;
}

int FileDescriptors::CreateDescriptor() {
	Common::LockGuard lock(m_mutex);

	auto file         = std::make_shared<File>();
	file->opened      = false;
	file->directory   = false;
	file->readable    = false;
	file->writable    = false;
	file->append      = false;
	file->sync_writes = false;
	file->special     = SpecialFile::None;

	int files_num = static_cast<int>(m_files.size());
	for (int index = 0; index < files_num; index++) {
		if (m_files[index] == nullptr) {
			m_files[index] = file;
			return index + DESCRIPTOR_MIN;
		}
	}

	m_files.push_back(file);
	return static_cast<int>(m_files.size()) + DESCRIPTOR_MIN - 1;
}

void FileDescriptors::DeleteDescriptor(int d) {
	Common::LockGuard lock(m_mutex);

	auto index = static_cast<size_t>(d - DESCRIPTOR_MIN);

	EXIT_IF(index >= m_files.size());
	EXIT_IF(m_files[index] == nullptr);
	EXIT_IF(m_files[index]->opened);

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	// Close host files opened before a failed descriptor setup.
	m_files[index]->f.Close();
#endif

	m_files[index].reset();
}

std::shared_ptr<File> FileDescriptors::GetFile(int d) {
	Common::LockGuard lock(m_mutex);

	auto index = static_cast<size_t>(d - DESCRIPTOR_MIN);

	if (index >= m_files.size()) {
		return nullptr;
	}

	return m_files[index];
}

std::shared_ptr<File> FileDescriptors::GetFile(const std::filesystem::path& real_name) {
	Common::LockGuard lock(m_mutex);

	for (const auto& f: m_files) {
		if (f != nullptr && f->real_name == real_name) {
			return f;
		}
	}

	return nullptr;
}

void FileDescriptors::CloseAll() {
	Common::LockGuard lock(m_mutex);

	for (auto& f: m_files) {
		if (f != nullptr) {
			if (f->opened) {
				f->f.Close();
			}
			f.reset();
		}
	}
}

void MountPoints::Mount(const std::filesystem::path& folder, const std::string& point) {
	Common::LockGuard lock(m_mutex);

	auto point_str  = Common::FixDirectorySlash(point);

	Umount(point_str);

	MountPair p;
	p.dir   = folder;
	p.point = point_str;
	p.archive = Common::OpenArchive(folder);

	m_mount_pairs.push_back(std::move(p));
}

void MountPoints::Umount(const std::string& folder_or_point) {
	Common::LockGuard lock(m_mutex);

	auto folder_or_point_str = Common::FixDirectorySlash(folder_or_point);

	const auto it = std::find_if(
	    m_mount_pairs.begin(), m_mount_pairs.end(), [&folder_or_point_str](const MountPair& p) {
		    return Common::FixDirectorySlash(Common::PathToGenericString(p.dir)) == folder_or_point_str ||
		           p.point == folder_or_point_str;
	    });
	if (it != m_mount_pairs.end()) {
		m_mount_pairs.erase(it);
	}
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// Resolve guest paths case-insensitively on case-sensitive hosts.
static std::filesystem::path ResolvePathIgnoringCase(const std::filesystem::path& path) {
	std::error_code ec;
	if (std::filesystem::exists(path, ec)) {
		return path;
	}

	// Preserve unmatched components for the caller's ENOENT path.
	std::filesystem::path resolved =
	    path.has_root_path() ? path.root_path() : std::filesystem::path(".");
	bool matched = true;

	for (const auto& component: path.relative_path()) {
		if (component.empty()) {
			continue;
		}

		auto candidate = resolved / component;
		if (!matched || std::filesystem::exists(candidate, ec)) {
			resolved = std::move(candidate);
			continue;
		}

		bool found = false;
		for (std::filesystem::directory_iterator entry(resolved, ec), end; entry != end;
		     entry.increment(ec)) {
			if (ec) {
				break;
			}
			if (Common::EqualNoCase(entry->path().filename().string(), component.string())) {
				resolved = entry->path();
				found    = true;
				break;
			}
		}

		if (!found) {
			resolved = std::move(candidate);
			matched  = false;
		}
	}

	return resolved;
}
#else
static bool HasWindowsForbiddenFilenameCharacter(const std::string& relative_path) {
	return relative_path.find_first_of("<>:\"|?*") != std::string::npos;
}
#endif

std::filesystem::path MountPoints::ResolvePath(const std::string& mounted_name) {
	Common::LockGuard lock(m_mutex);

	// Match the entire guest path so a mount root works with or without a trailing slash.
	const auto mounted_path = Common::FixDirectorySlash(mounted_name);
	const auto it           = std::find_if(
	    m_mount_pairs.begin(), m_mount_pairs.end(),
	    [&mounted_path](const MountPair& p) { return mounted_path.starts_with(p.point); });
	if (it != m_mount_pairs.end()) {
		const auto& p = *it;
		auto rel_path = Common::RemoveFirst(Common::FixFilenameSlash(mounted_name), p.point.size());
		while (rel_path.starts_with('/')) {
			rel_path = Common::RemoveFirst(rel_path, 1);
		}

		const auto native_rel_path = Common::PathFromUtf8(rel_path);
		if (Common::IsArchivePath(p.dir)) {
			return p.dir / native_rel_path;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (HasWindowsForbiddenFilenameCharacter(rel_path)) {
			::printf("FileSystem: Windows-incompatible guest filename: %s\n", mounted_name.c_str());
		}
		return p.dir / native_rel_path;
#else
		return ResolvePathIgnoringCase(p.dir / native_rel_path);
#endif
	}

	// Unmounted guest paths must not fall through to the host filesystem.
	return {};
}

void Initialize() {
	g_mount_points = new MountPoints;
	g_files        = new FileDescriptors;
}

void EmergencyShutdown() {
	if (g_files != nullptr) {
		g_files->CloseAll();
	}
}

void Shutdown() {
	EmergencyShutdown();
	delete g_files;
	delete g_mount_points;
	g_files        = nullptr;
	g_mount_points = nullptr;
}

void Mount(const std::filesystem::path& folder, const std::string& point) {

	g_mount_points->Mount(folder, point);
}

void Umount(const std::string& folder_or_point) {

	g_mount_points->Umount(folder_or_point);
}

std::filesystem::path GetRealFilename(const std::string& mounted_file_name) {

	return g_mount_points->ResolvePath(mounted_file_name);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int KYTY_SYSV_ABI KernelOpen(const char* path, int flags, uint16_t mode) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto flags_u = static_cast<uint32_t>(flags);

	LOGF("\t path = %s\n"
	     "\t flags = %08" PRIx32 "\n"
	     "\t mode = %04" PRIx16 "\n",
	     path, flags_u, mode);
	TraceOpenCaller(path, reinterpret_cast<uint64_t>(__builtin_return_address(0)),
	                __builtin_frame_address(0));
	// KYTY_DEBUG_FILE_OPENS=1: every guest open with its result ("missing" = ENOENT).
	static const bool log_opens = KYTY_GETENV("KYTY_DEBUG_FILE_OPENS") != nullptr;
	const auto        log_open  = [&](const char* result) {
		if (log_opens) [[unlikely]] {
			std::printf("file-open %s flags=%08" PRIx32 " -> %s\n", path,
			            static_cast<uint32_t>(flags), result);
			std::fflush(stdout);
		}
	};

	bool append    = (flags_u & 0x0008u) != 0;
	bool fsync     = (flags_u & 0x0080u) != 0;
	bool sync      = (flags_u & 0x0080u) != 0;
	bool creat     = (flags_u & 0x0200u) != 0;
	bool trunc     = (flags_u & 0x0400u) != 0;
	bool excl      = (flags_u & 0x0800u) != 0;
	bool dsync     = (flags_u & 0x1000u) != 0;
	bool direct    = (flags_u & 0x00010000u) != 0;
	bool directory = (flags_u & 0x00020000u) != 0;

	if (direct) {
		LOGF("\t O_DIRECT ignored on host file backend\n");
	}

	flags_u &= 0x3u;

	Common::File::Mode rw_mode = Common::File::Mode::Read;

	switch (flags_u) {
		case 0: rw_mode = Common::File::Mode::Read; break;
		case 1: rw_mode = Common::File::Mode::Write; break;
		case 2: rw_mode = Common::File::Mode::ReadWrite; break;
		default: EXIT("invalid flag_u: %u\n", flags_u);
	}

	EXIT_NOT_IMPLEMENTED(directory && rw_mode != Common::File::Mode::Read);
	EXIT_NOT_IMPLEMENTED(directory && (trunc || creat));

	int   descriptor = g_files->CreateDescriptor();
	auto  file       = g_files->GetFile(descriptor);

	EXIT_IF(file == nullptr || file->opened || file->directory);

	file->name          = path;
	file->readable      = rw_mode != Common::File::Mode::Write;
	file->writable      = rw_mode != Common::File::Mode::Read;
	file->append        = append;
	file->sync_writes   = fsync || sync || dsync;
	file->close_on_exec = (static_cast<uint32_t>(flags) & 0x00100000u) != 0;

	if (!directory && IsRandomDevice(file->name)) {
		file->real_name = file->name;
		file->special   = SpecialFile::Random;
		file->opened    = true;

		LOGF_COLOR(Log::Color::Green, "\tOpen device: %s, [ok]\n",
		           Common::PathToString(file->real_name).c_str());

		return descriptor;
	}

	file->real_name = g_mount_points->ResolvePath(file->name);
	if (file->real_name.empty()) {
		log_open("missing (no mount point)");
		g_files->DeleteDescriptor(descriptor);
		return KERNEL_ERROR_ENOENT;
	}
	if (Common::IsArchivePath(file->real_name) &&
	    (rw_mode != Common::File::Mode::Read || trunc || creat)) {
		g_files->DeleteDescriptor(descriptor);
		return KERNEL_ERROR_EROFS;
	}

	if (trunc && rw_mode == Common::File::Mode::Read) {
		g_files->DeleteDescriptor(descriptor);
		return KERNEL_ERROR_EACCES;
	}

	const auto info = Common::File::GetInfo(file->real_name);
	const bool dir_exist = info && !info->is_file;
	const bool file_exist = info && info->is_file;

	// A missing path opened without O_CREAT is ENOENT
	if (!creat && !dir_exist && !file_exist) {
		log_open("missing");
		g_files->DeleteDescriptor(descriptor);
		return KERNEL_ERROR_ENOENT;
	}
	log_open(dir_exist ? "dir" : "ok");

	if (directory || dir_exist) {
		if (!dir_exist) {
			g_files->DeleteDescriptor(descriptor);
			return KERNEL_ERROR_ENOTDIR;
		}

		EXIT_NOT_IMPLEMENTED(!directory && rw_mode != Common::File::Mode::Read);
		EXIT_NOT_IMPLEMENTED(!directory && (trunc || creat));

		const auto entries = Common::File::GetDirEntries(file->real_name);
		file->dirents      = PackDirents(entries);
		file->dents_offset = 0;
		file->directory    = true;

		LOGF_COLOR(Log::Color::Green, "\tOpen dir: %s, entries = %" PRIu32 ", [ok]\n",
		           Common::PathToString(file->real_name).c_str(),
		           static_cast<uint32_t>(entries.size()));

		for (const auto& f: entries) {
			LOGF("\t\t%s %s\n", f.is_file ? "[file]" : "[dir ]", f.name.c_str());
		}
	} else {
		bool result = false;

		if (excl && creat && file_exist) {
			g_files->DeleteDescriptor(descriptor);
			return KERNEL_ERROR_EEXIST;
		}

		if (creat && (!file_exist || trunc)) {
			Common::File::CreateDirectories(file->real_name.parent_path());
			result = file->f.Create(file->real_name);

			LOGF_COLOR(result ? Log::Color::Green : Log::Color::Red, "\tCreate: %s, %s\n",
			           Common::PathToString(file->real_name).c_str(), result ? "[ok]" : "[fail]");
		} else {
			result = file->f.Open(file->real_name, rw_mode);

			LOGF_COLOR(result ? Log::Color::Green : Log::Color::Red, "\tOpen: %s, %s\n",
			           Common::PathToString(file->real_name).c_str(), result ? "[ok]" : "[fail]");
		}

		if (result && trunc) {
			result = file->f.Truncate(0);
		}

		if (result && append) {
			file->f.Seek(file->f.Size());
		}

		if (!result || file->f.IsInvalid()) {
			g_files->DeleteDescriptor(descriptor);
			return KERNEL_ERROR_EACCES;
		}
	}

	file->opened = true;
	return descriptor;
}

int KYTY_SYSV_ABI KernelClose(int d) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (::Libs::Network::Net::IsSocket(d)) {
		const auto result = ::Libs::Network::Net::SocketClose(d);
		return (result == OK ? OK : PosixToKernel(::Libs::Network::NetToPosix(result)));
	}

	auto file = g_files->GetFile(d);

	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	{
		// Serialize against in-flight reads and writes, which hold this mutex
		// while they touch the host file.
		Common::LockGuard lock(file->mutex);

		if (!file->opened) {
			return KERNEL_ERROR_EBADF;
		}

		if (!file->directory && file->special == SpecialFile::None) {
			file->f.Close();
		}

		file->opened = false;
	}

	LOGF("\tClose: %s\n", Common::PathToString(file->real_name).c_str());

	g_files->DeleteDescriptor(d);

	return OK;
}

int KYTY_SYSV_ABI KernelFcntl(int d, int command, int arg) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	auto file = g_files->GetFile(d);
	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	switch (command) {
		case 1: return file->close_on_exec.load(std::memory_order_relaxed); // F_GETFD
		case 2: // F_SETFD, FD_CLOEXEC
			file->close_on_exec.store((arg & 1) != 0, std::memory_order_relaxed);
			return OK;
		default: return KERNEL_ERROR_EINVAL;
	}
}

int64_t KYTY_SYSV_ABI KernelRead(int d, void* buf, size_t nbytes) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (::Libs::Network::Net::IsSocket(d)) {
		const auto result = ::Libs::Network::Net::Recv(d, buf, nbytes, 0);
		return (result >= 0 ? result : PosixToKernel(*Posix::GetErrorAddr()));
	}

	auto file = g_files->GetFile(d);

	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	if (nbytes > INT_MAX) {
		return KERNEL_ERROR_EINVAL;
	}
	if (file->directory) {
		Common::LockGuard lock(file->mutex);
		const auto        bytes = ReadDirectory(file.get(), buf, nbytes);
		LOGF("\tRead %" PRIu64 " directory bytes from: %s\n", bytes,
		     Common::PathToString(file->real_name).c_str());
		return static_cast<int64_t>(bytes);
	}

	if (file->special == SpecialFile::Random) {
		FillRandomBuffer(buf, nbytes);

		LOGF("\tRead %" PRIu64 " random bytes from: %s\n", static_cast<uint64_t>(nbytes),
		     Common::PathToString(file->real_name).c_str());

		return static_cast<int64_t>(nbytes);
	}

	file->mutex.Lock();

	if (!file->opened) {
		file->mutex.Unlock();
		return KERNEL_ERROR_EBADF;
	}

	if (file->f.IsInvalid()) {
		file->mutex.Unlock();
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	const auto pos       = file->f.Tell();
	const auto file_size = file->f.Size();
	const auto remaining = pos < file_size ? file_size - pos : 0;
	Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buf),
	                         std::min<uint64_t>(nbytes, remaining));
	uint32_t bytes_read = 0;
	file->f.Read(buf, static_cast<uint32_t>(nbytes), &bytes_read);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileReadBytes, bytes_read);

	file->mutex.Unlock();

	LOGF("\tRead %u bytes from: %s\n", bytes_read, Common::PathToString(file->real_name).c_str());
	TraceTextureRead(file->real_name, buf, bytes_read, -1);
	NoteRepeatedFileRead(file->real_name, -1, bytes_read);

	return bytes_read;
}

int64_t KYTY_SYSV_ABI KernelWrite(int d, const void* buf, size_t nbytes) {
	PRINT_NAME();

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (d == 1 || d == 2) {
		auto*      out      = (d == 1 ? stdout : stderr);
		const auto written  = std::fwrite(buf, 1, nbytes, out);
		const auto flush_ok = std::fflush(out) == 0;

		if (written != nbytes || !flush_ok) {
			return KERNEL_ERROR_EIO;
		}

		return static_cast<int64_t>(written);
	}

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	if (::Libs::Network::Net::IsSocket(d)) {
		const auto result = ::Libs::Network::Net::Send(d, buf, nbytes, 0);
		return (result >= 0 ? result : PosixToKernel(*Posix::GetErrorAddr()));
	}

	auto file = g_files->GetFile(d);

	if (file == nullptr || !file->opened || !file->writable) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_NOT_IMPLEMENTED(file->directory);
	EXIT_NOT_IMPLEMENTED(file->special != SpecialFile::None);

	EXIT_NOT_IMPLEMENTED(nbytes > UINT_MAX);

	file->mutex.Lock();

	if (!file->opened) {
		file->mutex.Unlock();
		return KERNEL_ERROR_EBADF;
	}

	if (file->f.IsInvalid()) {
		file->mutex.Unlock();
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	uint32_t bytes_written = 0;
	if (file->append) {
		file->f.Seek(file->f.Size());
	}
	file->f.Write(buf, static_cast<uint32_t>(nbytes), &bytes_written);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileWriteBytes, bytes_written);
	if (file->sync_writes) {
		file->f.Flush();
	}

	file->mutex.Unlock();

	LOGF("\tWrite %u bytes to: %s\n", bytes_written, Common::PathToString(file->real_name).c_str());

	return bytes_written;
}

int64_t KYTY_SYSV_ABI KernelPread(int d, void* buf, size_t nbytes, int64_t offset) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (offset < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	auto file = g_files->GetFile(d);

	if (file == nullptr) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_NOT_IMPLEMENTED(file->directory);

	if (!file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_NOT_IMPLEMENTED(nbytes > UINT_MAX);

	if (file->special == SpecialFile::Random) {
		FillRandomBuffer(buf, nbytes);

		LOGF("\tRead %" PRIu64 " random bytes (pos = %" PRId64 ") from: %s\n",
		     static_cast<uint64_t>(nbytes), offset, Common::PathToString(file->real_name).c_str());

		return static_cast<int64_t>(nbytes);
	}

	file->mutex.Lock();

	if (!file->opened) {
		file->mutex.Unlock();
		return KERNEL_ERROR_EBADF;
	}

	if (file->f.IsInvalid()) {
		file->mutex.Unlock();
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	const auto pos       = file->f.Tell();
	const auto file_size = file->f.Size();
	const auto remaining =
	    static_cast<uint64_t>(offset) < file_size ? file_size - static_cast<uint64_t>(offset) : 0;
	Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buf),
	                         std::min<uint64_t>(nbytes, remaining));
	uint32_t bytes_read = 0;
	file->f.Seek(offset);
	file->f.Read(buf, static_cast<uint32_t>(nbytes), &bytes_read);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileReadBytes, bytes_read);
	file->f.Seek(pos);

	file->mutex.Unlock();

	LOGF("\tRead %u bytes (pos = %" PRId64 ") from: %s\n", bytes_read, offset,
	     Common::PathToString(file->real_name).c_str());
	TraceTextureRead(file->real_name, buf, bytes_read, offset);
	NoteRepeatedFileRead(file->real_name, offset, bytes_read);

	return bytes_read;
}

static int ValidateIovecs(const KernelIovec* iov, int iovcnt, int64_t offset,
                          std::vector<KernelIovec>* buffers, size_t* total) {
	constexpr int MaxIov = 1024;
	if (iovcnt < 0 || iovcnt > MaxIov || offset < 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if (iov == nullptr && iovcnt != 0) {
		return KERNEL_ERROR_EFAULT;
	}

	if (iovcnt != 0) {
		buffers->assign(iov, iov + iovcnt);
	}
	*total = 0;
	for (const auto& buffer: *buffers) {
		if (buffer.iov_len > INT_MAX - *total) {
			return KERNEL_ERROR_EINVAL;
		}
		if (buffer.iov_base == nullptr && buffer.iov_len != 0) {
			return KERNEL_ERROR_EFAULT;
		}
		*total += buffer.iov_len;
	}
	return OK;
}

int64_t KYTY_SYSV_ABI KernelPreadv(int d, const KernelIovec* iov, int iovcnt, int64_t offset) {
	PRINT_NAME();

	std::vector<KernelIovec> buffers;
	size_t                   total = 0;
	const int                error = ValidateIovecs(iov, iovcnt, offset, &buffers, &total);
	if (error != OK) {
		return error;
	}

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}
	if (::Libs::Network::Net::IsSocket(d)) {
		return KERNEL_ERROR_ESPIPE;
	}
	auto file = g_files->GetFile(d);
	if (file == nullptr || !file->opened || !file->readable) {
		return KERNEL_ERROR_EBADF;
	}

	Common::LockGuard lock(file->mutex);
	if (file->directory) {
		return KERNEL_ERROR_EISDIR;
	}
	if (total == 0) {
		return 0;
	}
	if (file->special == SpecialFile::Random) {
		for (const auto& buffer: buffers) {
			if (buffer.iov_len != 0) {
				Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buffer.iov_base),
				                         buffer.iov_len);
				FillRandomBuffer(buffer.iov_base, buffer.iov_len);
			}
		}
		return static_cast<int64_t>(total);
	}
	if (file->f.IsInvalid()) {
		return KERNEL_ERROR_EIO;
	}

	const auto position  = file->f.Tell();
	const auto file_size = file->f.Size();
	if (!file->f.Seek(static_cast<uint64_t>(offset))) {
		return KERNEL_ERROR_EIO;
	}
	auto    remaining  = static_cast<uint64_t>(offset) < file_size ? file_size - offset : 0;
	int64_t bytes_read = 0;
	for (const auto& buffer: buffers) {
		if (remaining == 0) {
			break;
		}
		if (buffer.iov_len == 0) {
			continue;
		}
		const auto count = static_cast<uint32_t>(std::min<uint64_t>(buffer.iov_len, remaining));
		Memory::InvalidateMemory(reinterpret_cast<uint64_t>(buffer.iov_base), count);
		uint32_t bytes = 0;
		file->f.Read(buffer.iov_base, count, &bytes);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileReadBytes, bytes);
		bytes_read += bytes;
		remaining -= bytes;
		if (bytes < count) {
			break;
		}
	}
	if (!file->f.Seek(position)) {
		return KERNEL_ERROR_EIO;
	}
	LOGF("\tReadv %" PRId64 " bytes (pos = %" PRId64 ", iovcnt = %d) from: %s\n", bytes_read,
	     offset, iovcnt, Common::PathToString(file->real_name).c_str());
	return bytes_read;
}

int64_t KYTY_SYSV_ABI KernelPwrite(int d, const void* buf, size_t nbytes, int64_t offset) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (offset < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	auto file = g_files->GetFile(d);

	if (file == nullptr || !file->opened || !file->writable) {
		return KERNEL_ERROR_EBADF;
	}

	EXIT_NOT_IMPLEMENTED(file->directory);
	EXIT_NOT_IMPLEMENTED(file->special != SpecialFile::None);

	EXIT_NOT_IMPLEMENTED(nbytes > UINT_MAX);

	file->mutex.Lock();

	if (!file->opened) {
		file->mutex.Unlock();
		return KERNEL_ERROR_EBADF;
	}

	if (file->f.IsInvalid()) {
		file->mutex.Unlock();
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	const auto pos           = file->f.Tell();
	uint32_t   bytes_written = 0;
	file->f.Seek(file->append ? file->f.Size() : static_cast<uint64_t>(offset));
	file->f.Write(buf, static_cast<uint32_t>(nbytes), &bytes_written);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileWriteBytes, bytes_written);
	if (file->sync_writes) {
		file->f.Flush();
	}
	file->f.Seek(pos);

	file->mutex.Unlock();

	LOGF("\tWrite %u bytes (pos = %" PRId64 ") to: %s\n", bytes_written, offset,
	     Common::PathToString(file->real_name).c_str());

	return bytes_written;
}

int64_t KYTY_SYSV_ABI KernelPwritev(int d, const KernelIovec* iov, int iovcnt, int64_t offset) {
	PRINT_NAME();

	std::vector<KernelIovec> buffers;
	size_t                   total = 0;
	const int                error = ValidateIovecs(iov, iovcnt, offset, &buffers, &total);
	if (error != OK) {
		return error;
	}

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}
	if (::Libs::Network::Net::IsSocket(d)) {
		return KERNEL_ERROR_ESPIPE;
	}
	auto file = g_files->GetFile(d);
	if (file == nullptr || !file->opened || !file->writable) {
		return KERNEL_ERROR_EBADF;
	}
	Common::LockGuard lock(file->mutex);
	if (file->directory) {
		return KERNEL_ERROR_EISDIR;
	}
	if (file->special != SpecialFile::None) {
		return KERNEL_ERROR_EINVAL;
	}
	if (total == 0) {
		return 0;
	}
	if (file->f.IsInvalid()) {
		return KERNEL_ERROR_EIO;
	}

	const auto position = file->f.Tell();
	const auto target   = (file->append ? file->f.Size() : static_cast<uint64_t>(offset));
	if (target > static_cast<uint64_t>(INT64_MAX) - total) {
		return KERNEL_ERROR_EFBIG;
	}
	if (!file->f.Seek(target)) {
		return KERNEL_ERROR_EIO;
	}

	int64_t bytes_written = 0;
	for (const auto& buffer: buffers) {
		if (buffer.iov_len == 0) {
			continue;
		}
		uint32_t bytes = 0;
		file->f.Write(buffer.iov_base, static_cast<uint32_t>(buffer.iov_len), &bytes);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::GuestFileWriteBytes, bytes);
		bytes_written += bytes;
		if (bytes < buffer.iov_len) {
			break;
		}
	}
	const bool flushed  = !file->sync_writes || file->f.Flush();
	const bool restored = file->f.Seek(position);
	if (!flushed || !restored) {
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tWritev %" PRId64 " bytes (pos = %" PRId64 ", iovcnt = %d) to: %s\n", bytes_written,
	     offset, iovcnt, Common::PathToString(file->real_name).c_str());
	return bytes_written;
}

int64_t KYTY_SYSV_ABI KernelLseek(int d, int64_t offset, int whence) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	auto file = g_files->GetFile(d);

	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	if (file->special != SpecialFile::None) {
		return KERNEL_ERROR_ESPIPE;
	}

	Common::LockGuard lock(file->mutex);

	if (!file->directory && file->f.IsInvalid()) {
		LOGF("\tfile is invalid\n");
		return KERNEL_ERROR_EIO;
	}

	uint64_t base = 0;
	switch (whence) {
		case 0: break;
		case 1: base = file->directory ? file->dents_offset : file->f.Tell(); break;
		case 2: base = file->directory ? file->dirents.size() : file->f.Size(); break;
		default: return KERNEL_ERROR_EINVAL;
	}

	uint64_t position = 0;
	if (offset >= 0) {
		if (base > static_cast<uint64_t>(INT64_MAX - offset)) {
			return KERNEL_ERROR_EOVERFLOW;
		}
		position = base + static_cast<uint64_t>(offset);
	} else {
		const auto distance = uint64_t {0} - static_cast<uint64_t>(offset);
		if (base < distance) {
			return KERNEL_ERROR_EINVAL;
		}
		position = base - distance;
		if (position > INT64_MAX) {
			return KERNEL_ERROR_EOVERFLOW;
		}
	}

	if (file->directory) {
		file->dents_offset = position;
	} else {
		file->f.Seek(position);
		EXIT_IF(file->f.Tell() != position);
	}

	LOGF("\tLseek (pos = %" PRIu64 ") to: %s\n", position,
	     Common::PathToString(file->real_name).c_str());

	return static_cast<int64_t>(position);
}

int KYTY_SYSV_ABI KernelStat(const char* path, FileStat* sb) {
	PRINT_NAME();

	if (path == nullptr || sb == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	LOGF("\t KernelStat: %s\n", path);

	auto real_file_name = g_mount_points->ResolvePath(path);

	const auto info = Common::File::GetInfo(real_file_name);
	static const bool log_opens = KYTY_GETENV("KYTY_DEBUG_FILE_OPENS") != nullptr;
	if (!info) {
		LOGF("\t file not found\n");
		if (log_opens) [[unlikely]] {
			std::printf("file-stat %s -> missing\n", path);
			std::fflush(stdout);
		}
		return KERNEL_ERROR_ENOENT;
	}

	const bool is_dir = !info->is_file;

	FileStat stat {};
	stat.st_mode = 0000777u | (is_dir ? 0040000u : 0100000u);

	auto at = SessionStartTime();
	auto wt = at;

	if (is_dir) {
		stat.st_size =
		    static_cast<int64_t>(PackDirents(Common::File::GetDirEntries(real_file_name)).size());
		stat.st_blksize = 512;
		stat.st_blocks  = stat.st_size / 512;
	} else {
		stat.st_size    = static_cast<int64_t>(info->size);
		stat.st_blksize = 512;
		stat.st_blocks  = (stat.st_size + 511) / 512;

		Common::File::GetLastAccessAndWriteTimeUTC(real_file_name, &at, &wt);
	}

	FillStatTimes(&stat, wt);
	*sb              = stat;

	return OK;
}

int KYTY_SYSV_ABI KernelFstat(int d, FileStat* sb) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EPERM;
	}

	if (sb == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	auto file = g_files->GetFile(d);

	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	LOGF("\tKernelFstat: %s\n", Common::PathToString(file->real_name).c_str());

	FileStat stat {};
	stat.st_mode = 0000777u | (file->directory ? 0040000u : 0100000u);

	auto at = SessionStartTime();
	auto wt = at;

	if (file->special != SpecialFile::None) {
		stat.st_mode    = 0020644u;
		stat.st_size    = 0;
		stat.st_blksize = 512;
		stat.st_blocks  = 0;
	} else if (!file->directory) {
		file->mutex.Lock();

		if (!file->opened) {
			file->mutex.Unlock();
			return KERNEL_ERROR_EBADF;
		}

		if (file->f.IsInvalid()) {
			file->mutex.Unlock();
			LOGF("\tfile is invalid\n");
			return KERNEL_ERROR_EIO;
		}

		auto size = file->f.Size();
		file->f.GetLastAccessAndWriteTimeUTC(&at, &wt);

		file->mutex.Unlock();

		stat.st_size    = static_cast<int64_t>(size);
		stat.st_blksize = 512;
		stat.st_blocks  = (stat.st_size + 511) / 512;
	} else {
		stat.st_size    = static_cast<int64_t>(file->dirents.size());
		stat.st_blksize = 512;
		stat.st_blocks  = stat.st_size / 512;
	}

	FillStatTimes(&stat, wt);
	*sb              = stat;

	return OK;
}

int KYTY_SYSV_ABI KernelFtruncate(int d, int64_t length) {
	PRINT_NAME();

	if (d < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	if (length < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	if (::Libs::Network::Net::IsSocket(d)) {
		return KERNEL_ERROR_EINVAL;
	}

	auto file = g_files->GetFile(d);

	if (file == nullptr || !file->opened) {
		return KERNEL_ERROR_EBADF;
	}

	if (!file->writable) {
		return KERNEL_ERROR_EBADF;
	}

	if (file->directory || file->special != SpecialFile::None) {
		return KERNEL_ERROR_EINVAL;
	}

	Common::LockGuard lock(file->mutex);

	if (file->f.IsInvalid() || !file->f.Truncate(static_cast<uint64_t>(length))) {
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tFtruncate (size = %" PRId64 ") file: %s\n", length,
	     Common::PathToString(file->real_name).c_str());

	return OK;
}

int KYTY_SYSV_ABI KernelUnlink(const char* path) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto real_file_name = g_mount_points->ResolvePath(path);
	if (Common::IsArchivePath(real_file_name)) {
		return KERNEL_ERROR_EROFS;
	}

	bool is_dir  = Common::File::IsDirectoryExisting(real_file_name);
	bool is_file = Common::File::IsFileExisting(real_file_name);

	if (is_dir) {
		return KERNEL_ERROR_EPERM;
	}

	if (!is_file) {
		return KERNEL_ERROR_ENOENT;
	}

	auto  open_file = g_files->GetFile(real_file_name);
	bool  ok        = (open_file != nullptr && open_file->opened && !open_file->directory
	                       ? open_file->f.Unlink()
	                       : Common::File::DeleteFile(real_file_name));

	if (!ok) {
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tKernelUnlink: %s\n", path);

	return OK;
}

int KYTY_SYSV_ABI KernelRename(const char* from, const char* to) {
	PRINT_NAME();

	if (from == nullptr || to == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto from_path = std::string(from);
	auto to_path   = std::string(to);
	auto real_from = g_mount_points->ResolvePath(from_path);
	auto real_to   = g_mount_points->ResolvePath(to_path);
	if (Common::IsArchivePath(real_from) || Common::IsArchivePath(real_to)) {
		return KERNEL_ERROR_EROFS;
	}

	if (real_to.empty() || !Common::File::IsFileExisting(real_from)) {
		return KERNEL_ERROR_ENOENT;
	}

	Common::File::CreateDirectories(real_to.parent_path());

	if (Common::File::IsFileExisting(real_to)) {
		Common::File::DeleteFile(real_to);
	}

	if (!Common::File::RenameFile(real_from, real_to)) {
		return KERNEL_ERROR_EIO;
	}

	LOGF("\tKernelRename: %s -> %s\n", from, to);

	return OK;
}

int KYTY_SYSV_ABI KernelGetdirentries(int fd, char* buf, int nbytes, int64_t* basep) {
	PRINT_NAME();

	if (fd < DESCRIPTOR_MIN) {
		return KERNEL_ERROR_EBADF;
	}

	if (buf == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	auto file = g_files->GetFile(fd);

	if (file == nullptr) {
		return KERNEL_ERROR_EBADF;
	}

	if (!file->directory || nbytes < static_cast<int>(DIR_BLOCK_SIZE)) {
		return KERNEL_ERROR_EINVAL;
	}

	if (!file->opened) {
		return KERNEL_ERROR_EBADF;
	}
	Common::LockGuard lock(file->mutex);
	if (file->dents_offset > file->dirents.size() || file->dents_offset % DIR_BLOCK_SIZE != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	LOGF("\t dir    = %s\n"
	     "\t nbytes = %d\n"
	     "\t offset = %" PRIu64 "\n",
	     Common::PathToString(file->real_name).c_str(), nbytes, file->dents_offset);

	if (basep != nullptr) {
		*basep = static_cast<int64_t>(file->dents_offset);
	}
	return static_cast<int>(
	    ReadDirectory(file.get(), buf, AlignDown(static_cast<uint64_t>(nbytes), DIR_BLOCK_SIZE)));
}

int KYTY_SYSV_ABI KernelGetdents(int fd, char* buf, int nbytes) {
	PRINT_NAME();

	return KernelGetdirentries(fd, buf, nbytes, nullptr);
}

int KYTY_SYSV_ABI KernelMkdir(const char* path, uint16_t mode) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	if (std::strlen(path) > 255) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	LOGF("\t path = %s\n"
	     "\t mode = %04" PRIx16 "\n",
	     path, mode);

	auto real_name = g_mount_points->ResolvePath(std::string(path));
	if (real_name.empty()) {
		return KERNEL_ERROR_ENOENT;
	}
	if (Common::IsArchivePath(real_name)) {
		return KERNEL_ERROR_EROFS;
	}

	if (Common::File::IsDirectoryExisting(real_name)) {
		return KERNEL_ERROR_EEXIST;
	}

	if (!Common::File::CreateDirectory(real_name)) {
		return KERNEL_ERROR_EIO;
	}

	if (!Common::File::IsDirectoryExisting(real_name)) {
		return KERNEL_ERROR_ENOENT;
	}

	return OK;
}

int KYTY_SYSV_ABI KernelRmdir(const char* path) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	LOGF("\t path = %s\n", path);

	auto real_name = g_mount_points->ResolvePath(std::string(path));
	if (Common::IsArchivePath(real_name)) {
		return KERNEL_ERROR_EROFS;
	}

	if (!Common::File::IsDirectoryExisting(real_name)) {
		return KERNEL_ERROR_ENOENT;
	}

	if (!Common::File::DeleteDirectory(real_name)) {
		return KERNEL_ERROR_EIO;
	}

	return OK;
}

int KYTY_SYSV_ABI KernelCheckReachability(const char* path) {
	PRINT_NAME();

	if (path == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	if (std::strlen(path) > 255) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	LOGF("\t KernelCheckReachability: %s\n", path);

	std::string mounted_path = std::string(path);

	if (IsRandomDevice(mounted_path)) {
		return OK;
	}

	auto real_name = g_mount_points->ResolvePath(mounted_path);

	if (Common::File::IsFileExisting(real_name) || Common::File::IsDirectoryExisting(real_name)) {
		return OK;
	}

	return KERNEL_ERROR_ENOENT;
}

} // namespace Libs::LibKernel::FileSystem
