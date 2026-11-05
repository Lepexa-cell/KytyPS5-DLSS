#include "graphics/shader/shaderCallLink.h"

#include "common/assert.h"
#include "common/switches.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shaderBindings.h"
#include "kernel/memory.h"

#include <magic_enum.hpp>
#include "xxhash.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Libs::Graphics::ShaderCallLink {

namespace {

namespace D = ShaderRecompiler::Decoder;

constexpr uint32_t NullSgpr        = 125u;
constexpr uint32_t MaxTableEntries = 64u;
constexpr uint32_t MaxLeafWords    = 48u;
constexpr uint32_t TraceWindow     = 64u;

uint32_t Sdst(uint32_t word) {
	return (word >> 16u) & 0x7fu;
}

bool IsCallWord(uint32_t word) {
	return (word >> 23u) == 0x17du && ((word >> 8u) & 0xffu) == 0x21u && Sdst(word) != NullSgpr;
}

bool IsCall(const D::Instruction& inst) {
	return inst.family == D::Family::SOP1 && IsCallWord(inst.raw[0]);
}

uint32_t WriteWidth(const D::Instruction& inst) {
	if (inst.family == D::Family::SMEM) {
		return inst.data_dwords;
	}
	const auto name = magic_enum::enum_name(inst.opcode);
	return name.ends_with("B64") || name.ends_with("U64") || name.ends_with("I64") ||
	               name.starts_with("V_CMP")
	           ? 2u
	           : 1u;
}

bool Writes(const D::Instruction& inst, uint32_t reg) {
	if (IsCall(inst)) {
		return reg == Sdst(inst.raw[0]) || reg == Sdst(inst.raw[0]) + 1u;
	}
	const auto covers = [&](const D::Operand& dst, uint32_t width) {
		return dst.kind == D::OperandKind::Sgpr && reg >= dst.reg && reg < dst.reg + width;
	};
	return covers(inst.dst, WriteWidth(inst)) || covers(inst.dst2, 2u);
}

// SOPP branch from word `from` to word `to`.
uint32_t Branch(uint32_t opcode, uint32_t from, uint32_t to) {
	const auto delta = static_cast<int64_t>(to) - static_cast<int64_t>(from) - 1;
	EXIT_IF(delta < INT16_MIN || delta > INT16_MAX);
	return 0xbf800000u | (opcode << 16u) | (static_cast<uint32_t>(delta) & 0xffffu);
}

constexpr uint32_t SBranch      = 0x02u;
constexpr uint32_t SCbranchScc0 = 0x04u;
constexpr uint32_t SCbranchScc1 = 0x05u;

bool ReadGuest(uint64_t address, void* data, uint64_t size) {
	return address != 0 && LibKernel::Memory::TryReadGpuCleanBacking(address, data, size);
}

struct Entry {
	bool                               scanned    = false;
	uint32_t                           code_words = 0;
	std::vector<Site>                  sites;
	std::vector<std::vector<Function>> functions;
	std::vector<uint64_t>              rejected;
	std::span<const uint32_t>          linked;
	uint64_t                           variant = 0;
	bool                               read_failed = false;
	std::string                        trace;
};

} // namespace

std::vector<Site> FindSites(std::span<const uint32_t> code, uint32_t user_data_count,
                            uint32_t& code_words) {
	std::vector<Site> sites;
	D::Program        program;
	try {
		D::DecodeProgram(code, program);
	} catch (const D::DecodeError&) {
		return sites;
	}
	const auto& insts = program.instructions;
	if (insts.empty()) {
		return sites;
	}
	code_words = insts.back().pc / 4u + insts.back().word_count;
	for (size_t i = 0; i < insts.size(); i++) {
		if (!IsCall(insts[i])) {
			continue;
		}
		const uint32_t target = insts[i].raw[0] & 0xffu;
		// The target pair back through plain copies to the table load (s_buffer_load_dwordx4 of
		// entry lo, hi, ...).
		uint32_t lo   = target;
		uint32_t hi   = target + 1u;
		size_t   load = SIZE_MAX;
		for (size_t j = i; j-- > 0 && i - j <= TraceWindow;) {
			const auto& in     = insts[j];
			const bool  low    = Writes(in, lo);
			const bool  high   = Writes(in, hi);
			const auto  name   = magic_enum::enum_name(in.opcode);
			if (!low && !high) {
				continue;
			}
			if (in.opcode == D::Opcode::S_MOV_B32 && in.src0.kind == D::OperandKind::Sgpr) {
				(low ? lo : hi) = in.src0.reg;
				continue;
			}
			if (in.family == D::Family::SMEM && name.starts_with("S_BUFFER_LOAD") &&
			    in.dst.reg == lo && hi == lo + 1u && in.data_dwords >= 2u) {
				load = j;
			}
			break;
		}
		if (load == SIZE_MAX) {
			continue;
		}
		// The table descriptor: an s_load_dwordx4 through a user-data pointer nothing rewrote.
		const uint32_t descriptor = insts[load].src0.reg;
		size_t         from       = SIZE_MAX;
		for (size_t j = load; j-- > 0 && load - j <= TraceWindow;) {
			const auto& in = insts[j];
			if (!Writes(in, descriptor) && !Writes(in, descriptor + 1u) &&
			    !Writes(in, descriptor + 2u) && !Writes(in, descriptor + 3u)) {
				continue;
			}
			if (in.opcode == D::Opcode::S_LOAD_DWORDX4 && in.dst.reg == descriptor &&
			    in.src0.kind == D::OperandKind::Sgpr && in.src1.kind == D::OperandKind::Null) {
				from = j;
			}
			break;
		}
		if (from == SIZE_MAX) {
			continue;
		}
		const uint32_t pointer = insts[from].src0.reg;
		// shortcut: only the code before the load is checked; a loop back over it would need flow.
		const bool rewritten = pointer + 1u >= user_data_count ||
		                       std::any_of(insts.begin(), insts.begin() + static_cast<ptrdiff_t>(from),
		                                   [&](const D::Instruction& in) {
			                                   return Writes(in, pointer) || Writes(in, pointer + 1u);
		                                   });
		if (rewritten) {
			continue;
		}
		sites.push_back({.call_word    = insts[i].pc / 4u,
		                 .target_reg   = target,
		                 .return_reg   = Sdst(insts[i].raw[0]),
		                 .table_sgpr   = pointer,
		                 .table_offset = static_cast<int32_t>(insts[from].offset)});
	}
	return sites;
}

bool ReadLeaf(std::span<const uint32_t> code, uint32_t return_reg, std::vector<uint32_t>& body) {
	body.clear();
	for (uint32_t word = 0; word + D::MaxInstructionRawWords <= code.size();) {
		D::Instruction inst;
		try {
			D::DecodeInstruction(code, word, inst);
		} catch (const D::DecodeError&) {
			return false;
		}
		const uint32_t raw = inst.raw[0];
		if (inst.family == D::Family::SOP1 && (raw & 0xffu) == return_reg &&
		    ((inst.opcode_id == 0x21u && Sdst(raw) == NullSgpr) || inst.opcode_id == 0x20u)) {
			return true;
		}
		const bool control =
		    (inst.family == D::Family::SOP1 && inst.opcode_id >= 0x1fu && inst.opcode_id <= 0x21u) ||
		    (inst.family == D::Family::SOPP && inst.opcode != D::Opcode::S_NOP &&
		     inst.opcode != D::Opcode::S_WAITCNT && inst.opcode != D::Opcode::S_WAITCNT_DEPCTR &&
		     inst.opcode != D::Opcode::S_INST_PREFETCH) ||
		    inst.opcode == D::Opcode::S_SUBVECTOR_LOOP_BEGIN ||
		    inst.opcode == D::Opcode::S_SUBVECTOR_LOOP_END;
		if (control || inst.opcode == D::Opcode::UNSUPPORTED || inst.opcode == D::Opcode::UNKNOWN ||
		    Writes(inst, return_reg) || Writes(inst, return_reg + 1u)) {
			return false;
		}
		body.insert(body.end(), code.begin() + word, code.begin() + word + inst.word_count);
		word += inst.word_count;
		if (body.size() > MaxLeafWords) {
			return false;
		}
	}
	return false;
}

std::vector<uint32_t> Build(std::span<const uint32_t> code, uint32_t code_words,
                            std::span<const Site> sites,
                            std::span<const std::vector<Function>> functions) {
	std::vector<uint32_t> out(code.begin(), code.begin() + code_words);
	const auto            here = [&] { return static_cast<uint32_t>(out.size()); };
	for (size_t s = 0; s < sites.size(); s++) {
		const auto& site    = sites[s];
		const auto& targets = functions[s];
		if (targets.empty()) {
			continue; // stays a call, decoded as KYTY_PRO_SWAPPC_SKIP decodes it
		}
		const uint32_t back = site.call_word + 1u;
		out[site.call_word] = Branch(SBranch, site.call_word, here());
		// The call's return address, as the S_SWAPPC_B64 would write it: S_GETPC_B64 here, moved
		// back to the call (the stub is always after it).
		const auto return_address = [&] {
			const auto at    = here();
			const auto bytes = (static_cast<int64_t>(back) - static_cast<int64_t>(at) - 1) * 4;
			const auto reg   = site.return_reg;
			out.push_back(0xbe801f00u | (reg << 16u));                               // s_getpc_b64
			out.push_back(0x8000ff00u | (reg << 16u) | reg);                         // s_add_u32 lit
			out.push_back(static_cast<uint32_t>(bytes));
			out.push_back(0x8200c100u | ((reg + 1u) << 16u) | (reg + 1u));           // s_addc_u32 -1
		};
		// shortcut: clobbers SCC, which the call leaves alone; no SCC reader follows d0f76103's call,
		// check the next shader that gets linked.
		std::vector<uint32_t> entries;
		for (const auto& function: targets) {
			out.push_back(0xbf06ff00u | site.target_reg); // s_cmp_eq_u32 target lo, lit
			out.push_back(static_cast<uint32_t>(function.target));
			const auto next = here();
			out.push_back(0);
			out.push_back(0xbf06ff00u | (site.target_reg + 1u)); // s_cmp_eq_u32 target hi, lit
			out.push_back(static_cast<uint32_t>(function.target >> 32u));
			entries.push_back(here());
			out.push_back(0);
			out[next] = Branch(SCbranchScc0, next, here());
		}
		return_address();
		out.push_back(Branch(SBranch, here(), back));
		for (size_t f = 0; f < targets.size(); f++) {
			out[entries[f]] = Branch(SCbranchScc1, entries[f], here());
			return_address();
			out.insert(out.end(), targets[f].body.begin(), targets[f].body.end());
			out.push_back(Branch(SBranch, here(), back));
		}
	}
	out.push_back(D::ExecutionOrderMarker);
	out.push_back(~D::ExecutionOrderMarker);
	return out;
}

bool Link(const ShaderParams& params, ShaderParams& linked) {
	static const bool enabled = [] {
		const char* text = KYTY_GETENV("KYTY_PRO_SWAPPC_LINK");
		return text != nullptr && text[0] == '1';
	}();
	if (!enabled || params.base_address != 0 || params.code.empty()) {
		return false;
	}
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, Entry>    entries;
	static std::deque<std::vector<uint32_t>>      codes;
	const std::lock_guard                         lock(mutex);
	auto&                                         entry = entries[params.hash];
	if (!entry.scanned) {
		entry.scanned = true;
		if (std::any_of(params.code.begin(), params.code.end(), IsCallWord)) {
			entry.sites = FindSites(params.code, params.user_data_count, entry.code_words);
			entry.functions.resize(entry.sites.size());
			for (const auto& site: entry.sites) {
				std::printf("KYTY_PRO_SWAPPC_LINK: shader 0x%016llx call at pc 0x%x, table at user "
				            "s%u+0x%x\n",
				            static_cast<unsigned long long>(params.hash), site.call_word * 4u,
				            site.table_sgpr, static_cast<unsigned>(site.table_offset));
			}
			std::fflush(stdout);
		}
	}
	if (entry.sites.empty()) {
		return false;
	}
	bool grew = false;
	for (size_t s = 0; s < entry.sites.size(); s++) {
		const auto&    site    = entry.sites[s];
		const uint64_t pointer = params.user_data[site.table_sgpr] |
		                         (uint64_t {params.user_data[site.table_sgpr + 1u]} << 32u);
		ShaderBufferResource descriptor {};
		if (!ReadGuest(pointer + static_cast<int64_t>(site.table_offset), descriptor.fields,
		               sizeof(descriptor.fields))) {
			continue;
		}
		const auto count =
		    static_cast<uint32_t>(std::min<uint64_t>(descriptor.GetSize() / 16u, MaxTableEntries));
		std::array<uint32_t, MaxTableEntries * 4u> table {};
		if (count == 0 || !ReadGuest(descriptor.Base48(), table.data(), count * 16u)) {
			continue;
		}
		auto& functions = entry.functions[s];
		for (uint32_t e = 0; e < count; e++) {
			const uint64_t target = table[e * 4u] | (uint64_t {table[e * 4u + 1u]} << 32u);
			const auto     known  = [&](const Function& f) { return f.target == target; };
			if (target == 0 || std::any_of(functions.begin(), functions.end(), known) ||
			    std::find(entry.rejected.begin(), entry.rejected.end(), target) !=
			        entry.rejected.end()) {
				continue;
			}
			std::array<uint32_t, MaxLeafWords + D::MaxInstructionRawWords + 8u> words {};
			if (!ReadGuest(target & ~uint64_t {3}, words.data(), sizeof(words))) {
				if (!entry.read_failed) {
					entry.read_failed = true;
					std::printf("KYTY_PRO_SWAPPC_LINK: shader 0x%016llx target 0x%llx unreadable, "
					            "tried again later\n",
					            static_cast<unsigned long long>(params.hash),
					            static_cast<unsigned long long>(target));
				}
				continue;
			}
			Function function {.target = target};
			const bool leaf = ReadLeaf(words, site.return_reg, function.body);
			std::printf("KYTY_PRO_SWAPPC_LINK: shader 0x%016llx pc 0x%x target 0x%llx %s:",
			            static_cast<unsigned long long>(params.hash), site.call_word * 4u,
			            static_cast<unsigned long long>(target),
			            leaf ? "linked" : "not a leaf, skipped");
			for (const auto word: function.body) {
				std::printf(" %08x", word);
			}
			std::printf("\n");
			std::fflush(stdout);
			if (!leaf) {
				entry.rejected.push_back(target);
				continue;
			}
			functions.push_back(std::move(function));
			grew = true;
		}
		// The same set gives the same code, whatever order the tables named it in.
		std::sort(functions.begin(), functions.end(),
		          [](const Function& a, const Function& b) { return a.target < b.target; });
		// KYTY_DEBUG_SWAPPC_TARGETS=1: which target entry 0 named, as a run of letters (a = the
		// lowest linked target, ...) per 60 dispatches, to see how the game picks its function.
		static const bool trace = KYTY_GETENV("KYTY_DEBUG_SWAPPC_TARGETS") != nullptr;
		if (trace) {
			const uint64_t first = table[0] | (uint64_t {table[1]} << 32u);
			const auto     found = std::find_if(functions.begin(), functions.end(),
			                                    [&](const Function& f) { return f.target == first; });
			entry.trace += found == functions.end() ? '?' : static_cast<char>('a' + (found - functions.begin()));
			if (entry.trace.size() == 60) {
				std::printf("KYTY_DEBUG_SWAPPC_TARGETS: pc 0x%x table 0x%llx %s\n", site.call_word * 4u,
				            static_cast<unsigned long long>(descriptor.Base48()), entry.trace.c_str());
				std::fflush(stdout);
				entry.trace.clear();
			}
		}
	}
	if (grew) {
		codes.push_back(Build(params.code, entry.code_words, entry.sites, entry.functions));
		entry.linked  = codes.back();
		entry.variant = XXH3_64bits(entry.linked.data(), entry.linked.size_bytes()) | 1u;
	}
	if (entry.linked.empty()) {
		return false;
	}
	linked              = params;
	linked.code         = entry.linked;
	linked.base_address = params.Base();
	linked.code_variant = entry.variant;
	return true;
}

} // namespace Libs::Graphics::ShaderCallLink
