#include "graphics/shader/recompiler/ir/passes/ExecSelectSinking.h"

#include "common/switches.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

bool IsSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

// Reads another lane's value (or neighbouring lanes, for implicit derivatives), or merges control
// flow: what an inactive lane holds must not reach it.
bool IsBarrierToSinking(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::Phi:
		case ValueOpcode::Ballot:
		case ValueOpcode::ReadFirstLane:
		case ValueOpcode::ReadLane:
		case ValueOpcode::WriteLane:
		case ValueOpcode::Permlane16U32:
		case ValueOpcode::PermuteU32:
		case ValueOpcode::BpermuteU32:
		case ValueOpcode::SwizzleU32:
		case ValueOpcode::DppMoveU32:
		case ValueOpcode::DppUpdateU32:
		case ValueOpcode::WqmU64:
		case ValueOpcode::ImageSampleRaw:
		case ValueOpcode::ImageGatherRaw:
		case ValueOpcode::ImageQueryLod: return true;
		default: return false;
	}
}

class Sinking {
public:
	ExecSelectSinkingStats Run(const BlockList& blocks) {
		std::vector<Inst*> selects;
		for (auto* block: blocks) {
			for (auto& inst: *block) {
				if (IsSelect(inst.GetOpcode())) selects.push_back(&inst);
			}
		}
		for (auto* select: selects) {
			m_stats.selects++;
			TrySink(*select);
		}
		if (m_report) {
			PrintReasons();
		}
		return m_stats;
	}

private:
	// Whether whatever an inactive lane computes into `value` is discarded by every consumer,
	// directly or further on. `value` is computed from a sunk select, not the select itself.
	bool Discarded(Inst& value, const Value& exec) {
		const auto key = std::make_pair(&value, exec.TryInstruction());
		if (const auto found = m_memo.find(key); found != m_memo.end()) return found->second;
		m_memo[key] = false; // a cycle needs a phi, which is refused anyway
		bool ok = true;
		for (const auto& use: value.Uses()) {
			if (!UseDiscards(use, exec, /*select_itself=*/false)) {
				ok = false;
				break;
			}
		}
		m_memo[key] = ok;
		return ok;
	}

	static bool SameExec(const Value& a, const Value& b) { return a.Resolve() == b.Resolve(); }

	bool UseDiscards(const Use& use, const Value& exec, bool select_itself) {
		auto&      user = *use.user;
		const auto op   = user.GetOpcode();
		if (user.Parent() == nullptr) return Refuse("planning", user);
		if (IsBarrierToSinking(op)) return Refuse(select_itself ? "direct" : "derived", user);
		if (IsSelect(op) && SameExec(user.Arg(0), exec)) {
			// Inactive lanes take the false operand: the true one is free, the false one is the
			// select itself (rewritten to its old value) and nothing derived from it.
			if (use.operand == 1 || (use.operand == 2 && select_itself)) return true;
			return Refuse("derived-false-operand", user);
		}
		if (user.MayHaveSideEffects()) {
			// Memory operations carry their EXEC as the last operand.
			const auto last = user.NumArgs() - 1;
			if (user.NumArgs() > 1 && use.operand != last && user.Arg(last).GetType() == Type::U1 &&
			    SameExec(user.Arg(last), exec)) {
				return true;
			}
			return Refuse("side-effect", user);
		}
		if (user.GetType() == Type::Void) return Refuse("void", user);
		return Discarded(user, exec);
	}

	void TrySink(Inst& select) {
		const auto exec = select.Arg(0).Resolve();
		if (exec.IsImmediate()) return; // constant selects fold elsewhere
		std::vector<Use> false_operands;
		for (const auto& use: select.Uses()) {
			if (!UseDiscards(use, exec, /*select_itself=*/true)) return;
			if (use.operand == 2 && IsSelect(use.user->GetOpcode()) &&
			    SameExec(use.user->Arg(0), exec)) {
				false_operands.push_back(use);
			}
		}
		const auto chosen = select.Arg(1);
		const auto old    = select.Arg(2);
		for (const auto& use: false_operands) {
			use.user->SetArg(use.operand, old);
		}
		select.ReplaceUsesWith(chosen);
		m_stats.sunk++;
	}

	// KYTY_DEBUG_SINK_REASONS=1: why selects stay, by the consumer that keeps them.
	bool Refuse(const char* why, const Inst& user) {
		if (m_report) {
			m_reasons[std::string(why) + " " + std::string(ValueOpcodeName(user.GetOpcode()))]++;
		}
		return false;
	}

	void PrintReasons() const {
		std::vector<std::pair<uint32_t, std::string>> sorted;
		for (const auto& [reason, count]: m_reasons) sorted.emplace_back(count, reason);
		std::ranges::sort(sorted, std::greater {});
		std::fprintf(stderr, "exec select sinking: selects=%u sunk=%u\n", m_stats.selects,
		             m_stats.sunk);
		for (size_t i = 0; i < sorted.size() && i < 16u; i++) {
			std::fprintf(stderr, "  kept by %6u %s\n", sorted[i].first, sorted[i].second.c_str());
		}
	}

	struct PairHash {
		size_t operator()(const std::pair<Inst*, Inst*>& key) const {
			return std::hash<Inst*>()(key.first) * 31u + std::hash<Inst*>()(key.second);
		}
	};

	std::unordered_map<std::pair<Inst*, Inst*>, bool, PairHash> m_memo;
	ExecSelectSinkingStats                                      m_stats;
	const bool m_report = KYTY_GETENV("KYTY_DEBUG_SINK_REASONS") != nullptr;
	std::map<std::string, uint32_t> m_reasons;
};

} // namespace

ExecSelectSinkingStats SinkExecSelects(const BlockList& blocks) {
	return Sinking().Run(blocks);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
