#ifndef EMULATOR_SRC_COMMON_SLOTVECTOR_H_
#define EMULATOR_SRC_COMMON_SLOTVECTOR_H_

#include "common/assert.h"
#include "common/abi.h"

#include <compare>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace Common {

struct SlotId {
	static constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();

	constexpr SlotId() noexcept = default;
	constexpr SlotId(uint32_t value) noexcept: index(value), generation(1) {}
	constexpr SlotId(uint32_t value, uint32_t slot_generation) noexcept
	    : index(value), generation(slot_generation) {}

	[[nodiscard]] constexpr explicit operator bool() const noexcept { return index != INVALID_INDEX; }
	constexpr auto operator<=>(const SlotId&) const noexcept = default;

	uint32_t index = INVALID_INDEX;
	uint32_t generation = 0;
};

// Stable-address slot storage for cache resources that are intentionally non-movable.
template <typename T>
class SlotVector {
public:
	SlotVector() = default;
	KYTY_CLASS_NO_COPY(SlotVector);

	[[nodiscard]] T& operator[](SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		return *m_values[id.index].value;
	}

	[[nodiscard]] const T& operator[](SlotId id) const noexcept {
		EXIT_IF(!is_allocated(id));
		return *m_values[id.index].value;
	}

	[[nodiscard]] T* try_get(SlotId id) noexcept {
		return is_allocated(id) ? &*m_values[id.index].value : nullptr;
	}

	[[nodiscard]] const T* try_get(SlotId id) const noexcept {
		return is_allocated(id) ? &*m_values[id.index].value : nullptr;
	}

	// Answered from a small array of its own: the generation used to sit behind the value, so each
	// check of a large resource (an image) touched a cache line nothing else on the path reads
	// (the hottest single line of the GPU thread in ASTRO BOT's Sky Garden, about 5%).
	[[nodiscard]] bool is_allocated(SlotId id) const noexcept {
		return id && id.index < m_states.size() &&
		       m_states[id.index] == ((uint64_t {id.generation} << 1u) | 1u);
	}

	template <typename... Args>
	[[nodiscard]] SlotId insert(Args&&... args) {
		uint32_t index = 0;
		if (m_free_list.empty()) {
			index = static_cast<uint32_t>(m_values.size());
			m_values.emplace_back();
			m_values.back().value.emplace(std::forward<Args>(args)...);
			if (m_states.capacity() == 0) {
				m_states.reserve(1u << 16u);
			}
			m_states.push_back(uint64_t {1} << 1u);
		} else {
			index = m_free_list.back();
			m_free_list.pop_back();
			EXIT_IF(m_values[index].value.has_value());
			m_values[index].value.emplace(std::forward<Args>(args)...);
		}
		m_states[index] |= 1u;
		++m_size;
		return SlotId {index, static_cast<uint32_t>(m_states[index] >> 1u)};
	}

	void erase(SlotId id) noexcept {
		EXIT_IF(!is_allocated(id));
		auto& slot = m_values[id.index];
		slot.value.reset();
		uint32_t generation = id.generation + 1u;
		if (generation == 0) {
			generation = 1;
		}
		m_states[id.index] = uint64_t {generation} << 1u;
		m_free_list.push_back(id.index);
		--m_size;
	}

	[[nodiscard]] size_t size() const noexcept { return m_size; }
	[[nodiscard]] size_t capacity() const noexcept { return m_values.size(); }

	template <typename F>
	void ForEach(F&& fn) {
		for (uint32_t index = 0; index < m_values.size(); ++index) {
			if (m_values[index].value) {
				fn(SlotId {index, static_cast<uint32_t>(m_states[index] >> 1u)}, *m_values[index].value);
			}
		}
	}

	template <typename F>
	void ForEach(F&& fn) const {
		for (uint32_t index = 0; index < m_values.size(); ++index) {
			if (m_values[index].value) {
				fn(SlotId {index, static_cast<uint32_t>(m_states[index] >> 1u)}, *m_values[index].value);
			}
		}
	}

private:
	struct Slot {
		std::optional<T> value;
	};

	std::deque<Slot>             m_values;
	// Per slot: its generation << 1 | allocated.
	std::vector<uint64_t>        m_states;
	std::vector<uint32_t>        m_free_list;
	size_t                       m_size = 0;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
	[[nodiscard]] size_t operator()(Common::SlotId id) const noexcept {
		return std::hash<uint64_t> {}((static_cast<uint64_t>(id.generation) << 32u) | id.index);
	}
};

#endif // EMULATOR_SRC_COMMON_SLOTVECTOR_H_
