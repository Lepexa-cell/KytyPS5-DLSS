#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

class DrawSpeculator;
namespace Pssr {
struct Request;
} // namespace Pssr

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

enum class Pm4ProcessResult { Complete, Blocked };

enum class PredicateSync { None, Download, Drain };

[[nodiscard]] constexpr PredicateSync ClassifyPredicateSync(bool image_gpu_modified,
                                                            bool pending_image_writeback,
                                                            bool buffer_gpu_dirty) {
	if (image_gpu_modified || pending_image_writeback) {
		return PredicateSync::Drain;
	}
	return buffer_gpu_dirty ? PredicateSync::Download : PredicateSync::None;
}

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }
	// The innermost buffer's commands from the packet the execution stopped at (for debugging).
	[[nodiscard]] std::span<const uint32_t> RemainingCommands() const noexcept {
		if (m_buffer_stack.empty()) {
			return {};
		}
		const auto& cursor = m_buffer_stack.back();
		return cursor.commands.subspan(std::min<size_t>(cursor.offset_dw, cursor.commands.size()));
	}

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
};

// Upstream 9f83fd5f: the packet names user SGPRs that receive the draw's vertex offset,
// first instance and first index (SH_NOP when unused).
struct IndirectDrawRegisters {
	uint32_t vertex_offset   = 0x280u;
	uint32_t instance_offset = 0x280u;
	uint32_t index_offset    = 0x280u;
	[[nodiscard]] bool Any() const noexcept {
		return vertex_offset != 0x280u || instance_offset != 0x280u || index_offset != 0x280u;
	}
};

class CommandProcessor {
public:
	// Where the last Process call left its execution (see Process: a resume keeps the walk).
	const Pm4Execution* m_spec_execution = nullptr;
	const uint32_t*     m_spec_packet    = nullptr;

	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	CommandProcessor(RenderContext& renderer, int interrupt_event_id);
	~CommandProcessor();

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);
	// PS5 Pro upscaling request (graphics/guest_gpu/pssr.h), recorded into the current buffer.
	void ProcessPssr(const Pssr::Request& request);

	void            BufferInit();
	void            BufferFlush();
	// Submits only when the GPU has retired every earlier submission.
	void            BufferFlushIfGpuIdle();
	// Submits for a queued interrupt, batching within the label flush interval.
	void            BufferFlushForInterrupt();
	void            BufferFlushAndWait();
	void            BufferWait();
	HW::Context&    GetCtx() { return m_ctx; }
	HW::UserConfig& GetUcfg() { return m_ucfg; }
	HW::Shader&     GetShCtx() { return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	// The NUM_INSTANCES state, reading it from guest memory if a GPU-args draw left it there.
	[[nodiscard]] uint32_t NumInstances();
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed,
	                  IndirectDrawRegisters registers = {});
	bool TryDrawIndexedIndirectOnGpu(uint64_t address);
	bool TryDrawIndirectOnGpu(uint64_t address);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       uint32_t draw_initiator, bool indexed,
	                       IndirectDrawRegisters registers = {});
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	void EmitGlobalBarrier(uint32_t source = 0);
	// Records a pending compute chain barrier (KYTY_COMPUTE_CHAIN) at a guest synchronization point
	// that records no host barrier of its own.
	void BreakComputeChain();
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete) { m_ce_complete = complete; }
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	void WaitForRewind(bool valid);
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	// A packet decides on guest memory the CPU may have just written: later draws reading memory
	// through addresses must see the CPU's writes (RenderContext::PrepareBda).
	void AdvanceBdaEpoch();
	// GET_LOD_STATS: see RenderContext::ReportMipStats.
	void ReportMipStats(void* dst, uint32_t size, bool reset);
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }
	// Process is about to run the constant engine's stream (no draws to speculate) or not.
	void SetConstantStream(bool constant) { m_constant_stream = constant; }

private:
	friend class DrawSpeculator;

	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	void ProcessPm4(Pm4Execution& execution);
	// After a draw that created a pipeline: walks the rest of the command stream without
	// executing it, replaying only register writes into a copy of the register state, and asks
	// the pipeline cache to prefetch the shader library parts of the draws it finds, so a loading
	// burst's pipelines compile in parallel instead of one after another.
	void RunPipelineLookahead(const Pm4Execution& execution);
	// Whether the background work a Prefetch-mode look-ahead left pending has progressed enough
	// to walk again (see m_lookahead_rewalk).
	[[nodiscard]] bool LookaheadWorkFinished() const;
	// At the start of every submission: the same walk for dispatches only, so a compute pipeline
	// the submission needs first starts compiling on a worker thread before its dispatch. Shaders
	// a dispatch already used are skipped without a lookup.
	void RunDispatchLookahead(const Pm4Execution& execution);
	// One walk of the look-ahead; `pending` is set when a shader it met is still translating.
	// With `dispatches_only`, draws are passed over.
	[[nodiscard]] static bool DispatchLookaheadDue(uint64_t compute_pipelines_created);
	void LookaheadPass(const Pm4Execution& execution, ProgramWait wait, uint32_t& draws,
	                   uint32_t& parts, bool& pending, bool dispatches_only = false);
	void SuspendPm4();
	// Whether a dispatch may suspend this queue while its compute pipeline compiles
	// (--defer-dispatches).
	[[nodiscard]] bool MayDeferDispatch() const;
	// Whether a dispatch may suspend this asynchronous compute queue while bytes its program lookup
	// reads are downloaded (KYTY_DEFER_DIRTY_READS, BufferCache::DeferDirtyReads).
	[[nodiscard]] bool MayDeferReads() const;
	// Starts the draw speculation walk at the current packet (see DrawSpeculator).
	void RestartSpeculation(const Pm4Execution& execution);
	void SynchronizePredicate(uint64_t address, uint64_t size);
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	RenderContext&   m_renderer;
	HW::Context      m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state: indirect draws update it for subsequent draws.
	uint32_t m_num_instances = 1;
	// A GPU-args indirect draw leaves the instance count in guest memory; read on demand.
	uint64_t m_num_instances_address = 0;

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	uint32_t m_const_ram[0x3000] = {0};

	FlipInfo  m_flip;
	const int m_interrupt_event_id;
	uint64_t  m_submit_id                   = 0;
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;
	// Draws the last look-ahead already covered; no new look-ahead runs until they are processed.
	uint32_t  m_lookahead_draws_left        = 0;
	// With asynchronous pipelines, the last look-ahead left shaders translating or compiling on
	// worker threads; the walk repeats as they finish, to take each prediction a step further.
	bool      m_lookahead_rewalk            = false;
	uint64_t  m_lookahead_jobs_finished     = 0;
	std::chrono::steady_clock::time_point m_lookahead_time {};
	// Pipelines created while processing the current and the previous submission.
	uint64_t  m_submission_created_start    = 0;
	uint64_t  m_last_submission_created     = 0;
	// Speculates the resources of this queue's draws ahead of it (KYTY_SPECULATE_DRAWS=1).
	std::unique_ptr<DrawSpeculator> m_speculator;
	// Process is running the constant engine's stream, which has no draws to speculate.
	bool                            m_constant_stream = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
