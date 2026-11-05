#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/alignment.h"
#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <compare>
#include <cstdlib>
#include <limits>
#include <optional>
#include <source_location>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
class TileManager;
struct ImageTestAccess;

using ImageId = Common::SlotId;

// Bumped whenever an image may stop being clean for a sampled binding (TextureCache::IsTextureClean):
// CPU or buffer writes, a rebind, tracking, depth or stencil association and registration changes.
// While it is unchanged, an image group that was clean is still clean (see ImageGroupStands).
inline std::atomic<uint64_t> g_image_clean_epoch {1};
// KYTY_DEBUG_IMAGE_EPOCH_STATS=1: bumps counted per source line (printed with the group check
// counts, see ImageGroupStands): which change keeps clean image groups from being skipped.
inline std::array<std::atomic<uint64_t>, 8192> g_image_epoch_bumps {};
[[nodiscard]] inline bool ImageEpochStatsEnabled() noexcept {
	static const bool enabled = std::getenv("KYTY_DEBUG_IMAGE_EPOCH_STATS") != nullptr;
	return enabled;
}
inline void BumpImageCleanEpoch(std::source_location site = std::source_location::current()) noexcept {
	g_image_clean_epoch.fetch_add(1, std::memory_order_acq_rel);
	if (ImageEpochStatsEnabled()) [[unlikely]] {
		// textureCache.cpp lines as they are; image.h lines offset by 8000 (both stay below 8192).
		const bool header = std::string_view(site.file_name()).ends_with("image.h");
		g_image_epoch_bumps[(site.line() + (header ? 8000u : 0u)) % 8192u].fetch_add(
		    1, std::memory_order_relaxed);
	}
}

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	bool texture       = false;
	bool storage       = false;
	bool render_target = false;
	bool depth_target  = false;
	bool video_out     = false;
};

struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             is_bound      = false;
	bool             is_target     = false;
	bool             needs_rebind  = false;
	bool             force_general = false;
	bool             shader_write  = false;
};

class Image final {
public:
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] Barriers GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer);
	// [offset, offset + size) is the guest data's range in `buffer`; `capacity` is the whole
	// VkBuffer's size. Regions that would run past `capacity` are clamped (see ClampCopyRegions).
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size, uint64_t capacity);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size, uint64_t capacity);
	// Copies the color or depth aspect, or with stencil=true only the stencil aspect.
	void CopyImage(Image& source, bool stencil = false);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	void InvalidateCpuWrite(uint64_t vaddr, uint64_t size) {
		if (ImageRangeOverlaps(info.data.address, info.data.size, vaddr, size)) {
			m_cpu_dirty        = true;
			m_maybe_cpu_dirty  = false;
			m_maybe_hash_valid = false;
			BumpImageCleanEpoch();
		} else if (ImagePageRangesOverlap(info.data.address, info.data.size, vaddr, size)) {
			m_maybe_cpu_dirty = true;
			BumpImageCleanEpoch();
			NoteMark("cpu-write-same-page", vaddr, size, 0);
		}
	}
	// The last reason this image was marked stale (a CPU write over it, or a texture cache mark such
	// as mark-gpu-write with the writing shader): the --rd upload marker prints it.
	void NoteMark(const char* reason, uint64_t address, uint64_t size, uint64_t shader) const noexcept {
		mark_reason  = reason;
		mark_address = address;
		mark_size    = size;
		mark_shader  = shader;
	}
	mutable const char* mark_reason  = nullptr;
	mutable uint64_t    mark_address = 0;
	mutable uint64_t    mark_size    = 0;
	mutable uint64_t    mark_shader  = 0;

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!m_cpu_dirty) {
			m_maybe_cpu_dirty = true;
		}
		BumpImageCleanEpoch();
	}
	[[nodiscard]] bool NeedsMaybeCpuHash() const {
		return m_maybe_cpu_dirty && !m_maybe_hash_valid;
	}
	void SetMaybeCpuHash(uint64_t hash) {
		if (!NeedsMaybeCpuHash()) {
			EXIT("image cannot initialize maybe-dirty hash\n");
		}
		m_maybe_cpu_hash   = hash;
		m_maybe_hash_valid = true;
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || !m_maybe_hash_valid || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty        = false;
		m_maybe_cpu_dirty  = false;
		m_maybe_hash_valid = false;
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept { m_gpu_modified = true; }
	void               ClearGpuModified() noexcept { m_gpu_modified = false; }

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept {
		m_buffer_modified = true;
		m_buffer_whole    = true;
		m_buffer_ranges.clear();
		BumpImageCleanEpoch();
	}
	// KYTY_GPU_WRITE_RANGES: a GPU buffer write over [begin, end) of the image only. The ranges stay
	// sorted and merged; past MaxBufferRanges, or with any whole mark, all of the image counts as
	// written (TextureCache::InitializeImage uploads the ranges alone).
	void MarkBufferModifiedRange(uint64_t begin, uint64_t end) {
		if (!m_buffer_modified) {
			m_buffer_whole = false;
			m_buffer_ranges.clear();
		}
		m_buffer_modified = true;
		if (!m_buffer_whole && begin < end) {
			auto at = std::lower_bound(m_buffer_ranges.begin(), m_buffer_ranges.end(), begin,
			                           [](const GuestRange& range, uint64_t value) { return range.End() < value; });
			auto last = at;
			while (last != m_buffer_ranges.end() && last->address <= end) {
				begin = std::min(begin, last->address);
				end   = std::max(end, last->End());
				++last;
			}
			at = m_buffer_ranges.erase(at, last);
			m_buffer_ranges.insert(at, GuestRange {begin, end - begin});
			if (m_buffer_ranges.size() > MaxBufferRanges) {
				m_buffer_whole = true;
				m_buffer_ranges.clear();
			}
		}
		BumpImageCleanEpoch();
	}
	// The written ranges; empty while all of the image counts as written (or none is).
	[[nodiscard]] std::span<const GuestRange> BufferModifiedRanges() const noexcept {
		return m_buffer_whole ? std::span<const GuestRange> {} : std::span<const GuestRange> {m_buffer_ranges};
	}
	void ClearBufferModified() noexcept {
		// Whatever cleared the mark uploaded the tracked DWORDs too: their bits are spent.
		dirty_stale       = dirty_stale || m_buffer_tracked;
		dirty_gpu_modified = false;
		m_buffer_modified = false;
		m_buffer_whole    = false;
		m_buffer_tracked  = false;
		m_buffer_ranges.clear();
	}
	// KYTY_GPU_DIRTY_TRACK: GPU buffer stores set the bits of the DWORDs they wrote in the image's
	// bitmap (dirty_slot); TextureCache::RefreshTracked copies those DWORDs alone.
	void MarkBufferTracked() noexcept {
		if (!m_buffer_modified) {
			m_buffer_whole = false;
			m_buffer_ranges.clear();
		}
		m_buffer_modified = true;
		m_buffer_tracked  = true;
		BumpImageCleanEpoch();
	}
	[[nodiscard]] bool IsBufferTracked() const noexcept { return m_buffer_tracked; }
	// Only the tracked DWORDs are newer in the buffer (no whole or range mark).
	[[nodiscard]] bool IsBufferTrackedOnly() const noexcept {
		return m_buffer_tracked && !m_buffer_whole && m_buffer_ranges.empty();
	}
	// RefreshTracked copied the tracked DWORDs and zeroed their bits.
	void ClearBufferTracked() noexcept { m_buffer_tracked = false; }

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(info.data.address, info.data.size, address, size)
		             : ImageRangeOverlaps(info.data.address, info.data.size, address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return IsGpuModified() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : Common::AlignUp(info.data.size, 1024);
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;

	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	// Referenced by a bindless table slot: the collector keeps it, and unregistering it tells the
	// table (TextureCache::on_bindless_unregister) to repoint the slot.
	bool             bindless_pinned = false;
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	// KYTY_GPU_DIRTY_TRACK (TextureCache): the image's bitmap slot, its words, whether its bits may
	// be spent (zeroed before the next tracked write), and the refresh target from the first check
	// (dirty_texel_words 0: the image does not qualify).
	uint32_t         dirty_slot           = UINT32_MAX;
	uint32_t         dirty_words          = 0;
	bool             dirty_stale          = true;
	bool             dirty_checked        = false;
	// The tracked write cleared GpuModified: the image's other texels are still newer than memory.
	bool             dirty_gpu_modified   = false;
	uint8_t          dirty_texel_words    = 0;
	uint32_t         dirty_blocks_per_row = 0;
	ImageId          depth_id {};
	// The stencil plane was filled from a stencil keeper (TextureCache::RestoreStencil) and its
	// next association record is current without an upload from guest memory.
	bool             stencil_restored = false;
	// The current stencil plane's mapping into the depth image, also retained by its association.
	ImageSubresourceRange stencil_subresources;
	uint64_t         tick_accessed_last = 0;
	// Alias coherence (TextureCache::SyncAliases): the order of the image's last GPU write among
	// all images, and whether that write is still only in the native image (not yet tiled into
	// the guest-memory buffer, where other images over the same bytes take it from).
	uint64_t         write_stamp        = 0;
	bool             unpublished_write  = false;
	// The image's range generation when it was last found to overlap no other image.
	uint64_t         alone_generation   = 0;
	size_t           lru_id             = 0;
	// The GC tick the LRU entry last received, so that touching it again in the same tick,
	// which the LRU ignores, skips the LRU lookup.
	uint64_t         lru_tick           = 0;
	// TextureCache::MaterializeColorClear's last call for this image when it changed nothing, and
	// the state that result depended on (see ColorClearUnchanged).
	struct ColorClearCheck {
		ImageMetadataInfo metadata;
		ImageViewInfo     view;
		ImageSubresources resources;
		vk::Extent3D      extent;
		uint32_t          image_type              = 0;
		uint32_t          metadata_base_layer     = 0;
		uint8_t           binding_type            = 0;
		bool              valid                   = false;
		// A GPU-written metadata check that found every slice checked (or no key it could
		// apply); otherwise a surface without single-mip metadata slices.
		bool              gpu_checked             = false;
		// A check of metadata the GPU had not written that found no slice's first key to be a
		// clear: it holds while those keys are unchanged and no GPU write came since
		// (gpu_dirty_generation), so the metadata is still not GPU-written.
		bool              cpu_checked             = false;
		struct Keys {
			uint32_t                count = 0;
			std::array<uint64_t, 4> addresses {};
			std::array<uint8_t, 4>  codes {};
		} keys;
		uint64_t          surface_meta_generation = 0;
		uint64_t          gpu_dirty_generation    = 0;
		uint64_t          dcc_checked_generation  = 0;
	} color_clear_check;

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	uint64_t          m_maybe_cpu_hash   = 0;
	bool              m_cpu_dirty        = false;
	bool              m_maybe_cpu_dirty  = false;
	bool              m_maybe_hash_valid = false;
	bool              m_gpu_modified     = false;
	bool              m_buffer_modified  = false;
	bool              m_buffer_tracked   = false;
	// With m_buffer_modified: all of the image written (or where is not known), else m_buffer_ranges.
	static constexpr size_t MaxBufferRanges = 16;
	bool                    m_buffer_whole = false;
	std::vector<GuestRange> m_buffer_ranges;
};

// Debugging aid: GPU work recorded on an image without a VkImage (a stencil association image has
// none) would hand the driver a null handle; logs the guest range, the operation and the zone key.
void NoteNullImage(const Image& image, const char* what);

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
