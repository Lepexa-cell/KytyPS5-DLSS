#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/dccClearResolver.h"
#include "graphics/host_gpu/renderer/cache/dirtyRefresh.h"
#include "graphics/host_gpu/renderer/cache/htileWriter.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <array>
#include <atomic>
#include <optional>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
struct TextureCacheTestAccess;

class TextureCache {
public:
	// Called before a bindless-pinned image is unregistered.
	std::function<void(ImageId)> on_bindless_unregister;
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	// When the lookup finds exactly one image with the same backing, *unique_generation receives
	// the generation of the images over the request's range (RangeGeneration), else 0: until an
	// image over the range is registered or unregistered, the same request finds the same image,
	// and RefindImage does the rest of FindImage for it. With KYTY_OVERLAP_TARGETS, so does a
	// render or depth target request resolved through an overlap (see LastFindOverlap).
	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false,
	                                      uint64_t* unique_generation = nullptr);
	// Whether the last FindImage resolved its request through an overlap with an image of another
	// backing and still gave a generation: a depth target over the image a texture registered, a
	// layer or level of a larger image. Thread_Gpu only.
	[[nodiscard]] bool          LastFindOverlap() const noexcept { return m_last_find_overlap; }
	// KYTY_OVERLAP_TARGETS=0 gives overlap lookups no generation (every draw looks its targets up
	// again, as before); KYTY_DEBUG_AB=overlap alternates.
	[[nodiscard]] static bool   OverlapTargetsEnabled() noexcept;
	// KYTY_VERIFY_OVERLAP_TARGETS=1: checks a target kept from an overlap lookup against a full one.
	void                        VerifyOverlapRefind(ImageId id, const ImageDesc& desc, bool exact_format);
	// FindImage's bookkeeping for a request FindImage resolved to id under generation (see
	// unique_generation). Returns false, doing nothing, when the images over the request's range
	// changed since.
	[[nodiscard]] bool          RefindImage(ImageId id, uint64_t generation, const ImageDesc& desc,
	                                        uint32_t metadata_base_layer);
	// RefindImage's bookkeeping for images whose lookups are known to repeat (the image set and GC
	// tick they were found under are current): each is accessed in this tick, all under one lock.
	// The caller has checked what FinishFind would apply (IsColorClearCurrent). Thread_Gpu only.
	void                        MarkImagesAccessed(std::span<const ImageId> ids);
	// The newest registration or unregistration of an image over `range`, as a stamp of the whole
	// image set's generation (never 0): unchanged while no image over the range came or went.
	// Registering an image stamps its range with a newer value, so a range that gained an image
	// never shows an older one, whatever image slots are reused. Reads without m_lock.
	[[nodiscard]] uint64_t      RangeGeneration(GuestRange range) const noexcept;
	// RangeGeneration over the image's range.
	[[nodiscard]] uint64_t      ImageGeneration(ImageId id) const noexcept {
		return RangeGeneration(m_slot_images[id].info.data);
	}
	// Whether FindTexture for a sampled texture that returned a view while the image had this
	// generation (ImageGeneration) would now only touch the image and return that view again: the
	// image and its views still live, and it needs no refresh. Reads the image without m_lock, as
	// GetImage does.
	[[nodiscard]] bool          IsTextureCurrent(ImageId id, uint64_t generation) const noexcept;
	// IsTextureCurrent's conditions on the image itself, for a caller that knows no image was
	// registered or unregistered since it took the generation (see CurrentTargetState): the slot
	// still holds the image, and the range's generation needs no lookup.
	[[nodiscard]] bool          IsTextureClean(ImageId id) const noexcept;
	// Whether FinishFind's color clear for this request on `id` would do nothing: the surface has
	// no color metadata, or its last check was for the same request and still holds.
	[[nodiscard]] bool IsColorClearCurrent(ImageId id, const ImageDesc& desc,
	                                       uint32_t metadata_base_layer) const;
	// The same for FindRenderTarget: the target is also GPU-owned already, so marking it written
	// changes nothing.
	[[nodiscard]] bool IsRenderTargetCurrent(ImageId id, uint64_t generation) const noexcept;
	// The same for FindTexture with a storage binding: the image is also GPU-owned already, so
	// marking it written and committing the write change nothing, and it is not enrolled for
	// downloads.
	[[nodiscard]] bool IsStorageCurrent(ImageId id, uint64_t generation) const noexcept;
	// The same for FindDepthTarget with a depth-only request: the target is also GPU-owned and a
	// depth target already, with the request's metadata, and no surface metadata entry was added
	// or removed since meta_generation (SurfaceMetaGeneration before that acquisition), so the
	// entry it made is still there.
	[[nodiscard]] bool     IsDepthTargetCurrent(ImageId id, uint64_t generation,
	                                            uint64_t meta_generation,
	                                            const ImageDesc& desc) const noexcept;
	// Bumped whenever an image becomes GPU-modified.
	[[nodiscard]] uint64_t GpuModifiedGeneration() const noexcept {
		return m_gpu_modified_generation.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t SurfaceMetaGeneration() const noexcept {
		return m_surface_meta_generation.load(std::memory_order_acquire);
	}
	// What a successful RefindImage depends on besides its request: the image set, and the tick
	// and GC tick its bookkeeping keys on. While this is unchanged, repeating a request that
	// RefindImage accepted succeeds again and changes nothing, except for FinishFind's color clear
	// (see IsColorClearCurrent). Thread_Gpu only.
	struct TargetState {
		uint64_t image_set = 0;
		uint64_t tick      = 0;
		uint64_t gc_tick   = 0;
		bool     operator==(const TargetState&) const = default;
	};
	[[nodiscard]] TargetState CurrentTargetState() const noexcept;
	void                        UpdateImage(ImageId id);
	// starts_here: set when a registered image over the range starts at `address`, whatever
	// the result (see HasImageStartingAt).
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true,
	                                               bool* starts_here = nullptr);
	// Whether a registered image over the range starts at `address`: without one,
	// FindImageFromRange finds nothing, and it stays so while RangeGeneration of the range is
	// unchanged (KYTY_DRAW_MEMO keeps texel buffer bindings on it).
	[[nodiscard]] bool          HasImageStartingAt(uint64_t address, uint64_t size);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	// Read-only, without touching the LRU (diagnostics); nullptr for a free slot.
	[[nodiscard]] const Image* PeekImage(ImageId id) const { return m_slot_images.try_get(id); }
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);
	[[nodiscard]] BlitHelper& GetBlitHelper() { return m_blit_helper; }

	// `block_bytes` != 0: only an image of that many bytes per texel block qualifies.
	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear, uint32_t block_bytes = 0);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	// What wrote guest memory through a buffer (for the KYTY_GPU_ZONES refresh log); shader
	// storage-buffer stores are the default.
	enum class GpuWriteSource : uint8_t { Fill, Copy, Shader };
	// KYTY_GPU_DIRTY_TRACK: a storage-buffer write of a tracked program (BindingLayout::track) may
	// mark its image as tracked instead of written; then `bitmap` is the device address of the
	// image's bitmap, which the shader's stores fill (see DirtyRefresh).
	struct TrackWrite {
		vk::DeviceAddress bitmap        = 0;
		uint64_t          image_address = 0;
		uint64_t          image_size    = 0;
	};
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size,
	                                           GpuWriteSource source = GpuWriteSource::Shader,
	                                           TrackWrite*    track  = nullptr);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t address, uint64_t size);
	// Queued publications remain authoritative until their guest backing writes complete.
	[[nodiscard]] bool HasPendingDownload(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);
	// Whether every image over the range's pages is a sampled texture only (never a target or
	// storage image, not GPU-modified) and no download covers those pages: a CPU write there
	// needs no image data first, only InvalidateMemory (KYTY_EOP_IMAGE_LABELS). Buffer writes into
	// such an image may exist; a later upload reads them from the cached buffer. GPU thread only:
	// the download check runs outside m_lock, and only the GPU thread records downloads.
	[[nodiscard]] bool OnlyCpuWrittenImages(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsHtileMeta(uint64_t address);
	// Drain statistics only: native DCC metadata ranges seen by image lookups.
	[[nodiscard]] bool IsKnownDccMetadata(uint64_t address, uint64_t size);
	// A GPU write was recorded to guest memory. DCC slices checked on the GPU before it must be
	// checked again.
	void OnBufferGpuWrite(uint64_t address, uint64_t size);
	// A draw or dispatch starts binding its resources (alias sync: writes committed after this
	// belong to it and stay unpublished until it is recorded).
	void BeginBindings() noexcept { m_bindings_begin_stamp = m_write_clock.load(std::memory_order_relaxed); }
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	// stencil_cleared: the clearing write's HTile word also says "stencil cleared" (SMEM, bits 9:8, is
	// 0). On the PS5 such a fill clears a depth target's stencil along with its depth when the stencil
	// lives in that HTile; the title clears its scene depth this way every frame, and with only the depth
	// cleared the previous frame's sky mark (stencil bit 3) hid the ceiling from the depth prepass
	// (black upper hall, frozen by a history pass). Unknown words leave the stencil as it is.
	[[nodiscard]] bool ClearMeta(uint64_t address, bool stencil_cleared = false);
	// Whether [address, address + size) is exactly a known HTile range (a dispatch writing just that is
	// an HTile writer; one writing other bytes there reuses the memory).
	[[nodiscard]] bool IsExactHtileRange(uint64_t address, uint64_t size);
	// Whether the last clear of this HTile also cleared the stencil of <slice>; takes that state.
	[[nodiscard]] bool TakeMetaStencilClear(uint64_t address, uint32_t slice);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);
	// Late HTile clears (KYTY_HTILE_LATE_CLEAR=1). A uniform buffer fill that leaves every HTile
	// word "cleared" (ZMASK 0) is a depth fast clear, but it is only recognized at an address the
	// cache already holds as HTile. The title binds one depth surface in two forms per frame (with and
	// without stencil, each with its own HTile), so each form's image replaces the other's and
	// its HTile entry goes with it: both fills ran as plain writes and the fighter was drawn
	// against the previous frame's depth (blocks of missing pixels in the menus). The fill is
	// remembered here (with its HTile word) and taken when a depth target whose HTile it covers
	// binds before the next flip but one (the cache's own flip counter). Any later write to the
	// range that is not such a fill drops it (DropPendingHtileFills): on the PS5 the clear holds
	// until the HTile memory is overwritten. GPU thread only.
	void NotePendingHtileFill(uint64_t address, uint64_t size, uint32_t value);
	void DropPendingHtileFills(uint64_t address, uint64_t size);
	[[nodiscard]] bool HasPendingHtileFills() const noexcept {
		return m_pending_htile_any.load(std::memory_order_relaxed);
	}
	// The fill's HTile word, or nullopt when no pending fill covers the HTile range.
	[[nodiscard]] std::optional<uint32_t> TakePendingHtileFill(uint64_t address, uint64_t size);
	// HTile synthesis. On the PS5 the DB keeps a depth target's HTile (min/max depth per 8x8 tile)
	// current while it renders; Ghost of Yotei bins its shadow-cascade tiles by reading it, and
	// with HTile left at zero no tile reached a cascade (the moonlight was gone). The host never
	// writes HTile, so a guest read of a surface whose depth target was bound since the last
	// synthesis rebuilds it from the depth image first. GPU thread only.
	[[nodiscard]] bool HasStaleHtile() const noexcept { return !m_htile_stale.empty(); }
	void               SyncHtile(uint64_t address, uint64_t size);
	// A guest write (here: a compute dispatch's written buffer) over a stale surface's HTile range:
	// the guest owns those bytes now, so the entry is dropped instead of synthesized later. The title
	// reuses a retired depth target's HTile memory for its tile-lighting lists; a later read of the
	// list synthesized HTile over it (0xF per tile) and the skin went black
	// (2026-10-23). GPU thread only.
	void               DropStaleHtile(uint64_t address, uint64_t size, const char* writer);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	// KYTY_DEBUG_DUMP_IMAGES (see textureCache.cpp): writes the listed images to files at a frame's
	// end while the trigger file exists.
	void DebugDumpImages();
	// KYTY_DEBUG_DUMP_AT_CS=<hash>: the listed images right after that compute shader's dispatch, in the
	// frames the film dumps (images the game keeps in memory it reuses within the frame).
	void DebugDumpListedNow(std::span<const ImageId> bound = {});
	// On a device fault: the last images created and freed (guest range, extent, format, the
	// VkImage handle and its memory block), to see whether a dispatch or a bindless slot still
	// pointed at an image that was retired or recreated.
	void DescribeRecentImages() const;
	// Diagnostic: prints the images registered over [address, address + size).
	void DescribeImagesIn(uint64_t address, uint64_t size);
	void TickFrame();
	void RunGarbageCollector();

private:
	enum class TransferDirection { Upload, Download };
	struct TextureTransfer;
	struct ImageDownload;

	struct MetaDataInfo {
		enum class Type : uint8_t { CMask, FMask, HTile };

		Type     type;
		uint32_t clear_mask = UINT32_MAX;
		// Slices whose stencil the last clear also cleared (ClearMeta's stencil_cleared).
		uint32_t stencil_clear_mask = 0;
		// Bytes of the HTile range (0 = not known, CMask/FMask entries).
		uint64_t size = 0;
		// KYTY_HTILE_SURFACE_KEEP=1 (upstream PR 1743): the depth surface (native image base) this
		// HTile was last bound with; its clear state belongs to that surface.
		uint64_t depth_address = 0;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 44, 14>;

	// Callers have validated the nonempty 44-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id, std::vector<ImageId>* retired = nullptr);
	void                      FreeImage(ImageId id, std::vector<ImageId>* retired = nullptr);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	// What MaterializeColorClearNow did: nothing it would repeat for the same surface (no
	// single-mip metadata slices, GPU-written slices already checked, or metadata the GPU had not
	// written whose keys, in `keys`, hold no clear), or something else.
	enum class ColorClearOutcome : uint8_t { NoSlices, GpuChecked, CpuChecked, Other };
	[[nodiscard]] ColorClearOutcome MaterializeColorClearNow(ImageId id, const ImageDesc& desc,
	                                                         uint32_t metadata_base_layer,
	                                                         bool&    changed,
	                                                         Image::ColorClearCheck::Keys& keys);
	[[nodiscard]] bool              ColorClearUnchanged(const Image& image, const ImageDesc& desc,
	                                                    uint32_t metadata_base_layer) const;
	[[nodiscard]] static bool ColorClearKeysUnchanged(const Image::ColorClearCheck::Keys& keys);
	void                        FinishFind(ImageId id, const ImageDesc& desc,
	                                       uint32_t metadata_base_layer);
	// Applies clears found in GPU-written color metadata (DCC or CMASK keys) with conditional
	// rendering. Returns false when the target needs the CPU readback path. `no_op`: set when it
	// returned true without doing anything (no applicable key, or the slices were checked already).
	[[nodiscard]] bool MaterializeDccClearOnGpu(ImageId id, const ImageDesc& desc,
	                                            uint64_t slices_address, uint64_t slice_size,
	                                            uint32_t image_first, uint32_t count,
	                                            bool* no_op = nullptr);
	[[nodiscard]] bool DccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count,
	                                    uint32_t code_mask);
	void MarkDccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count,
	                          uint32_t code_mask);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] TextureTransfer
	BuildTextureTransfer(const Image& image, BindingType binding, TransferDirection direction) const;
	[[nodiscard]] ImageDownload BuildDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	// KYTY_GPU_WRITE_RANGES: the 64 KiB blocks over the image's GPU-written ranges alone; false
	// (nothing recorded): not such an image, or the ranges cover all of it.
	[[nodiscard]] bool UploadImageRanges(Image& image);
	// KYTY_GPU_DIRTY_TRACK: whether the image qualifies for tracked writes (cached in the image),
	// readies its bitmap for one, and copies its tracked DWORDs (false: upload it otherwise).
	[[nodiscard]] bool DirtyTarget(Image& image);
	[[nodiscard]] bool PrepareTrackedWrite(Image& image);
	[[nodiscard]] bool RefreshTracked(Image& image);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                       uint64_t destination_size, ImageDownload transfer);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	void               CopyStencil(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil, ImageId prefer = {});
	[[nodiscard]] ImageId StencilPlaneDepth(const ImageInfo& info);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	[[nodiscard]] bool DownloadImageMemory(ImageId id);
	// Render-target aliasing (the title places its transient targets in one memory pool, several
	// surfaces at one address a frame): images over the same guest bytes in the same layout (one
	// surface at another extent or view format) share their contents through the guest-memory
	// buffer, as the console's surfaces share the memory. An image whose
	// latest GPU write is newer than an overlapping image's contents is tiled into the buffer
	// (PublishImage, on the GPU), and the older images over its bytes upload from there when next
	// used. Off by default; KYTY_ALIAS_SYNC=1 turns it on.
	[[nodiscard]] static bool AliasSyncEnabled() noexcept;
	// Whether a write newer than the image's own lies over its bytes: a GPU buffer write (pages
	// stamped in OnBufferGpuWrite) or an overlapping image's write in any layout. Retiring such an
	// image must not publish its older bytes over them.
	[[nodiscard]] bool NewerWriteOver(ImageId id);
	[[nodiscard]] bool        CanPublish(const Image& image) const;
	bool                      PublishImage(ImageId id);
	void                      SyncAliases(ImageId id);
	std::atomic<uint64_t>     m_write_clock {0};
	uint64_t                  m_bindings_begin_stamp = 0;
	// Alias sync: 64 KiB page -> write stamp of the latest GPU buffer write there.
	std::mutex                                   m_buffer_write_mutex;
	std::unordered_map<uint64_t, uint64_t>       m_buffer_write_pages;
	// The stencil plane of a depth image that a color image is about to replace (ResolveDepthOverlap:
	// Ghost of Yotei binds its depth buffer as a storage image every frame), kept in a buffer until
	// the stencil's next image, so the stencil survives the replacement instead of reading stale
	// guest memory.
	struct StencilKeeper {
		std::unique_ptr<Buffer> buffer;
		vk::Extent3D            extent {};
		uint32_t                layers = 0;
		uint32_t                pitch  = 0;
		uint64_t                size   = 0;
	};
	void               KeepStencil(ImageId depth_id);
	// Drops the keeper of `address`; its buffer lives on until the GPU and the record thread are
	// done with the copies recorded from or into it (see RetireStencilBuffer).
	void               EraseStencilKeeper(uint64_t address);
	void               RetireStencilBuffer(std::unique_ptr<Buffer> buffer);
	[[nodiscard]] bool RestoreStencil(Image& image, bool stencil_aspect);
	[[nodiscard]] bool CollectGarbage(bool pressure_only);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	std::unordered_map<vk::Format, ImageId>           m_null_images;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::unordered_map<uint64_t, StencilKeeper>       m_stencil_keepers;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	// Bumped when a surface's clear state changes (ClearMeta, TouchMeta). With
	// m_surface_meta_generation it validates m_meta_clear_memo.
	std::atomic<uint64_t> m_meta_clear_generation {0};
	// IsMetaCleared's last answer: draws ask about the same depth surface over and over. Its callers
	// are GPU-thread only.
	struct MetaClearMemo {
		uint64_t address            = 0;
		uint64_t surface_generation = 0;
		uint64_t clear_generation   = 0;
		uint32_t clear_mask         = 0;
		bool     found              = false;
		bool     valid              = false;
	} m_meta_clear_memo;
	// Bumped when m_surface_metas gains or loses an entry (see ColorClearUnchanged and
	// IsDepthTargetCurrent).
	std::atomic<uint64_t>                             m_surface_meta_generation {0};
	RangeSet                                          m_dcc_metadata_seen;
	struct DccCheckedSlice {
		uint64_t size      = 0;
		uint32_t code_mask = 0;
	};
	DccClearResolver                   m_dcc_resolver;
	HtileWriter                        m_htile_writer;
	std::unique_ptr<DirtyRefresh>      m_dirty_refresh;
	// HTile surfaces whose depth target was bound since their last synthesis (see SyncHtile).
	// The depth target as the guest bound it: the cached image it was drawn into can be larger.
	struct StaleHtile {
		uint64_t address      = 0;
		uint64_t size         = 0;
		uint64_t data_address = 0;
		uint32_t width        = 0;
		uint32_t height       = 0;
		uint32_t layers       = 0;
		ImageId  image {};
	};
	std::vector<StaleHtile>            m_htile_stale;
	void MarkHtileStale(const ImageDesc& desc, ImageId image);
	void SynthesizeHtile(const StaleHtile& surface);
	// Uniform HTile-clearing fills at addresses not (yet) known as HTile (see NotePendingHtileFill).
	struct PendingHtileFill {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t tick    = 0; // m_gc_tick when noted
		uint32_t value   = 0;
	};
	std::vector<PendingHtileFill>      m_pending_htile_fills;
	std::atomic<bool>                  m_pending_htile_any {false};
	bool                               m_dcc_gpu_clear  = false;
	uint64_t                           m_dcc_gpu_checks = 0;
	// Slice address -> slices whose GPU check still reflects the metadata. Any recorded GPU write
	// to the slice erases its entry; CPU writes leave the metadata CPU-dirty, which bypasses it.
	std::mutex                         m_dcc_checked_mutex;
	std::map<uint64_t, DccCheckedSlice> m_dcc_checked;
	// Bumped whenever m_dcc_checked changes (see ColorClearUnchanged).
	std::atomic<uint64_t>              m_dcc_checked_generation {0};
	std::mutex                                        m_pending_download_mutex;
	std::vector<GuestRange>                           m_pending_downloads;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	uint64_t                                          m_last_pressure_gc_tick  = UINT64_MAX;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_readback_linear_images = false;

	// FindImage results where the lookup found exactly one image with the same backing, keyed by
	// every request field SameBacking and the lookup's checks read. A lookup only sees registered
	// images, whose fields SameBacking reads never change, so a result stays right until an image
	// is registered or unregistered, which bumps m_image_set_generation. Caller holds m_lock.
	// A scene looks up more distinct textures than a small direct-mapped table holds (Sky Garden
	// missed 55% of lookups in 512 entries), so the table is set-associative and replaces entries
	// of an older image set first, then the least recently used.
	struct ImageLookup {
		uint64_t            generation = 0;
		uint64_t            last_use   = 0;
		GuestRange          data;
		vk::Extent3D        extent;
		ImageSubresources   resources;
		uint32_t            samples         = 0;
		uint32_t            bytes_per_block = 0;
		Prospero::TileMode  tile_mode       = Prospero::TileMode::kLinear;
		vk::Format          pixel_format    = vk::Format::eUndefined;
		Prospero::ImageType type            = Prospero::ImageType::kColor2D;
		bool                exact_format    = false;
		ImageId             id;

		[[nodiscard]] bool Matches(const ImageInfo& info, bool exact) const noexcept {
			return data == info.data && extent == info.extent && resources == info.resources &&
			       samples == info.samples && bytes_per_block == info.bytes_per_block &&
			       tile_mode == info.tile_mode && pixel_format == info.pixel_format &&
			       type == info.type && exact_format == exact;
		}
	};
	static constexpr size_t ImageLookupWays = 4;
	static constexpr size_t ImageLookupSets = 1024;
	// The first entry of the request's set.
	[[nodiscard]] ImageLookup* ImageLookupSet(const ImageInfo& info, bool exact) noexcept {
		auto hash = info.data.address ^ (info.data.size * 0x9e3779b97f4a7c15ull) ^
		            (static_cast<uint64_t>(info.pixel_format) << 1u) ^ (exact ? 1u : 0u);
		hash ^= hash >> 29u;
		hash *= 0xbf58476d1ce4e5b9ull;
		hash ^= hash >> 32u;
		return &m_image_lookups[static_cast<size_t>(hash % ImageLookupSets) * ImageLookupWays];
	}
	std::vector<ImageLookup> m_image_lookups =
	    std::vector<ImageLookup>(ImageLookupSets * ImageLookupWays);
	uint64_t m_image_lookup_clock = 0;
	bool     m_last_find_overlap  = false;
	// Each cache counts from its own base, so a generation remembered from one cache (a test's
	// earlier context, say) never matches another's.
	[[nodiscard]] static uint64_t        NextGenerationBase() noexcept {
		static std::atomic<uint64_t> caches {0};
		return (caches.fetch_add(1, std::memory_order_relaxed) + 1) << 40u;
	}
	// Changed under m_lock; IsTextureCurrent reads it without.
	std::atomic<uint64_t>                m_image_set_generation {NextGenerationBase()};
	const uint64_t m_image_set_base = m_image_set_generation.load(std::memory_order_relaxed);
	// The stamp (an m_image_set_generation value) of the newest registration or unregistration
	// over each 2 MiB guest region, hashed into buckets: a bucket shared by two regions only costs
	// extra misses. Stamped under m_lock; RangeGeneration reads without.
	static constexpr uint32_t RegionGenerationBits    = 21;
	static constexpr size_t   RegionGenerationBuckets = size_t {1} << 16u;
	std::vector<std::atomic<uint64_t>> m_region_generations =
	    std::vector<std::atomic<uint64_t>>(RegionGenerationBuckets);
	// Advances the image set's generation and stamps the image's regions with it.
	void AdvanceImageSetGeneration(GuestRange range) noexcept;
	// Bumped whenever an image becomes GPU-modified (MarkGpuModified); with the image set
	// generation it validates IsRegionGpuModified's per-thread record of clean pages.
	std::atomic<uint64_t>                m_gpu_modified_generation {NextGenerationBase()};
	void MarkGpuModified(Image& image) noexcept {
		if (!image.IsGpuModified()) {
			image.MarkGpuModified();
			m_gpu_modified_generation.fetch_add(1, std::memory_order_release);
		}
	}

	friend struct TextureCacheTestAccess;
	friend struct PerformanceMemoryTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
};

// The tick the recent-image ring (device-fault report) saw `vk_image` created at; 0 if the ring no
// longer holds that event (it keeps the last 96 creations and frees).
[[nodiscard]] uint64_t RecentImageCreatedTick(uint64_t vk_image);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
