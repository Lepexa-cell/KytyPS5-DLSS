#include "graphics/host_gpu/renderer/debug.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/switches.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/presentation/renderDoc.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

// __rdtsc: <intrin.h> is MSVC's (and clang-cl's); GCC and Clang elsewhere declare it in
// <x86intrin.h>.
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace Libs::Graphics {

uint32_t render_target_mask_slot(uint32_t mask, uint32_t slot) {
	return (mask >> (slot * 4u)) & 0x0fu;
}

static bool RenderTargetMaskHasMrt(uint32_t mask) {
	return (mask & ~0x0fu) != 0;
}

static bool RenderTargetMaskHasBoundMrt(const CommandBuffer& buffer) {
	const auto& hw   = buffer.GetRegisters();
	const auto  mask = hw.GetRenderTargetMask();

	if (!RenderTargetMaskHasMrt(mask)) {
		return false;
	}

	uint32_t bound_targets = 0;
	for (uint32_t i = 0; i < 8; i++) {
		if (render_target_mask_slot(mask, i) != 0 && hw.GetRenderTarget(i).base.addr != 0) {
			bound_targets++;
		}
	}

	return bound_targets > 1;
}

uint32_t render_target_first_bound_slot(const CommandBuffer& buffer) {
	const auto& hw   = buffer.GetRegisters();
	const auto  mask = hw.GetRenderTargetMask();
	for (uint32_t i = 0; i < 8; i++) {
		if (render_target_mask_slot(mask, i) != 0 && hw.GetRenderTarget(i).base.addr != 0) {
			return i;
		}
	}

	return 0;
}

bool graphics_debug_dump_enabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

void uc_print(const char* func, const HW::UserConfig& uc) {
	LOGF("%s\n", func);

	const auto& ge_cntl = uc.GetGeControl();
	const auto& user_en = uc.GetGeUserVgprEn();

	LOGF("\t GetPrimType()         = 0x%08" PRIx32 "\n"
	     "\t GetIndexOffset()      = 0x%08" PRIx32 "\n"
	     "\t GetObjectId()         = 0x%08" PRIx32 "\n"
	     "\t primitive_reset       = 0x%08" PRIx32 "\n"
	     "\t primitive_group_size  = 0x%04" PRIx16 "\n"
	     "\t vertex_group_size     = 0x%04" PRIx16 "\n"
	     "\t en_user_vgpr1         = %s\n"
	     "\t en_user_vgpr2         = %s\n"
	     "\t en_user_vgpr3         = %s\n",
	     static_cast<uint32_t>(uc.GetPrimType()), uc.GetIndexOffset(), uc.GetObjectId(),
	     uc.GetPrimitiveResetControl(), ge_cntl.primitive_group_size, ge_cntl.vertex_group_size,
	     user_en.vgpr1 ? "true" : "false", user_en.vgpr2 ? "true" : "false",
	     user_en.vgpr3 ? "true" : "false");
}

void uc_check(const HW::UserConfig& uc) {
	const auto& user_en = uc.GetGeUserVgprEn();

	EXIT_NOT_IMPLEMENTED(user_en.vgpr1 != false);
	EXIT_NOT_IMPLEMENTED(user_en.vgpr2 != false);
	EXIT_NOT_IMPLEMENTED(user_en.vgpr3 != false);
}

std::string rt_print(const char* func, const HW::RenderTarget& rt) {
	std::string dst;
	dst.reserve(4096);

	dst += fmt::format("{}\n", func);

	dst += fmt::format("\t base.addr                       = 0x{:016x}\n", rt.base.addr);
	dst += fmt::format("\t view.base_array_slice_index     = 0x{:08x}\n",
	                  rt.view.base_array_slice_index);
	dst += fmt::format("\t view.last_array_slice_index     = 0x{:08x}\n",
	                  rt.view.last_array_slice_index);
	dst += fmt::format("\t view.current_mip_level          = 0x{:08x}\n", rt.view.current_mip_level);
	dst += fmt::format("\t info.fmask_compression_enable   = {}\n",
	                  rt.info.fmask_compression_enable ? "true" : "false");

	dst += fmt::format("\t info.fmask_data_compression_disable = {}\n",
	                  rt.info.fmask_data_compression_disable ? "true" : "false");
	dst += fmt::format("\t info.fmask_one_frag_mode        = {}\n",
	                  rt.info.fmask_one_frag_mode ? "true" : "false");

	dst += fmt::format("\t info.cmask_fast_clear_enable    = {}\n",
	                  rt.info.cmask_fast_clear_enable ? "true" : "false");
	dst += fmt::format("\t info.dcc_compression_enable     = {}\n",
	                  rt.info.dcc_compression_enable ? "true" : "false");
	dst += fmt::format("\t info.format                     = 0x{:08x}\n",
	                  static_cast<uint32_t>(rt.info.format));
	dst += fmt::format("\t info.channel_type               = 0x{:08x}\n",
	                  static_cast<uint32_t>(rt.info.channel_type));
	dst += fmt::format("\t info.channel_order              = 0x{:08x}\n",
	                  static_cast<uint32_t>(rt.info.channel_order));
	dst += fmt::format("\t info.blend_bypa                 = {}\n",
	                  rt.info.blend_bypass ? "true" : "false");
	dst += fmt::format("\t info.blend_clamp                = {}\n",
	                  rt.info.blend_clamp ? "true" : "false");
	dst += fmt::format("\t info.round_mode                 = {}\n",
	                  rt.info.round_mode ? "true" : "false");
	dst += fmt::format("\t attrib.force_dest_alpha_to_one  = {}\n",
	                  rt.attrib.force_dest_alpha_to_one ? "true" : "false");
	dst += fmt::format("\t attrib.num_samples              = 0x{:08x}\n", rt.attrib.num_samples);
	dst += fmt::format("\t attrib.num_fragments            = 0x{:08x}\n", rt.attrib.num_fragments);
	dst += fmt::format("\t attrib2.width                   = 0x{:08x}\n", rt.attrib2.width);
	dst += fmt::format("\t attrib2.height                  = 0x{:08x}\n", rt.attrib2.height);
	dst += fmt::format("\t attrib2.num_mip_levels          = 0x{:08x}\n", rt.attrib2.num_mip_levels);
	dst += fmt::format("\t attrib3.depth                   = 0x{:08x}\n", rt.attrib3.depth);
	dst += fmt::format("\t attrib3.tile_mode               = 0x{:08x}\n",
	                  static_cast<uint32_t>(rt.attrib3.tile_mode));
	dst += fmt::format("\t attrib3.dimension               = 0x{:08x}\n", rt.attrib3.dimension);
	dst += fmt::format("\t attrib3.metadata_pipe_aligned   = {}\n",
	                  rt.attrib3.metadata_pipe_aligned ? "true" : "false");
	dst += fmt::format("\t attrib3.write_vrs_rate_hint_to_cmask = {}\n",
	                  rt.attrib3.write_vrs_rate_hint_to_cmask ? "true" : "false");
	dst += fmt::format("\t dcc.max_uncompressed_block_size = 0x{:08x}\n",
	                  rt.dcc.max_uncompressed_block_size);
	dst += fmt::format("\t dcc.max_compressed_block_size   = 0x{:08x}\n",
	                  rt.dcc.max_compressed_block_size);
	dst += fmt::format("\t dcc.color_transform             = 0x{:08x}\n", rt.dcc.color_transform);
	dst += fmt::format("\t dcc.overwrite_combiner_disable  = {}\n",
	                  rt.dcc.overwrite_combiner_disable ? "true" : "false");
	dst += fmt::format("\t dcc.independent_block_size      = 0x{:02x}\n",
	                  static_cast<uint8_t>(rt.dcc.independent_block_size));
	dst += fmt::format("\t data_write_on_dcc_clear_to_reg  = {}\n",
	                  rt.dcc.data_write_on_dcc_clear_to_reg ? "true" : "false");
	dst += fmt::format("\t dcc.dcc_clear_key_enable        = {}\n",
	                  rt.dcc.dcc_clear_key_enable ? "true" : "false");
	dst += fmt::format("\t cmask.addr                      = 0x{:016x}\n", rt.cmask.addr);
	dst += fmt::format("\t fmask.addr                      = 0x{:016x}\n", rt.fmask.addr);
	dst += fmt::format("\t clear_word0.word0               = 0x{:08x}\n", rt.clear_word0.word0);
	dst += fmt::format("\t clear_word1.word1               = 0x{:08x}\n", rt.clear_word1.word1);
	dst += fmt::format("\t dcc_addr.addr                   = 0x{:016x}\n", rt.dcc_addr.addr);

	return dst;
}

static bool RenderIsColorTileMode(Prospero::TileMode tile_mode) {
	// AGC CxRenderTarget::TileMode shifted by CB_COLOR*_ATTRIB3.COLOR_SW_MODE.
	switch (tile_mode) {
		case Prospero::TileMode::kLinear:
		case Prospero::TileMode::kStandard256B:
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
		case Prospero::TileMode::kPrt:
		case Prospero::TileMode::kDepth:
		case Prospero::TileMode::kRenderTarget: return true;
		default: return false;
	}
}

bool RenderIsColorTileModeLinear(Prospero::TileMode tile_mode) {
	return tile_mode == Prospero::TileMode::kLinear;
}

static bool RenderIsColorDimension(uint32_t dimension) {
	return dimension <= 0x02;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void RtCheck(const HW::RenderTarget& rt) {
	if (rt.base.addr != 0) {
		//  EXIT_NOT_IMPLEMENTED(rt.base_addr == 0);

		EXIT_NOT_IMPLEMENTED(rt.view.base_array_slice_index > rt.view.last_array_slice_index);
		if (rt.view.base_array_slice_index != 0x00000000 ||
		    rt.view.last_array_slice_index != 0x00000000) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: using color target array slice range %" PRIu32 "..%" PRIu32
				     "\n",
				     rt.view.base_array_slice_index, rt.view.last_array_slice_index);
				logged = true;
			}
		}
		if (rt.view.current_mip_level != 0x00000000) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: using PS5 color target mip level %" PRIu32 "\n",
				     rt.view.current_mip_level);
				logged = true;
			}
		}
		if (rt.info.fmask_compression_enable) {
			EXIT_NOT_IMPLEMENTED(rt.attrib.num_samples == 0 && rt.attrib.num_fragments == 0);
			// Native MSAA stores expanded samples, independent of FMASK metadata compression.
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: using expanded native Vulkan MSAA samples, "
				     "fmask=0x%016" PRIx64 "\n",
				     rt.fmask.addr);
				logged = true;
			}
		}

		if (rt.info.blend_bypass) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: temporary: using PS5 blend bypass as disabled Vulkan "
				     "blending\n");
				logged = true;
			}
		}
		// EXIT_NOT_IMPLEMENTED(rt.info.blend_clamp != false);
		if (rt.info.round_mode) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: temporary: ignoring PS5 color round mode\n");
				logged = true;
			}
		}
		//		 EXIT_NOT_IMPLEMENTED(rt.format != 0x0000000a);
		// EXIT_NOT_IMPLEMENTED(rt.channel_type != 0x00000006);
		// EXIT_NOT_IMPLEMENTED(rt.channel_order != 0x00000001);
		if (rt.attrib.force_dest_alpha_to_one) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: temporary: accepting PS5 force destination alpha-to-one\n");
				logged = true;
			}
		}
		if (rt.attrib.num_samples != 0x00000000 || rt.attrib.num_fragments != 0x00000000) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: using native PS5 MSAA color target, "
				     "samples=0x%08" PRIx32 " fragments=0x%08" PRIx32 "\n",
				     rt.attrib.num_samples, rt.attrib.num_fragments);
				logged = true;
			}
		}

		if (rt.attrib2.width == 0x00000000 || rt.attrib2.height == 0x00000000) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: temporary: accepting PS5 raw 1-pixel color target extent "
				     "fields width_minus1=0x%08" PRIx32 " height_minus1=0x%08" PRIx32 "\n",
				     rt.attrib2.width, rt.attrib2.height);
				logged = true;
			}
		}
		if (rt.attrib2.num_mip_levels != 0x00000000) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: using PS5 color target mip count field 0x%08" PRIx32 "\n",
				     rt.attrib2.num_mip_levels);
				logged = true;
			}
		}
		if (!RenderIsColorTileMode(rt.attrib3.tile_mode)) {
			EXIT("unknown PS5 render-target tile mode: 0x%08" PRIx32 "\n",
			     static_cast<uint32_t>(rt.attrib3.tile_mode));
		}
		if (!RenderIsColorDimension(rt.attrib3.dimension)) {
			EXIT("unknown PS5 render-target dimension: 0x%08" PRIx32 "\n", rt.attrib3.dimension);
		}
		if (!rt.attrib3.metadata_pipe_aligned) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: temporary: accepting unaligned PS5 metadata pipe flag\n");
				logged = true;
			}
		}
		EXIT_NOT_IMPLEMENTED(rt.attrib3.write_vrs_rate_hint_to_cmask);

		// EXIT_NOT_IMPLEMENTED(rt.dcc_max_uncompressed_block_size != 0x00000002);
		// EXIT_NOT_IMPLEMENTED(rt.dcc.max_compressed_block_size != 0x00000000);
		// EXIT_NOT_IMPLEMENTED(rt.dcc.color_transform != 0x00000000);
		EXIT_NOT_IMPLEMENTED(rt.dcc.overwrite_combiner_disable != false);
		// EXIT_NOT_IMPLEMENTED(rt.dcc.data_write_on_dcc_clear_to_reg != false);
		EXIT_NOT_IMPLEMENTED(rt.dcc.dcc_clear_key_enable != false);
		if (rt.cmask.addr != 0x0000000000000000 || rt.fmask.addr != 0x0000000000000000 ||
		    rt.dcc_addr.addr != 0x0000000000000000) {
			static bool logged = false;
			if (!logged) {
				LOGF("RenderTarget: temporary: ignoring PS5 metadata addresses cmask=0x%016" PRIx64
				     " fmask=0x%016" PRIx64 " dcc=0x%016" PRIx64 "\n",
				     rt.cmask.addr, rt.fmask.addr, rt.dcc_addr.addr);
				logged = true;
			}
		}
	}
}

static void ZPrint(const char* func, const HW::DepthRenderTarget& z) {
	LOGF("%s\n", func);

	LOGF("\t z_info.format                         = 0x%08" PRIx32 "\n"
	     "\t z_info.num_samples                    = 0x%08" PRIx32 "\n"
	     "\t z_info.texture_compatibility          = 0x%08" PRIx32 "\n"
	     "\t z_info.htile_acceleration             = %s\n"
	     "\t z_info.expclear_enabled               = %s\n"
	     "\t z_info.z_compare_base                 = 0x%08" PRIx32 "\n"
	     "\t z_info.partially_resident             = %s\n"
	     "\t z_info.max_mip_level                  = 0x%02" PRIx8 "\n"
	     "\t stencil_info.format                   = 0x%08" PRIx32 "\n"
	     "\t stencil_info.texture_compatibility    = 0x%08" PRIx32 "\n"
	     "\t stencil_info.htile_stencil_disabled   = %s\n"
	     "\t stencil_info.expclear_enabled         = %s\n"
	     "\t stencil_info.partially_resident       = %s\n"
	     "\t depth_view.slice_start                = 0x%08" PRIx32 "\n"
	     "\t depth_view.slice_max                  = 0x%08" PRIx32 "\n"
	     "\t depth_view.current_mip_level          = 0x%02" PRIx8 "\n"
	     "\t depth_view.depth_write_disable        = %s\n"
	     "\t depth_view.stencil_write_disable      = %s\n"
	     "\t z_read_base_addr                      = 0x%016" PRIx64 "\n"
	     "\t stencil_read_base_addr                = 0x%016" PRIx64 "\n"
	     "\t z_write_base_addr                     = 0x%016" PRIx64 "\n"
	     "\t stencil_write_base_addr               = 0x%016" PRIx64 "\n"
	     "\t htile_data_base_addr                  = 0x%016" PRIx64 "\n"
	     "\t shading_rate_encoding                 = 0x%02" PRIx8 "\n"
	     "\t size.x_max                            = 0x%04" PRIx16 "\n"
	     "\t size.y_max                            = 0x%04" PRIx16 "\n"
	     "\t size.valid                            = %s\n",
	     static_cast<uint32_t>(z.z_info.format), z.z_info.num_samples,
	     static_cast<uint32_t>(z.z_info.texture_compatibility),
	     z.z_info.htile_acceleration ? "true" : "false",
	     z.z_info.expclear_enabled ? "true" : "false",
	     static_cast<uint32_t>(z.z_info.z_compare_base),
	     z.z_info.partially_resident ? "true" : "false", z.z_info.max_mip_level,
	     static_cast<uint32_t>(z.stencil_info.format),
	     static_cast<uint32_t>(z.stencil_info.texture_compatibility),
	     z.stencil_info.htile_stencil_disabled ? "true" : "false",
	     z.stencil_info.expclear_enabled ? "true" : "false",
	     z.stencil_info.partially_resident ? "true" : "false", z.depth_view.slice_start,
	     z.depth_view.slice_max, z.depth_view.current_mip_level,
	     z.depth_view.depth_write_disable ? "true" : "false",
	     z.depth_view.stencil_write_disable ? "true" : "false", z.z_read_base_addr,
	     z.stencil_read_base_addr, z.z_write_base_addr, z.stencil_write_base_addr,
	     z.htile_data_base_addr, z.shading_rate_encoding, z.size.x_max, z.size.y_max,
	     z.size.valid ? "true" : "false");
}

static void ClipPrint(const char* func, const HW::ClipControl& c) {
	LOGF("%s\n", func);

	LOGF("\t user_clip_planes                    = 0x%02" PRIx8 "\n"
	     "\t user_clip_plane_mode                = 0x%02" PRIx8 "\n"
	     "\t dx_clip_space                       = %s\n"
	     "\t vertex_kill_any                     = %s\n"
	     "\t min_z_clip_disable                  = %s\n"
	     "\t max_z_clip_disable                  = %s\n"
	     "\t user_clip_plane_negate_y            = %s\n"
	     "\t clip_disable                        = %s\n"
	     "\t user_clip_plane_cull_only           = %s\n"
	     "\t cull_on_clipping_error_disable      = %s\n"
	     "\t linear_attribute_clip_enable        = %s\n"
	     "\t force_viewport_index_from_vs_enable = %s\n",
	     c.user_clip_planes, c.user_clip_plane_mode, c.dx_clip_space ? "true" : "false",
	     c.vertex_kill_any ? "true" : "false", c.min_z_clip_disable ? "true" : "false",
	     c.max_z_clip_disable ? "true" : "false", c.user_clip_plane_negate_y ? "true" : "false",
	     c.clip_disable ? "true" : "false", c.user_clip_plane_cull_only ? "true" : "false",
	     c.cull_on_clipping_error_disable ? "true" : "false",
	     c.linear_attribute_clip_enable ? "true" : "false",
	     c.force_viewport_index_from_vs_enable ? "true" : "false");
}

static void ClipCheck(const HW::ClipControl& c) {
	// dx_linear_attr_clip_enable preserves linear (noperspective) attributes at clip-generated
	// vertices, which Vulkan provides as part of clipping and interpolation.
	EXIT_NOT_IMPLEMENTED(c.user_clip_planes != 0 || c.user_clip_plane_mode != 0 ||
	                     c.vertex_kill_any || c.user_clip_plane_negate_y ||
	                     c.user_clip_plane_cull_only || c.cull_on_clipping_error_disable ||
	                     c.force_viewport_index_from_vs_enable);
}

static void RcPrint(const char* func, const HW::RenderControl& c) {
	LOGF("%s\n", func);

	LOGF("\t depth_clear_enable       = %s\n"
	     "\t stencil_clear_enable     = %s\n"
	     "\t resummarize_enable       = %s\n"
	     "\t stencil_compress_disable = %s\n"
	     "\t depth_compress_disable   = %s\n"
	     "\t copy_centroid            = %s\n"
	     "\t copy_sample              = %" PRIu8 "\n",
	     c.depth_clear_enable ? "true" : "false", c.stencil_clear_enable ? "true" : "false",
	     c.resummarize_enable ? "true" : "false", c.stencil_compress_disable ? "true" : "false",
	     c.depth_compress_disable ? "true" : "false", c.copy_centroid ? "true" : "false",
	     c.copy_sample);
}

static void RcCheck(const HW::RenderControl& c) {
	// EXIT_NOT_IMPLEMENTED(c.depth_clear_enable != false);
	// EXIT_NOT_IMPLEMENTED(c.stencil_clear_enable != false);
	// EXIT_NOT_IMPLEMENTED(c.stencil_compress_disable != false);
	// EXIT_NOT_IMPLEMENTED(c.depth_compress_disable != false);
	EXIT_NOT_IMPLEMENTED(c.copy_centroid != false);
	EXIT_NOT_IMPLEMENTED(c.copy_sample != 0);
}

static void McPrint(const char* func, const HW::ModeControl& c) {
	LOGF("%s\n", func);

	LOGF("\t cull_front               = %s\n"
	     "\t cull_back                = %s\n"
	     "\t face                     = %s\n"
	     "\t poly_mode                = %" PRIu8 "\n"
	     "\t polymode_front_ptype     = %" PRIu8 "\n"
	     "\t polymode_back_ptype      = %" PRIu8 "\n"
	     "\t poly_offset_front_enable = %s\n"
	     "\t poly_offset_back_enable  = %s\n"
	     "\t vtx_window_offset_enable = %s\n"
	     "\t provoking_vtx_last       = %s\n"
	     "\t persp_corr_dis           = %s\n",
	     c.cull_front ? "true" : "false", c.cull_back ? "true" : "false", c.face ? "true" : "false",
	     c.poly_mode, c.polymode_front_ptype, c.polymode_back_ptype,
	     c.poly_offset_front_enable ? "true" : "false",
	     c.poly_offset_back_enable ? "true" : "false",
	     c.vtx_window_offset_enable ? "true" : "false", c.provoking_vtx_last ? "true" : "false",
	     c.persp_corr_dis ? "true" : "false");
}

static void McCheck(const HW::ModeControl& c) {
	// EXIT_NOT_IMPLEMENTED(c.cull_front != false);
	// EXIT_NOT_IMPLEMENTED(c.cull_back != false);
	// EXIT_NOT_IMPLEMENTED(c.face != false);
	if (c.vtx_window_offset_enable) {
		static bool logged = false;
		if (!logged) {
			LOGF("\t temporary: PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE is not fully "
			     "implemented; continuing without vertex window "
			     "offset adjustment\n");
			logged = true;
		}
	}
	EXIT_NOT_IMPLEMENTED(c.persp_corr_dis != false);
}

static void BcPrint(const char* func, const HW::BlendControl& c, const HW::BlendColor& color,
                    const HW::ColorControl& cc) {
	LOGF("%s\n", func);

	LOGF("\t color_srcblend       = %" PRIu8 "\n"
	     "\t color_comb_fcn       = %" PRIu8 "\n"
	     "\t color_destblend      = %" PRIu8 "\n"
	     "\t alpha_srcblend       = %" PRIu8 "\n"
	     "\t alpha_comb_fcn       = %" PRIu8 "\n"
	     "\t alpha_destblend      = %" PRIu8 "\n"
	     "\t separate_alpha_blend = %s\n"
	     "\t enable               = %s\n"
	     "\t red                  = %f\n"
	     "\t green                = %f\n"
	     "\t blue                 = %f\n"
	     "\t alpha                = %f\n"
	     "\t cc.mode              = %" PRIu8 "\n"
	     "\t cc.op                = %" PRIu8 "\n",
	     c.color_srcblend, c.color_comb_fcn, c.color_destblend, c.alpha_srcblend, c.alpha_comb_fcn,
	     c.alpha_destblend, c.separate_alpha_blend ? "true" : "false", c.enable ? "true" : "false",
	     color.red, color.green, color.blue, color.alpha, cc.mode, cc.op);
}

static void BcCheck(const HW::BlendColor& color, const HW::ColorControl& cc) {
	if (color.red != 0.0f || color.green != 0.0f || color.blue != 0.0f || color.alpha != 0.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("BlendControl: temporary: accepting nonzero blend constants (%f, %f, %f, %f)\n",
			     color.red, color.green, color.blue, color.alpha);
			logged = true;
		}
	}
	if (cc.mode != 1 && cc.mode != 0 && cc.mode != 2 && cc.mode != 3 && cc.mode != 5 &&
	    cc.mode != 6) {
		static bool logged = false;
		if (!logged) {
			LOGF("BlendControl: temporary: accepting unsupported color-control mode %" PRIu8 "\n",
			     cc.mode);
			logged = true;
		}
	}
	if (cc.op != 0xCC) {
		static bool logged = false;
		if (!logged) {
			LOGF("BlendControl: temporary: accepting unsupported raster op 0x%02" PRIx8 "\n",
			     cc.op);
			logged = true;
		}
	}
}

static void DPrint(const char* func, const HW::DepthControl& c, const HW::StencilControl& s,
                   const HW::StencilMask& sm) {
	LOGF("%s\n", func);

	LOGF("\t stencil_enable       = %s\n"
	     "\t z_enable             = %s\n"
	     "\t z_write_enable       = %s\n"
	     "\t depth_bounds_enable  = %s\n"
	     "\t zfunc                = %" PRIu8 "\n"
	     "\t backface_enable      = %s\n"
	     "\t stencilfunc          = %" PRIu8 "\n"
	     "\t stencilfunc_bf       = %" PRIu8 "\n"
	     "\t stencil_fail         = %" PRIu8 "\n"
	     "\t stencil_zpass        = %" PRIu8 "\n"
	     "\t stencil_zfail        = %" PRIu8 "\n"
	     "\t stencil_fail_bf      = %" PRIu8 "\n"
	     "\t stencil_zpass_bf     = %" PRIu8 "\n"
	     "\t stencil_zfail_bf     = %" PRIu8 "\n"
	     "\t stencil_testval      = %" PRIu8 "\n"
	     "\t stencil_mask         = %" PRIu8 "\n"
	     "\t stencil_writemask    = %" PRIu8 "\n"
	     "\t stencil_opval        = %" PRIu8 "\n"
	     "\t stencil_testval_bf   = %" PRIu8 "\n"
	     "\t stencil_mask_bf      = %" PRIu8 "\n"
	     "\t stencil_writemask_bf = %" PRIu8 "\n"
	     "\t stencil_opval_bf     = %" PRIu8 "\n",
	     c.stencil_enable ? "true" : "false", c.z_enable ? "true" : "false",
	     c.z_write_enable ? "true" : "false", c.depth_bounds_enable ? "true" : "false", c.zfunc,
	     c.backface_enable ? "true" : "false", c.stencilfunc, c.stencilfunc_bf, s.stencil_fail,
	     s.stencil_zpass, s.stencil_zfail, s.stencil_fail_bf, s.stencil_zpass_bf,
	     s.stencil_zfail_bf, sm.stencil_testval, sm.stencil_mask, sm.stencil_writemask,
	     sm.stencil_opval, sm.stencil_testval_bf, sm.stencil_mask_bf, sm.stencil_writemask_bf,
	     sm.stencil_opval_bf);
}

static void EqaaPrint(const char* func, const HW::EqaaControl& c) {
	LOGF("%s\n", func);

	LOGF("\t max_anchor_samples         = %" PRIu8 "\n"
	     "\t ps_iter_samples            = %" PRIu8 "\n"
	     "\t mask_export_num_samples    = %" PRIu8 "\n"
	     "\t alpha_to_mask_num_samples  = %" PRIu8 "\n"
	     "\t high_quality_intersections = %s\n"
	     "\t incoherent_eqaa_reads      = %s\n"
	     "\t interpolate_comp_z         = %s\n"
	     "\t static_anchor_associations = %s\n",
	     c.max_anchor_samples, c.ps_iter_samples, c.mask_export_num_samples,
	     c.alpha_to_mask_num_samples, c.high_quality_intersections ? "true" : "false",
	     c.incoherent_eqaa_reads ? "true" : "false", c.interpolate_comp_z ? "true" : "false",
	     c.static_anchor_associations ? "true" : "false");
}

static void EqaaCheck(const HW::EqaaControl& c, const HW::AaConfig& cf) {
	// The EQAA controls only take effect while multisampling is on, so a single-sample
	// target leaves them inert and has nothing to warn about.
	if (cf.msaa_num_samples == 0) {
		return;
	}
	if (c.max_anchor_samples != 0 || c.ps_iter_samples != 0 || c.mask_export_num_samples != 0 ||
	    c.alpha_to_mask_num_samples != 0 || c.high_quality_intersections ||
	    c.incoherent_eqaa_reads || c.interpolate_comp_z || c.static_anchor_associations) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("\t warning: unsupported PS5 EQAA controls use native Vulkan MSAA defaults\n");
		}
	}
}

static void AaPrint(const char* func, const HW::AaSampleControl& c, const HW::AaConfig& cf) {
	LOGF("%s\n", func);

	LOGF("\t centroid_priority = %016" PRIx64 "\n", c.centroid_priority);
	for (int i = 0; i < 16; i++) {
		LOGF("\t locations[%d] = %08" PRIx32 "\n", i, c.locations[i]);
	}
	LOGF("\t msaa_num_samples      = %" PRIu8 "\n"
	     "\t aa_mask_centroid_dtmn = %s\n"
	     "\t max_sample_dist       = %" PRIu8 "\n"
	     "\t msaa_exposed_samples  = %" PRIu8 "\n",
	     cf.msaa_num_samples, cf.aa_mask_centroid_dtmn ? "true" : "false", cf.max_sample_dist,
	     cf.msaa_exposed_samples);
}

static void AaCheck(const HW::AaSampleControl& c, const HW::AaConfig& cf) {
	bool non_default_locations = (c.centroid_priority != 0);
	for (uint32_t l: c.locations) {
		non_default_locations |= (l != 0);
	}

	if (non_default_locations || cf.msaa_num_samples != 0 || cf.aa_mask_centroid_dtmn ||
	    cf.max_sample_dist != 0 || cf.msaa_exposed_samples != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("\t warning: unsupported PS5 sample locations use native Vulkan locations: "
			     "samples=%" PRIu8 ", exposed=%" PRIu8 ", max_dist=%" PRIu8 "\n",
			     cf.msaa_num_samples, cf.msaa_exposed_samples, cf.max_sample_dist);
		}
	}
}

uint64_t DrawPhaseTimer::Hash() {
	static const uint64_t hash = [] {
		const char* text = KYTY_GETENV("KYTY_DEBUG_DRAW_PHASES");
		if (text == nullptr) {
			return uint64_t {0};
		}
		return std::strcmp(text, "all") == 0 ? AllDraws : std::strtoull(text, nullptr, 16);
	}();
	return hash;
}

uint64_t DrawPhaseTimer::Now() {
	return __rdtsc();
}

// KYTY_DEBUG_FULL_BARRIERS_FILE=<path> turns them on only while that file exists (checked twice a
// second): a route that needs the normal frame rate to get somewhere creates it on arrival.
bool DebugFullBarriers() noexcept {
	static const bool  enabled = KYTY_GETENV("KYTY_DEBUG_FULL_BARRIERS") != nullptr;
	static const char* file    = KYTY_GETENV("KYTY_DEBUG_FULL_BARRIERS_FILE");
	if (enabled || file == nullptr) {
		return enabled;
	}
	static std::atomic<int64_t> next_check {0};
	static std::atomic<bool>    present {false};
	const auto                  now = std::chrono::steady_clock::now().time_since_epoch();
	if (now.count() >= next_check.load(std::memory_order_relaxed)) {
		next_check.store((now + std::chrono::milliseconds(500)).count(), std::memory_order_relaxed);
		std::error_code error;
		present.store(std::filesystem::exists(file, error), std::memory_order_relaxed);
	}
	return present.load(std::memory_order_relaxed);
}

bool ComputeChainEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = KYTY_GETENV("KYTY_COMPUTE_CHAIN");
		return value != nullptr && value[0] != '0' && KYTY_GETENV("KYTY_DEBUG_FULL_BARRIERS") == nullptr &&
		       KYTY_GETENV("KYTY_DEBUG_FULL_BARRIERS_FILE") == nullptr;
	}();
	return enabled;
}

bool ReadbackAgeEnabled() noexcept {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_READBACK_AGE") != nullptr;
	return enabled;
}

// Dispatches recorded while a rendering instance was open (render thread only).
static uint64_t g_dispatch_pass_breaks = 0;

void NoteDispatchPassBreak(bool rendering) noexcept {
	g_dispatch_pass_breaks += rendering ? 1u : 0u;
}

void NoteComputeChain(bool continued) noexcept {
	// Dispatches are recorded on the render thread only.
	static uint64_t total     = 0;
	static uint64_t chained   = 0;
	total++;
	chained += continued ? 1u : 0u;
	if (total % 50000u == 0u) {
		std::printf("compute chain: %" PRIu64 " of %" PRIu64
		            " dispatches without barriers before them (%.1f%%), %" PRIu64
		            " ended a rendering instance\n",
		            chained, total, 100.0 * static_cast<double>(chained) / static_cast<double>(total),
		            g_dispatch_pass_breaks);
		std::fflush(stdout);
	}
}

// KYTY_DEBUG_SKIP_SHADERS_LIVE=<path>: the file holds pixel or vertex shader hashes (hex,
// separated by commas or whitespace), read again twice a second, so groups of draws can be left
// out and brought back while the game runs, to see what they cost and what they draw.
static bool LiveSkipListed(const char* file, uint64_t shader_hash) {
	static std::mutex            mutex;
	static std::vector<uint64_t> hashes;
	static int64_t               next_check = 0;
	const std::lock_guard        lock(mutex);
	const auto                   now = std::chrono::steady_clock::now().time_since_epoch().count();
	if (now >= next_check) {
		next_check = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
		                       std::chrono::milliseconds(500))
		                       .count();
		hashes.clear();
		if (auto* in = std::fopen(file, "rb"); in != nullptr) {
			char text[4096] {};
			const auto read = std::fread(text, 1, sizeof(text) - 1, in);
			std::fclose(in);
			text[read] = '\0';
			for (const char* p = text; *p != '\0';) {
				char*      end   = nullptr;
				const auto value = std::strtoull(p, &end, 16);
				if (end == p) {
					p++;
					continue;
				}
				if (value != 0) {
					hashes.push_back(value);
				}
				p = end;
			}
		}
	}
	return std::ranges::find(hashes, shader_hash) != hashes.end();
}

// KYTY_DEBUG_SKIP_SHADERS_FILE=<path> skips them only while that file exists (checked twice a
// second), like KYTY_DEBUG_FULL_BARRIERS_FILE.
// Each shader is printed the first time it is skipped ("skip-shader: ps <hash>"), so a bisection
// ends with the list of the shaders in its last half.
bool DebugSkipShader(uint64_t shader_hash, DebugShaderKind kind, bool bisect) noexcept {
	// KYTY_RENDERDOC_ON_SHADER=<hex hash>: a RenderDoc capture (--rd) is requested the first time
	// a draw or dispatch with this shader is recorded; it covers the next two frames (a state that
	// goes wrong on a shader's first frame: the title cloth cs 0c7e4ec4e387a0fa).
	static const uint64_t capture_hash = [] {
		const char* text = KYTY_GETENV("KYTY_RENDERDOC_ON_SHADER");
		return text != nullptr ? std::strtoull(text, nullptr, 16) : 0ull;
	}();
	if (capture_hash != 0 && shader_hash == capture_hash) [[unlikely]] {
		// KYTY_RENDERDOC_ON_SHADER_DELAY_MS=<n>: the request waits until the shader has run for n ms
		// (a capture during the very first frame lost the device in the title's fight load).
		static const int64_t delay_ms = [] {
			const char* text = KYTY_GETENV("KYTY_RENDERDOC_ON_SHADER_DELAY_MS");
			return text != nullptr ? std::strtoll(text, nullptr, 10) : 0ll;
		}();
		static const auto first_seen = std::chrono::steady_clock::now();
		static std::atomic<bool> requested {false};
		if (std::chrono::steady_clock::now() - first_seen >= std::chrono::milliseconds(delay_ms) &&
		    !requested.exchange(true)) {
			std::printf("renderdoc-on-shader: 0x%016llx first recorded, capture requested\n",
			            static_cast<unsigned long long>(shader_hash));
			std::fflush(stdout);
			RenderDocRequestCapture();
		}
	}
	// KYTY_DEBUG_SHADER_ACTIVITY=1 prints each shader the first time it runs and again whenever it
	// runs after 3 s unused, to find the shaders behind a short effect (a splash, a flash).
	static const bool activity = KYTY_GETENV("KYTY_DEBUG_SHADER_ACTIVITY") != nullptr;
	if (activity && shader_hash != 0) {
		static std::mutex                             mutex;
		static std::unordered_map<uint64_t, int64_t> last_used;
		static const auto                            start = std::chrono::steady_clock::now();
		const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		                        std::chrono::steady_clock::now() - start)
		                        .count();
		const std::lock_guard lock(mutex);
		auto [it, inserted] = last_used.try_emplace(shader_hash, now_ms);
		if (inserted || now_ms - it->second > 3000) {
			std::printf("shader-active t=%.2f %s %016" PRIx64 "%s\n", static_cast<double>(now_ms) / 1000.0,
			            kind == DebugShaderKind::Pixel    ? "ps"
			            : kind == DebugShaderKind::Vertex ? "vs"
			                                              : "cs",
			            shader_hash, inserted ? " new" : "");
			std::fflush(stdout);
		}
		it->second = now_ms;
	}
	static const char* live_file = KYTY_GETENV("KYTY_DEBUG_SKIP_SHADERS_LIVE");
	if (live_file != nullptr && shader_hash != 0 && kind != DebugShaderKind::Compute &&
	    LiveSkipListed(live_file, shader_hash)) {
		return true;
	}
	// KYTY_DEBUG_AB=skipshaders: the list below applies only in "on" windows (upper bound of
	// leaving those draws out, judged inside one run).
	static const bool skip_ab = AbSelected("skipshaders");
	if (skip_ab && AbFeatureOff()) {
		return false;
	}
	struct Entry {
		uint64_t value;
		uint64_t mask;
		bool     pixel;
		bool     vertex;
		bool     compute;
	};
	static const std::vector<Entry> entries = [] {
		std::vector<Entry> list;
		for (const char* text = KYTY_GETENV("KYTY_DEBUG_SKIP_SHADERS"); text != nullptr;) {
			const bool only_pixel   = std::strncmp(text, "p:", 2) == 0;
			const bool only_compute = std::strncmp(text, "c:", 2) == 0;
			text += only_pixel || only_compute ? 2 : 0;
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) {
				break;
			}
			Entry entry {value, ~uint64_t {0}, true, true, true};
			if (*end == '/') {
				text       = end + 1;
				entry.mask = std::strtoull(text, &end, 16);
				entry.value &= entry.mask;
				entry.pixel   = !only_compute;
				entry.vertex  = false;
				entry.compute = !only_pixel;
			}
			list.push_back(entry);
			text = *end == ',' ? end + 1 : nullptr;
		}
		return list;
	}();
	if (entries.empty() || shader_hash == 0) {
		return false;
	}
	const bool listed = std::ranges::any_of(entries, [&](const Entry& entry) {
		const bool kind_match = kind == DebugShaderKind::Pixel    ? entry.pixel
		                        : kind == DebugShaderKind::Vertex ? entry.vertex
		                                                          : entry.compute;
		return kind_match && (bisect || entry.mask == ~uint64_t {0}) &&
		       (shader_hash & entry.mask) == entry.value;
	});
	if (!listed) {
		return false;
	}
	static const char* file = KYTY_GETENV("KYTY_DEBUG_SKIP_SHADERS_FILE");
	if (file != nullptr) {
		static std::atomic<int64_t> next_check {0};
		static std::atomic<bool>    present {false};
		const auto                  now = std::chrono::steady_clock::now().time_since_epoch();
		if (now.count() >= next_check.load(std::memory_order_relaxed)) {
			next_check.store((now + std::chrono::milliseconds(500)).count(),
			                 std::memory_order_relaxed);
			std::error_code error;
			present.store(std::filesystem::exists(file, error), std::memory_order_relaxed);
		}
		if (!present.load(std::memory_order_relaxed)) {
			return false;
		}
	}
	static std::mutex            mutex;
	static std::vector<uint64_t> reported;
	const std::lock_guard        lock(mutex);
	if (std::ranges::find(reported, shader_hash) == reported.end()) {
		reported.push_back(shader_hash);
		std::printf("skip-shader: %s %016" PRIx64 "\n",
		            kind == DebugShaderKind::Pixel    ? "ps"
		            : kind == DebugShaderKind::Vertex ? "vs"
		                                              : "cs",
		            shader_hash);
	}
	return true;
}

// Outside rendering: all earlier commands finish, and their writes are visible to everything
// later.
void RecordFullBarrier(vk::CommandBuffer command) noexcept {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 1, &barrier, 0, nullptr,
	                        0, nullptr);
}

static std::atomic<bool> g_ab_off {false};

// KYTY_DEBUG_AB_OFF_FILE=<path>: the selected features are off while that file exists, so a run
// script can alternate them on its own clock (the title Pro: 10 s phases); no KYTY_DEBUG_DRAW_PHASES needed.
static const char* AbOffFile() {
	static const char* path = KYTY_GETENV("KYTY_DEBUG_AB_OFF_FILE");
	return path;
}

static bool AbEnabled() {
	static const bool enabled = (DrawPhaseTimer::Hash() != 0 || AbOffFile() != nullptr) &&
	                            KYTY_GETENV("KYTY_DEBUG_AB") != nullptr;
	return enabled;
}

bool AbSelected(const char* feature) noexcept {
	if (!AbEnabled()) {
		return false;
	}
	const std::string_view features = KYTY_GETENV("KYTY_DEBUG_AB");
	if (features == "1") {
		return std::string_view(feature) == "reuse";
	}
	for (size_t begin = 0; begin <= features.size();) {
		const auto end = std::min(features.find(',', begin), features.size());
		if (features.substr(begin, end - begin) == feature) {
			return true;
		}
		begin = end + 1;
	}
	return false;
}

// KYTY_DEBUG_AB_OFF=1 keeps the selected features off for the whole run instead of alternating
// (to see a scene without them, e.g. for visual bisection).
bool AbFeatureOff() noexcept {
	static const bool forced = KYTY_GETENV("KYTY_DEBUG_AB_OFF") != nullptr;
	if (AbOffFile() != nullptr && !forced) {
		static FileSwitch off(AbOffFile(), "debug-ab");
		return off.Exists();
	}
	return forced || g_ab_off.load(std::memory_order_relaxed);
}

bool FileSwitch::Exists() noexcept {
	if (m_path == nullptr) {
		return false;
	}
	// Looked at by time, not by calls (some features ask once a frame or less, others
	// thousands of times, and a file check costs ~10-30 us on the thread being measured).
	const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                           std::chrono::steady_clock::now().time_since_epoch())
	                           .count();
	if (now_ns >= m_next_check.load(std::memory_order_relaxed)) {
		m_next_check.store(now_ns + 100'000'000, std::memory_order_relaxed);
		std::error_code error;
		const bool      now = std::filesystem::exists(m_path, error);
		if (m_exists.exchange(now, std::memory_order_relaxed) != now) {
			std::printf("%s: %s at %.3f s\n", m_name, now ? "off" : "on", static_cast<double>(now_ns) / 1e9);
			std::fflush(stdout);
		}
	}
	return m_exists.load(std::memory_order_relaxed);
}

namespace {

struct ImageUploadEntry {
	uint64_t size            = 0;
	uint32_t width           = 0;
	uint32_t height          = 0;
	uint32_t guest_format    = 0;
	uint32_t tile_mode       = 0;
	uint64_t count           = 0;
	uint64_t buffer_modified = 0;
	// Staged uploads compared with the image's previous staged upload (RecordImageChunks): bytes
	// compared, bytes in changed chunks, and runs of consecutive changed chunks.
	uint64_t compared        = 0;
	uint64_t changed         = 0;
	uint64_t changed_runs    = 0;
};

// The previous staged upload of an image address: its size and a hash per chunk.
struct ImageChunkHashes {
	uint64_t              size = 0;
	std::vector<uint64_t> hashes;
};

struct UploadStats {
	std::mutex                                      mutex;
	std::array<uint64_t, size_t(UploadSource::Count)> calls {};
	std::array<uint64_t, size_t(UploadSource::Count)> bytes {};
	// Keyed by source << 56 | 4 MiB region.
	std::unordered_map<uint64_t, uint64_t>         regions;
	std::unordered_map<uint64_t, ImageUploadEntry> images;
	// Write faults per 4 KiB page.
	std::unordered_map<uint64_t, uint64_t>         fault_pages;
	// Kept across windows.
	std::unordered_map<uint64_t, ImageChunkHashes> chunk_hashes;
	uint64_t                                       compared = 0;
	uint64_t                                       changed  = 0;
};

UploadStats& GetUploadStats() {
	static UploadStats stats;
	return stats;
}

void PrintUploadStats(double seconds) {
	static constexpr std::array<const char*, size_t(UploadSource::Count)> Names {
	    "buffer", "bda", "stream", "image", "fault", "bda-pass", "bda-sync", "kernel"};
	auto&            stats = GetUploadStats();
	std::scoped_lock lock {stats.mutex};
	std::string      line = "uploads:";
	for (size_t i = 0; i < stats.calls.size(); i++) {
		line += fmt::format(" {}={:.0f}/s,{:.0f}KiB/s", Names[i], stats.calls[i] / seconds,
		                    stats.bytes[i] / 1024.0 / seconds);
	}
	std::vector<std::pair<uint64_t, uint64_t>> regions(stats.regions.begin(), stats.regions.end());
	std::sort(regions.begin(), regions.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	line += " | regions:";
	for (size_t i = 0; i < std::min<size_t>(regions.size(), 12); i++) {
		line += fmt::format(" {}@0x{:x}={:.0f}KiB/s", Names[regions[i].first >> 56u],
		                    (regions[i].first & ((uint64_t {1} << 56u) - 1)) << 22u,
		                    regions[i].second / 1024.0 / seconds);
	}
	// How often each faulting page faulted in the window (about 90 frames at 18 fps): pages with
	// 1, 2-15, 16-63, 64-127, 128-255, 256-511 and 512+ faults.
	static constexpr std::array<uint64_t, 6> Limits {2, 16, 64, 128, 256, 512};
	std::array<uint64_t, Limits.size() + 1>  buckets {};
	for (const auto& [page, count]: stats.fault_pages) {
		size_t bucket = 0;
		while (bucket < Limits.size() && count >= Limits[bucket]) {
			bucket++;
		}
		buckets[bucket]++;
	}
	line += fmt::format(" | fault pages={} (x1={} x2-15={} x16-63={} x64-127={} x128-255={} "
	                    "x256-511={} x512+={})",
	                    stats.fault_pages.size(), buckets[0], buckets[1], buckets[2], buckets[3],
	                    buckets[4], buckets[5], buckets[6]);
	line += fmt::format(" | staged changed={:.0f}KiB/s of compared={:.0f}KiB/s",
	                    stats.changed / 1024.0 / seconds, stats.compared / 1024.0 / seconds);
	std::printf("%s\n", line.c_str());
	std::vector<std::pair<uint64_t, ImageUploadEntry>> images(stats.images.begin(),
	                                                          stats.images.end());
	std::sort(images.begin(), images.end(), [](const auto& a, const auto& b) {
		return a.second.count * a.second.size > b.second.count * b.second.size;
	});
	for (size_t i = 0; i < std::min<size_t>(images.size(), 10); i++) {
		const auto& [address, image] = images[i];
		std::printf("uploads: image 0x%" PRIx64 " size=0x%" PRIx64 " %ux%u fmt=%u tile=%u "
		            "uploads/s=%.1f KiB/s=%.0f buffer-modified=%" PRIu64 " changed=%.1f%%"
		            " runs/upload=%.1f\n",
		            address, image.size, image.width, image.height, image.guest_format,
		            image.tile_mode, image.count / seconds,
		            image.count * image.size / 1024.0 / seconds, image.buffer_modified,
		            image.compared != 0 ? 100.0 * image.changed / image.compared : 0.0,
		            image.compared != 0 ? static_cast<double>(image.changed_runs) * image.size /
		                                      image.compared
		                                : 0.0);
	}
	stats.compared = 0;
	stats.changed  = 0;
	stats.calls.fill(0);
	stats.bytes.fill(0);
	stats.regions.clear();
	stats.images.clear();
	stats.fault_pages.clear();
}

} // namespace

namespace {
struct StampTrace {
	std::vector<uint64_t>                  surfaces;
	std::vector<std::pair<char, uint32_t>> events; // Event and surface index.
	std::string                            last_shape;
	uint64_t                               frame = 0;
};
StampTrace& GetStampTrace() {
	static StampTrace trace;
	return trace;
}
int StampSurface(uint64_t address, bool add) {
	auto& surfaces = GetStampTrace().surfaces;
	for (size_t i = 0; i < surfaces.size(); i++) {
		if (surfaces[i] == address) {
			return static_cast<int>(i);
		}
	}
	if (!add || address == 0 || surfaces.size() >= 9) {
		return -1;
	}
	surfaces.push_back(address);
	std::printf("stamp-trace: surface %zu = 0x%016" PRIx64 "\n", surfaces.size() - 1, address);
	return static_cast<int>(surfaces.size() - 1);
}
} // namespace

bool StampTraceEnabled() noexcept {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_STAMP_TRACE") != nullptr;
	return enabled;
}

void StampTraceDraw(uint64_t pixel_hash, uint64_t target_address) noexcept {
	const bool stamp = pixel_hash == 0x63722404bb8c7b26ull;
	const int  index = StampSurface(target_address, stamp);
	if (index >= 0) {
		GetStampTrace().events.emplace_back(stamp ? 'D' : 'O', static_cast<uint32_t>(index));
	}
}

void StampTraceEvent(char event, uint64_t address) noexcept {
	if (const int index = StampSurface(address, false); index >= 0) {
		GetStampTrace().events.emplace_back(event, static_cast<uint32_t>(index));
	}
}

void StampTraceDispatch(uint64_t compute_hash) noexcept {
	if (compute_hash == 0xfb948a435a4e295eull) {
		GetStampTrace().events.emplace_back('S', 0u);
	}
}

void StampTraceFrame() noexcept {
	auto& trace = GetStampTrace();
	trace.frame++;
	std::string shape;
	std::string text;
	for (size_t i = 0; i < trace.events.size();) {
		size_t run = 1;
		while (i + run < trace.events.size() && trace.events[i + run] == trace.events[i]) {
			run++;
		}
		const auto [event, surface] = trace.events[i];
		shape += event;
		text += event;
		if (event != 'S') {
			shape += static_cast<char>('0' + surface);
			text += static_cast<char>('0' + surface);
		}
		if (run > 1) {
			text += "x" + std::to_string(run);
		}
		text += ' ';
		i += run;
	}
	trace.events.clear();
	if (shape != trace.last_shape) {
		std::printf("stamp-trace: frame=%" PRIu64 " t=%.2f : %s\n", trace.frame,
		            std::chrono::duration<double>(
		                std::chrono::steady_clock::now().time_since_epoch())
		                .count(),
		            text.empty() ? "-" : text.c_str());
		std::fflush(stdout);
		trace.last_shape = shape;
	}
}

bool UploadStatsEnabled() noexcept {
	static const bool enabled =
	    DrawPhaseTimer::Hash() != 0 && KYTY_GETENV("KYTY_DEBUG_UPLOADS") != nullptr;
	return enabled;
}

void RecordUpload(UploadSource source, uint64_t address, uint64_t bytes) noexcept {
	if (!UploadStatsEnabled()) {
		return;
	}
	auto&            stats = GetUploadStats();
	const auto       index = static_cast<uint64_t>(source);
	std::scoped_lock lock {stats.mutex};
	stats.calls[index]++;
	stats.bytes[index] += bytes;
	if (bytes != 0) {
		stats.regions[index << 56u | address >> 22u] += bytes;
	}
	if (source == UploadSource::Fault) {
		stats.fault_pages[address >> 12u]++;
	}
}

void RecordImageUpload(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                       uint32_t guest_format, uint32_t tile_mode, bool buffer_modified) noexcept {
	if (!UploadStatsEnabled()) {
		return;
	}
	auto&            stats = GetUploadStats();
	std::scoped_lock lock {stats.mutex};
	auto&            entry = stats.images[address];
	entry.size             = size;
	entry.width            = width;
	entry.height           = height;
	entry.guest_format     = guest_format;
	entry.tile_mode        = tile_mode;
	entry.count++;
	entry.buffer_modified += buffer_modified ? 1 : 0;
}

void RecordUploadPages(uint64_t address, uint64_t size) noexcept {
	static const bool enabled = KYTY_GETENV("KYTY_DEBUG_UPLOAD_CHANGES") != nullptr;
	if (!enabled || size == 0) {
		return;
	}
	struct State {
		std::mutex                             mutex;
		std::unordered_map<uint64_t, uint64_t> hashes; // page -> hash of its last upload
		uint64_t                               pages     = 0;
		uint64_t                               unchanged = 0;
		uint64_t                               first     = 0;
		std::chrono::steady_clock::time_point  window    = std::chrono::steady_clock::now();
	};
	static State state;
	std::array<uint8_t, 4096> page {};
	std::scoped_lock          lock {state.mutex};
	for (uint64_t at = address & ~uint64_t {0xfff}; at < address + size; at += 0x1000) {
		if (!Libs::LibKernel::Memory::TryReadBacking(at, page.data(), page.size())) {
			continue;
		}
		const auto hash     = XXH3_64bits(page.data(), page.size());
		auto [it, inserted] = state.hashes.try_emplace(at, hash);
		state.pages++;
		if (inserted) {
			state.first++;
		} else if (it->second == hash) {
			state.unchanged++;
		} else {
			it->second = hash;
		}
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - state.window < std::chrono::seconds(5)) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - state.window).count();
	const auto   share   = [&](uint64_t part) {
        return state.pages != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(state.pages)
                                    : 0.0;
	};
	std::printf("upload-changes: pages/s=%.0f (%.1f MB/s) unchanged=%.1f%% first-time=%.1f%% "
	            "distinct pages=%zu\n",
	            static_cast<double>(state.pages) / seconds,
	            static_cast<double>(state.pages) * 4096.0 / 1e6 / seconds, share(state.unchanged),
	            share(state.first), state.hashes.size());
	std::fflush(stdout);
	state.pages     = 0;
	state.unchanged = 0;
	state.first     = 0;
	state.window    = now;
}

void RecordImageChunks(uint64_t address, uint64_t size) noexcept {
	if (!UploadStatsEnabled() || size == 0) {
		return;
	}
	static constexpr uint64_t                ChunkSize = 64 * 1024;
	static thread_local std::vector<uint8_t> bytes;
	bytes.resize(size);
	if (!Libs::LibKernel::Memory::TryReadBacking(address, bytes.data(), size)) {
		return;
	}
	const auto            chunks = (size + ChunkSize - 1) / ChunkSize;
	std::vector<uint64_t> hashes(chunks);
	for (uint64_t i = 0; i < chunks; i++) {
		const auto begin = i * ChunkSize;
		hashes[i]        = XXH3_64bits(bytes.data() + begin, std::min(ChunkSize, size - begin));
	}
	auto&            stats = GetUploadStats();
	std::scoped_lock lock {stats.mutex};
	auto&            previous = stats.chunk_hashes[address];
	if (previous.size == size) {
		uint64_t changed = 0;
		uint64_t runs    = 0;
		for (uint64_t i = 0; i < chunks; i++) {
			if (hashes[i] != previous.hashes[i]) {
				changed += std::min(ChunkSize, size - i * ChunkSize);
				runs += (i == 0 || hashes[i - 1] == previous.hashes[i - 1]) ? 1 : 0;
			}
		}
		auto& entry = stats.images[address];
		entry.compared += size;
		entry.changed += changed;
		entry.changed_runs += runs;
		stats.compared += size;
		stats.changed += changed;
	}
	previous.size   = size;
	previous.hashes = std::move(hashes);
}

bool BindStatsEnabled() noexcept {
	static const bool enabled =
	    DrawPhaseTimer::Hash() != 0 && KYTY_GETENV("KYTY_DEBUG_BIND_STATS") != nullptr;
	return enabled;
}

// The image-groups and depth-only lines (see BindStats), per second or as shares.
static void PrintBindStats(double seconds, double to_us) {
	auto&          stats = g_bind_stats;
	const auto     share = [](uint64_t part, uint64_t whole) {
		return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
	};
	static constexpr std::array<const char*, BindStats::GroupOutcomeCount> GroupNames {
	    "no-images", "kept",    "disabled", "not-recorded", "other-program", "other-images",
	    "image-set", "tick",    "gc-tick",  "other-image",  "dirty",         "color-clear"};
	static constexpr std::array<const char*, BindStats::RefusalCount> RefusalNames {
	    "none", "type", "lookup", "view", "dynamic-mip"};
	uint64_t stages = 0;
	for (const auto count: stats.group) {
		stages += count;
	}
	const auto with_images = stages - stats.group[BindStats::NoImages];
	std::string line = fmt::format("image-groups: stages/s={:.0f} with-images/s={:.0f} |",
	                               stages / seconds, with_images / seconds);
	for (uint32_t i = 0; i < BindStats::GroupOutcomeCount; i++) {
		line += fmt::format(" {}%={:.1f}", GroupNames[i],
		                    share(stats.group[i], i == BindStats::NoImages ? stages : with_images));
	}
	line += fmt::format(" kept-lost%={:.1f} | refused%:", share(stats.kept_lost,
	                                                            stats.group[BindStats::Kept]));
	uint64_t refused = 0;
	for (const auto count: stats.refused) {
		refused += count;
	}
	for (uint32_t i = 1; i < BindStats::RefusalCount; i++) {
		line += fmt::format(" {}={:.1f}", RefusalNames[i], share(stats.refused[i], refused));
	}
	const auto per = [&](uint64_t ticks, uint64_t count) {
		return count != 0 ? static_cast<double>(ticks) * to_us / static_cast<double>(count) : 0.0;
	};
	line += fmt::format(" | us/stage prepare kept={:.3f} other={:.3f} rebind kept={:.3f} other={:.3f}",
	                    per(stats.prepare_ticks[0], stats.prepare_count[0]),
	                    per(stats.prepare_ticks[1], stats.prepare_count[1]),
	                    per(stats.rebind_ticks[0], stats.rebind_count[0]),
	                    per(stats.rebind_ticks[1], stats.rebind_count[1]));
	std::printf("%s\n", line.c_str());
	uint64_t breaks = 0;
	for (const auto count: stats.run_breaks) {
		breaks += count;
	}
	std::string depth = fmt::format(
	    "depth-only: draws/s={:.0f} depth%={:.1f} | of depth: no-ps%={:.1f} ps-images%={:.1f} "
	    "vs-images%={:.1f} | run draws% 1={:.1f} 2-3={:.1f} 4-7={:.1f} 8-15={:.1f} 16-63={:.1f} "
	    "64+={:.1f} | breaks% pipeline={:.1f} vs={:.1f} target={:.1f} color-draw={:.1f}",
	    stats.draws / seconds, share(stats.depth_draws, stats.draws),
	    share(stats.depth_no_ps, stats.depth_draws), share(stats.depth_ps_images, stats.depth_draws),
	    share(stats.depth_vs_images, stats.depth_draws), share(stats.run_draws[0], stats.depth_draws),
	    share(stats.run_draws[1], stats.depth_draws), share(stats.run_draws[2], stats.depth_draws),
	    share(stats.run_draws[3], stats.depth_draws), share(stats.run_draws[4], stats.depth_draws),
	    share(stats.run_draws[5], stats.depth_draws), share(stats.run_breaks[0], breaks),
	    share(stats.run_breaks[1], breaks), share(stats.run_breaks[2], breaks),
	    share(stats.run_breaks[3], breaks));
	std::printf("%s\n", depth.c_str());
	std::printf("set-repeats: set0 payloads/s=%.0f repeat%%=%.1f set1 payloads/s=%.0f repeat%%=%.1f\n",
	            stats.set_payloads[0] / seconds, share(stats.set_repeats[0], stats.set_payloads[0]),
	            stats.set_payloads[1] / seconds, share(stats.set_repeats[1], stats.set_payloads[1]));
	// Keep the open run.
	const auto run_length   = stats.run_length;
	const auto run_pipeline = stats.run_pipeline;
	const auto run_program  = stats.run_program;
	const auto run_target   = stats.run_target;
	stats                   = {};
	stats.run_length        = run_length;
	stats.run_pipeline      = run_pipeline;
	stats.run_program       = run_program;
	stats.run_target        = run_target;
}

void DrawPhaseTimer::End(uint64_t pixel_hash) {
	if (!active) {
		return;
	}
	Mark(Tail);
	active = false;
	static constexpr std::array<const char*, Count> Names {
	    "setup",    "vs-params", "ps-params",  "ps-program", "vs-program", "targets",  "stage-tex",
	    "stage-smp", "stage-bind", "find-buf", "rebind-img", "buf-views",  "gfx-bind", "rt-acquire",
	    "pipeline", "records",   "commit",     "record",     "tail"};
	static constexpr std::array<const char*, ProbeCount> ProbeNames {
	    "rt-image", "tex-image", "tex-describe", "buf-written", "buf-read", "buf-invalidate",
	    "upload",   "find-finish", "stream-copy", "pending-ops", "bda",       "spec-reads",
	    "refresh",  "prog-match",  "inputs-check", "upload-copy", "commit-scan", "commit-transit",
	    "commit-writes", "commit-payload"};
	static std::array<uint64_t, Count>      totals {};
	static std::array<uint64_t, ProbeCount> probe_totals {};
	static uint64_t draws        = 0;
	static uint64_t other_draws  = 0;
	static uint64_t other_ticks  = 0;
	static auto     window_start = std::chrono::steady_clock::now();
	static uint64_t window_tsc   = Now();
	static uint64_t window_gpu   = g_gpu_busy_ns.load(std::memory_order_relaxed);
	static std::array<uint64_t, 8> window_allocations {};
	static std::array<uint64_t, 5> window_repeats {};
	uint64_t        sum          = 0;
	for (const auto ticks: current) {
		sum += ticks;
	}
	// DRAW_INDEX_AUTO draws, also on their own (see Begin).
	static std::array<uint64_t, Count> auto_totals {};
	static uint64_t                    auto_draws = 0;
	// Depth-only draws (see SetDepthOnly), also on their own.
	static std::array<uint64_t, Count>      depth_totals {};
	static std::array<uint64_t, ProbeCount> depth_probes {};
	static uint64_t                         depth_draws = 0;
	// Of those, the draws without a pixel shader (pixel_hash 0: shadow and prepass depth).
	static std::array<uint64_t, Count>      nops_totals {};
	static std::array<uint64_t, ProbeCount> nops_probes {};
	static uint64_t                         nops_draws = 0;
	if (pixel_hash == Hash() || Hash() == AllDraws) {
		for (uint32_t i = 0; i < Count; i++) {
			totals[i] += current[i];
		}
		for (uint32_t i = 0; i < ProbeCount; i++) {
			probe_totals[i] += probes[i];
		}
		draws++;
		if (auto_kind) {
			for (uint32_t i = 0; i < Count; i++) {
				auto_totals[i] += current[i];
			}
			auto_draws++;
		}
		if (depth_kind) {
			for (uint32_t i = 0; i < Count; i++) {
				depth_totals[i] += current[i];
			}
			for (uint32_t i = 0; i < ProbeCount; i++) {
				depth_probes[i] += probes[i];
			}
			depth_draws++;
			if (pixel_hash == 0) {
				for (uint32_t i = 0; i < Count; i++) {
					nops_totals[i] += current[i];
				}
				for (uint32_t i = 0; i < ProbeCount; i++) {
					nops_probes[i] += probes[i];
				}
				nops_draws++;
			}
		}
	} else {
		other_draws++;
		other_ticks += sum;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - window_start < std::chrono::seconds(5)) {
		return;
	}
	// The window calibrates the time stamp counter against the steady clock.
	const auto   tsc     = Now();
	const double seconds = std::chrono::duration<double>(now - window_start).count();
	const double to_us   = seconds * 1e6 / static_cast<double>(tsc - window_tsc);
	const auto   gpu_ns  = g_gpu_busy_ns.load(std::memory_order_relaxed);
	const std::array<uint64_t, 8> allocations {
	    g_allocation_counters.buffers_created.load(std::memory_order_relaxed),
	    g_allocation_counters.buffers_destroyed.load(std::memory_order_relaxed),
	    g_allocation_counters.images_created.load(std::memory_order_relaxed),
	    g_allocation_counters.images_destroyed.load(std::memory_order_relaxed),
	    g_allocation_counters.game_buffers_created.load(std::memory_order_relaxed),
	    g_allocation_counters.game_buffers_joined.load(std::memory_order_relaxed),
	    g_allocation_counters.game_buffers_collected.load(std::memory_order_relaxed),
	    g_allocation_counters.buffer_create_ns.load(std::memory_order_relaxed)};
	uint64_t     all     = 0;
	for (const auto ticks: totals) {
		all += ticks;
	}
	std::string line = fmt::format("draw-phases: {:.1f}s draws/s={:.0f} us/draw={:.2f} ms/s={:.1f} "
	                               "other draws/s={:.0f} other ms/s={:.1f} gpu-ms/s={:.1f}{} |",
	                               seconds, draws / seconds, draws != 0 ? all * to_us / draws : 0.0,
	                               all * to_us / 1000.0 / seconds, other_draws / seconds,
	                               other_ticks * to_us / 1000.0 / seconds,
	                               static_cast<double>(gpu_ns - window_gpu) / 1e6 / seconds,
	                               !AbEnabled() ? "" : (AbFeatureOff() ? " ab=off" : " ab=on"));
	for (uint32_t i = 0; i < Count; i++) {
		line += fmt::format(" {}={:.2f}", Names[i], draws != 0 ? totals[i] * to_us / draws : 0.0);
	}
	const std::array<uint64_t, 5> repeats {
	    g_binding_repeats.stages.load(std::memory_order_relaxed),
	    g_binding_repeats.same_program.load(std::memory_order_relaxed),
	    g_binding_repeats.same_images.load(std::memory_order_relaxed),
	    g_binding_repeats.same_buffers.load(std::memory_order_relaxed),
	    g_binding_repeats.same_all.load(std::memory_order_relaxed)};
	const auto stages = static_cast<double>(std::max<uint64_t>(repeats[0] - window_repeats[0], 1));
	line += fmt::format(" rep-prog%={:.0f} rep-img%={:.0f} rep-buf%={:.0f} rep-all%={:.0f}",
	                    100.0 * static_cast<double>(repeats[1] - window_repeats[1]) / stages,
	                    100.0 * static_cast<double>(repeats[2] - window_repeats[2]) / stages,
	                    100.0 * static_cast<double>(repeats[3] - window_repeats[3]) / stages,
	                    100.0 * static_cast<double>(repeats[4] - window_repeats[4]) / stages);
	window_repeats = repeats;
	const auto per_second = [&](size_t i) {
		return static_cast<double>(allocations[i] - window_allocations[i]) / seconds;
	};
	line += fmt::format(" | buf+/s={:.0f} buf-/s={:.0f} img+/s={:.0f} img-/s={:.0f} game-buf+/s={:.0f}"
	                    " game-buf-join/s={:.0f} game-buf-gc/s={:.0f} buf-create-ms/s={:.2f}",
	                    per_second(0), per_second(1), per_second(2), per_second(3), per_second(4),
	                    per_second(5), per_second(6), per_second(7) / 1e6);
	line += " | probes:";
	for (uint32_t i = 0; i < ProbeCount; i++) {
		line += fmt::format(" {}={:.2f}", ProbeNames[i],
		                    draws != 0 ? probe_totals[i] * to_us / draws : 0.0);
	}
	std::printf("%s\n", line.c_str());
	if (auto_draws != 0) {
		uint64_t auto_all = 0;
		for (const auto ticks: auto_totals) {
			auto_all += ticks;
		}
		std::string auto_line = fmt::format("draw-phases auto: draws/s={:.0f} us/draw={:.2f} |",
		                                    auto_draws / seconds, auto_all * to_us / auto_draws);
		for (uint32_t i = 0; i < Count; i++) {
			auto_line += fmt::format(" {}={:.2f}", Names[i], auto_totals[i] * to_us / auto_draws);
		}
		std::printf("%s\n", auto_line.c_str());
		auto_totals.fill(0);
		auto_draws = 0;
	}
	if (depth_draws != 0) {
		uint64_t depth_all = 0;
		for (const auto ticks: depth_totals) {
			depth_all += ticks;
		}
		std::string depth_line = fmt::format("draw-phases depth: draws/s={:.0f} us/draw={:.2f} |",
		                                     depth_draws / seconds, depth_all * to_us / depth_draws);
		for (uint32_t i = 0; i < Count; i++) {
			depth_line += fmt::format(" {}={:.2f}", Names[i], depth_totals[i] * to_us / depth_draws);
		}
		depth_line += " | probes:";
		for (uint32_t i = 0; i < ProbeCount; i++) {
			depth_line +=
			    fmt::format(" {}={:.2f}", ProbeNames[i], depth_probes[i] * to_us / depth_draws);
		}
		std::printf("%s\n", depth_line.c_str());
		depth_totals.fill(0);
		depth_probes.fill(0);
		depth_draws = 0;
	}
	if (nops_draws != 0) {
		uint64_t nops_all = 0;
		for (const auto ticks: nops_totals) {
			nops_all += ticks;
		}
		std::string nops_line = fmt::format("draw-phases depth-nops: draws/s={:.0f} us/draw={:.2f} |",
		                                    nops_draws / seconds, nops_all * to_us / nops_draws);
		for (uint32_t i = 0; i < Count; i++) {
			nops_line += fmt::format(" {}={:.2f}", Names[i], nops_totals[i] * to_us / nops_draws);
		}
		nops_line += " | probes:";
		for (uint32_t i = 0; i < ProbeCount; i++) {
			nops_line += fmt::format(" {}={:.2f}", ProbeNames[i], nops_probes[i] * to_us / nops_draws);
		}
		std::printf("%s\n", nops_line.c_str());
		nops_totals.fill(0);
		nops_probes.fill(0);
		nops_draws = 0;
	}
	if (BindStatsEnabled()) {
		PrintBindStats(seconds, to_us);
	}
	if (UploadStatsEnabled()) {
		PrintUploadStats(seconds);
	}
	{
		// The PM4 handlers that took the most time, as opcode:packets/s:ms/s.
		auto&                                          ops = g_pm4_ops;
		std::array<std::pair<uint64_t, uint32_t>, 256> ranked {};
		uint64_t                                       handled = 0;
		for (uint32_t op = 0; op < 256; op++) {
			ranked[op] = {ops.ticks[op], op};
			handled += ops.ticks[op];
		}
		std::ranges::sort(ranked, std::greater {});
		std::string pm4 = fmt::format("pm4-ops: handlers ms/s={:.1f} between ms/s={:.1f} |",
		                              handled * to_us / 1000.0 / seconds,
		                              ops.between * to_us / 1000.0 / seconds);
		for (uint32_t i = 0; i < 12 && ranked[i].first != 0; i++) {
			const auto op = ranked[i].second;
			pm4 += fmt::format(" {:02x}:{:.0f}:{:.2f}", op, ops.counts[op] / seconds,
			                   ops.ticks[op] * to_us / 1000.0 / seconds);
		}
		// NOP packets (0x10) by their custom code R (0 = plain NOP or user data marker).
		std::array<std::pair<uint64_t, uint32_t>, 64> nops {};
		for (uint32_t r = 0; r < 64; r++) {
			nops[r] = {ops.nop_ticks[r], r};
		}
		std::ranges::sort(nops, std::greater {});
		pm4 += " | nop-r:";
		for (uint32_t i = 0; i < 5 && nops[i].first != 0; i++) {
			const auto r = nops[i].second;
			pm4 += fmt::format(" {:02x}:{:.0f}:{:.2f}", r, ops.nop_counts[r] / seconds,
			                   ops.nop_ticks[r] * to_us / 1000.0 / seconds);
		}
		// Draw packets by how their handler ended, as outcome:packets/s:ms/s.
		static constexpr std::array<const char*, Pm4OpTimer::OutcomeCount> OutcomeNames {
		    "drawn", "empty", "metadata", "depth-copy", "resolve", "no-shader", "not-prepared",
		    "rect-skip"};
		for (uint32_t kind = 0; kind < 2; kind++) {
			pm4 += kind == 0 ? " | indexed:" : " | auto:";
			for (uint32_t outcome = 0; outcome < Pm4OpTimer::OutcomeCount; outcome++) {
				if (ops.outcome_counts[kind][outcome] != 0) {
					pm4 += fmt::format(" {}:{:.0f}:{:.2f}", OutcomeNames[outcome],
					                   ops.outcome_counts[kind][outcome] / seconds,
					                   ops.outcome_ticks[kind][outcome] * to_us / 1000.0 / seconds);
				}
			}
		}
		std::printf("%s\n", pm4.c_str());
		ops = {};
	}
	if (AbEnabled()) {
		g_ab_off.store(!AbFeatureOff(), std::memory_order_relaxed);
	}
	totals.fill(0);
	probe_totals.fill(0);
	draws        = 0;
	other_draws  = 0;
	other_ticks  = 0;
	window_start = now;
	window_tsc   = tsc;
	window_gpu   = gpu_ns;
	window_allocations = allocations;
}

void LogDrawPhase(const char* draw_name, const char* phase) {
	if (graphics_debug_dump_enabled()) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
			LOGF("DrawPhase: %s %s\n", draw_name, phase);
		}
	}
}

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id) {
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: %s VS=%" PRIu64 " PS=%" PRIu64 "\n", phase, vertex_program_id,
		     pixel_program_id);
	}
}

static void VpPrint(const char* func, const HW::ScreenViewport& vp,
                    const HW::ScanModeControl& smc) {
	LOGF("%s\n", func);

	LOGF("\t msaa_enable                    = %s\n"
	     "\t vport_scissor_enable           = %s\n"
	     "\t line_stipple_enable            = %s\n"
	     "\t vp[0].zmin                     = %f\n"
	     "\t vp[0].zmax                     = %f\n"
	     "\t vp[0].xscale                   = %f\n"
	     "\t vp[0].xoffset                  = %f\n"
	     "\t vp[0].yscale                   = %f\n"
	     "\t vp[0].yoffset                  = %f\n"
	     "\t vp[0].zscale                   = %f\n"
	     "\t vp[0].zoffset                  = %f\n"
	     "\t vp[0].viewport_scissor_left    = %d\n"
	     "\t vp[0].viewport_scissor_top     = %d\n"
	     "\t vp[0].viewport_scissor_right   = %d\n"
	     "\t vp[0].viewport_scissor_bottom  = %d\n"
	     "\t transform_control              = 0x%08" PRIx32 "\n"
	     "\t screen_scissor_left            = %d\n"
	     "\t screen_scissor_top             = %d\n"
	     "\t screen_scissor_right           = %d\n"
	     "\t screen_scissor_bottom          = %d\n"
	     "\t window_scissor_left            = %d\n"
	     "\t window_scissor_top             = %d\n"
	     "\t window_scissor_right           = %d\n"
	     "\t window_scissor_bottom          = %d\n"
	     "\t generic_scissor_left           = %d\n"
	     "\t generic_scissor_top            = %d\n"
	     "\t generic_scissor_right          = %d\n"
	     "\t generic_scissor_bottom         = %d\n"
	     "\t window_offset_x                = %d\n"
	     "\t window_offset_y                = %d\n"
	     "\t hw_offset_x                    = %u\n"
	     "\t hw_offset_y                    = %u\n"
	     "\t guard_band_horz_clip           = %f\n"
	     "\t guard_band_vert_clip           = %f\n"
	     "\t guard_band_horz_discard        = %f\n"
	     "\t guard_band_vert_discard        = %f\n"
	     "\t clip_rect_rule                 = 0x%04" PRIx16 "\n"
	     "\t clip_rect_0                    = (%d,%d)-(%d,%d), window_offset = %s\n"
	     "\t window_scissor_window_offset_enable                = %s\n"
	     "\t generic_scissor_window_offset_enable               = %s\n",
	     smc.msaa_enable ? "true" : "false", smc.vport_scissor_enable ? "true" : "false",
	     smc.line_stipple_enable ? "true" : "false", vp.viewports[0].zmin, vp.viewports[0].zmax,
	     vp.viewports[0].xscale, vp.viewports[0].xoffset, vp.viewports[0].yscale,
	     vp.viewports[0].yoffset, vp.viewports[0].zscale, vp.viewports[0].zoffset,
	     vp.viewports[0].viewport_scissor_left, vp.viewports[0].viewport_scissor_top,
	     vp.viewports[0].viewport_scissor_right, vp.viewports[0].viewport_scissor_bottom,
	     vp.transform_control, vp.screen_scissor_left, vp.screen_scissor_top,
	     vp.screen_scissor_right, vp.screen_scissor_bottom, vp.window_scissor_left,
	     vp.window_scissor_top, vp.window_scissor_right, vp.window_scissor_bottom,
	     vp.generic_scissor_left, vp.generic_scissor_top, vp.generic_scissor_right,
	     vp.generic_scissor_bottom, vp.window_offset_x, vp.window_offset_y, vp.hw_offset_x,
	     vp.hw_offset_y, vp.guard_band_horz_clip, vp.guard_band_vert_clip,
	     vp.guard_band_horz_discard, vp.guard_band_vert_discard, vp.clip_rect_rule,
	     vp.clip_rect_left[0], vp.clip_rect_top[0], vp.clip_rect_right[0], vp.clip_rect_bottom[0],
	     vp.clip_rect_window_offset_enable[0] ? "true" : "false",
	     vp.window_scissor_window_offset_enable ? "true" : "false",
	     vp.generic_scissor_window_offset_enable ? "true" : "false");
	LOGF("\t viewports[0].viewport_scissor_window_offset_enable = %s\n",
	     vp.viewports[0].viewport_scissor_window_offset_enable ? "true" : "false");
}

static void VpCheck(const HW::ScreenViewport& vp, const HW::ScanModeControl& smc) {

	if (smc.msaa_enable) {

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("\t warning: unsupported PS5 MSAA raster controls use native Vulkan defaults\n");
		}
	}
	// EXIT_NOT_IMPLEMENTED(smc.vport_scissor_enable);
	EXIT_NOT_IMPLEMENTED(smc.line_stipple_enable);

	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].xscale != 960.000000);
	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].xoffset != 960.000000);
	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].yscale != -540.000000);
	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].yoffset != 540.000000);
	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].zscale != 0.500000);
	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].zoffset != 0.500000);
	if (vp.transform_control != 1087) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("\t warning: non-default viewport transform control = 0x%08" PRIx32
			     ", applying enabled scale/offset bits\n",
			     vp.transform_control);
		}
	}
	// EXIT_NOT_IMPLEMENTED(vp.hw_offset_x != 60);
	// EXIT_NOT_IMPLEMENTED(vp.hw_offset_y != 32);
	// EXIT_NOT_IMPLEMENTED(fabsf(vp.guard_band_horz_clip - 33.133327f) > 0.001f);
	// EXIT_NOT_IMPLEMENTED(fabsf(vp.guard_band_vert_clip - 59.629623f) > 0.001f);

	if (vp.guard_band_horz_discard != 1.0f || vp.guard_band_vert_discard != 1.0f) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("\t warning: unsupported PS5 guard band discard = %f, %f, continuing\n",
			     vp.guard_band_horz_discard, vp.guard_band_vert_discard);
		}
	}

	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].viewport_scissor_left != 0);
	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].viewport_scissor_top != 0);
	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].viewport_scissor_right != 0);
	// EXIT_NOT_IMPLEMENTED(vp.viewports[0].viewport_scissor_bottom != 0);
	// EXIT_NOT_IMPLEMENTED(viewport_scissor &&
	// vp.viewports[0].viewport_scissor_window_offset_enable != true);
}

static bool ScissorRectValid(const ScissorRect& r) {
	return r.right > r.left && r.bottom > r.top;
}

static ScissorRect ScissorRectOffset(ScissorRect r, int x, int y) {
	r.left += x;
	r.right += x;
	r.top += y;
	r.bottom += y;
	return r;
}

static ScissorRect ScissorRectIntersect(const ScissorRect& a, const ScissorRect& b) {
	return {std::max(a.left, b.left), std::max(a.top, b.top),
	        std::min(a.right, b.right), std::min(a.bottom, b.bottom)};
}

static ScissorRect ScissorRectClamp(ScissorRect r, uint32_t width, uint32_t height) {
	int max_right  = static_cast<int>(width);
	int max_bottom = static_cast<int>(height);

	r.left   = std::clamp(r.left, 0, max_right);
	r.right  = std::clamp(r.right, 0, max_right);
	r.top    = std::clamp(r.top, 0, max_bottom);
	r.bottom = std::clamp(r.bottom, 0, max_bottom);

	if (!ScissorRectValid(r)) {
		r.right  = r.left;
		r.bottom = r.top;
	}

	return r;
}

static constexpr std::array<uint16_t, 16> MakeScissorIntersectionRules() {
	std::array<uint16_t, 16> rules {};
	for (uint32_t candidate = 0; candidate < rules.size(); candidate++) {
		for (uint32_t combination = 0; combination < 16; combination++) {
			if ((combination & candidate) == candidate) {
				rules[candidate] |= static_cast<uint16_t>(1u << combination);
			}
		}
	}
	return rules;
}

static bool ScissorClipRuleToIntersectionMask(uint16_t rule, uint8_t* mask) {
	EXIT_IF(mask == nullptr);

	static constexpr auto rules = MakeScissorIntersectionRules();
	for (uint32_t candidate = 0; candidate < rules.size(); candidate++) {
		if (rules[candidate] == rule) {
			*mask = static_cast<uint8_t>(candidate);
			return true;
		}
	}

	return false;
}

ScissorRect calc_final_scissor(const HW::ScreenViewport& vp, const HW::ScanModeControl& smc,
                               vk::Extent2D extent, uint32_t viewport_index) {
	EXIT_IF(viewport_index >= std::size(vp.viewports));
	ScissorRect final {vp.screen_scissor_left, vp.screen_scissor_top, vp.screen_scissor_right,
	                   vp.screen_scissor_bottom};
	const auto intersect = [&](ScissorRect rect, bool window_offset) {
		if (window_offset) {
			rect = ScissorRectOffset(rect, vp.window_offset_x, vp.window_offset_y);
		}
		final = ScissorRectIntersect(final, rect);
	};
	intersect({vp.window_scissor_left, vp.window_scissor_top, vp.window_scissor_right,
	           vp.window_scissor_bottom}, vp.window_scissor_window_offset_enable);
	intersect({vp.generic_scissor_left, vp.generic_scissor_top, vp.generic_scissor_right,
	           vp.generic_scissor_bottom}, vp.generic_scissor_window_offset_enable);

	const auto& viewport = vp.viewports[viewport_index];
	if (smc.vport_scissor_enable) {
		intersect({viewport.viewport_scissor_left, viewport.viewport_scissor_top,
		           viewport.viewport_scissor_right, viewport.viewport_scissor_bottom},
		          viewport.viewport_scissor_window_offset_enable);
	}

	if (vp.clip_rect_rule == 0) {
		final = {0, 0, 0, 0};
	} else if (vp.clip_rect_rule != 0xffffu) {
		uint8_t clip_rect_mask = 0;
		if (ScissorClipRuleToIntersectionMask(vp.clip_rect_rule, &clip_rect_mask)) {
			for (uint32_t i = 0; i < 4; i++) {
				if ((clip_rect_mask & (1u << i)) == 0) {
					continue;
				}

				intersect({vp.clip_rect_left[i], vp.clip_rect_top[i], vp.clip_rect_right[i],
				           vp.clip_rect_bottom[i]}, vp.clip_rect_window_offset_enable[i]);
			}
		} else {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
				LOGF("unsupported clip-rect rule 0x%04" PRIx16 ", leaving scissor unchanged\n",
				     vp.clip_rect_rule);
			}
		}
	}

	return ScissorRectClamp(final, extent.width, extent.height);
}

void hw_check(const CommandBuffer& buffer) {
	const auto& hw      = buffer.GetRegisters();
	const auto  rt_slot = render_target_first_bound_slot(buffer);
	const auto& rt      = hw.GetRenderTarget(rt_slot);
	const auto& bclr    = hw.GetBlendColor();
	const auto& vp      = hw.GetScreenViewport();
	const auto& c       = hw.GetClipControl();
	const auto& rc      = hw.GetRenderControl();
	const auto& mc      = hw.GetModeControl();
	const auto& eqaa    = hw.GetEqaaControl();
	const auto& cc      = hw.GetColorControl();
	const auto& smc     = hw.GetScanModeControl();
	const auto& aa      = hw.GetAaSampleControl();
	const auto& ac      = hw.GetAaConfig();

	auto log_phase = [](const char* phase) {
		if (graphics_debug_dump_enabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 512) {
				LOGF("HwCheckPhase: %s\n", phase);
			}
		}
	};

	log_phase("rt");
	RtCheck(rt);
	log_phase("vp");
	VpCheck(vp, smc);
	log_phase("clip");
	ClipCheck(c);
	log_phase("rc");
	RcCheck(rc);
	log_phase("depth");
	log_phase("mode");
	McCheck(mc);
	log_phase("blend");
	BcCheck(bclr, cc);
	log_phase("eqaa");
	EqaaCheck(eqaa, ac);
	log_phase("aa");
	AaCheck(aa, ac);
	log_phase("done");

	if (graphics_debug_dump_enabled() && RenderTargetMaskHasBoundMrt(buffer)) {
		LOGF("MRT render target mask: 0x%08" PRIx32 "\n", hw.GetRenderTargetMask());
		for (uint32_t i = 0; i < 8; i++) {
			const auto& mrt = hw.GetRenderTarget(i);
			LOGF("\t RT%u addr=0x%010" PRIx64 " mask=0x%x fmt=0x%08" PRIx32
			     " width=%u height=%u tile=%u\n",
			     i, mrt.base.addr, (hw.GetRenderTargetMask() >> (i * 4u)) & 0x0fu,
			     static_cast<uint32_t>(mrt.info.format), mrt.attrib2.width + 1,
			     mrt.attrib2.height + 1, static_cast<uint32_t>(mrt.attrib3.tile_mode));
		}
	}
	if (rc.depth_clear_enable && hw.GetDepthClearValue() != 0.0f &&
	    hw.GetDepthClearValue() != 1.0f) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 16) {
			LOGF("\t temporary: accepting non-default depth clear value %f\n",
			     hw.GetDepthClearValue());
		}
	}
	// EXIT_NOT_IMPLEMENTED(hw.GetStencilClearValue() != 0);
}

void hw_print(const CommandBuffer& buffer) {
	const auto& hw      = buffer.GetRegisters();
	const auto  rt_slot = render_target_first_bound_slot(buffer);
	const auto& rt      = hw.GetRenderTarget(rt_slot);
	const auto& bc      = hw.GetBlendControl(rt_slot);
	const auto& bclr    = hw.GetBlendColor();
	const auto& vp      = hw.GetScreenViewport();
	const auto& z       = hw.GetDepthRenderTarget();
	const auto& c       = hw.GetClipControl();
	const auto& rc      = hw.GetRenderControl();
	const auto& d       = hw.GetDepthControl();
	const auto& s       = hw.GetStencilControl();
	const auto& sm      = hw.GetStencilMask();
	const auto& mc      = hw.GetModeControl();
	const auto& eqaa    = hw.GetEqaaControl();
	const auto& cc      = hw.GetColorControl();
	const auto& smc     = hw.GetScanModeControl();
	const auto& aa      = hw.GetAaSampleControl();
	const auto& ac      = hw.GetAaConfig();

	if (graphics_debug_dump_enabled()) {
		LOGF("Context\n"
		     "\t GetRenderTargetMask()   = 0x%08" PRIx32 "\n"
		     "\t GetDepthClearValue()    = %f\n"
		     "\t GetStencilClearValue()  = %" PRIu8 "\n"
		     "\t GetLineWidth()          = %f\n"
		     "\t primitive_reset_index   = 0x%08" PRIx32 "\n",
		     hw.GetRenderTargetMask(), hw.GetDepthClearValue(), hw.GetStencilClearValue(),
		     hw.GetLineWidth(), hw.GetPrimitiveResetIndex());

		LOGF("%s", rt_print("RenderTraget:", rt).c_str());

		ZPrint("DepthRenderTraget:", z);
		VpPrint("ScreenViewport:", vp, smc);
		ClipPrint("ClipControl:", c);
		RcPrint("RenderControl:", rc);
		DPrint("DepthStencilControlMask:", d, s, sm);
		McPrint("ModeControl:", mc);
		BcPrint("BlendColorControl:", bc, bclr, cc);
		EqaaPrint("EqaaControl:", eqaa);
		AaPrint("AaSampleControl:", aa, ac);
	}
}

} // namespace Libs::Graphics
