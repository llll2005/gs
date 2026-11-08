/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy, float pp_shifty,
		int* radii,
		float2* points_xy_image,
		float* depths,
		// float* isovals,
		// float3* normals,
		float* transMats,
		float* colors,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		ushort4* rects,
		bool exact_conic_aabb,
		bool prefiltered,
		const float* shs_rest = nullptr);   // ★ 2026-10-24 非 nullptr => SH 分開存放（shs 是 dc，M 是含 dc 的總係數數）

	// Main rasterization method.
	// ★ 2026-11-07 record 模式累加 Σ T 而不是 Σ T*alpha（lean bit 256；見 forward.cu g_record_T）
	void setRecordT(bool on);
	// ★ 2026-11-07 alpha 上限（預設 0.99；lean bit 512 => 0.999；見 forward.cu g_alpha_max）
	void setAlphaMax(float a);
	// ★ 2026-11-08 飽和門檻（預設 1e-4；lean bit 1024 => 1e-3；見 forward.cu g_t_sat）
	void setTSat(float t);

	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float2* points_xy_image,
		const float* features,
		const float* transMats,
		const float* depths,
		const float4* normal_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_others,
		float* transmittance,
		int* num_covered_pixels,
		bool record_transmittance,
		int mode = 0,
		uint8_t* pair_used = nullptr,   // ★ 2026-10-16 配對貢獻 bit（只在 mode 1 寫）
		const uint8_t* pix_skip = nullptr,   // ★ 2026-11-05 逐像素不計分遮罩（只在 mode 1／5 讀）
		bool wacc = false);   // ★ 2026-11-05 訓練中累加 V/C 走 warp 先加總（只在 mode 1／5 且有累加時）
}


#endif
