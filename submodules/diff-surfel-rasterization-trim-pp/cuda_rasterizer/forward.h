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
		const float* shs_rest = nullptr,   // ★ 2026-10-24 非 nullptr => SH 分開存放（shs 是 dc，M 是含 dc 的總係數數）
		float* aih = nullptr);   // ★ 2026-11-21 反單應 9 個數（非 nullptr 時 preprocess 順便寫；見 forward.cu aihFromGeom）

	// Main rasterization method.
	// ★ 2026-11-07 record 模式累加 Σ T 而不是 Σ T*alpha（lean bit 256；見 forward.cu g_record_T）
	void setRecordT(bool on);
	// ★ 2026-11-07 alpha 上限（預設 0.99；lean bit 512 => 0.999；見 forward.cu g_alpha_max）
	void setAlphaMax(float a);
	// ★ 2026-11-08 飽和門檻（預設 1e-4；lean bit 1024 => 1e-3、bit 2048 => 1e-5；見 forward.cu g_t_sat）
	void setTSat(float t);
	void setSkipFlags(int f);   // ★ 2026-11-16 A-1／A-3：1＝A-1 forward（逐 warp 精確支撐盒跳過）、4＝A-3（pair_used 存 8 個 warp 的遮罩）
	// ★ 2026-11-11 逐像素深度上限圖（裝置端指標＋像素數；nullptr／0＝清掉）
	void setDepthCap(const float* cap_inv, int n);
	void setShBasis(const float* U, int K);   // ★ 2026-11-21 晚 SH 可觀測子空間基底（U 225 個 float、列距 15；K＝0 關）
	void setDepthCapAcc(float* bh, float* tot);
	void setSurfTex(const float* tex);   // ★ 2026-11-28 面內亮度漸變（float2 x P；nullptr＝關）   // ★ 2026-11-11 逐顆落在上限後面的 T·α

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
		bool wacc = false,   // ★ 2026-11-05 訓練中累加 V/C 走 warp 先加總（只在 mode 1／5 且有累加時）
		const float* aih = nullptr);   // ★ 2026-11-21 反單應交點（只在 mode 1 讀；nullptr＝原寫法）

	// ★ 2026-11-21 反單應交點的每顆 9 個數（q0, q1, x_mu, y_mu, 1/Z_c）；只寫 radii > 0 的顆粒
	void computeAIH(int P, const float* means3D, const glm::vec2* scales, const glm::vec4* rotations,
		const float* viewmatrix, const float focal_x, const float focal_y, const int W, const int H, const float pp_shifty,
		const int* radii, float* aih);
}


#endif
