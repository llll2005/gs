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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy, float pp_shifty,
			const bool prefiltered,
			float* out_color,
			float* out_others,
			float* transmittance,
			int* num_covered_pixels,
			bool record_transmittance,
			int* radii = nullptr,
			bool debug = false,
			// ★ 2026-09-21：逐顆的 tile 數（= binning 成本本身）。光柵器每步都算好，
			//   但過去只被 prefix sum 消費掉、從未暴露。Python 端的 `cost_budget` /
			//   `vpc` / `cost_aware_densify` 一直在用 `radii^2` 估它，而那個估計
			//   假設正方形、忽略 tile 量化、忽略螢幕裁切。nullptr = 不輸出（零成本）。
			int* out_tiles = nullptr,
			// ★ 2026-09-21 執行期旗標（見 auxiliary.h 的長註解）：編譯期版本讓 lab 的
			//   三槽平行 A/B **無法同時跑**，且誤用不會報錯。
			bool exact_conic_aabb = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy, float pp_shifty,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_depths,
			float* dL_dmean2D,
			float* dL_dnormal,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dtransMat,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif
