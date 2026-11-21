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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* transMat;
		float* aih;          // ★ 2026-11-21 反單應交點（P*9；只在 lean bit 16384 時填，見 FORWARD::computeAIH）
		float4* normal_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;
		ushort4* rects;   // 逐顆的 tile 盒（不對稱），preprocess 存、duplicateWithKeys 讀

		static GeometryState fromChunk(char*& chunk, size_t P);
		// ★ 2026-11-21 aih 放在緩衝最後面、只有 with_aih 時才算進大小（forward 依旗標設；backward 不動＝只算指標，
		//   緩衝裡沒有它時那個指標不會被讀：backward 只在反單應／幾何配對遮罩旗標開時用，那時 forward 一定配置了）。
		//   原本無條件配置 => 旗標全關也每顆多 36 B（全場景 20M 顆一次渲染 +0.54 GiB，紀錄/實驗分析/39 §4.7）。
		static bool with_aih;
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;
		uint8_t* pair_used;   // ★ 2026-10-16 配對貢獻 bit：forward（lean 訓練）記「這個 (tile,顆粒) 配對有沒有被任何像素混合」，backward 擠掉 0 的

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};