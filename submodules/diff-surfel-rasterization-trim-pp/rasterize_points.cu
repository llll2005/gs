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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

// ⚠ 2026-09-21 多回傳一個 `tiles`（逐顆 tile 數＝精確的 binning 成本）=> 10 個元素
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const float pp_shifty,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool record_transmittance,
	const bool debug,
	const bool exact_conic_aabb,
	const int lean)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  if (scales.ndimension() != 2 || scales.size(1) != 2) {
	AT_ERROR("scales must have dimensions (num_points, 2)");
  }

  if (rotations.ndimension() != 2 || rotations.size(1) != 4) {
	AT_ERROR("rotations must have dimensions (num_points, 4)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(colors);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+2, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  // ★ 2026-09-21：逐顆 tile 數（精確的 binning 成本），見 rasterizer.h 的註解
  torch::Tensor tiles = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

  torch::Tensor transmittance = torch::full({P}, 0.0, float_opts);
  torch::Tensor num_occluder = torch::full({P}, 0, int_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
	  }

	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(),
		opacity.contiguous().data<float>(),
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		pp_shifty,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		transmittance.contiguous().data<float>(),
		num_occluder.contiguous().data<int>(),
		record_transmittance,
		radii.contiguous().data<int>(),
		debug,
		tiles.contiguous().data<int>(),
		exact_conic_aabb,
		lean);
  }
  return std::make_tuple(rendered, out_color, out_others, radii, geomBuffer, binningBuffer, imgBuffer, transmittance, num_occluder, tiles);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	 const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const float pp_shifty,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const int geom_grad)
{

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(radii);
  CHECK_INPUT(colors);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(sh);
  CHECK_INPUT(campos);
  CHECK_INPUT(binningBuffer);
  CHECK_INPUT(imageBuffer);
  CHECK_INPUT(geomBuffer);

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if(sh.size(0) != 0)
  {
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  if(P != 0)
  {
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H,
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  transMat_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  pp_shifty,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_others.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dnormal.contiguous().data<float>(),
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dtransMat.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug,
	  geom_grad);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{
  const int P = means3D.size(0);

  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }

  return present;
}

// ★ 2026-10-19 融合／稀疏 Adam（使用者 10-19「照建議做」；實驗分析 15 §2）
//   為什麼：Adam 佔每步 26%（b6、N 2.34M：增生期 15.3／收割期 13.9 ms），而 torch 2.0.1 的 foreach Adam
//   對全部參數掃約 8 遍（mul／add／mul／addcmul／sqrt／div／add／addcdiv 各一個 kernel＋暫存張量）。
//   torch 內建 fused=True 在 2.0.1 反而慢 2.3~3.2 倍（已關閉的線）=> 自己寫：一個 kernel 一遍，讀 p/g/m/v、寫 p/m/v。
//   算式與運算順序照 torch 2.0.1 _multi_tensor_adam（非 capturable、無 weight decay／amsgrad／maximize）：
//     m = m*b1;  m = m + (1-b1)*g;  v = v*b2;  v = v + (1-b2)*(g*g)
//     d = sqrt(v) / sqrt(bc2) + eps;  p = p + step_size*(m/d)        step_size = -lr/bc1（Python 端以 double 算好）
//   VAR：0＝每個運算各自捨入；1＝a + b*c 收縮成 fma（torch 的 foreach kernel 由 nvcc 預設 --fmad=true 編譯）
//        => 兩種都留著，用 tools/check_lean_render.py adamcheck 找出與 torch 逐位元一致的那個。
//   MASK：mask（N 個 bool）非空 => 稀疏：該列（顆粒）不可見就整列不動（p／m／v 都不更新）
//         ＝ gsplat SelectiveAdam／Taming 3DGS 的做法，但保留 torch 的 bias correction 與逐張量 step 計數。
template <int VAR, bool MASK>
__global__ void fusedAdamCUDA(const int64_t n, const int64_t M,
	float* __restrict__ p, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v,
	const bool* __restrict__ mask,
	const float b1, const float omb1, const float b2, const float omb2,
	const float eps, const float step_size, const float bc2s)
{
	const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	if (MASK && !mask[i / M])
		return;
	const float gi = g[i];
	float mi = __fmul_rn(m[i], b1);
	float vi = __fmul_rn(v[i], b2);
	if (VAR == 0) {
		mi = __fadd_rn(mi, __fmul_rn(omb1, gi));
		vi = __fadd_rn(vi, __fmul_rn(omb2, __fmul_rn(gi, gi)));
	} else {
		mi = __fmaf_rn(omb1, gi, mi);
		vi = __fmaf_rn(omb2, __fmul_rn(gi, gi), vi);     // torch addcmul＝self + value*(t1*t2)
	}
	const float d = __fadd_rn(__fdiv_rn(__fsqrt_rn(vi), bc2s), eps);
	const float q = __fdiv_rn(mi, d);
	p[i] = (VAR == 0) ? __fadd_rn(p[i], __fmul_rn(step_size, q)) : __fmaf_rn(step_size, q, p[i]);
	m[i] = mi;
	v[i] = vi;
}

void FusedAdamCUDA(
	torch::Tensor& param, const torch::Tensor& grad, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq,
	const torch::Tensor& mask,
	const float b1, const float omb1, const float b2, const float omb2,
	const float eps, const float step_size, const float bc2s, const int variant)
{
	for (const torch::Tensor* t : std::initializer_list<const torch::Tensor*>{&param, &grad, &exp_avg, &exp_avg_sq}) {
		TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat && t->is_contiguous(),
		            "fused_adam：p/g/m/v 必須是 contiguous 的 CUDA float32");
		TORCH_CHECK(t->numel() == param.numel(), "fused_adam：p/g/m/v 元素數不同");
	}
	const int64_t n = param.numel();
	if (n == 0)
		return;
	const bool use_mask = mask.numel() > 0;
	int64_t M = 1;
	if (use_mask) {
		TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == at::kBool && mask.is_contiguous(), "fused_adam：mask 必須是 contiguous 的 CUDA bool");
		TORCH_CHECK(param.dim() >= 1 && param.size(0) == mask.numel(), "fused_adam：mask 長度必須等於參數第 0 維（顆數）");
		M = n / mask.numel();
	}
	const int threads = 256;
	const int64_t blocks = (n + threads - 1) / threads;
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	float* p = param.data_ptr<float>();
	const float* g = grad.data_ptr<float>();
	float* m = exp_avg.data_ptr<float>();
	float* v = exp_avg_sq.data_ptr<float>();
	const bool* k = use_mask ? mask.data_ptr<bool>() : nullptr;
#define FADAM_L(VV, MM) fusedAdamCUDA<VV, MM><<<blocks, threads, 0, stream>>>(n, M, p, g, m, v, k, b1, omb1, b2, omb2, eps, step_size, bc2s)
	if (variant == 0) { if (use_mask) FADAM_L(0, true); else FADAM_L(0, false); }
	else              { if (use_mask) FADAM_L(1, true); else FADAM_L(1, false); }
#undef FADAM_L
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}
