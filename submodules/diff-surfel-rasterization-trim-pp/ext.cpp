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

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("set_depth_cap", &SetDepthCap);
  m.def("set_depth_cap_grad_mode", &SetDepthCapGradMode);   // ★ 2026-11-13 規格 3 §4-3 只補前面   // ★ 2026-11-11 逐貢獻深度上限
  m.def("set_depth_cap_acc", &SetDepthCapAcc);   // ★ 2026-11-11 逐顆落在上限後面的 T·α
  m.def("fused_adam", &FusedAdamCUDA);   // ★ 2026-10-19 融合／稀疏 Adam
}