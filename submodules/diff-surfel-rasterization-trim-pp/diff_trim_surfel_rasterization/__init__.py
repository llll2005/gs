#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.pp_shifty,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.record_transmittance,
            raster_settings.debug,
            raster_settings.exact_conic_aabb,
            3 if getattr(raster_settings, "audit_tiles", False) else (
                (1 if getattr(raster_settings, "lean_render", False) else 0)
                | (4 if getattr(raster_settings, "record_reduce", False) else 0)
                | (8 if getattr(raster_settings, "tile_cull", False) else 0)),
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, transmittance, num_covered_pixels, tiles = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, transmittance, num_covered_pixels, tiles = _C.rasterize_gaussians(*args)

        if getattr(raster_settings, "audit_tiles", False):
            # audit：transmittance＝reached_tiles、num_covered_pixels＝useful_tiles、depth＝逐像素 [迴圈次數, tile 配對數, 混合次數]
            return transmittance, num_covered_pixels, radii, depth, tiles
        if raster_settings.record_transmittance:
            return transmittance, num_covered_pixels, radii

        # ★ 2026-10-02 lean_render：幾何 7 通道沒算（只剩 alpha）=> backward 一律走 GEOM=false。
        #   關掉 materialize_grads，沒進 loss 的輸出在 backward 拿到 None 而不是全 0 張量
        #   => 有人把幾何輸出接進 loss 時 grad_depth 不是 None => 在 backward 當場報錯（不靜默算錯）。
        if getattr(raster_settings, "lean_render", False):
            ctx.set_materialize_grads(False)
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        # ★ 2026-09-21：`tiles` = 逐顆 tile 數（精確的 binning 成本）。整數張量、不可微，
        #   明確標記以免 autograd 為它建圖。見 rasterizer.h 的長註解。
        if not getattr(raster_settings, "return_tiles", False):
            return color, radii, depth          # 歷史介面：其他 checkout 靠它
        ctx.mark_non_differentiable(tiles)
        return color, radii, depth, tiles

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth, grad_tiles=None):
        # grad_tiles 有預設值：`record_transmittance` 分支的 forward 只回傳 3 個值，
        # 兩條路徑的 arity 不同（既有設計；該分支一律在 no_grad 下用）。

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        assert not raster_settings.record_transmittance, 'should not execute backward for calculate transmittance'
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        geom_grad = True
        if getattr(raster_settings, "lean_render", False):
            if grad_depth is not None:
                raise RuntimeError(
                    "lean_render=True 時幾何輸出（深度／法線／distortion，只有 alpha 有算）沒有計算，"
                    "但它們被接進了 loss（normal／dist／depth 權重 > 0？）=> 關掉 lean_render")
            geom_grad = False
            grad_depth = torch.empty(0, device=means3D.device)
            if grad_out_color is None:
                grad_out_color = torch.zeros((3, raster_settings.image_height, raster_settings.image_width),
                                             device=means3D.device)

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.pp_shifty,
                grad_out_color,
                grad_depth,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                # ★ 2026-10-09 int 旗標：bit0 = geom、bit1 = 關 absgrad（absgrad_gate；舊 bool 介面等價於 bit1=0）
                (1 if geom_grad else 0) | (0 if getattr(raster_settings, "absgrad", True) else 2)
                # ★ 2026-10-09 bit2 = 飽和跳過、bit3 = block 內先加總（只在 lean／geom 關時生效，backward.cu renderCUDA_opt）
                | (4 if getattr(raster_settings, "bwd_sat_skip", False) else 0)
                | (8 if getattr(raster_settings, "bwd_reduce", False) else 0))

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    record_transmittance: bool
    debug : bool
    pp_shifty : float = 0.0
    # ★ 2026-09-21：精確圓錐不對稱外接盒（見 cuda_rasterizer/auxiliary.h 的長註解）。
    #   **執行期**旗標而非 #define —— lab 三槽平行時 A/B 兩臂要能同時跑，
    #   且誤用編譯期旗標不會報錯（兩邊都吃最後編的那份）。
    exact_conic_aabb : bool = False
    # ⚠⚠ 2026-09-22：`tiles` 必須**可選**。這個套件裝在**共用的 conda 環境**裡，而同一台機器上
    #   還有別的 checkout 在用它（例：`cityGS_origin/` 那份未改動原始碼的官方參考線）。
    #   把回傳從 3 個改成 4 個 => 那份的 `rendered_image, radii, allmap = output` 直接
    #   `ValueError: too many values to unpack`，官方參考線**整條陣亡**（2026-09-22 實際發生）。
    #   ⇒ 預設 False（回傳 3 個，與歷史一致）；要 tiles 的呼叫端自己開。
    #   本修正**只動 Python 包裝層**，C++ 本來就一直回傳 10 個 => **不需要重編**。
    return_tiles : bool = False
    bwd_sat_skip : bool = False   # ★ 2026-10-09 backward 從 tile 內最大的最後貢獻者開始（結果逐位元相同）
    bwd_reduce : bool = False     # ★ 2026-10-09 backward 梯度 block 內先加總再寫全域（只差浮點加總順序）
    absgrad : bool = True   # ★ 2026-10-09：False => backward 不做 absgrad 的 atomicAdd（只有 dL_dmean2D.z 變 0）
    # ★ 2026-10-02：剃除沒人用的計算（見 forward.cu 的 MODE 與 backward.cu 的 GEOM）。
    #   訓練：只算顏色＋alpha、backward 跳過幾何梯度；record（trim pass）：只算 T*alpha 與覆蓋數。
    #   預設 False => 與舊行為完全相同（舊 kernel 的原始碼一字未動，只是多了模板參數）。
    lean_render : bool = False
    # ★ 2026-10-02：純量測（auditCUDA）——每個 (tile, 顆粒) 配對的幾何有效／遮擋可達／逐像素工作量。不可微、不參與訓練。
    audit_tiles : bool = False
    # ★ 2026-10-03：record（trim pass）改成 block 內先加總、每個 (tile,顆粒) 一次 global atomic（forward.cu recordReduceCUDA）
    record_reduce : bool = False
    # ★ 2026-10-03：非 record 呼叫只綁「可能 alpha >= 1/255」的 tile（rasterizer_impl.cu tileMayContribute）；渲染逐位元相同
    tile_cull : bool = False

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()


        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

