# -*- coding: utf-8 -*-
"""Tiling feasibility verification (route 2, camera-crop) for the unified-budget K lever.

Answers Gemini's cold-water #1 and my own gate: does rendering a view in K horizontal
strips (via camera-crop: height=H/K, cy shifted) actually cut PEAK training VRAM toward
~1/K of the full-image forward+backward — the 96% rasterizer/backward buffer — while
producing the identical composited image?

Mechanism (verified against gsplat stages in ../gsplat-src):
  - projection: O(N), NOT reduced by strips (all gaussians projected each strip) -> the 4%
  - isect_tiles + rasterize fwd/bwd: only strip-intersecting gaussians -> the 96%, ~/K
Camera-crop math: a world point at original row v projects to row v - k*h in strip k's
viewport (cy' = cy - k*h, height = h). Rows outside [0,h) are culled. Exact crop.

Runs on b12's pathological geometry (the block that OOMs) for a real test. CPU data,
alloc-conf, backward included (the backward activation buffer is the dominant term).

Usage:
  python probes/p5_cost_aware/verify_tiling.py --block_id 12 \
    --init_ply data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
    --Ks 1 2 4 8 --out outputs/verify_tiling_b12.txt
"""
import argparse
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization_2dgs
from train_p5 import init_params, build_sets  # reuse proven loaders


def make_K_cam(fx, fy, cx, cy, W, H, k, K, device):
    """Return (viewmat unchanged, K-matrix, W, h, row_offset) for strip k of K."""
    h = H // K
    v0 = k * h
    # last strip absorbs the remainder so strips tile H exactly
    if k == K - 1:
        h = H - v0
    Kmat = torch.tensor([[fx, 0, cx], [0, fy, cy - v0], [0, 0, 1]], device=device)
    return Kmat[None], int(W), int(h), v0


@torch.no_grad()
def _peak_reset():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def render_full(params, viewmat, K3, W, H, sh_degree, backward):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out = rasterization_2dgs(params["means"], params["quats"], torch.exp(params["scales"]),
                             torch.sigmoid(params["opacities"]), colors, viewmat, K3, W, H,
                             sh_degree=sh_degree, packed=True, render_mode="RGB")
    img = out[0][0]  # [H,W,3]
    if backward:
        loss = img.square().mean()
        loss.backward()
    return img.detach()


def render_strips(params, viewmat, fx, fy, cx, cy, W, H, sh_degree, K, backward):
    """Render in K strips, composite, backward per-strip (bounded peak).
    Activations (colors/scales/ops) are recomputed INSIDE the loop so each strip owns a
    fresh graph that is freed after its own backward — this is what bounds peak memory
    (recompute is the ~epsilon(K) overhead). Grads accumulate into the leaf params."""
    strips = []
    for k in range(K):
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        scales = torch.exp(params["scales"])
        ops = torch.sigmoid(params["opacities"])
        Kmat, w, h, v0 = make_K_cam(fx, fy, cx, cy, W, H, k, K, params["means"].device)
        out = rasterization_2dgs(params["means"], params["quats"], scales, ops, colors,
                                 viewmat, Kmat, w, h, sh_degree=sh_degree, packed=True,
                                 render_mode="RGB")
        simg = out[0][0]  # [h,W,3]
        if backward:
            (simg.square().sum() / (H * W * 3)).backward()
        strips.append(simg.detach())
    return torch.cat(strips, dim=0)  # [H,W,3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_id", type=int, default=12)
    ap.add_argument("--init_ply", required=True)
    ap.add_argument("--config", default="configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml")
    ap.add_argument("--data_path", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--scale_mult", type=float, default=1.0,
                    help=">1 inflates gaussian footprints to force the high-overdraw / "
                         "intersection-buffer-dominated regime that actually OOMs b12 in training")
    ap.add_argument("--out", default="outputs/verify_tiling.txt")
    args = ap.parse_args()

    device = "cuda"
    train_set, _ = build_sets(args.config, args.data_path, args.block_id, "/tmp/vt")
    params = init_params(args.init_ply, device, args.sh_degree)
    if args.scale_mult != 1.0:
        with torch.no_grad():
            params["scales"].add_(float(np.log(args.scale_mult)))  # log-space multiply
    # keep grads on (training-memory test)
    for p in params.values():
        p.requires_grad_(True)
    n = params["means"].shape[0]

    # pick the camera with the most content in front (proxy for worst overdraw): just use
    # the median-index camera; b12's geometry is uniformly heavy.
    cam = train_set.cameras[len(train_set.cameras) // 2].to_device(device)
    viewmat = cam.world_to_camera.T.to(device)[None]
    fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
    W, H = int(cam.width), int(cam.height)
    print(f"[verify] block {args.block_id}: N={n:,}, view {W}x{H}")

    lines = [f"tiling verification: block {args.block_id}, N={n:,}, view {W}x{H}, sh{args.sh_degree}"]
    ref = None
    for K in args.Ks:
        for p in params.values():
            if p.grad is not None:
                p.grad = None
        _peak_reset()
        t0 = time.time()
        try:
            if K == 1:
                img = render_full(params, viewmat, torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                  device=device)[None], W, H, args.sh_degree, backward=True)
            else:
                img = render_strips(params, viewmat, fx, fy, cx, cy, W, H, args.sh_degree, K, backward=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            dt = time.time() - t0
            match = "" if ref is None else f" | vs K=1 MSE {F.mse_loss(img, ref).item():.2e}"
            if ref is None:
                ref = img
            msg = f"K={K:>2}: peakVRAM {peak:.3f}G  time {dt*1000:.0f}ms{match}"
        except torch.cuda.OutOfMemoryError as e:
            msg = f"K={K:>2}: OOM ({str(e)[:60]})"
        print(msg); lines.append(msg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saved] {args.out}")
    # verdict
    print("\nEXPECT: peakVRAM should fall toward ~1/K (the 96% rasterizer buffer);")
    print("MSE vs K=1 should be ~0 (strips composite to the same image).")


if __name__ == "__main__":
    main()
