# -*- coding: utf-8 -*-
"""Calibrate the closed-form budget constants for N_max = (V_target - V_os)/(4MF + γτ/K).

Measures on real b12 geometry (the pathological block):
  V_os     : OS/CUDA-context + non-model overhead (GB)
  V_model  : N * F * 4 bytes * M  (params + grad + 2*Adam) — computed, cross-checked
  γ (gamma): effective render bytes per screen-tile intersection (isect + backward activations)
  τ (tau)  : screen tiles per gaussian in a view (mean AND worst — worst drives N_max)
Then prints predicted N_max for K = 1,2,4,8 (with a fragmentation safety factor).

Usage: python probes/p5_cost_aware/calibrate_budget.py --block_id 12 \
  --init_ply data/matrix_city/aerial/train/block_all/depth_init/block_12.ply --n_views 12
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization_2dgs
from train_p5 import init_params, build_sets, make_optimizers, cam_to_gsplat, load_gt

GB = 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_id", type=int, default=12)
    ap.add_argument("--init_ply", required=True)
    ap.add_argument("--config", default="configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml")
    ap.add_argument("--data_path", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--n_views", type=int, default=12)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--vram_target_gb", type=float, default=5.4)
    ap.add_argument("--frag_safety", type=float, default=0.9, help="fragmentation safety factor on V_target")
    args = ap.parse_args()

    device = "cuda"
    train_set, _ = build_sets(args.config, args.data_path, args.block_id, "/tmp/cal")
    params = init_params(args.init_ply, device, args.sh_degree)
    for p in params.values():
        p.requires_grad_(True)
    N = params["means"].shape[0]

    # F = floats per point (from actual tensors); M = params+grad+2*Adam = 4 at peak
    F_floats = sum(int(np.prod(p.shape[1:])) for p in params.values())  # per-point floats
    M = 4
    scene_scale = float((params["means"].max(0).values - params["means"].min(0).values).norm() / 2)
    opts, _ = make_optimizers(params, scene_scale, 30000)

    # --- baseline: model resident, no render (V_os + V_model, before Adam state exists) ---
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base_before_adam = torch.cuda.memory_allocated() / GB
    V_model_params_grad = N * F_floats * 4 * 2 / GB   # params + grad (Adam state lazily created on first step)

    # --- per-view render fwd+bwd: peak VRAM increment and intersection count ---
    idxs = np.linspace(0, len(train_set.cameras) - 1, args.n_views).astype(int)
    gammas, taus, peaks = [], [], []
    for i in idxs:
        cam = train_set.cameras[int(i)].to_device(device)
        viewmat, K3, W, H = cam_to_gsplat(cam, device)
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        rc, _, _, _, _, _, info = rasterization_2dgs(
            params["means"], params["quats"], torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors, viewmat, K3, W, H,
            sh_degree=args.sh_degree, packed=True, render_mode="RGB")
        img = rc[0, ..., :3]
        img.square().mean().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / GB
        tpg = info.get("tiles_per_gauss")
        I = float(tpg.float().sum()) if tpg is not None else float("nan")
        base = torch.cuda.memory_allocated() / GB  # after this view (incl Adam now)
        V_render = peak - base
        gammas.append(V_render * GB / max(I, 1))     # bytes per intersection
        taus.append(I / N)
        peaks.append(peak)

    gamma = float(np.median(gammas))
    tau_mean, tau_max = float(np.mean(taus)), float(np.max(taus))
    V_model = N * F_floats * 4 * M / GB
    base_after = torch.cuda.memory_allocated() / GB
    V_os = max(0.0, base_after - V_model)

    print("\n===== BUDGET CALIBRATION (b%d, N=%d, F=%d floats, M=%d) =====" % (args.block_id, N, F_floats, M))
    print("V_os (os+ctx+non-model)   = %.3f GB" % V_os)
    print("V_model @N=%d             = %.3f GB  (N*F*4*M)" % (N, V_model))
    print("gamma (bytes/intersection)= %.1f B   (median over %d views)" % (gamma, args.n_views))
    print("tau (tiles/point)         = mean %.1f, MAX %.1f  (MAX drives N_max)" % (tau_mean, tau_max))
    print("peak VRAM per view        = mean %.2f GB, max %.2f GB" % (np.mean(peaks), np.max(peaks)))

    Vt = args.vram_target_gb * args.frag_safety
    print("\nN_max = (%.2f*%.2f - %.3f) / (4*%d*%d/GB + gamma*tau_MAX/K)" % (
        args.vram_target_gb, args.frag_safety, V_os, F_floats, M) + "  [V_target*frag=%.2fGB]" % Vt)
    per_pt_model = M * F_floats * 4          # bytes/point (model)
    for K in (1, 2, 4, 8):
        denom = per_pt_model + gamma * tau_max / K          # bytes per point
        n_max = (Vt - V_os) * GB / denom
        print("  K=%d:  N_max = %.2fM  (model %.0fB/pt + render %.0fB/pt)" % (
            K, n_max / 1e6, per_pt_model, gamma * tau_max / K))
    print("\n(τ_MAX used for safety; K bounds render/point, Adam-offload would cut M 4->2)")


if __name__ == "__main__":
    main()
