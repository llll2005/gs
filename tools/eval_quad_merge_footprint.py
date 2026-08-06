# -*- coding: utf-8 -*-
"""Fair subset-merge evaluation: compare merged vs per-block models ONLY on val
cameras whose view frustum footprint lies inside the merged region.

Why: a subset merge (e.g. quad {7,8,12,13} of 25) crops each block to its
partition cell, so any val view that sees beyond the subset renders background
holes there (measured: naive full-image eval craters to ~12-15 PSNR while the
per-block models score 22.4-24.3 — the gap is dominated by missing out-of-subset
content, not by merge seams). Filtering to fully-covered cameras isolates the
true cross-block consistency / seam effect.

Usage:
  python tools/eval_quad_merge_footprint.py \
    --merged outputs/quad_merge_aggr17/checkpoints/merged.ckpt \
    --runs 7=outputs/mcmc_60k_sh3_aggr17_b7 8=outputs/mcmc_60k_sh3_aggr17_b8 \
           12=outputs/mcmc_60k_sh3_aggr17_b12 13=outputs/mcmc_60k_sh3_aggr17_b13 \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml
"""
import os
import sys
import glob
import argparse
import dataclasses

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.estimated_depth_colmap_block_dataparser import EstimatedDepthBlockColmap

# quad {7,8,12,13} cell union in SfM frame, from partitions.pt partition_coordinates
QUAD_X = (-1.1103, 4.5549)
QUAD_Y = (-3.7766, 1.1385)
GROUND_Z = 0.5          # ray-plane intersection height (between ground ~0 and roofs ~2-3.5)
MIN_DOWN = 0.05         # rays with dir_z > -MIN_DOWN (near-horizontal) => footprint unbounded => reject


def build_val_set(cfg_path, data_path, block, out_dir):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    init_args = dict(cfg["data"]["parser"].get("init_args", {}))
    init_args["block_id"] = block
    valid = {f.name for f in dataclasses.fields(EstimatedDepthBlockColmap)}
    init_args = {k: v for k, v in init_args.items() if k in valid}
    dp = EstimatedDepthBlockColmap(**init_args).instantiate(path=data_path, output_path=out_dir, global_rank=0)
    return dp.get_outputs().val_set


def footprint_ok(cam, nx=5, ny=3):
    """True iff a grid of pixel rays all intersect z=GROUND_Z inside the quad rect."""
    W, H = int(cam.width), int(cam.height)
    fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
    c = cam.camera_center.cpu().numpy().reshape(3)
    Rinv = torch.inverse(cam.world_to_camera[:3, :3]).cpu().numpy()  # row conv: dir_w = dir_c @ Rinv
    for v in np.linspace(0, H - 1, ny):
        for u in np.linspace(0, W - 1, nx):
            d_c = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            d_w = d_c @ Rinv
            d_w /= np.linalg.norm(d_w)
            if d_w[2] > -MIN_DOWN:
                return False
            t = (GROUND_Z - c[2]) / d_w[2]
            p = c + t * d_w
            if not (QUAD_X[0] <= p[0] <= QUAD_X[1] and QUAD_Y[0] <= p[1] <= QUAD_Y[1]):
                return False
    return True


def load_gt(image_path, W, H, device):
    img = Image.open(image_path).convert("RGB")
    if img.size != (W, H):
        img = img.resize((W, H), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1).to(device) / 255.0


@torch.no_grad()
def psnr(a, b):
    mse = F.mse_loss(a, b)
    return float(-10.0 * torch.log10(mse))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="block=run_dir pairs")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_path", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--out", default="outputs/quad_merge_aggr17/footprint_eval.txt")
    args = ap.parse_args()

    device = "cuda"
    runs = {}
    for kv in args.runs:
        k, v = kv.split("=")
        runs[int(k)] = v

    merged_model, merged_renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        args.merged, device, eval_mode=True)
    print(f"[load] merged: {merged_model.get_xyz.shape[0]:,} gaussians")
    bg = torch.zeros(3, device=device)

    lines = []
    all_m, all_s = [], []
    for block, run_dir in sorted(runs.items()):
        val_set = build_val_set(args.config, args.data_path, block, run_dir)
        keep = [i for i, cam in enumerate(val_set.cameras) if footprint_ok(cam)]
        print(f"[block {block}] val={len(val_set.cameras)} footprint-covered={len(keep)}")
        if not keep:
            lines.append(f"block {block}: 0 covered cameras")
            continue

        blk_ckpt = GaussianModelLoader.search_load_file(os.path.join(run_dir, "blocks", f"block_{block}"))
        blk_model, blk_renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            blk_ckpt, device, eval_mode=True)

        for i in keep:
            cam = val_set.cameras[i].to_device(device)
            gt = load_gt(val_set.image_paths[i], int(cam.width), int(cam.height), device)
            r_m = merged_renderer(cam, merged_model, bg_color=bg)["render"].clamp(0, 1)
            r_s = blk_renderer(cam, blk_model, bg_color=bg)["render"].clamp(0, 1)
            pm, ps = psnr(r_m, gt), psnr(r_s, gt)
            all_m.append(pm); all_s.append(ps)
            name = val_set.image_names[i]
            lines.append(f"block {block} {name}: merged {pm:.2f} | single {ps:.2f} | drop {ps - pm:+.2f}")
            print(lines[-1])

        del blk_model, blk_renderer
        torch.cuda.empty_cache()

    if all_m:
        summary = (f"\nCOVERED-VIEW SUMMARY ({len(all_m)} views): "
                   f"merged {np.mean(all_m):.3f} | single {np.mean(all_s):.3f} | "
                   f"merge drop {np.mean(all_s) - np.mean(all_m):+.3f} dB")
        lines.append(summary)
        print(summary)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
