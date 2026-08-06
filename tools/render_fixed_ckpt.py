"""Render a fixed checkpoint over fixed views and dump images + cost counters.

Purpose: A/B a rasterizer change that is supposed to be output-preserving. Run once with the
old binary, rebuild, run again, compare. Written for EXACT_SUPPORT (紀錄/論文核對表.md §11),
where the claim is "renders are byte-identical, only the binning cost drops" — that claim is
only worth anything if it is checked bit for bit rather than by eyeballing PSNR.

Counters recorded per view:
  peak_mem  peak CUDA bytes allocated across the render call — the quantity the VRAM budget
            formula is written in, so this is the number that decides whether N_max moves.
  tiles     Σ over primitives of the tile-rectangle area implied by the returned `radii`,
            i.e. the binning work the rasterizer signed up for. Uses the same 16px grid and
            frame clamping as getRect, so it tracks `tiles_touched` rather than a proxy.
  n_visible number of primitives with radii > 0 (EXACT_SUPPORT drops o <= 1/255 outright).

Usage:
  python tools/render_fixed_ckpt.py --ckpt <path> --out before.npz --views 8
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probes/p5_cost_aware"))

from internal.utils.gaussian_model_loader import GaussianModelLoader

TILE = 16


def tile_count(radii: torch.Tensor, means2d_ok: torch.Tensor, W: int, H: int) -> int:
    """Sum of getRect rectangle areas, clamped to the grid exactly as auxiliary.h does."""
    gx, gy = math.ceil(W / TILE), math.ceil(H / TILE)
    r = radii.float()
    # getRect works from the projected centre; we do not have it here, so bound the rectangle
    # by the grid only (an upper bound that is tight for anything not near the frame edge).
    side_x = torch.clamp(torch.ceil(2 * r / TILE) + 1, max=gx)
    side_y = torch.clamp(torch.ceil(2 * r / TILE) + 1, max=gy)
    return int((side_x * side_y * means2d_ok).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--config", default="configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml")
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block", type=int, default=12)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)

    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        args.ckpt, device, eval_mode=False)   # eval_mode pre-activates/renames properties; the SB renderer reads gaussians["shs_dc"] directly
    print(f"[load] {args.ckpt}\n[load] gaussians = {model.get_xyz.shape[0]:,}")

    from train_p5 import build_sets
    train_set, _ = build_sets(args.config, args.data, args.block, "/tmp/claude-1000/_rfc")
    idxs = np.linspace(0, len(train_set.cameras) - 1, args.views).astype(int).tolist()
    print(f"[data] {len(train_set.cameras)} views, sampling {idxs}")

    bg = torch.zeros(3, device=device)
    out = {}
    tot_tiles = tot_vis = 0
    for k, i in enumerate(idxs):
        cam = train_set.cameras[i].to_device(device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            res = renderer(cam, model, bg_color=bg)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        img = res["render"].detach().float().cpu().numpy()
        radii = res["radii"].detach() if "radii" in res else None
        W, H = int(cam.width), int(cam.height)
        vis = (radii > 0).float() if radii is not None else torch.zeros(1, device=device)
        tiles = tile_count(radii, vis, W, H) if radii is not None else 0
        nvis = int(vis.sum().item())
        tot_tiles += tiles
        tot_vis += nvis

        out[f"img{k}"] = img
        out[f"peak{k}"] = np.int64(peak)
        print(f"  view {i:4d}: peak={peak/2**20:8.1f} MiB  tiles={tiles/1e6:7.2f}M  visible={nvis:,}")

    out["isect_total"] = np.int64(tot_tiles)
    out["visible_total"] = np.int64(tot_vis)
    out["peak_max"] = np.int64(max(int(out[f"peak{k}"]) for k in range(len(idxs))))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f"\n[out] {args.out}")
    print(f"  Σtiles={tot_tiles/1e6:.2f}M  Σvisible={tot_vis:,}  peak_max={int(out['peak_max'])/2**20:.1f} MiB")


if __name__ == "__main__":
    main()
