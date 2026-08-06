"""Do verified anchors stay put, and do they survive pruning better than guessed points?

Stage 0 of confidence-graded init seeds 15.5% of the cloud from SfM points with track >= 4 --
positions that multiple independent views agreed on -- and fills the rest from monocular depth.
It applies NO protection to the anchors (that is Stage 1). So the run answers two questions that
decide whether Stage 1 is worth building, and neither of them is PSNR:

  DRIFT     Does photometric optimization pull verified points off the surface?
            If yes, the differential position lr of Stage 1 is required, and this measures by how
            much. If no, the whole init line is weaker than hypothesized and RAIN-GS is right that
            optimization, not initialization, is the lever.

  SURVIVAL  v1's schedule is net-decaying (interval 350 > break-even 231.5, -22% per 3500 steps),
            so contribution pruning executes most of the population. If anchors -- known real
            surface -- do not survive at a higher rate than fill, that is a sixth independent
            demonstration that contribution pruning is blind to geometry.

PRIMITIVES HAVE NO PERSISTENT ID. Pruning and MCMC relocation rewrite the array every few hundred
steps, so an anchor cannot be followed individually. Both measurements are therefore defined on
OCCUPANCY of the initial voxel grid, which needs no identity:

    retention   fraction of cells that held a tier-X seed and still hold some primitive
    drift       distance from each seed to the nearest surviving primitive, in units of the
                initial grid spacing. ~0 means something is still sitting where the seed was.

Run it on the graded arm and the depth-init arm at matching steps; the depth-init arm has no
anchors, so pass its own PLY and every seed counts as fill.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floater_label import surface_distance

STEP = re.compile(r"step=(\d+)")


def load_seed(ply_path):
    v = PlyData.read(ply_path)["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)
    wp = ply_path + ".w.npy"
    w = np.load(wp) if os.path.exists(wp) else np.zeros(len(xyz), np.float32)
    return xyz, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_ply", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    sxyz, w = load_seed(a.seed_ply)
    anchor, fill = w > 0.5, w <= 0.5
    tree = cKDTree(sxyz)
    rng = np.random.default_rng(0)
    sub = sxyz[rng.choice(len(sxyz), min(200_000, len(sxyz)), replace=False)]
    grid = float(np.median(tree.query(sub, k=2)[0][:, 1]))
    print(f"[seed] {a.seed_ply}")
    print(f"  N={len(sxyz):,}  錨點 {int(anchor.sum()):,} ({100 * anchor.mean():.1f}%)  "
          f"填充 {int(fill.sum()):,}   格距 {grid:.5f}")

    d0, unit = surface_distance(sxyz, a.data)
    print(f"  起始離表面：全體 p50={np.median(d0) / unit:5.1f}x", end="")
    if anchor.any():
        print(f"   錨點 p50={np.median(d0[anchor]) / unit:5.1f}x"
              f"   填充 p50={np.median(d0[fill]) / unit:5.1f}x")
    else:
        print()

    hdr = (f"\n{'step':>7}{'N':>11}{'離表面p50':>11}"
           f"{'錨點留存':>10}{'填充留存':>10}{'錨點漂移':>10}{'填充漂移':>10}")
    print(hdr)
    for ck in sorted(a.ckpts, key=lambda p: int(STEP.search(p).group(1))):
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        cur = sd["gaussian_model.gaussians.means"].float().numpy().astype(np.float64)
        step = int(STEP.search(ck).group(1))
        ct = cKDTree(cur)
        # distance from every seed to the nearest surviving primitive
        dd, _ = ct.query(sxyz)
        # "retained" = something still occupies that seed's own grid cell
        ret_a = float((dd[anchor] < grid).mean()) if anchor.any() else float("nan")
        ret_f = float((dd[fill] < grid).mean())
        dr_a = float(np.median(dd[anchor]) / grid) if anchor.any() else float("nan")
        dr_f = float(np.median(dd[fill]) / grid)
        d, _ = surface_distance(cur, a.data)
        print(f"{step:>7,}{len(cur):>11,}{np.median(d) / unit:>10.1f}x"
              f"{100 * ret_a:>9.1f}%{100 * ret_f:>9.1f}%"
              f"{dr_a:>9.2f}g{dr_f:>9.2f}g")

    print("\n  留存＝該種子的格子裡還有粒子的比例；漂移＝種子到最近存活粒子的中位距離（格距倍數）")
    print("  ⚠ 離表面距離的定義就是『到最近 SfM 點』，錨點起點必然 0x —— 不可用它判勝負，"
          "勝負看 PSNR/SSIM/LPIPS/紋理比")


if __name__ == "__main__":
    main()
