"""錨點在訓練中漂多遠、存活率是否高於單目填充點。

⚠ 粒子沒有持久 ID（剪枝/relocation 每幾百步重寫陣列），所以兩個量測都定義在**初始體素格的
佔用率**上。⚠ 漂移欄被剪枝污染，分不出「移動了」和「被刪了」。
結論（剪枝盲於多視角驗證）見 紀錄/研究總覽.md §5.3。
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
