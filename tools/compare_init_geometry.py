"""Compare initializations and trained models on frame-independent geometry.

The 27-28 dB runs (`citygsv2_mc_aerial_sh0_trim`, 2026-04) were trained on the OLD dataset, which
was regenerated on 2026-05-29 into a different SfM frame, and the old images are gone. So absolute
scales cannot be compared and the old cloud cannot be projected into any camera we still have.

What survives the frame change is RATIOS. And the ratio that matters is the one that sets overdraw:

    coverage = 3 * scale / own-median-NN-spacing

A cloud whose Gaussians span 3 sigma across k times their own spacing paints each pixel about
pi*k^2 times over per layer. Multiply by the number of layers -- points per spacing-sized cell --
and you have the overdraw, all from quantities internal to the cloud.

    overdraw ~ pi * coverage^2 * layers

Measured on b12 at 1.2x: our depth-init projects to a 22.42 px median radius while the visible
surface supports at most ~362k primitives (~4 px each), so the initial Gaussians are an order of
magnitude larger than the surface can use. This tool asks whether the coarse model that produced
the 28.9 dB result had the same property or not.

`layers` is also the shell measure: 1.0 means a clean single surface, 3+ means the per-image depth
estimates were stacked rather than fused.
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


def load(path):
    """-> xyz, scales[n,k], opacity(sigmoid-space)"""
    if path.endswith(".ply"):
        v = PlyData.read(path)["vertex"]
        names = v.data.dtype.names
        xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1)
        sk = sorted(n for n in names if re.fullmatch(r"scale_\d+", n))
        sc = np.exp(np.stack([np.asarray(v[k]) for k in sk], 1).astype(np.float64))
        op = 1 / (1 + np.exp(-np.asarray(v["opacity"]).astype(np.float64)))
        return xyz.astype(np.float64), sc, op
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if "state_dict" in sd else sd
    g = "gaussian_model.gaussians."
    xyz = sd[g + "means"].float().numpy().astype(np.float64)
    sc = np.exp(sd[g + "scales"].float().numpy().astype(np.float64))
    op = torch.sigmoid(sd[g + "opacities"]).float().numpy().astype(np.float64).reshape(-1)
    return xyz, sc, op


def stats(name, path, sample=300_000):
    xyz, sc, op = load(path)
    n = len(xyz)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(sample, n), replace=False)
    tree = cKDTree(xyz)
    d = tree.query(xyz[idx], k=2)[0][:, 1]
    spacing = float(np.median(d))                       # the cloud's own resolution
    s_med = float(np.median(sc[idx].mean(1)))           # mean of the 2D scales
    coverage = 3 * s_med / spacing                      # 3-sigma extent in units of spacing

    # SHELL THICKNESS by local PCA. A first version counted points per spacing-sized cell, which is
    # circular for a voxel-downsampled cloud: dedup guarantees ~1 point per cell, and `spacing` is
    # itself set by the voxel size, so it returned 1.06 for a cloud whose layers were measured to be
    # 0.0600 apart -- 2.4x the spacing, i.e. in DIFFERENT cells, one point each.
    #
    # The smallest principal axis of a local neighbourhood is the quantity that actually separates
    # them: for a single-layer surface the neighbourhood is planar and the third eigenvalue is at
    # the noise floor, while a stack of disagreeing per-image estimates is thick along the view
    # direction. Reported in units of the cloud's own spacing so it survives the frame change.
    k = 24
    _, nb = tree.query(xyz[idx], k=k)
    P = xyz[nb]                                          # [m, k, 3]
    P = P - P.mean(1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", P, P) / k
    ev = np.linalg.eigvalsh(cov)                         # ascending
    thick = np.sqrt(np.maximum(ev[:, 0], 0))             # smallest axis = out-of-plane extent
    planar = np.sqrt(np.maximum(ev[:, 2], 0))            # largest axis, for the ratio
    thickness = float(np.median(thick)) / spacing
    flatness = float(np.median(thick / np.maximum(planar, 1e-12)))
    overdraw = np.pi * coverage ** 2 * max(2 * thickness, 1.0)
    aniso = float(np.median(sc[idx].max(1) / np.maximum(sc[idx].min(1), 1e-12))) if sc.shape[1] > 1 else 1.0
    return {
        "name": name, "N": n, "spacing": spacing, "scale": s_med,
        "coverage": coverage, "thickness": thickness, "flatness": flatness,
        "overdraw": overdraw,
        "aniso": aniso, "op_med": float(np.median(op)),
        "op_lo": float((op < 0.05).mean()), "op_hi": float((op > 0.9).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path ...")
    a = ap.parse_args()

    rows = []
    for spec in a.models:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"  {name}: 找不到 {path}，跳過")
            continue
        rows.append(stats(name, path))
        print(f"  ...{name}")

    print(f"\n{'模型':<26}{'N':>12}{'自身間距':>11}{'覆蓋倍數':>10}"
          f"{'厚度/間距':>11}{'厚/平坦':>9}{'overdraw':>10}{'opacity中位':>11}")
    for r in rows:
        print(f"{r['name']:<26}{r['N']:>12,}{r['spacing']:>11.5f}"
              f"{r['coverage']:>10.2f}{r['thickness']:>11.2f}{r['flatness']:>9.3f}"
              f"{r['overdraw']:>10.1f}{r['op_med']:>11.3f}")

    print(f"\n  覆蓋倍數 = 3σ 延展 ÷ 自身點間距（1.0 = 剛好鋪滿不重疊，座標系無關）")
    print(f"  厚度/間距 = 局部 24 鄰域 PCA 最小主軸 ÷ 自身間距 ← **殼的判準**")
    print(f"             單層表面應 <1（面內解析度就是間距，面外應更薄）；>2 = 堆疊的殼")
    print(f"  厚/平坦   = 最小主軸 ÷ 最大主軸（0 = 完美平面，1 = 各向同性團塊）")
    print(f"  overdraw = π × 覆蓋倍數² × 厚度（每個像素被畫幾次的估計）")
    print(f"\n  ⚠⚠ 只在【同類之間】可比：init（opacity 一律 0.99）vs 訓練後模型（中位 0.03~0.11）")
    print(f"     不可跨類比較。本指標完全不看 opacity，而一顆又大又透明的高斯在 binning 成本上很貴、")
    print(f"     在成像上幾乎不存在——兩者被混為一談。實測訓練後模型的覆蓋倍數常態就在 8~18")
    print(f"     （oreg 8.79 拿 23.848、official_ft_blk5 17.76 拿 23.342），所以拿 init 的 3.67")
    print(f"     去判訓練後的 8.60 為『異常』是比錯對象（2026-08-06 修正）。")
    print(f"  ⚠ 舊資料的模型只有比值可比，絕對尺度與螢幕像素不可跨資料集比較")


if __name__ == "__main__":
    main()
