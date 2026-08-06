"""信心分級初始化：SfM track>=k 的錨點 + 深度點只補沒錨點的體素。

⛔ 重投影誤差**不可**當信心指標，它與驗證度**反相關**（track<=2 誤差 0.142px < track>=6 的
0.549px）。用 track 長度。⚠ 取點區域要用「離 depth-init 雲夠近」不是 partition AABB
（AABB 只涵蓋足跡的 1/6）。Stage 0 結果與後續見 紀錄/研究總覽.md §7.1。
"""
import argparse
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.utils.colmap import read_points3D_binary

FIELDS = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ("opacity", "f4"),
    ("scale_0", "f4"), ("scale_1", "f4"),
    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
]


def read_depth_init(path):
    v = PlyData.read(path)["vertex"]
    return {n: np.asarray(v[n], dtype=np.float32) for n, _ in FIELDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--track_min", type=int, default=4,
                    help="Tier A threshold. 4 keeps 54.3%% of points at p90 3.9x coverage loss")
    ap.add_argument("--region_radius", type=float, default=3.0,
                    help="an SfM point counts as in-region if it lies within this many depth-init "
                         "spacings of the depth-init cloud. Replaces the partition AABB, which "
                         "covers only ~1/6 of what the block's cameras actually reconstruct")
    ap.add_argument("--fill_radius", type=float, default=None,
                    help="drop a depth point if an anchor sits within this distance; "
                         "default = the depth-init cloud's own median nearest-neighbour spacing, "
                         "i.e. replace at exactly the resolution depth-init already works at")
    ap.add_argument("--anchor_voxel", type=float, default=None,
                    help="downsample anchors to this spacing; default = fill_radius, so the two "
                         "tiers land at one common resolution and the total stays comparable to "
                         "depth-init (otherwise SfM's native 0.0036 spacing over-provisions)")
    ap.add_argument("--depth_init", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    di_path = a.depth_init or f"{a.data}/depth_init/block_{a.block}.ply"

    D = read_depth_init(di_path)
    dxyz = np.stack([D["x"], D["y"], D["z"]], 1).astype(np.float64)
    print(f"[depth-init] {di_path}\n  N={len(dxyz):,}")

    dtree = cKDTree(dxyz)
    rng = np.random.default_rng(0)
    sub = dxyz[rng.choice(len(dxyz), min(200_000, len(dxyz)), replace=False)]
    spacing = float(np.median(dtree.query(sub, k=2)[0][:, 1]))
    r = a.fill_radius if a.fill_radius is not None else spacing
    vox = a.anchor_voxel if a.anchor_voxel is not None else r
    print(f"  自身中位間距 = {spacing:.5f}   →  fill_radius={r:.5f}  anchor_voxel={vox:.5f}")

    P = read_points3D_binary(f"{a.data}/sparse/0/points3D.bin")
    sxyz = np.array([p.xyz for p in P.values()], dtype=np.float64)
    nobs = np.array([len(p.image_ids) for p in P.values()], dtype=np.int32)

    # REGION = the depth-init cloud's own footprint, NOT the partition AABB.
    # depth-init back-projects every pixel of all 284 block images, and those cameras see far
    # beyond the block boundary: measured on b12, the partition AABB holds 196,631 SfM points
    # while the depth-init cloud spans ~6x that area. Selecting anchors by AABB therefore covered
    # only 2.5% of the arm's actual content. "Near an existing depth point" is the right test --
    # an anchor is only useful where this arm has something to anchor, and it drops the COLMAP
    # outliers (z down to -202) for free.
    d_to_depth, _ = dtree.query(sxyz)
    inb = d_to_depth < a.region_radius * spacing
    print(f"[SfM] 全域 {len(sxyz):,}  →  離 depth-init 雲 < "
          f"{a.region_radius:.0f}x 間距 的有 {int(inb.sum()):,}")

    sel = inb & (nobs >= a.track_min)
    axyz_full = sxyz[sel]
    print(f"[Tier A] track >= {a.track_min}: {len(axyz_full):,}"
          f"  ({100 * len(axyz_full) / max(int(inb.sum()), 1):.1f}% of 區域內)")

    # voxel-downsample anchors to the common resolution (keep the longest-track point per cell)
    key = np.floor(axyz_full / vox).astype(np.int64)
    order = np.argsort(-nobs[sel])            # longest track wins its cell
    key_s = key[order]
    _, keep_s = np.unique(key_s, axis=0, return_index=True)
    axyz = axyz_full[order][keep_s]
    atrack = nobs[sel][order][keep_s]
    print(f"[Tier A] 體素降採樣 @ {vox:.5f} → {len(axyz):,}"
          f"  (track 中位 {int(np.median(atrack))}, p90 {int(np.percentile(atrack, 90))})")

    # Tier B: depth points with no anchor nearby
    atree = cKDTree(axyz)
    d_to_anchor, _ = atree.query(dxyz)
    fill = d_to_anchor > r
    print(f"[Tier B] 深度點在 {r:.5f} 內沒有錨點的: {int(fill.sum()):,}"
          f"  ({100 * fill.mean():.1f}% of depth-init)")

    # anchors inherit orientation/scale from their nearest depth point -> identical surfel
    # convention across tiers, so POSITION is the only variable against the depth-init arm
    _, nn = dtree.query(axyz)

    n = len(axyz) + int(fill.sum())
    el = np.empty(n, dtype=FIELDS)
    for name, _ in FIELDS:
        el[name] = np.concatenate([D[name][nn], D[name][fill]])
    # the anchor block above inherited the NEAREST depth point's position too; replace it with the
    # anchor's own metric position. The fill block already carries its own.
    na = len(axyz)
    el["x"][:na] = axyz[:, 0].astype(np.float32)
    el["y"][:na] = axyz[:, 1].astype(np.float32)
    el["z"][:na] = axyz[:, 2].astype(np.float32)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(a.out)
    w = np.concatenate([np.ones(len(axyz), np.float32), np.zeros(int(fill.sum()), np.float32)])
    np.save(a.out + ".w.npy", w)
    np.save(a.out + ".track.npy",
            np.concatenate([atrack.astype(np.int32), np.zeros(int(fill.sum()), np.int32)]))

    print(f"\n[out] {a.out}")
    print(f"  總數 {n:,}   錨點 {len(axyz):,} ({100 * len(axyz) / n:.1f}%)"
          f"   填充 {int(fill.sum()):,} ({100 * fill.sum() / n:.1f}%)")
    print(f"  對照 depth-init {len(dxyz):,}  →  比值 {n / len(dxyz):.2f}x")
    print(f"  w_i 存於 {a.out}.w.npy（Stage 1 用）")


if __name__ == "__main__":
    main()
