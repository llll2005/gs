"""Predict what the start-of-training trim will keep, and clear the spurious front layer.

TWO MEASURED FACTS THIS IS BUILT ON
-----------------------------------
1. The start trim (`sep_depth_trim_2dgs_renderer.py:185-218`) keeps a roughly FIXED number of
   primitives -- 360,817 from a 1,316,223-point init and 363,190 from a 991,841-point one. What
   survives is the front-most layer, and its size is set by the scene's visible surface area, not
   by how many points we supply.

2. It keeps the WRONG layer. On the original graded init, survivors sat 21.4x from the true
   surface while the culled points sat 18.3x -- what got deleted was CLOSER to the truth. 69.1% of
   culled points lie behind their nearest survivor, at a median radial gap of 0.0600, which matches
   the pseudo-depth error scale (0.0795) rather than the voxel size (0.03).

So depth-init is not a surface, it is a shell of per-image surface estimates that disagree by the
depth error, and the trim performs a self-reinforcing selection: whatever happens to be in front at
step 1 survives and locks the error in, deleting the verified geometry behind it.

WHY SPHERE CLEARING FAILED, AND WHAT REPLACES IT
------------------------------------------------
`make_graded_init.py --fill_radius 0.08` clears fill within a SPHERE around each anchor. That only
removes the immediate neighbours; a fill point 0.3 further along the line of sight still occludes.
Measured effect on anchor survival: 10.7% -> 13.1%, and fill survival rose by the same relative
amount, i.e. it was the smaller cloud, not the clearing.

The correct test is in RAY space: a fill point is spurious if, at the pixel it lands on, an anchor
sits just BEHIND it. "Just" matters -- a fill point far in front is a genuinely different surface
(a roof edge in front of a wall) and must be kept. The margin is therefore bounded by the depth
error, so only shell duplicates of the same surface are removed.

A single view can be unlucky (grazing angle, a real thin structure), so removal requires a majority
vote across the views in which the point is actually visible.

PREDICTING SURVIVAL WITHOUT TRAINING
------------------------------------
The same depth buffer answers "will the trim keep this point?" -- a primitive survives if it is
front-most at its pixel in at least one view. Run with --predict_only to score a candidate init on
CPU in minutes instead of launching a run to find out. The predictor is calibrated against two
measured runs (see --validate).
"""
import argparse
import os
import sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras

DATA = "data/matrix_city/aerial/train/block_all"
FIELDS = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ("opacity", "f4"),
    ("scale_0", "f4"), ("scale_1", "f4"),
    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
]


def block_camera_indices(data, block, block_dim):
    names, cams = load_test_cameras(data, 1.2)
    by, bx = block // block_dim[0], block % block_dim[0]
    want = {l.strip() for l in open(os.path.join(
        data, "partition", f"partitions-dim_{block_dim[0]}_{block_dim[1]}_visibility_0.08",
        f"{bx:03d}_{by:03d}.txt")) if l.strip()}
    return [i for i, n in enumerate(names) if n in want], cams


def project(cam, xyz, scale_world, footprint):
    """-> (valid_mask, cell_index, camera_space_z, n_cells, median_radius_px)

    PRIMITIVES HAVE AREA. A first version binned each point into one pixel and predicted 99.7%
    survival against a measured 27.4%: with 1.3M points on 1.44M pixels almost everyone gets their
    own pixel, so nothing occludes anything. The real Gaussians carry scale 0.03, which at the
    typical depth 7.39 and fx~1100 projects to a ~4.5 px radius -- ~64 px each, i.e. ~58x overdraw,
    which is the right order for the observed cull.

    So bin at the FOOTPRINT scale instead of the pixel scale: cell size = footprint * median
    projected radius. `footprint` is the one free parameter and is calibrated against the measured
    survivor count (see --footprint).
    """
    R = cam.R.numpy().astype(np.float64)
    t = cam.T.numpy().astype(np.float64)
    fx, fy, cx, cy = (float(getattr(cam, k)) for k in ("fx", "fy", "cx", "cy"))
    H, W = int(cam.height), int(cam.width)
    Xc = (R @ xyz.T).T + t
    z = Xc[:, 2]
    ok = z > 0.05
    r_px = np.zeros(len(xyz))
    r_px[ok] = fx * scale_world[ok] / z[ok]
    med_r = float(np.median(r_px[ok])) if ok.any() else 1.0
    cell = max(footprint * med_r, 1.0)
    gw, gh = int(np.ceil(W / cell)), int(np.ceil(H / cell))
    u = np.full(len(xyz), -1, np.int64)
    v = np.full(len(xyz), -1, np.int64)
    u[ok] = (fx * Xc[ok, 0] / z[ok] + cx) // cell
    v[ok] = (fy * Xc[ok, 1] / z[ok] + cy) // cell
    ok &= (u >= 0) & (u < gw) & (v >= 0) & (v < gh)
    return ok, v * gw + u, z, gw * gh, med_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_ply", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--views", type=int, default=80, help="cameras to sample from the block's set")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="a fill point counts as a shell duplicate only if the anchor behind it is "
                         "within this distance. Bounded by the pseudo-depth error (0.0795) so that "
                         "genuinely nearer surfaces are preserved")
    ap.add_argument("--vote", type=float, default=0.5,
                    help="remove a fill point when this fraction of the views that see it agree")
    ap.add_argument("--footprint", type=float, default=2.0,
                    help="cell size = footprint x median projected radius. The one free parameter; "
                         "calibrate against a measured start-trim survivor count (b12: 360,817 "
                         "survived from a 1,316,223-point graded init)")
    ap.add_argument("--predict_only", action="store_true",
                    help="only report predicted trim survival; write nothing")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    v = PlyData.read(a.seed_ply)["vertex"]
    cols = {n: np.asarray(v[n], dtype=np.float32) for n, _ in FIELDS}
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    wpath = a.seed_ply + ".w.npy"
    w = np.load(wpath) if os.path.exists(wpath) else np.zeros(len(xyz), np.float32)
    anchor = w > 0.5
    n = len(xyz)
    print(f"[種子] {a.seed_ply}\n  N={n:,}  錨點 {int(anchor.sum()):,} ({100 * anchor.mean():.1f}%)")

    pool, cams = block_camera_indices(a.data, a.block, a.block_dim)
    probe = [pool[j] for j in np.linspace(0, len(pool) - 1, min(a.views, len(pool))).astype(int)]
    print(f"[相機] 該塊 {len(pool)} 台，取樣 {len(probe)} 台")

    front_any = np.zeros(n, bool)      # 在任一視角是最前面的（＝trim 會留它）
    seen_cnt = np.zeros(n, np.int32)
    kill_cnt = np.zeros(n, np.int32)

    scale_world = np.exp(cols["scale_0"].astype(np.float64))
    med_r_all = []
    for j, ci in enumerate(probe):
        cam = cams[ci]
        ok, pix, z, ncell, med_r = project(cam, xyz, scale_world, a.footprint)
        med_r_all.append(med_r)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            continue
        seen_cnt[idx] += 1

        # 全體深度緩衝：每個格子最前面的那顆
        buf = np.full(ncell, np.inf)
        np.minimum.at(buf, pix[idx], z[idx])
        front_any |= ok & (z <= buf[np.clip(pix, 0, ncell - 1)] + 1e-9)

        # 錨點深度緩衝：每個像素最前面的錨點
        ai = idx[anchor[idx]]
        if len(ai) == 0:
            continue
        abuf = np.full(ncell, np.inf)
        np.minimum.at(abuf, pix[ai], z[ai])

        # 填充點：同像素、在錨點前面、且距離在 margin 內 ⇒ 同一面的殼複本
        fi = idx[~anchor[idx]]
        za = abuf[pix[fi]]
        gap = za - z[fi]
        kill_cnt[fi[(gap > 1e-6) & (gap < a.margin)]] += 1

        if (j + 1) % 20 == 0:
            print(f"  ...{j + 1}/{len(probe)}")

    live = seen_cnt > 0
    print(f"\n[足跡] 投影半徑中位 {np.median(med_r_all):.2f} px  →  格子邊長 "
          f"{max(a.footprint * np.median(med_r_all), 1.0):.2f} px  (footprint={a.footprint})")
    print(f"[預測 trim 留存]（＝在任一視角是最前面的）")
    print(f"  整體 {100 * front_any[live].mean():5.1f}%   "
          f"錨點 {100 * front_any[live & anchor].mean():5.1f}%   "
          f"填充 {100 * front_any[live & ~anchor].mean():5.1f}%")
    print(f"  預測存活數 ≈ {int(front_any.sum()):,}")
    print(f"  ⚠ 這是幾何近似（點視為無面積），與實際 trim 的 transmittance 判準不同；"
          f"用途是同口徑比較不同 init，不是預測絕對值")

    frac = np.where(seen_cnt > 0, kill_cnt / np.maximum(seen_cnt, 1), 0.0)
    kill = (~anchor) & (seen_cnt > 0) & (frac >= a.vote)
    print(f"\n[射線清除] margin={a.margin}  vote>={a.vote:.0%}")
    print(f"  判定為偽前層的填充點 {int(kill.sum()):,}  ({100 * kill.mean():.1f}% of all)")

    keep = ~kill
    print(f"  清除後 N={int(keep.sum()):,}  錨點佔 {100 * anchor[keep].mean():.1f}%")
    # 清除後重新預測：錨點應該大幅浮到前面
    print(f"  清除後錨點在『被保留集合』中的預測留存 "
          f"{100 * front_any[keep & anchor].sum() / max(int((keep & anchor).sum()), 1):5.1f}%"
          f"  （未重算深度緩衝，重跑本工具於輸出檔可得真值）")

    if a.predict_only or not a.out:
        return
    el = np.empty(int(keep.sum()), dtype=FIELDS)
    for name, _ in FIELDS:
        el[name] = cols[name][keep]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(a.out)
    np.save(a.out + ".w.npy", w[keep].astype(np.float32))
    tp = a.seed_ply + ".track.npy"
    if os.path.exists(tp):
        np.save(a.out + ".track.npy", np.load(tp)[keep])
    print(f"\n[out] {a.out}  ({int(keep.sum()):,} 顆)")


if __name__ == "__main__":
    main()
