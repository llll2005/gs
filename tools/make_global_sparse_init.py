"""Build a sparse GLOBAL depth-init, matching the official coarse's density.

WHY
---
The coarse-vs-depth-init A/B confounds three things at once:

    geometry source   SfM triangulation + 30k steps of global photometric training
                      vs monocular back-projection, never trained
    density           499,682 points globally (~100-200k in a block)
                      vs 1,699,313 in one block -- roughly 10x
    appearance        the coarse ckpt carries trained sh2; the depth PLY carries DC only

This removes the density term. It also tests a prediction worth having on record: the start trim
keeps a roughly FIXED number of primitives -- 360,817 from a 1,316,223-point init and 363,190 from
a 991,841-point one -- so if feeding it 500k lands in the same place, init DENSITY does not matter
and only its geometry does.

WHAT THIS IS NOT
----------------
Not a coarse model. A coarse is "SfM points + 30k steps of global training", and the training is
where it earns its positions and its appearance. This is only the untrained cloud at coarse
density. The matching object would be "train a global model starting from depth-init", which is a
separate and much more expensive thing.

HOW
---
The per-block PLYs already exist and cover the scene many times over (b12's cloud spans ~6x its own
partition AABB, so neighbouring blocks overlap heavily). Concatenating them and voxel-downsampling
to the target count reproduces what a global run would have produced far more cheaply than
re-reading 5621 depth maps. Voxel dedup keeps one point per cell, so the overlap collapses.

The voxel size is solved for, not guessed: bisect on the cell size until the surviving count lands
within tolerance of the target.
"""
import argparse
import glob
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIELDS = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ("opacity", "f4"),
    ("scale_0", "f4"), ("scale_1", "f4"),
    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/matrix_city/aerial/train/block_all/depth_init_4x4")
    ap.add_argument("--target", type=int, default=499_682,
                    help="point count to match; default = official_coarse_sh2's")
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--rescale_to_spacing", type=float, default=None,
                    help="after downsampling the cloud is sparser, so the inherited scale (sized "
                         "for the dense grid) leaves gaps. Set the 2D scale to this multiple of "
                         "the new median spacing; 0.33 reproduces the coarse's ~3x coverage. "
                         "Omit to keep the original scales")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src_dir, "block_*.ply")))
    print(f"[來源] {a.src_dir}  {len(files)} 個分塊 PLY")
    cols = {n: [] for n, _ in FIELDS}
    for f in files:
        v = PlyData.read(f)["vertex"]
        for n, _ in FIELDS:
            cols[n].append(np.asarray(v[n], dtype=np.float32))
    cols = {n: np.concatenate(v) for n, v in cols.items()}
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(np.float64)
    print(f"[合併] {len(xyz):,} 顆（分塊重疊尚未去除）")

    # solve for the voxel size that lands on the target count
    lo, hi = 1e-3, 5.0
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        n = len(np.unique(np.floor(xyz / mid).astype(np.int64), axis=0))
        if abs(n - a.target) <= a.tol * a.target:
            break
        lo, hi = (mid, hi) if n > a.target else (lo, mid)
    vox = mid
    key = np.floor(xyz / vox).astype(np.int64)
    _, keep = np.unique(key, axis=0, return_index=True)
    print(f"[體素] 解出 {vox:.5f}  →  {len(keep):,} 顆（目標 {a.target:,}）")

    el = np.empty(len(keep), dtype=FIELDS)
    for name, _ in FIELDS:
        el[name] = cols[name][keep]

    if a.rescale_to_spacing is not None:
        from scipy.spatial import cKDTree
        sub_xyz = xyz[keep]
        tree = cKDTree(sub_xyz)
        rng = np.random.default_rng(0)
        s = sub_xyz[rng.choice(len(sub_xyz), min(100_000, len(sub_xyz)), replace=False)]
        spacing = float(np.median(tree.query(s, k=2)[0][:, 1]))
        new_scale = np.log(a.rescale_to_spacing * spacing).astype(np.float32)
        old = float(np.median(np.exp(el["scale_0"].astype(np.float64))))
        el["scale_0"] = el["scale_1"] = new_scale
        print(f"[尺寸] 新間距 {spacing:.5f} → scale {np.exp(new_scale):.5f}"
              f"（原 {old:.5f}）  覆蓋倍數 {3 * np.exp(new_scale) / spacing:.2f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(a.out)
    print(f"\n[out] {a.out}  ({len(keep):,} 顆)")


if __name__ == "__main__":
    main()
