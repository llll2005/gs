"""Geometry-side numbers that the photometric metrics cannot see. CPU only, no rendering.

Removing the two geometric priors buys photometric score and pays in geometry, and PSNR/SSIM/LPIPS
/texture ratio show only the first half of that trade:

    sh3_viewdep_refix  (both priors)     25.435   floaters 0.94%
    sh3_nonormal       (no normal)       25.828   floaters 1.19%   +27%
    sh3_nodepth        (no depth loss)   25.675   floaters 1.45%   +54%

So report this alongside every recipe change, or "improvements" that are really trades stay hidden.

FLOATERS -- primitives sitting within `near` of a training camera centre while still opaque enough
to render. A primitive that close projects a huge footprint, so a handful of them dominate whole
regions of the image.
⚠ It counts distance to the nearest camera CENTRE, so it includes primitives behind the camera as
  well as in front. Fine as a relative measure across runs on the same cameras; not an absolute
  "how many artefacts will I see".

SPREAD -- median distance from each primitive to its k nearest neighbours, as a proxy for whether
the cloud has collapsed onto surfaces or diffused into a haze. Reported per run; only comparable
between runs on the same block.
"""
import argparse
import glob
import re

import numpy as np
import torch
from scipy.spatial import cKDTree


def cameras_of(data, block, block_dim):
    import os, sys
    sys.path.insert(0, "."); sys.path.insert(0, "tools")
    from eval_official_test import load_test_cameras
    names, cams = load_test_cameras(data, 1.2)
    by, bx = block // block_dim[0], block % block_dim[0]
    p = os.path.join(data, "partition",
                     f"partitions-dim_{block_dim[0]}_{block_dim[1]}_visibility_0.08",
                     f"{bx:03d}_{by:03d}.txt")
    want = {l.strip() for l in open(p) if l.strip()}
    idx = [i for i, n in enumerate(names) if n in want]
    return np.stack([(cams[i].world_to_camera.T.inverse()[:3, 3]).cpu().numpy() for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--near", type=float, default=0.5, help="世界單位；距相機中心多近算 floater")
    ap.add_argument("--min_opacity", type=float, default=0.1)
    ap.add_argument("--knn", type=int, default=8)
    a = ap.parse_args()

    C = cameras_of(a.data, a.block, a.block_dim)
    tree = cKDTree(C)
    print(f"[相機] {len(C)} 台   [floater 判準] 距相機 <{a.near} 且 opacity>{a.min_opacity}")
    print(f"\n{'run':<24}{'N':>11}{'floater':>10}{'佔比':>9}{'其不透明':>10}{'鄰距中位':>10}")
    for run in a.runs:
        f = glob.glob(f"outputs/{run}/blocks/block_{a.block}/checkpoints/*.ckpt")
        if not f:
            print(f"{run[:23]:<24}  無 ckpt")
            continue
        f.sort(key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
        sd = torch.load(f[-1], map_location="cpu")["state_dict"]
        xyz = sd[[k for k in sd if k.endswith("means")][0]].numpy()
        op = torch.sigmoid(sd[[k for k in sd if k.endswith("opacities")][0]]).numpy().ravel()
        d, _ = tree.query(xyz)
        sel = (d < a.near) & (op > a.min_opacity)
        sub = xyz[np.random.default_rng(0).choice(len(xyz), min(50_000, len(xyz)), replace=False)]
        nn, _ = cKDTree(xyz).query(sub, k=a.knn + 1)
        print(f"{run[:23]:<24}{len(xyz):>11,}{int(sel.sum()):>10,}{100*sel.mean():>8.3f}%"
              f"{(op[sel].mean() if sel.any() else 0):>10.3f}{np.median(nn[:, 1:].mean(1)):>10.4f}")
    print("\n  floater 佔比上升 = 用幾何品質換光度分數。任何配方變更都要連同 PSNR/LPIPS/紋理比一起報。")


if __name__ == "__main__":
    main()
