"""Do floaters sit over low-texture content? Premise test for the gradient-starvation account.

2026-08-09: near-camera floaters and the ghost buildings on water were unified under one mechanism.
A primitive close to the camera projects a huge screen footprint; over flat content it can explain
the mean colour cheaply and then receives almost no gradient telling it to move. Two things already
support that over an initialisation-based account:

  - uniform_60k_b12, whose init has no shape at all, still produces floaters (0.463% of primitives
    within 0.5 of a camera and opaque, vs depth-init's 1.802%). Init amplifies ~4x, it does not
    cause.
  - the same starvation explains the water ghosts, which appear in both arms.

The account's remaining prediction is testable without training: floaters should land over
low-texture regions more often than primitives in general. If they are spread evenly across texture
levels the mechanism is wrong, and the cross-view geometric consistency idea built on it should not
be implemented.

CPU only: it projects primitive centres and reads GT gradients, never renders.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--near", type=float, default=0.5, help="世界單位；距相機中心多近算 floater")
    ap.add_argument("--min_opacity", type=float, default=0.1)
    a = ap.parse_args()

    names, cams = load_test_cameras(a.data, 1.2)
    by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
    want = {l.strip() for l in open(os.path.join(
        a.data, "partition", f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08",
        f"{bx:03d}_{by:03d}.txt")) if l.strip()}
    files = sorted(f for f in os.listdir(f"{a.data}/images_1.2") if f.lower().endswith(".png"))
    idx = [i for i, n in enumerate(names) if n in want and 0 <= int(n[:-4]) - 1 < len(files)]
    picked = idx[:: max(1, len(idx) // a.views)][: a.views]
    W, H = int(cams.width[0]), int(cams.height[0])

    # per-view GT gradient map and camera centre, computed once and reused for every run
    views = []
    for i in picked:
        g = Image.open(f"{a.data}/images_1.2/{files[int(names[i][:-4]) - 1]}").convert("RGB")
        t = torch.from_numpy(np.array(g.resize((W, H), Image.LANCZOS), np.uint8)).float().permute(2, 0, 1) / 255.
        tex = (t[:, 1:] - t[:, :-1]).abs().mean(0)
        tex = torch.nn.functional.pad(tex[None, None], (0, 0, 0, 1))[0, 0]
        centre = cams[i].world_to_camera.T.inverse()[:3, 3]
        views.append((cams[i].full_projection, centre, tex))
    print(f"[視角] {len(views)} 張   [判準] 距相機 <{a.near} 且 opacity>{a.min_opacity}")

    print(f"\n{'run':<22}{'floater 紋理':>13}{'全體紋理':>11}{'混合比':>8}{'逐視角比':>10}{'floater 數':>11}")
    for run in a.runs:
        f = glob.glob(f"outputs/{run}/blocks/block_{a.block}/checkpoints/*.ckpt")
        if not f:
            print(f"{run:<22} 無 ckpt")
            continue
        f.sort(key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
        sd = torch.load(f[-1], map_location="cpu")["state_dict"]
        xyz = sd[[k for k in sd if k.endswith("means")][0]]
        op = torch.sigmoid(sd[[k for k in sd if k.endswith("opacities")][0]]).ravel()

        # Per-view normalisation. Pooling texture across views confounds "floaters prefer flat
        # REGIONS" with "floaters concentrate in flat VIEWS" -- and b12 is half water, so the
        # second is entirely possible on its own. Comparing each floater against the mean of the
        # image it lands in removes that.
        tex_float, tex_all, n_float, per_view = [], [], 0, []
        for proj, centre, tex in views:
            d = (xyz - centre).norm(dim=-1)
            p = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ proj.cpu()
            vis = p[:, 3] > 1e-6
            uv = torch.zeros(xyz.shape[0], 2)
            uv[vis] = p[vis, :2] / p[vis, 3:4]
            u = ((uv[:, 0] * 0.5 + 0.5) * W).long().clamp(0, W - 1)
            v = ((uv[:, 1] * 0.5 + 0.5) * H).long().clamp(0, H - 1)
            on = vis & (uv[:, 0].abs() < 1) & (uv[:, 1].abs() < 1)
            t_at = tex[v, u]
            fl = on & (d < a.near) & (op > a.min_opacity)
            if fl.any():
                tex_float.append(t_at[fl])
                n_float += int(fl.sum())
                per_view.append((float(t_at[fl].mean()), float(t_at[on].mean()), int(fl.sum())))
            tex_all.append(t_at[on])

        if not tex_float:
            print(f"{run[:21]:<22}  這些視角裡沒有符合判準的 floater")
            continue
        tf = float(torch.cat(tex_float).mean())
        ta = float(torch.cat(tex_all).mean())
        # weighted mean of the per-view ratios: each floater compared to ITS OWN image
        w = sum(n for _, _, n in per_view)
        rv = sum((f / max(a2, 1e-9)) * n for f, a2, n in per_view) / max(w, 1)
        print(f"{run[:21]:<22}{tf:>13.5f}{ta:>11.5f}{tf / max(ta, 1e-9):>8.2f}{rv:>10.2f}{n_float:>11,}")

    print("\n  ★逐視角比才是判準（混合比會把「集中在低紋理視角」誤判成「偏好低紋理區」）")
    print("  逐視角比 <1 ⇒ floater 確實偏好低紋理區，支持梯度飢餓假說")
    print("  逐視角比 ≈1 ⇒ 與紋理無關，假說錯，不要據此實作跨視角一致性約束")


if __name__ == "__main__":
    main()
