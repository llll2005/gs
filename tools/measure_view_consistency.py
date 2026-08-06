"""Which primitives only work from a few angles?

A floater is not a primitive with low contribution -- it is one whose contribution is CONCENTRATED.
It was placed to explain a handful of training views and it does explain them, so every criterion we
have tried scores it as valuable: dust harvesting, v/c pruning, opacity_reg, condensation annealing
all rank by contribution SUMMED over views, and summing is exactly what destroys the signal. That is
why none of them removed floaters, and why 55-69% of the primitives in every model we have measured
still sit above the local ground.

So keep the view axis instead of collapsing it. Per primitive, over K views:

    n_views       how many views it contributes to at all
    concentration max_view / sum_views   -- 1.0 means one view carries everything,
                                            1/K means perfectly spread
    gini          inequality of the per-view contribution vector

Real surface is seen from many angles and contributes broadly. A floater spikes in a few.

This needs no held-out data, which is the point: using the official test set to steer pruning would
consume the only clean measurement we have. A train/val split (split_mode=experiment) would be
legitimate but costs images and a retrain; multi-view concentration is computable from the training
views the model already has.

The tool VALIDATES the metric rather than assuming it: it checks whether primitives that sit above
the local ground surface -- an independent geometric signal, built from the COLMAP points, not from
contribution at all -- really do score as more concentrated. If the two disagree, concentration is
not finding floaters and the idea is wrong.

Memory: contributions are accumulated as running statistics, never as an N x K matrix (1.8M x 284
floats would be 2 GB).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.colmap import read_points3D_binary
from internal.utils.gaussian_model_loader import GaussianModelLoader

DATA = "data/matrix_city/aerial/train/block_all"


def local_ground(data, cell=0.05):
    """Height of the real surface per (x, y) cell, from the triangulated points.

    A global threshold does not work: the scene's p99 is 2.14 (the tallest roofs) while the median
    ground is 0.20, so anything hovering at z=1.0 -- well above the street, right between the camera
    and the ground -- is missed entirely. That mistake reported 0.1% floaters where the local
    measure finds 64%.
    """
    P = np.array([p.xyz for p in read_points3D_binary(f"{data}/sparse/0/points3D.bin").values()])
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    P = P[np.all((P > lo) & (P < hi), 1)]
    c = ((P[:, :2] - lo[:2]) / cell).astype(np.int64)
    key = c[:, 0] * 100_000 + c[:, 1]
    order = np.argsort(key)
    key_s, z_s = key[order], P[order, 2]
    uk, start = np.unique(key_s, return_index=True)
    ends = list(start[1:]) + [len(z_s)]
    tops = np.array([np.percentile(z_s[a:b], 95) for a, b in zip(start, ends)])
    return dict(zip(uk.tolist(), tops.tolist())), lo, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--block_dim", type=int, nargs=2, default=None)
    ap.add_argument("--views", type=int, default=60, help="取樣視角數；越多越準但越慢")
    ap.add_argument("--eps", type=float, default=1e-6, help="低於此值視為該視角沒有貢獻")
    ap.add_argument("--air_margin", type=float, default=0.2, help="高出局部地表多少才算懸空")
    ap.add_argument("--save", default=None, help="把逐顆統計存成 .npz 供後續剪枝用")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    dp = ck.get("datamodule_hyper_parameters", {}).get("parser")
    blk = a.block if a.block is not None else getattr(dp, "block_id", None)
    bdim = a.block_dim or getattr(dp, "block_dim", None) or [5, 5]

    names, cams = load_test_cameras(a.data, 1.2)
    if blk is None:
        pool = list(range(len(names)))
    else:
        by, bx = blk // bdim[0], blk % bdim[0]
        want = {l.strip() for l in open(os.path.join(
            a.data, "partition", f"partitions-dim_{bdim[0]}_{bdim[1]}_visibility_0.08",
            f"{bx:03d}_{by:03d}.txt")) if l.strip()}
        pool = [i for i, n in enumerate(names) if n in want]
    probe = [pool[j] for j in np.linspace(0, len(pool) - 1, min(a.views, len(pool))).astype(int)]

    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        a.ckpt, "cuda", eval_mode=True)
    n = model.get_xyz.shape[0]
    print(f"[模型] N={n:,}  block={blk}  取樣 {len(probe)} 視角")

    total = torch.zeros(n, device="cuda", dtype=torch.float64)
    sq = torch.zeros(n, device="cuda", dtype=torch.float64)
    peak = torch.zeros(n, device="cuda")
    seen = torch.zeros(n, device="cuda", dtype=torch.int32)
    bg = torch.zeros(3, device="cuda")
    with torch.no_grad():
        for j, i in enumerate(probe):
            out = renderer(cams[i].to_device("cuda"), model, bg_color=bg, record_transmittance=True)
            c = (out if torch.is_tensor(out) else out["transmittance"]).float()
            total += c.double()
            sq += (c.double() ** 2)
            peak = torch.maximum(peak, c)
            seen += (c > a.eps).int()
            if (j + 1) % 20 == 0:
                print(f"  ...{j + 1}/{len(probe)}")

    tot = total.cpu().numpy()
    conc = (peak.cpu().numpy() / np.maximum(tot, 1e-12))     # 1.0 = 全部來自單一視角
    nv = seen.cpu().numpy()
    K = len(probe)

    # independent geometric signal: is this primitive above the local ground?
    H, lo, cell = local_ground(a.data)
    m = model.get_xyz.detach().cpu().numpy()
    c2 = ((m[:, :2] - lo[:2]) / cell).astype(np.int64)
    key = c2[:, 0] * 100_000 + c2[:, 1]
    gh = np.array([H.get(int(k), np.nan) for k in key])
    known = ~np.isnan(gh)
    air = known & (m[:, 2] > gh + a.air_margin)
    ground = known & ~air

    live = tot > 1e-12
    print(f"\n[覆蓋] 有貢獻的顆粒 {100 * live.mean():.1f}%   "
          f"可判定地表高度的 {100 * known.mean():.1f}%")
    print(f"[幾何] 高於局部地表 >{a.air_margin} 的佔 {100 * air[known].mean():.1f}%")

    print(f"\n{'群組':<22}{'n':>10}{'視角數中位':>11}{'集中度中位':>11}{'貢獻總量中位':>13}")
    for lab, msk in [("懸空（獨立幾何判定）", air & live), ("貼地（獨立幾何判定）", ground & live)]:
        if msk.sum() == 0:
            continue
        print(f"{lab:<22}{int(msk.sum()):>10,}{np.median(nv[msk]):>11.0f}"
              f"{np.median(conc[msk]):>11.3f}{np.median(tot[msk]):>13.4g}")

    if (air & live).sum() and (ground & live).sum():
        ca, cg = np.median(conc[air & live]), np.median(conc[ground & live])
        va, vg = np.median(nv[air & live]), np.median(nv[ground & live])
        print(f"\n[判定] 懸空顆粒的集中度 {'高於' if ca > cg else '低於'}貼地顆粒"
              f"（{ca:.3f} vs {cg:.3f}，比值 {ca / max(cg, 1e-9):.2f}×）")
        print(f"       懸空顆粒出現在 {va:.0f}/{K} 個視角，貼地在 {vg:.0f}/{K}")
        ok = ca > cg * 1.3 or va < vg * 0.7
        print(f"       ⇒ {'✅ 集中度確實能辨識 floater，可拿來當剪枝判準' if ok else '❌ 兩者分不開，這個判準不成立'}")

        # what a concentration-based cut would actually remove
        for q in (10, 20, 30):
            thr = np.percentile(conc[live], 100 - q)
            cut = live & (conc >= thr)
            print(f"       若剪掉集中度最高的 {q}%：命中懸空 {100 * air[cut].mean():.1f}%"
                  f"（隨機的話應為 {100 * air[live].mean():.1f}%）")

    if a.save:
        np.savez(a.save, total=tot, concentration=conc, n_views=nv, air=air, K=K)
        print(f"\n[out] {a.save}")


if __name__ == "__main__":
    main()
