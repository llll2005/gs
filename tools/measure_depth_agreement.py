"""Do the primitives themselves sit where the depth map says the surface is?

The depth loss constrains a STATISTIC OF THE RAY, not the positions. `surf_depth` is the median
depth of the accumulated alpha, and many arrangements produce the same median: one primitive at the
right depth, or a faint one in front and a solid one behind, or a whole cloud whose median happens
to land correctly. So the loss can be satisfied while the primitives are scattered -- which is
exactly what we measure on our own models:

    rendered depth slope  1.082   correct at the ray level
    distance to surface    29x    wrong at the primitive level

This tool checks the primitive level directly. Project each primitive into each view, read the
pseudo-depth map at the pixel it lands on, and compare against the primitive's own camera-space z.
No rendering, so it does not compete with a training job for the card.

The pseudo-depth is good enough to serve as a position reference: measured 2026-08-04 against the
COLMAP points, the depth it implies is unbiased (median +0.18%) with a median error of 8.6%.

AGGREGATION IS PER-PRIMITIVE MINIMUM OVER VIEWS, not the mean. A depth map only records the FIRST
visible surface along each ray, so a primitive that legitimately represents an occluded wall
disagrees in every view where something is in front of it. Taking the best view asks the right
question: is there ANY view in which this primitive sits on the recorded surface?

The output that matters is the correlation with distance-to-surface. If disagreement tracks
distance, this is a pruning criterion that needs no held-out data and, unlike contribution, is not
fooled by a floater that explains its training views.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras

DATA = "data/matrix_city/aerial/train/block_all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--block_dim", type=int, nargs=2, default=None)
    ap.add_argument("--views", type=int, default=40)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    xyz = ck["state_dict"]["gaussian_model.gaussians.means"].float().numpy().astype(np.float64)
    dp = ck.get("datamodule_hyper_parameters", {}).get("parser")
    blk = a.block if a.block is not None else getattr(dp, "block_id", None)
    bdim = a.block_dim or getattr(dp, "block_dim", None) or [5, 5]
    n = len(xyz)

    import json
    scales = json.load(open(f"{a.data}/estimated_depth_scales.json"))
    names, cams = load_test_cameras(a.data, 1.2)
    if blk is None:
        pool = list(range(len(names)))
    else:
        by, bx = blk // bdim[0], blk % bdim[0]
        want = {l.strip() for l in open(os.path.join(
            a.data, "partition", f"partitions-dim_{bdim[0]}_{bdim[1]}_visibility_0.08",
            f"{bx:03d}_{by:03d}.txt")) if l.strip()}
        pool = [i for i, nm in enumerate(names) if nm in want]
    probe = [pool[j] for j in np.linspace(0, len(pool) - 1, min(a.views, len(pool))).astype(int)]
    print(f"[模型] N={n:,}  block={blk}  取樣 {len(probe)} 視角")

    best = np.full(n, np.inf)      # 跨視角最佳（最小）相對不一致度
    seen = np.zeros(n, dtype=np.int32)
    for j, i in enumerate(probe):
        nm = names[i]
        f = f"{a.data}/estimated_depths/{int(nm[:-4]):06d}.png.npy"
        if nm not in scales or not os.path.exists(f):
            continue
        s = scales[nm]
        inv = np.load(f) * s["scale"] + s["offset"]          # 逆深度（disparity）
        cam = cams[i]
        R, t = cam.R.numpy().astype(np.float64), cam.T.numpy().astype(np.float64)
        fx, fy, cx, cy = (float(getattr(cam, k)) for k in ("fx", "fy", "cx", "cy"))
        Xc = (R @ xyz.T).T + t
        z = Xc[:, 2]
        m = z > 0.05
        u = (fx * Xc[m, 0] / z[m] + cx).astype(np.int64)
        v = (fy * Xc[m, 1] / z[m] + cy).astype(np.int64)
        H, W = inv.shape
        k = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(m)[0][k]
        gt_inv = inv[v[k], u[k]]
        ok = gt_inv > 1e-6
        idx, gt_inv = idx[ok], gt_inv[ok]
        z_ref = 1.0 / gt_inv                                  # 深度圖說的表面深度
        rel = np.abs(z[idx] - z_ref) / z_ref
        np.minimum.at(best, idx, rel)
        seen[idx] += 1
        if (j + 1) % 10 == 0:
            print(f"  ...{j + 1}/{len(probe)}")

    live = seen > 0
    b = best[live]
    print(f"\n[覆蓋] 至少投影進一個視角的粒子 {100 * live.mean():.1f}%")
    print(f"[不一致度] 跨視角最佳值 p10/p50/p90 = "
          f"{np.percentile(b, 10):.3f} / {np.percentile(b, 50):.3f} / {np.percentile(b, 90):.3f}")
    print(f"           <10% 的佔 {100 * (b < 0.10).mean():.1f}%   "
          f"<25% 的佔 {100 * (b < 0.25).mean():.1f}%   "
          f">50% 的佔 {100 * (b > 0.50).mean():.1f}%")
    print(f"           （深度圖自身的誤差中位是 8.6%，所以 <10% 已接近其解析極限）")

    from floater_label import surface_distance
    d, unit = surface_distance(xyz, a.data)
    dr = d / unit
    print(f"\n[離表面] 中位 {np.median(dr):.1f}×（單位＝真實點彼此的中位距離 {unit:.4f}）")

    r = float(np.corrcoef(b, dr[live])[0, 1])
    print(f"\n[相關] 深度不一致度 vs 離表面距離：r = {r:+.3f}")
    print(f"{'不一致度分位':>14}{'n':>10}{'離表面中位':>12}")
    qs = np.percentile(b, [20, 40, 60, 80])
    groups = [("最一致 20%", b <= qs[0]), ("20-40%", (b > qs[0]) & (b <= qs[1])),
              ("40-60%", (b > qs[1]) & (b <= qs[2])), ("60-80%", (b > qs[2]) & (b <= qs[3])),
              ("最不一致 20%", b > qs[3])]
    sub = dr[live]
    for lab, msk in groups:
        print(f"{lab:>14}{int(msk.sum()):>10,}{np.median(sub[msk]):>11.1f}×")
    lo, hi = np.median(sub[groups[0][1]]), np.median(sub[groups[-1][1]])
    print(f"\n[判定] 最不一致的 20% 離表面 {hi / max(lo, 1e-9):.1f}× 於最一致的 20%")
    print(f"       ⇒ {'✅ 深度不一致度可當位置層級的判準' if (r > 0.3 or hi > lo * 2) else '❌ 兩者無關，深度圖對位置沒有約束力'}")

    if a.save:
        np.savez(a.save, best=best, seen=seen, surf_dist_rel=dr)
        print(f"\n[out] {a.save}")


if __name__ == "__main__":
    main()
