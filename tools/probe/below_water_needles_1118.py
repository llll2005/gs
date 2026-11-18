#!/usr/bin/env python
"""★ 11-18 水面高度以下的面元（使用者 viewer：v9 平視時陸地底下衝上來的白色針狀面元）。只讀、CPU。
每個模型：水面（海面物件的平面）以下 > 0.5 格、o > 0.3 的自由面元，分「海面格上」與「陸地底下」；細長（長短軸比 > 5）與「細長且亮（平均 RGB > 0.4）」另計。
用法：python tools/probe/below_water_needles_1118.py --obj <sea npz> 名=<ckpt glob> ...
"""
import numpy as np, glob, sys, os, torch
sys.path.insert(0, ".")
os.environ.setdefault("CITYGS_SFM_FAST", "1")
from internal.utils import colmap_fast
from scipy.spatial import cKDTree
D = "data/matrix_city/aerial/train/block_all"
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--obj", required=True); ap.add_argument("models", nargs="+"); a = ap.parse_args()
z = np.load(a.obj); g = float(z["g"]); org = z["origin"]; e1 = z["e1"]; e2 = z["e2"]; c = float(z["c"]); keys = np.sort(z["keys"])
i = np.round(keys / 1000003).astype(np.int64); j = keys - i * 1000003
cen = org + np.outer((i + 0.5) * g, e1) + np.outer((j + 0.5) * g, e2); tc = cKDTree(cen[:, :2])
d = colmap_fast.points3D_arrays(f"{D}/sparse/0/points3D.bin"); S = d["xyz"][(np.diff(d["off"]) >= 3) & (d["err"] < 1.5)].astype(np.float64); ts = cKDTree(S)
for spec in a.models:
    nm, pat = spec.split("=", 1); f = sorted(x for x in glob.glob(pat) if "xyz" not in x)[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]; p = "gaussian_model.gaussians."
    m = sd[p + "means"].double().numpy(); o = torch.sigmoid(sd[p + "opacities"].float()).reshape(-1).numpy(); s = torch.exp(sd[p + "scales"].float()).numpy()
    dc = sd[p + "shs_dc"].float().numpy().reshape(len(m), -1)[:, :3]; rgb = np.clip(dc * 0.28209479 + 0.5, 0, 1); del sd
    s2 = np.sort(s[:, :2], 1) if s.shape[1] >= 2 else s
    el = s2[:, -1] / np.maximum(s2[:, 0], 1e-9)
    h = m[:, 2] - c; dsea, _ = tc.query(m[:, :2]); over = dsea < 0.7 * g
    base_ = (h < -0.5 * g) & (o > 0.3) & (np.abs(h) > 2e-4)
    land_ = base_ & ~over; br = rgb.mean(1) > 0.4
    print(f"{nm}｜水面下 > 0.5 格 o > 0.3：海面格上 {int((base_ & over).sum()):,}、陸地底下 {int(land_.sum()):,}（細長 {int((land_ & (el > 5)).sum()):,}、細長且亮 {int((land_ & (el > 5) & br).sum()):,}）", flush=True)
    below = (h < -0.5 * g) & (o > 0.3) & over
    needle = below & (el > 5)
    dS, _ = ts.query(m[needle]) if needle.any() else (np.zeros(0), None)
    print(f"{nm}｜水面下 > 0.5 格、在海面格上、o > 0.3：{int(below.sum()):,} 顆｜其中細長（長短軸比 > 5）{int(needle.sum()):,} 顆"
          + (f"：長軸中位 {np.median(s2[needle, -1]) * 100:.1f} m、p90 {np.percentile(s2[needle, -1], 90) * 100:.1f} m｜深度中位 {np.median(-h[needle]) * 100:.1f} m｜"
             f"亮度中位 {np.median(rgb[needle].mean(1)):.2f}（海面約 0.26）｜離 SfM 中位 {np.median(dS) * 100:.1f} m、> 3 格的 {100 * (dS > 3 * g).mean():.0f}%" if needle.any() else ""), flush=True)
print("DONE needles")
