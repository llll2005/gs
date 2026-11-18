#!/usr/bin/env python
"""★ 11-18 港口之間的浮塊診斷（使用者 viewer：b3 碼頭旁有深色浮塊往上冒，A＋B 與 v5 都還在）。CPU。
對一個塊的 ckpt：不在海平面上的自由顆粒（o > 0.5），中心投到海平面落在海面格（第 0 層＋遠海層）上的，依
  高度 h（離平面，以格邊長 g 為單位）× 離可靠 SfM 的距離（以 g 為單位）分桶，報顆數與 Σ o·面積；
  浮塊收回（seafl）只收 h > 10 g 且離 SfM > 10 g => 看殘留落在哪一格（太低？離碼頭 SfM 太近？）。
另報殘留顆粒的位置分群（世界 xy 以 4 g 網格聚，列最大的幾群）與尺度，方便在 viewer 對位。
用法（lab）：python tools/probe/harbor_floaters_1118.py --ckpt <ckpt> --obj data/.../sea_object_v5/block_3.npz
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--obj", required=True)
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
a = ap.parse_args()
import torch
from scipy.spatial import cKDTree
os.environ.setdefault("CITYGS_SFM_FAST", "1")
from internal.utils import colmap_fast
z = np.load(a.obj); n = z["n"]; c = float(z["c"]); org = z["origin"]; e1 = z["e1"]; e2 = z["e2"]; g = float(z["g"]); base = int(z["base"]); keys = np.sort(z["keys"])
FAR = [(g * 2 ** int(L), np.sort(z[f"far_keys_{int(L)}"])) for L in (z["far_L"].tolist() if "far_L" in z.files else [])]
d = colmap_fast.points3D_arrays(os.path.join(a.D, "sparse/0/points3D.bin"))
S = d["xyz"][(np.diff(d["off"]) >= 3) & (d["err"] < 1.5)].astype(np.float64); tr = cKDTree(S)
def insea(X):
    q = X - org; i = np.floor(q @ e1 / g).astype(np.int64); j = np.floor(q @ e2 / g).astype(np.int64); m = np.isin(i * base + j, keys)
    for gL, kL in FAR:
        i = np.floor(q @ e1 / gL).astype(np.int64); j = np.floor(q @ e2 / gL).astype(np.int64); m |= np.isin(i * base + j, kL)
    return m
for f in a.ckpt:
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]; p = "gaussian_model.gaussians."
    m = sd[p + "means"].double().numpy(); o = torch.sigmoid(sd[p + "opacities"].float()).reshape(-1).numpy(); s = torch.exp(sd[p + "scales"].float()).numpy()
    h = m @ n - c
    free = (np.abs(h) > 2e-4) & (o > 0.5)
    P = m - np.outer(h, n); on = free & (h > g) & insea(P)
    dist, _ = tr.query(m[on]); hu = h[on] / g; du = dist / g; area = o[on] * 2 * np.pi * s[on, 0] * s[on, 1] * 1e4
    print(f"== {f}\n   海面格上方 > 1 g（g＝{g * 100:.2f} m）的自由不透明顆粒 {int(on.sum()):,}（Σo·面積 {area.sum() / 1e3:,.1f}k m²）", flush=True)
    hb = [1, 3, 10, 30, 1e9]; db = [0, 3, 10, 30, 1e9]
    print("   高度（g）\\ 離 SfM（g）  " + "  ".join(f"{db[k]:g}～{db[k + 1]:g}" if db[k + 1] < 1e8 else f">{db[k]:g}" for k in range(4)))
    for a_ in range(4):
        row = []
        for b_ in range(4):
            mm = (hu >= hb[a_]) & (hu < hb[a_ + 1]) & (du >= db[b_]) & (du < db[b_ + 1])
            row.append(f"{int(mm.sum()):>6,}／{area[mm].sum() / 1e3:>6.1f}k")
        print(f"   {hb[a_]:g}～{hb[a_ + 1]:g}".ljust(14) + "  ".join(row) if hb[a_ + 1] < 1e8 else f"   >{hb[a_]:g}".ljust(14) + "  ".join(row))
    bel = free & (h < -0.5 * g) & insea(P)                  # ★ 11-18 水面下（v6 viewer 的暗色楔形）
    ab = o[bel] * 2 * np.pi * s[bel, 0] * s[bel, 1] * 1e4
    lab_ = s[bel].max(1) > 5 * 0.01
    print(f"   水面下 > 0.5 g 的自由不透明顆粒 {int(bel.sum()):,}（Σo·面積 {ab.sum() / 1e3:,.1f}k m²）｜其中長軸 > 5 m 的 {int(lab_.sum()):,}（{ab[lab_].sum() / 1e3:,.1f}k m²）", flush=True)
    caught = (hu > 10) & (du > 10)
    print(f"   浮塊收回會收的（h > 10 g 且離 SfM > 10 g）：{int(caught.sum()):,}｜殘留（其他）：{int((~caught).sum()):,}（Σo·面積 {area[~caught].sum() / 1e3:,.1f}k m²）", flush=True)
    R = m[on][~caught]
    if len(R):
        cell = np.floor(R[:, :2] / (4 * g)).astype(np.int64); u_, inv, cn = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
        top = np.argsort(-cn)[:6]
        for t_ in top:
            mm = inv == t_
            print(f"   殘留群：世界 xy ≈ ({R[mm, 0].mean():.3f}, {R[mm, 1].mean():.3f})｜{int(cn[t_]):,} 顆｜高度中位 {np.median(hu[~caught][mm]):.1f} g｜離 SfM 中位 {np.median(du[~caught][mm]):.1f} g｜長軸中位 {np.median(s[on][~caught][mm].max(1)) * 100:.2f} m", flush=True)
print("DONE harbor_floaters", flush=True)
