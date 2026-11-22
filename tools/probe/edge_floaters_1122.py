#!/usr/bin/env python
"""★ 11-22 使用者 viewer：Mill-19 兩個場景邊界破碎、霧化、有脫離本體的塊、遠處大白浮塊、很多針狀浮塊；MatrixCity 的邊界是切平的。不訓練、CPU。
量（距離全用場景相對量：L＝可靠 SfM 平面範圍 p0.5～p99.5 的最長邊；不用公尺）：
  1. 分區盒外框 vs 可靠 SfM 範圍（外框被離群 SfM 撐大 => 合併時遠處的東西全留下）
  2. 顆粒按「平面位置」分：SfM 核心範圍內／外框內但核心外；每類的顆數、Σ o·面積、針狀比例
  3. 離最近可靠 SfM 點（3D）的距離：> 0.02L、> 0.05L 的不透明面積佔比（脫離本體／遠處浮塊）
  4. 高度：比 SfM 高度 p99.5 還高 > 0.02L 的不透明面積（天空浮塊）
針狀：o > 0.3 且 長軸／短軸 > 10。面積：π s1 s2（2DGS 兩個尺度）。
用法：python tools/probe/edge_floaters_1122.py --ckpt <merged> --data <資料集（有 sparse/0 與 partition）> [--tag 名稱]
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--tag", default="")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--nn_sample", type=int, default=400000)
a = ap.parse_args()
from internal.utils.colmap_fast import points3D_arrays
from scipy.spatial import cKDTree

P = torch.load(os.path.join(a.data, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
Rp = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
xy0 = P["partition_coordinates"]["xy"].double().numpy(); sz = P["scene_config"]["partition_size"].double().numpy()
sz = np.broadcast_to(sz, xy0.shape)
box_lo, box_hi = (xy0).min(0), (xy0 + sz).max(0)
S = points3D_arrays(os.path.join(a.data, "sparse/0/points3D.bin"))
xyz = S["xyz"]; err = S["err"]; ntrk = np.diff(S["off"]) if "off" in S else None
ok = (err < 1.5) & ((ntrk >= 3) if ntrk is not None else True)
X = xyz[ok] @ Rp.T
lo, hi = np.percentile(X[:, :2], 0.5, 0), np.percentile(X[:, :2], 99.5, 0); L = float((hi - lo).max())
zlo, zhi = np.percentile(X[:, 2], [0.5, 99.5])
allo, allhi = (xyz @ Rp.T)[:, :2].min(0), (xyz @ Rp.T)[:, :2].max(0)
print(f"=== {a.tag or a.ckpt}")
print(f"可靠 SfM {len(X):,}／全部 {len(xyz):,}｜核心範圍（p0.5～p99.5）{np.round(lo, 2)}～{np.round(hi, 2)}，最長邊 L={L:.3f}｜高度 p0.5～p99.5 {zlo:.3f}～{zhi:.3f}")
print(f"分區盒外框 {np.round(box_lo, 2)}～{np.round(box_hi, 2)} => 每邊比核心多出 {np.round((lo - box_lo) / L, 3)}／{np.round((box_hi - hi) / L, 3)} L｜全部 SfM（含不可靠）外框 {np.round(allo, 1)}～{np.round(allhi, 1)}")

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); sd = ck["state_dict"]
g = lambda k: sd[f"gaussian_model.gaussians.{k}"].float().numpy()
mu = g("means").astype(np.float64) @ Rp.T; o = 1 / (1 + np.exp(-g("opacities")[:, 0])); s = np.exp(g("scales"))
area = np.pi * s[:, 0] * s[:, 1]; w = o * area
needle = (o > 0.3) & (s.max(1) / np.maximum(s.min(1), 1e-12) > 10)
core = (mu[:, 0] >= lo[0]) & (mu[:, 0] <= hi[0]) & (mu[:, 1] >= lo[1]) & (mu[:, 1] <= hi[1])
N = len(mu); W = w.sum()
def row(name, m):
    return (f"  {name:<14} 顆數 {m.sum():>11,}（{100 * m.mean():5.1f}%）｜Σo·面積 {100 * w[m].sum() / W:5.1f}%｜針狀 {needle[m].sum():>9,}（佔該類 {100 * needle[m].mean():4.1f}%）"
            f"｜大面元（長軸 > 0.01L）{(s.max(1)[m] > 0.01 * L).sum():,}")
print(f"合併 N={N:,}"); print(row("SfM 核心內", core)); print(row("核心外", ~core))
# 離最近可靠 SfM 點的距離（3D，抽樣）
tree = cKDTree(X[np.random.default_rng(0).choice(len(X), min(len(X), 2_000_000), replace=False)])
idx = np.random.default_rng(1).choice(N, min(N, a.nn_sample), replace=False)
d, _ = tree.query(mu[idx], k=1, workers=-1)
for t in (0.02, 0.05, 0.1):
    m = d > t * L
    print(f"  離最近 SfM > {t:g}L：顆數 {100 * m.mean():5.2f}%｜Σo·面積 {100 * w[idx][m].sum() / w[idx].sum():5.2f}%｜其中核心外 {100 * (~core[idx][m]).mean():4.0f}%｜針狀 {100 * needle[idx][m].mean():4.1f}%")
hiz = mu[:, 2] > zhi + 0.02 * L; loz = mu[:, 2] < zlo - 0.02 * L
print(f"  比 SfM 高度 p99.5 還高 > 0.02L：顆數 {hiz.sum():,}｜Σo·面積 {100 * w[hiz].sum() / W:.2f}%｜針狀 {needle[hiz].sum():,}")
print(f"  比 SfM 高度 p0.5 還低 > 0.02L：顆數 {loz.sum():,}｜Σo·面積 {100 * w[loz].sum() / W:.2f}%")
print(f"  針狀合計 {needle.sum():,}（{100 * needle.mean():.2f}%）：核心內 {needle[core].sum():,}、核心外 {needle[~core].sum():,}")
