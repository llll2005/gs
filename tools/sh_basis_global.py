#!/usr/bin/env python
"""★ 2026-11-22 紀錄/實驗分析/40 §1.7：全場景共用的 SH 可觀測子空間基底（訓練前算一次、零成本；不訓練）。

每塊：盒內可靠 SfM 點（track >= 3、誤差 < 1.5，最多 --per_block 個）x 該塊訓練相機（視錐內才算），
M_i = Σ_相機 Y(d)Y(d)^T（16 x 16，Y＝3 階 SH 基底、d＝相機 -> 點），M̄_b = mean_i M_i / tr(M_i)；全域 M̄ = 各塊 M̄_b 的平均（每塊等權）。
renderer 的 sh_basis_k 用 M̄ 做保留 DC 的 Schur 補數基底（40 §1.4）。全場景共用一組 => 各塊的低維係數在同一個座標系，合併後可以直接低維存檔。
用法：python tools/sh_basis_global.py data/matrix_city/aerial/train/block_all --out configs/sh_basis_matrixcity_aerial.npz
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--per_block", type=int, default=30000)
ap.add_argument("--down", type=float, default=1.2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
from internal.utils.colmap import read_points3D_binary
from internal.utils.sh_utils import eval_sh_basis16
import eval_official_test as eot

P = torch.load(os.path.join(a.dataset, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
Rp = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = P["partition_coordinates"]["id"].numpy(); xys = P["partition_coordinates"]["xy"].double().numpy(); size = P["scene_config"]["partition_size"].double().numpy()
nx = int(P["scene_config"]["block_dim"][0])
pts = read_points3D_binary(os.path.join(a.dataset, "sparse/0/points3D.bin"))
xyz = np.stack([p.xyz for p in pts.values()])[np.array([len(p.image_ids) >= 3 and p.error < 1.5 for p in pts.values()])]
xy = (xyz @ Rp.T)[:, :2]
names, cams = eot.load_test_cameras(a.dataset, a.down)
pos = {n: i for i, n in enumerate(names)}
rng = np.random.default_rng(0)
Mbs, used = [], []
for k in range(len(ids)):
    b = int(ids[k][0] + nx * ids[k][1])
    fp = os.path.join(a.dataset, a.part, "%03d_%03d.txt" % (b % nx, b // nx))
    if not os.path.exists(fp):
        continue
    inb = np.where((xy[:, 0] >= xys[k][0]) & (xy[:, 0] < xys[k][0] + size[k][0]) & (xy[:, 1] >= xys[k][1]) & (xy[:, 1] < xys[k][1] + size[k][1]))[0]
    if len(inb) == 0:
        continue
    Xs = torch.from_numpy(xyz[rng.choice(inb, min(a.per_block, len(inb)), replace=False)]).double()
    M = torch.zeros(len(Xs), 16, 16, dtype=torch.float64); cnt = torch.zeros(len(Xs))
    for nm in (l.strip() for l in open(fp) if l.strip()):
        i = pos.get(nm)
        if i is None:
            continue
        R = cams.R[i].double(); T = cams.T[i].double()
        Xc = Xs @ R.T + T; z = Xc[:, 2]
        u = float(cams.fx[i]) * Xc[:, 0] / z.clamp_min(1e-6) + float(cams.cx[i]); v = float(cams.fy[i]) * Xc[:, 1] / z.clamp_min(1e-6) + float(cams.cy[i])
        m = (z > 0.2) & (u >= 0) & (u < float(cams.width[i])) & (v >= 0) & (v < float(cams.height[i]))
        if not bool(m.any()):
            continue
        d = Xs[m] + (R.T @ T)[None]; d = d / d.norm(dim=1, keepdim=True)
        Y = eval_sh_basis16(d)
        M[m] += Y[:, :, None] * Y[:, None, :]; cnt[m] += 1
    ok = cnt > 0
    Mb = (M[ok] / M[ok].diagonal(dim1=1, dim2=2).sum(1)[:, None, None]).mean(0)
    ev = torch.linalg.eigvalsh(Mb).flip(0)
    print(f"b{b:>2}：{int(ok.sum()):,} 點｜M̄ 特徵值（%）" + " ".join(f"{100 * float(x / ev.sum()):.2g}" for x in ev), flush=True)
    Mbs.append(Mb.numpy()); used.append(b)
Mg = np.mean(Mbs, axis=0)
ev = np.linalg.eigvalsh(Mg)[::-1]
print("全域 M̄ 特徵值（%）：" + " ".join(f"{100 * x / ev.sum():.3g}" for x in ev))
# 與各塊的高階 Schur 子空間（k=10 => 9 維）的重疊
def schur_U(M, kr):
    m00, m0r, Mrr = M[0, 0], M[0, 1:], M[1:, 1:]
    e, U = np.linalg.eigh(Mrr - np.outer(m0r, m0r) / m00)
    return U[:, ::-1][:, :kr]
Ug = schur_U(Mg, 9)
ov = [np.linalg.norm(Ug.T @ schur_U(M, 9)) ** 2 / 9 for M in Mbs]
print("全域與各塊高階 9 維子空間的重疊：最小 %.4f、中位 %.4f" % (min(ov), np.median(ov)))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, Mbar=Mg, Mbar_blocks=np.stack(Mbs), blocks=np.array(used))
print("=>", a.out)
