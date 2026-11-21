#!/usr/bin/env python
"""★ 11-21 合併模型裡「場景範圍外」的顆粒（使用者 viewer：m39 4x4 合併左邊遠處多了 floater，base9c15 沒有）。純 CPU。

場景範圍＝可靠 SfM 點（track >= 3、誤差 < 1.5）在分區座標（partitions.pt 的旋轉）裡的 xy 外接盒（0.5／99.5 百分位）；
不用公尺門檻：「外面」= 超出外接盒 f x 盒子邊長（預設 f = 0.05）。每顆報：落在哪一塊的分區盒（邊緣塊的盒往外延伸）、
離盒子幾倍邊長、不透明度、尺度（相對 SfM 最近鄰距離中位）、Σ o·面積。
用法：python tools/probe/far_floaters_1121.py --ckpts 名=<ckpt> ... [--f 0.05] [--blocks_root <跑次>/blocks]（逐塊 ckpt 的話給 --per_block）
"""
import argparse, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True, help="名=<ckpt 路徑或 glob>（glob＝逐塊 ckpt，合起來算）")
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--f", type=float, default=0.05)
ap.add_argument("--list", type=int, default=12, help="列出最大（o·面積）的前幾顆")
a = ap.parse_args()

from internal.utils.colmap import read_points3D_binary
P = torch.load(os.path.join(a.D, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
R = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = P["partition_coordinates"]["id"].numpy(); xys = P["partition_coordinates"]["xy"].double().numpy()
size = P["scene_config"]["partition_size"].double().numpy()
pts = read_points3D_binary(os.path.join(a.D, "sparse/0/points3D.bin"))
xyz = np.stack([p.xyz for p in pts.values()]); rel = np.array([len(p.image_ids) >= 3 and p.error < 1.5 for p in pts.values()])
q = xyz[rel] @ R.T
lo, hi = np.percentile(q[:, :2], 0.5, axis=0), np.percentile(q[:, :2], 99.5, axis=0)
zlo, zhi = np.percentile(q[:, 2], 0.5), np.percentile(q[:, 2], 99.5)
L = float((hi - lo).max()); m = a.f * L
print(f"場景 xy 盒（可靠 SfM 0.5／99.5 百分位，分區座標）{lo.round(2)}～{hi.round(2)}｜邊長 {L:.2f}｜外面＝超出 {a.f} x 邊長 = {m:.2f}｜SfM z 範圍 {zlo:.2f}～{zhi:.2f}")


def block_of(xy):
    b = np.full(len(xy), -1)
    for k in range(len(ids)):
        x0, y0 = xys[k]; x1, y1 = xys[k] + size[k]
        inb = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
        b[inb] = int(ids[k][0] + 4 * ids[k][1])
    return b


for spec in a.ckpts:
    name, path = spec.split("=", 1)
    files = sorted(glob.glob(path))
    M, O, S, BLK = [], [], [], []
    for f in files:
        sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
        pre = next(k for k in sd if k.endswith("gaussians.means"))[: -len("means")]
        M.append(sd[pre + "means"].double().numpy()); O.append(torch.sigmoid(sd[pre + "opacities"].double()).numpy()[:, 0])
        S.append(torch.exp(sd[pre + "scales"].double()).numpy())
        del sd
    M, O, S = np.concatenate(M), np.concatenate(O), np.concatenate(S)
    X = M @ R.T
    dx = np.maximum(np.maximum(lo[0] - X[:, 0], X[:, 0] - hi[0]), 0); dy = np.maximum(np.maximum(lo[1] - X[:, 1], X[:, 1] - hi[1]), 0)
    d = np.sqrt(dx ** 2 + dy ** 2)
    out = d > m
    area = np.pi * S[:, 0] * S[:, 1]
    b = block_of(X[:, :2])
    print(f"\n══ {name}（{len(files)} 個檔、N={len(M):,}）：場景範圍外 {int(out.sum()):,} 顆（{100 * out.mean():.3f}%）"
          f"｜o>0.5 的 {int((out & (O > 0.5)).sum()):,}｜Σ o·面積 {float((O * area)[out].sum()):.3g}（全模型 {float((O * area).sum()):.3g}）")
    if out.any():
        for lab, sel in (("超出 0.05～0.25 邊長", (d > m) & (d <= 0.25 * L)), ("0.25～1 邊長", (d > 0.25 * L) & (d <= L)), ("> 1 邊長", d > L)):
            print(f"   {lab:<16} {int(sel.sum()):>7,} 顆｜o>0.5 {int((sel & (O > 0.5)).sum()):>6,}｜Σ o·面積 {float((O * area)[sel].sum()):.3g}")
        ub, uc = np.unique(b[out], return_counts=True)
        print("   落在哪一塊的分區盒：", ", ".join(f"b{x}:{c}" for x, c in zip(ub, uc)))
        z = X[out, 2]
        print(f"   高度：在 SfM z 範圍內 {int(((z >= zlo) & (z <= zhi)).sum()):,}、上方 {int((z > zhi).sum()):,}、下方 {int((z < zlo).sum()):,}")
        idx = np.where(out)[0]; idx = idx[np.argsort(-(O * area)[idx])][: a.list]
        for i in idx:
            print(f"   #{i:<9} b{b[i]:<3} 離盒 {d[i] / L:6.2f} 邊長｜z {X[i, 2]:8.2f}｜o {O[i]:.3f}｜尺度 {S[i, 0]:.3g} x {S[i, 1]:.3g}｜o·面積 {O[i] * area[i]:.3g}")
