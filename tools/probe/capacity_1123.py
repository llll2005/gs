#!/usr/bin/env python
"""★ 11-23 使用者：Mill-19 的邊界霧化、針狀、遠處大面元，是不是起點／上限顆數太高（式子要改），還是消融拿掉的機制造成的？不訓練、CPU。
量「容量 vs 監督」：
  1. 訓練像素總數（訓練解析度）÷ 合併顆數＝每顆分到的像素
  2. 每塊：起點顆數（initialize_from 的 PLY）、上限（平均 x SfM 權重）、最後顆數（run_summary.json）
  3. 每顆被幾台訓練相機的視錐包含（不算遮擋；抽樣）：<= 5、<= 20 台的比例（顆數、Σo·面積），針狀顆粒另算
用法：python tools/probe/capacity_1123.py --tag MC --run outputs/lab/<跑次> --data <資料集> --down 1.2 --capavg 1500000 --alloc configs/cap_alloc_sfm_g44.json
"""
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--run", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--down", type=float, required=True); ap.add_argument("--capavg", type=float, default=1.5e6); ap.add_argument("--alloc", required=True)
ap.add_argument("--ckpt", default=""); ap.add_argument("--nsample", type=int, default=30000)
a = ap.parse_args()
from internal.utils.colmap import read_images_binary, read_cameras_binary

ims = read_images_binary(os.path.join(a.data, "sparse/0/images.bin")); cams = read_cameras_binary(os.path.join(a.data, "sparse/0/cameras.bin"))
px = sum(int(cams[im.camera_id].width / a.down) * int(cams[im.camera_id].height / a.down) for im in ims.values())
W = json.load(open(a.alloc))["weights"]
rows = []
for b in range(16):
    d = os.path.join(a.run, "blocks", f"block_{b}")
    if not os.path.isdir(d):
        continue
    nf = json.load(open(os.path.join(d, "run_summary.json"))).get("n_final") if os.path.exists(os.path.join(d, "run_summary.json")) else None
    ninit = None
    cfg = os.path.join(d, "config.yaml")
    if os.path.exists(cfg):
        for l in open(cfg):
            if "initialize_from:" in l:
                p = l.split("initialize_from:")[1].strip().strip("'\"")
                if p and p != "null" and os.path.exists(p):
                    with open(p, "rb") as f:
                        for hl in f.read(4096).split(b"\n"):
                            if hl.startswith(b"element vertex"):
                                ninit = int(hl.split()[-1])
                break
    rows.append((b, ninit, a.capavg * float(W[str(b)]), nf))
ck = a.ckpt or os.path.join(a.run, "checkpoints", "merged.ckpt")
sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
g = lambda k: sd[f"gaussian_model.gaussians.{k}"].float().numpy()
mu = g("means").astype(np.float64); o = 1 / (1 + np.exp(-g("opacities")[:, 0])); s = np.exp(g("scales")); N = len(mu)
w = o * np.pi * s[:, 0] * s[:, 1]; needle = (o > 0.3) & (s.max(1) / np.maximum(s.min(1), 1e-12) > 10)
print(f"=== {a.tag}｜訓練影像 {len(ims):,} 張、訓練像素 {px / 1e9:.2f} G｜合併 N {N / 1e6:.2f}M｜每顆分到 {px / N:.0f} 像素")
tot = [sum(x[i] or 0 for x in rows) for i in (1, 2, 3)]
print(f"  每塊合計：起點 {tot[0] / 1e6:.2f}M｜上限 {tot[1] / 1e6:.2f}M｜最後（合併前，含盒外）{tot[2] / 1e6:.2f}M（＝上限的 {100 * tot[2] / max(tot[1], 1):.0f}%）｜起點／上限 {100 * tot[0] / max(tot[1], 1):.0f}%")
print("  逐塊（起點／上限／最後，M）：" + "  ".join(f"b{b} {(ni or 0) / 1e6:.2f}/{c / 1e6:.2f}/{(nf or 0) / 1e6:.2f}" for b, ni, c, nf in rows))
# 可見相機數（抽樣、不算遮擋）
rng = np.random.default_rng(0); idx = rng.choice(N, min(N, a.nsample), replace=False); m = mu[idx]; vis = np.zeros(len(idx), np.int32)
for im in ims.values():
    c = cams[im.camera_id]; R = im.qvec2rotmat(); t = im.tvec
    f = c.params[0] / a.down; cx, cy = ((c.params[1], c.params[2]) if c.model.startswith("SIMPLE") else (c.params[2], c.params[3]))
    cx, cy = cx / a.down, cy / a.down; Wd, Hd = c.width / a.down, c.height / a.down
    pc = m @ R.T + t; z = pc[:, 2]; u = f * pc[:, 0] / np.maximum(z, 1e-9) + cx; v = f * pc[:, 1] / np.maximum(z, 1e-9) + cy
    vis += ((z > 0) & (u >= 0) & (u < Wd) & (v >= 0) & (v < Hd)).astype(np.int32)
ws = w[idx]; nd = needle[idx]
q = lambda x: "／".join(f"{np.percentile(x, p):.0f}" for p in (10, 50, 90))
print(f"  可見訓練相機數 p10/50/90：{q(vis)}｜<= 5 台：顆數 {100 * np.mean(vis <= 5):.1f}%、Σo·面積 {100 * ws[vis <= 5].sum() / ws.sum():.1f}%｜<= 20 台：顆數 {100 * np.mean(vis <= 20):.1f}%、Σo·面積 {100 * ws[vis <= 20].sum() / ws.sum():.1f}%")
print(f"  針狀 {100 * needle.mean():.2f}%：其可見相機數 p10/50/90 {q(vis[nd]) if nd.any() else '-'}｜針狀裡 <= 20 台的比例 {100 * np.mean(vis[nd] <= 20) if nd.any() else 0:.1f}%")
