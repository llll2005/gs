#!/usr/bin/env python
"""★ 11-21 純數學化簡 #2（離線、不訓練）：warp 層級「精確橢圓∪低通圓盤 vs 16x2 像素條」測試，能比現行 skip_box（外接矩形）多刪多少 warp 迭代？

背景：光柵器 forward／backward 的時間跟「(warp, 面元) 迭代次數」成正比；11-19 量過每像素迴圈 89.5 次只有 20.6 次真的混合。
skip_box（11-16）＝每個 warp 先看面元精確支撐的「外接矩形」有沒有碰到自己的 16x2 像素條，沒有就整個 warp 跳過。
但斜放的細長橢圓（空拍很多）的外接矩形很鬆。精確測試：
  支撐＝{ rho3d <= tau } ∪ { rho2d <= tau }，tau = 2 ln(255 o)
  rho3d <= tau  <=>  p'x^2 + p'y^2 - tau p'z^2 <= 0，p' = Q (x, y, 1) 對像素仿射（反單應，見 invhomog_verify_1121.py）
  像素條只有兩列像素中心 y0、y0+1 => 每列 F(x) = A x^2 + B x + C，問 [x0, x1] 上 min F <= 0（凸取頂點、凹取端點）；圓盤＝點到線段距離。
本腳本對真實 ckpt＋相機，對每個（面元, 像素條）配對比三件事：
  box   ＝現行 skip_box 不跳（＝實際執行的 warp 迭代）
  exact ＝精確測試不跳
  truth ＝32 個像素中心至少一個 alpha >= 1/255 且過近平面（離散真值；不含 T 飽和）
用法：python tools/probe/conic_strip_cull_1121.py [--ckpt ...] [--n 30000] [--cams 3]
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from internal.utils.colmap import read_images_binary, read_cameras_binary, qvec2rotmat

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/lab/view_block_2_nogate/checkpoints/view_block_2_nogate.ckpt")
ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08/002_000.txt")
ap.add_argument("--n", type=int, default=30000)
ap.add_argument("--cams", type=int, default=3)
ap.add_argument("--dsf", type=float, default=1.2)
a = ap.parse_args()
rng = np.random.default_rng(0)
FS = 0.7071067811865476; NEAR = 0.2

sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
g = {k.split(".")[-1]: v for k, v in sd.items() if k.startswith("gaussian_model.gaussians.")}
mu = g["means"].double().numpy(); sc = np.exp(g["scales"].double().numpy())
q = g["rotations"].double().numpy(); q /= np.linalg.norm(q, axis=1, keepdims=True)
op = 1 / (1 + np.exp(-g["opacities"].double().numpy()[:, 0]))
w, x, y, z = q.T
Rg = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
               np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
               np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
imgs = read_images_binary(os.path.join(a.data, "sparse/0/images.bin")); cams = read_cameras_binary(os.path.join(a.data, "sparse/0/cameras.bin"))
names = [l.strip() for l in open(os.path.join(a.data, a.part)) if l.strip()]
byname = {v.name: v for v in imgs.values()}
pick = [byname[n] for n in names[:: max(1, len(names) // a.cams)][: a.cams] if n in byname]


def aabb_R(Tu, Tv, Tw, R2):           # forward.cu computeAABB_R（R2＝R^2；R2=1 時就是 computeAABB 的中心）
    d = R2 * (Tw[:, 0] ** 2 + Tw[:, 1] ** 2) - Tw[:, 2] ** 2
    f = np.stack([R2, R2, -np.ones_like(R2)], 1) / d[:, None]
    px = (f * Tu * Tw).sum(1); py = (f * Tv * Tw).sum(1)
    hx = np.sqrt(np.maximum(px ** 2 - (f * Tu * Tu).sum(1), 0)); hy = np.sqrt(np.maximum(py ** 2 - (f * Tv * Tv).sum(1), 0))
    return px, py, hx, hy


tot = np.zeros(4)            # box、exact、truth、exact 但 truth=0（保守度）
per_cam = []
for im in pick:
    cam = cams[im.camera_id]; fx, fy = cam.params[0] / a.dsf, cam.params[1] / a.dsf
    Wd, Hd = int(round(cam.width / a.dsf)), int(round(cam.height / a.dsf)); cx, cy = Wd / 2, Hd / 2   # 同 preprocess：cx = W/2
    Rwc = qvec2rotmat(im.qvec); t = np.asarray(im.tvec); pv = mu @ Rwc.T + t
    xs = fx * pv[:, 0] / pv[:, 2] + cx; ys = fy * pv[:, 1] / pv[:, 2] + cy
    vis = np.where((pv[:, 2] > NEAR) & (xs > -Wd / 4) & (xs < Wd * 1.25) & (ys > -Hd / 4) & (ys < Hd * 1.25) & (op > 1 / 255))[0]
    idx = rng.choice(vis, size=min(a.n, len(vis)), replace=False)
    c0 = (Rg[idx][:, :, 0] @ Rwc.T) * sc[idx, 0:1]; c1 = (Rg[idx][:, :, 1] @ Rwc.T) * sc[idx, 1:2]; c2 = pv[idx]
    C = np.stack([c0, c1, c2], 2)
    Tu = fx * C[:, 0] + cx * C[:, 2]; Tv = fy * C[:, 1] + cy * C[:, 2]; Tw = C[:, 2]
    M = np.stack([Tu, Tv, Tw], 1); Q = np.linalg.inv(M)
    tau = 2 * np.log(255 * op[idx]); RE = np.sqrt(np.maximum(tau, 1e-12))
    xy0x, xy0y, _, _ = aabb_R(Tu, Tv, Tw, np.ones(len(idx)))             # 低通中心＝r=1 圓錐中心（points_xy_image）
    ccx, ccy, cex, cey = aabb_R(Tu, Tv, Tw, tau)
    fsr = FS * RE
    bx0 = np.minimum(ccx - cex, xy0x - fsr) - 1; bx1 = np.maximum(ccx + cex, xy0x + fsr) + 1
    by0 = np.minimum(ccy - cey, xy0y - fsr) - 1; by1 = np.maximum(ccy + cey, xy0y + fsr) + 1
    nb = ne = nt = nfp = 0
    for gi in range(len(idx)):
        if not np.isfinite([bx0[gi], bx1[gi], by0[gi], by1[gi]]).all():
            continue
        # 盒子碰到的像素條（像素中心 sx0 = 16m+0.5..16m+15.5、sy0 = 2k+0.5、sy1 = sy0+1；限畫面內）
        m0 = max(0, int(np.floor((bx0[gi] - 15.5) / 16)) ); m1 = min((Wd - 1) // 16, int(np.floor((bx1[gi] - 0.5) / 16)))
        k0 = max(0, int(np.floor((by0[gi] - 1.5) / 2))); k1 = min((Hd - 1) // 2, int(np.floor((by1[gi] - 0.5) / 2)))
        if m1 < m0 or k1 < k0:
            continue
        mm, kk = np.meshgrid(np.arange(m0, m1 + 1), np.arange(k0, k1 + 1), indexing="ij")
        mm = mm.ravel(); kk = kk.ravel()
        sx0 = 16 * mm + 0.5; sx1 = sx0 + 15; sy0 = 2 * kk + 0.5; sy1 = sy0 + 1
        boxp = ~((bx1[gi] < sx0) | (bx0[gi] > sx1) | (by1[gi] < sy0) | (by0[gi] > sy1))
        if not boxp.any():
            continue
        sx0, sx1, sy0, sy1 = sx0[boxp], sx1[boxp], sy0[boxp], sy1[boxp]
        Qg = Q[gi]
        ex = np.zeros(len(sx0), bool)
        for yrow in (sy0, sy1):
            # p'(x) = a + x b，a = Q (0, y, 1)、b = Q (1, 0, 0)；F = |p'xy|^2 - tau p'z^2
            av = np.outer(yrow, Qg[:, 1]) + Qg[:, 2]; bv = Qg[:, 0]
            A = bv[0] ** 2 + bv[1] ** 2 - tau[gi] * bv[2] ** 2
            B = 2 * (av[:, 0] * bv[0] + av[:, 1] * bv[1] - tau[gi] * av[:, 2] * bv[2])
            Cc = av[:, 0] ** 2 + av[:, 1] ** 2 - tau[gi] * av[:, 2] ** 2
            F = lambda xx: A * xx ** 2 + B * xx + Cc
            cand = [F(sx0), F(sx1)]
            if A > 0:
                xv = np.clip(-B / (2 * A), sx0, sx1); cand.append(F(xv))
            ex |= np.minimum.reduce(cand) <= 0
            # 低通圓盤：中心 xy0 到線段的距離 <= FS * RE
            dxs = np.clip(xy0x[gi], sx0, sx1) - xy0x[gi]; dys = yrow - xy0y[gi]
            ex |= dxs ** 2 + dys ** 2 <= (fsr[gi]) ** 2
        # 離散真值：32 個像素中心
        px = sx0[:, None] + np.arange(16)[None, :]; px = np.concatenate([px, px], 1)
        py = np.concatenate([np.repeat(sy0[:, None], 16, 1), np.repeat(sy1[:, None], 16, 1)], 1)
        P = px[..., None] * Qg[:, 0] + py[..., None] * Qg[:, 1] + Qg[:, 2]
        rho3 = (P[..., 0] ** 2 + P[..., 1] ** 2) / P[..., 2] ** 2
        rho2 = ((px - xy0x[gi]) ** 2 + (py - xy0y[gi]) ** 2) / FS ** 2
        depth = 1 / P[..., 2]
        tr = ((np.minimum(rho3, rho2) <= tau[gi]) & (depth >= NEAR)).any(1)
        nb += len(sx0); ne += int(ex.sum()); nt += int(tr.sum()); nfp += int((ex & ~tr).sum())
        if (tr & ~ex).any():
            raise RuntimeError(f"精確測試漏掉有混合的像素條（面元 {idx[gi]}）—— 測試不保守，推導有錯")
    per_cam.append((im.name, nb, ne, nt))
    tot += [nb, ne, nt, nfp]
    print(f"{im.name}：{len(idx):,} 顆｜盒子不跳 {nb:,}｜精確不跳 {ne:,}（{ne / max(nb, 1):.1%}）｜真的有混合 {nt:,}（{nt / max(nb, 1):.1%}）")
nb, ne, nt, nfp = tot
print(f"合計：現行 skip_box 執行的 warp 迭代 {int(nb):,}；精確橢圓∪圓盤測試只需 {int(ne):,}（{ne / nb:.1%}，可再刪 {1 - ne / nb:.1%}）；"
      f"離散真值 {int(nt):,}（{nt / nb:.1%}）；精確測試多放行（連續 vs 離散）{int(nfp):,}")
