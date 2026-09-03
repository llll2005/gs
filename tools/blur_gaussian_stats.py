#!/usr/bin/env python
"""持續糊掉的 tile 上，**我方的高斯**長什麼樣？（純 CPU，讀 ckpt + SfM）

## 一次檢驗三個外部提出的假說

外部諮詢（2026-09-03）給了三個彼此不同的解釋，它們對**同一組量**做出**不同方向**的預測：

| 假說 | 來源 | 對「糊掉 tile」的預測 |
|---|---|---|
| A 空間頻寬不足（footprint 太粗） | GPT | 投影半徑**明顯大**、每像素高斯數**明顯少** |
| B 大高斯低通層 / 梯度鎖死 | Gemini | 投影半徑**明顯大** + opacity **明顯高**（前層厚） |
| C partition / 內容不歸本塊所有 | GPT | 高斯數**明顯少**（那裡本來就沒配到質量），半徑不必然大 |

⇒ **三者可由「半徑」「密度」「opacity」三個量的組合分辨**，而且全部免費。

⚠ 先前已排除的（見 §11.72）：遠場（深度比 1.09x）、SfM 取樣不足（點數 1.43x **更多**）、
   深度不連續（離散比 0.83x **更平坦**）。所以本次量的是**我方高斯**，不是 SfM 點。

## 另外量「角度覆蓋」（multi-view conflict 假說）
外部指出：`track` 數量 != 監督的**幾何多樣性**。同一個點被 20 台幾乎同方向的相機看到，
與被 8 台分散方向看到，多視角約束完全不同。
⇒ 對每個 tile 內的 SfM 點，量觀測相機方向的**最大兩兩夾角**（視差角）。

用法: python tools/blur_gaussian_stats.py agd2_b12 sched30_b12 --blk 12
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.colmap import (read_images_binary, read_cameras_binary,  # noqa: E402
                                   read_points3D_binary)
from tools.veil_detect import final_test_dir  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="至少兩個配方，取糊掉遮罩交集")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--down-sample", type=float, default=1.2)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--model-run", default=None, help="取哪個跑次的高斯（預設第一個）")
    args = ap.parse_args()

    # ---- 高斯 ----
    mr = args.model_run or args.runs[0]
    ck = sorted(glob.glob(f"outputs/{mr}/**/*step=60000.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {mr} 的 60k ckpt")
    sd = torch.load(ck[0], map_location="cpu")["state_dict"]
    XYZ = sd["gaussian_model.gaussians.means"].double().numpy()
    SC = torch.exp(sd["gaussian_model.gaussians.scales"].float()).numpy()
    OP = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).squeeze(-1).numpy()
    s_max = SC.max(axis=1)
    print(f"高斯來源 {mr}：N = {len(XYZ):,}")

    # ---- 相機 ----
    sp = os.path.join(args.data, "sparse/0")
    images = read_images_binary(os.path.join(sp, "images.bin"))
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    c0 = list(cams.values())[0]
    fx = float(c0.params[0]) / args.down_sample
    W = int(round(c0.width / args.down_sample))
    H = int(round(c0.height / args.down_sample))
    by_name = {im.name: im for im in images.values()}
    centers = {iid: -im.qvec2rotmat().T @ im.tvec for iid, im in images.items()}

    pts = read_points3D_binary(os.path.join(sp, "points3D.bin"))
    PXYZ = np.stack([p.xyz for p in pts.values()])
    PIMG = [np.unique(p.image_ids) for p in pts.values()]
    print(f"SfM 點 {len(PXYZ):,}   影像 {W}x{H}")

    dirs = {r: final_test_dir(r, args.blk) for r in args.runs}
    fs = sorted(f for f in os.listdir(dirs[args.runs[0]]) if f.endswith(".png"))
    P, N_, miss = [], [], 0

    for f in fs:
        outs = {r: per_image(os.path.join(dirs[r], f), args.tile, args.contrast_q)
                for r in args.runs}
        if any(o is None for o in outs.values()):
            continue
        keep, corr, *_r, nx, ny = outs[args.runs[0]]
        ms = np.stack([outs[r][1] <= args.r_min for r in args.runs])
        allb, neverb = ms.all(0), ~ms.any(0)
        name = f[:-4] if f.endswith(".png.png") else f
        im = by_name.get(name)
        if im is None:
            miss += 1
            continue
        R, T = im.qvec2rotmat(), im.tvec

        def tile_stats(X, extra=None):
            """把 3D 點投影到 tile，回傳 (tile 索引, 該點的附加量)。"""
            pc = X @ R.T + T
            z = pc[:, 2]
            v = z > 0.2
            u = fx * pc[v, 0] / z[v] + W / 2
            w_ = fx * pc[v, 1] / z[v] + H / 2
            ok = (u >= 0) & (u < nx * args.tile) & (w_ >= 0) & (w_ < ny * args.tile)
            ti = (w_[ok] // args.tile).astype(int) * nx + (u[ok] // args.tile).astype(int)
            idx = np.where(v)[0][ok]
            return ti, idx, z[v][ok]

        # 高斯
        gti, gidx, gz = tile_stats(XYZ)
        grad_px = 3.0 * fx * s_max[gidx] / gz          # 螢幕半徑（px）
        gop = OP[gidx]
        nt = nx * ny
        cnt = np.bincount(gti, minlength=nt).astype(float)
        med_r = np.full(nt, np.nan)
        med_o = np.full(nt, np.nan)
        o_ = np.argsort(gti, kind="stable")
        gt_s, r_s, o_s = gti[o_], grad_px[o_], gop[o_]
        uq, st = np.unique(gt_s, return_index=True)
        en = np.append(st[1:], len(gt_s))
        for t_, a_, b_ in zip(uq, st, en):
            med_r[t_] = np.median(r_s[a_:b_])
            med_o[t_] = np.median(o_s[a_:b_])

        # SfM 點的角度覆蓋
        pti, pidx, _ = tile_stats(PXYZ)
        ang = np.full(nt, np.nan)
        o2 = np.argsort(pti, kind="stable")
        pt_s, pi_s = pti[o2], pidx[o2]
        uq2, st2 = np.unique(pt_s, return_index=True)
        en2 = np.append(st2[1:], len(pt_s))
        rng = np.random.default_rng(0)
        for t_, a_, b_ in zip(uq2, st2, en2):
            sel = pi_s[a_:b_]
            if len(sel) > 40:
                sel = rng.choice(sel, 40, replace=False)
            angs = []
            for pi in sel:
                obs = PIMG[pi]
                C = np.array([centers[o] for o in obs if o in centers])
                if len(C) < 2:
                    continue
                d = C - PXYZ[pi]
                d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                angs.append(np.degrees(np.arccos(np.clip(d @ d.T, -1, 1).min())))
            if angs:
                ang[t_] = np.median(angs)

        area = args.tile ** 2
        for m, acc in ((allb, P), (neverb, N_)):
            if m.sum():
                k = keep[m]
                acc.append(np.stack([cnt[k] / area, med_r[k], med_o[k], ang[k]], 1))

    if miss:
        print(f"⚠ {miss} 張對不到 COLMAP，已略過")
    P = np.concatenate(P) if P else np.zeros((0, 4))
    N_ = np.concatenate(N_) if N_ else np.zeros((0, 4))
    LAB = ["高斯/像素", "投影半徑px", "opacity", "視差角°"]
    print(f"\n{'':>12} " + " ".join(f"{l:>12}" for l in LAB) + f" {'n':>8}")
    med = {}
    for nm, A in (("持續糊掉", P), ("從不糊掉", N_)):
        if not len(A):
            continue
        m = [np.nanmedian(A[:, i]) for i in range(4)]
        med[nm] = m
        print(f"{nm:>12} " + " ".join(f"{x:>12.4f}" for x in m) + f" {len(A):>8,}")
    if len(med) == 2:
        a, b = med["持續糊掉"], med["從不糊掉"]
        print(f"{'比值 糊/不糊':>12} " + " ".join(f"{a[i]/max(b[i],1e-9):>12.2f}" for i in range(4)))
    print("""
判讀（三個假說的預測方向）：
  A 空間頻寬不足   投影半徑 >> 1  且  高斯/像素 << 1
  B 大高斯低通層   投影半徑 >> 1  且  opacity   >> 1
  C partition      高斯/像素 << 1  且  投影半徑 ~ 1
  全部 ~1          三者皆不成立，糊掉與高斯的空間統計無關 => 往 optimizer / 多視角衝突找
  視差角 << 1      監督的幾何多樣性不足（與 track 數量是不同的東西）""")


if __name__ == "__main__":
    main()
