#!/usr/bin/env python
"""糊掉的 tile 在不同配方之間是不是同一批？如果是，它們長什麼樣子？（純 CPU，讀 test 圖）

## 為什麼問這個

使用者目視觀察：**清楚區與糊掉區的位置跨配方固定**。而八個配方的糊掉%
全部落在 46 +- 0.8（b12）：
```
absgrad w=0/1/2/3      49.31 / 46.17 / 46.08 / 46.03
收割期移除 L1          46.04     densify_until 36k  46.68
trim stride=4          46.82     lambda_normal      49.45     depth_loss  48.52
```
**如果糊掉是優化問題，不同配方應該在不同地方失敗。** 位置固定 =>
病灶由**輸入**（影像/姿態/SfM/偽深度）決定，不由訓練動力學決定
=> 這一句就解釋了為何八次密度控制介入全部無效。

本工具做兩件事：
1. **取交集**：跨配方的「糊掉 tile 遮罩」重疊多少。
   ~100% => 確認資料決定；低 => 上面的推論垮掉，回頭重想。
2. **描述**：持續糊掉 vs 從不糊掉的 tile，在 GT/render 上有什麼差別
   （紋理、亮度、render 是否為低變異的灰、在畫面中的位置/半徑）。
   這直接分辨三個候選：內容超出塊範圍（=> 位置偏邊緣/遠場）／
   視角相依外觀（=> 特定亮度-紋理簽名）／未受約束質量。

⚠ tile 的保留遮罩（`sg > quantile(sg, contrast_q)`）只依 **GT** 計算
   => 跨配方完全相同 => tile 索引可直接對齊。

用法: python tools/blur_persistence.py agd2_b12 agd2oru_b12 agd3_b12 --blk 12
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402


def per_image(path, t, contrast_q):
    """回傳 (keep_idx, corr, GT 統計, render 統計)；keep_idx 只依 GT => 跨配方一致。"""
    im = Image.open(path)
    w = im.width // 2
    g = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
    r = np.asarray(im.crop((w, 0, im.width, im.height)).convert("L"), np.float32) / 255.
    G, R = tiles(g, t), tiles(r, t)
    sg = G.std(axis=1)
    keep = np.where(sg > np.quantile(sg, contrast_q))[0]
    if len(keep) < 8:
        return None
    G, R = G[keep], R[keep]
    sg2, sr2 = G.std(axis=1), R.std(axis=1)
    gm, rm = G.mean(axis=1, keepdims=True), R.mean(axis=1, keepdims=True)
    cov = ((G - gm) * (R - rm)).mean(axis=1)
    corr = cov / np.maximum(sg2 * sr2, 1e-8)
    nx = w // t
    return keep, corr, G.mean(1), sg2, R.mean(1), sr2, nx, (im.height // t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.40)
    args = ap.parse_args()

    masks, ref = {}, None
    for run in args.runs:
        d = final_test_dir(run, args.blk)
        if d is None:
            print(f"⚠ 找不到 {run} 的 test 圖，略過")
            continue
        blur, tot = {}, 0
        for f in sorted(os.listdir(d)):
            if not f.endswith(".png"):
                continue
            out = per_image(os.path.join(d, f), args.tile, args.contrast_q)
            if out is None:
                continue
            keep, corr, gmean, gstd, rmean, rstd, nx, ny = out
            blur[f] = corr <= args.r_min
            tot += len(keep)
            if ref is None or f not in ref[0]:
                if ref is None:
                    ref = ({}, {})
                ref[0][f] = (keep, gmean, gstd, nx, ny)
            ref[1].setdefault(f, {})[run] = (rmean, rstd, corr)
        masks[run] = blur
        n_b = sum(int(v.sum()) for v in blur.values())
        print(f"{run:>16}  糊掉 {n_b:>6,}/{tot:>6,} = {n_b / max(tot, 1):.2%}")

    runs = [r for r in args.runs if r in masks]
    if len(runs) < 2:
        print("需要至少兩個配方才能取交集")
        return

    # ---- 1) 交集 ----
    common = sorted(set.intersection(*[set(masks[r].keys()) for r in runs]))
    inter = uni = 0
    per_run_only = {r: 0 for r in runs}
    for f in common:
        ms = np.stack([masks[r][f] for r in runs])       # (R, T)
        allb = ms.all(0)
        anyb = ms.any(0)
        inter += int(allb.sum())
        uni += int(anyb.sum())
        for i, r in enumerate(runs):
            per_run_only[r] += int((ms[i] & ~allb).sum())
    print(f"\n===== 跨 {len(runs)} 個配方的糊掉 tile =====")
    print(f"  聯集（任一配方糊掉） {uni:>7,}")
    print(f"  交集（全部都糊掉）   {inter:>7,}   = 聯集的 **{inter / max(uni, 1):.1%}**")
    print("  判讀：接近 100% => 位置由**輸入**決定，不由訓練動力學決定；"
          "偏低 => 是優化/隨機性問題。")

    # ---- 2) 描述持續糊掉 vs 從不糊掉 ----
    rows_p, rows_n = [], []
    for f in common:
        ms = np.stack([masks[r][f] for r in runs])
        allb, neverb = ms.all(0), ~ms.any(0)
        keep, gmean, gstd, nx, ny = ref[0][f]
        rmean, rstd, corr = ref[1][f][runs[0]]
        ty, tx = keep // nx, keep % nx
        # 到畫面中心的正規化半徑（0=中心, 1=角落）
        rad = np.sqrt(((tx - nx / 2) / (nx / 2)) ** 2 + ((ty - ny / 2) / (ny / 2)))
        for m, acc in ((allb, rows_p), (neverb, rows_n)):
            if m.sum():
                acc.append(np.stack([gmean[m], gstd[m], rmean[m], rstd[m], rad[m]], 1))
    P = np.concatenate(rows_p) if rows_p else np.zeros((0, 5))
    N = np.concatenate(rows_n) if rows_n else np.zeros((0, 5))
    print(f"\n===== 特徵對照（中位數）=====")
    print(f"{'':>16} {'GT亮度':>8} {'GT紋理':>8} {'render亮度':>11} {'render紋理':>11} {'離中心半徑':>10} {'n':>8}")
    for nm, A in (("持續糊掉", P), ("從不糊掉", N)):
        if len(A) == 0:
            continue
        m = np.median(A, 0)
        print(f"{nm:>16} {m[0]:>8.3f} {m[1]:>8.4f} {m[2]:>11.3f} {m[3]:>11.4f} {m[4]:>10.3f} {len(A):>8,}")
    if len(P) and len(N):
        mp, mn = np.median(P, 0), np.median(N, 0)
        print(f"{'比值 糊/不糊':>16} {mp[0]/max(mn[0],1e-9):>8.2f} {mp[1]/max(mn[1],1e-9):>8.2f} "
              f"{mp[2]/max(mn[2],1e-9):>11.2f} {mp[3]/max(mn[3],1e-9):>11.2f} {mp[4]/max(mn[4],1e-9):>10.2f}")
    print("""
判讀（三個候選各自的預測）：
  內容超出塊的空間範圍 => 持續糊掉的 tile **離中心半徑明顯大**（邊緣/遠場）
  視角相依外觀（水面反光等）=> GT 紋理不低但 **render 紋理塌掉**、render 亮度趨中（灰）
  未受約束的質量       => render 紋理塌掉且亮度偏離 GT（沒有特定位置偏好）
⚠ 「render 紋理/GT 紋理」比值遠小於 1 = 結構被抹平，這是「一坨灰」的直接量化。""")


if __name__ == "__main__":
    main()
