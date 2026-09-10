#!/usr/bin/env python
"""失敗區在**總損失**裡佔多少？—— O0 之後最樸素的那個解釋，先量它值不值得做。

## 為什麼問這個

§11.87 的 O0 證明：凍結拓撲、只優化影響集、**tile 局部的 loss**，
失敗 tile 就能從 corr 0.273 拉到 0.933。訓練做不到，而 O0 與訓練有四個差異：
```
① 步長（means_lr 末期 6.4e-7 vs oracle 1e-3 = 1,563 倍）   -> O0 第二臂在測
② **損失被平均稀釋**（O0 只看 48x48，訓練看整張）           -> **本工具**
③ 多視角衝突（O0 只用一台相機）                            -> O0 的多視角覆核在測
④ Adam 二階動量（O0 用全新的 optimizer state）             -> 尚未測
```

## 量什麼

L1 對每個像素的梯度大小相同（sign 而已），所以**失敗區在 L1 梯度裡的份額 = 它的像素份額**。
真正決定「訊號多稀」的是：
```
像素份額            失敗 tile 的像素佔整張的比例
殘差份額            失敗 tile 的 |殘差| 總和佔整張的比例
稀釋倍率 = 殘差份額 / 像素份額
```
```
稀釋倍率 >> 1 => 失敗區**已經**在損失裡佔優勢（每像素殘差大），
                 重新加權買不到多少 => ② 不是主因
稀釋倍率 ~ 1  => 失敗區在損失裡完全沒有優先權，
                 而它只佔少數像素 => **加權是便宜且直接的介入**
```
⚠ 這只量 L1 那一項。SSIM 是局部視窗的，行為不同，但權重只有 0.2。

⚠ **本工具用 test 圖（render|gt 並排）算，不需要 GPU，也不需要載模型。**

用法: python tools/loss_dilution.py agd2_b12 sched30_b12 --blk 12
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    args = ap.parse_args()

    d0 = final_test_dir(args.runs[0], args.blk)
    dirs = {r: final_test_dir(r, args.blk) for r in args.runs}
    fs = sorted(f for f in os.listdir(d0) if f.endswith(".png"))
    T = args.tile

    tot_px = fail_px = 0
    tot_res = fail_res = 0.0
    tot_res2 = fail_res2 = 0.0
    hi_px = hi_res = 0.0
    n_img = 0
    for f in fs:
        outs = {r: per_image(os.path.join(dirs[r], f), T, args.contrast_q) for r in args.runs}
        if any(o is None for o in outs.values()):
            continue
        keep, *_z, nx, ny = outs[args.runs[0]]
        allb = np.stack([outs[r][1] <= args.r_min for r in args.runs]).all(0)

        im = Image.open(os.path.join(d0, f))
        w = im.width // 2
        G = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
        R = np.asarray(im.crop((w, 0, im.width, im.height)).convert("L"), np.float32) / 255.
        E = np.abs(G - R)

        # 全張（含低對比區）：這是訓練 loss 真正平均的分母
        tot_px += E.size
        tot_res += float(E.sum())
        tot_res2 += float((E ** 2).sum())

        # 失敗 tile（keep[allb] 是 tile 編號）
        Et = tiles(E, T)
        nt_row = E.shape[1] // T
        fail_tiles = keep[allb]
        if len(fail_tiles):
            sel = Et[fail_tiles]
            fail_px += sel.size
            fail_res += float(sel.sum())
            fail_res2 += float((sel ** 2).sum())
        # 高對比 tile 整體（keep），當作「有紋理可失敗的地方」的參照
        hi = Et[keep]
        hi_px += hi.size
        hi_res += float(hi.sum())
        n_img += 1

    print(f"共 {n_img} 張圖，tile {T}x{T}，失敗判準 = 所有配方 corr <= {args.r_min}\n")
    print(f"{'':>16} {'像素份額':>10} {'|殘差| 份額':>12} {'稀釋倍率':>10}")
    for lab, px, rs in (("失敗 tile", fail_px, fail_res), ("高對比 tile 全體", hi_px, hi_res)):
        fp, fr = px / tot_px, rs / tot_res
        print(f"{lab:>16} {100*fp:>9.2f}% {100*fr:>11.2f}% {fr/max(fp,1e-12):>10.3f}x")

    fp = fail_px / tot_px
    fr = fail_res / tot_res
    fr2 = fail_res2 / tot_res2
    print(f"\n  失敗區每像素平均 |殘差| / 全張平均 = **{fr/max(fp,1e-12):.3f}x**")
    print(f"  以 L2 計（SSIM/梯度更接近平方項）    = {fr2/max(fp,1e-12):.3f}x")
    print(f"""
判讀：
  稀釋倍率 >> 1  => 失敗區在 L1 損失裡**已經**每像素佔優，
                    「重新加權」買到的倍率就是 (1/份額) 之外的那一點點 => ② 不是主因
  稀釋倍率 ~ 1   => 失敗區完全沒有優先權，且只佔 {100*fp:.1f}% 像素
                    => 把權重提到 1/{fp:.3f} = {1/max(fp,1e-12):.0f}x 是**便宜且直接**的介入
⚠ 注意這個量與 §11.80 的「殘差多 86%」不同：那個是**高對比 tile 之間**的比較，
  這裡的分母是**整張圖**（含大片低對比區）—— 訓練 loss 平均的正是後者。""")


if __name__ == "__main__":
    main()
