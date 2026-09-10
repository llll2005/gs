#!/usr/bin/env python
"""我們的病在哪個頻段？—— 決定「頻率平衡 loss」（f-loss / FreGS 家族）值不值得碰。

## 起因

使用者 2026-09-05 提供 `參考論文/2609.02748v1.pdf`
（Balancing Frequencies and Pixels in Flow Matching, f-loss）。核心主張：
```
自然影像功率譜 ~ 1/f^2  =>  能量集中在低頻
像素域 loss 一視同仁     =>  **低頻主宰梯度**，細節（紋理/邊緣）學得慢
=> 頻率平衡的目標函數：收斂快 40%、FID 更好、drop-in 不動架構
```

## 為什麼不能直接照做

本專案的記憶 `feedback_experiment_methodology` 寫著：
**「FreGS/distortion＝高頻懲罰，但我們的病是低頻/on-surface 欠擬合」**，
而 FreGS 實測也輸過（21.71 vs 22.18）。
⚠ 但那是 **GT 錯位期**（2026-08-12 前）的數據，分數作廢；
   而今晚的量測顯示失敗的是**高對比 tile 的紋理不見了**，聽起來像高頻欠擬合。
⇒ **兩個說法直接衝突，必須用當前資料上的量測裁決，不能用論證。**

## 量什麼

對每張 test 圖（render|gt 並排）做 2D FFT，徑向分頻段：
```
1 GT 的能量分布          驗證 1/f^2（前提）
2 **殘差**的能量分布     像素域 loss 的梯度就正比於這個 => 誰在主宰優化
3 平衡後高頻的增益倍率    = (1/高頻現有份額) / 頻段數 => **這條線的天花板**
4 失敗 tile vs 成功 tile 的殘差差異落在**哪個頻段**  <- 決定性的一欄
```
```
差異集中在高頻 => 病是高頻欠擬合 => f-loss 對症，舊記憶要修
差異集中在低頻 => 舊記憶是對的，f-loss 打錯頻段 => 收線（省一次 9.6h）
```
⚠ 本工具量的是**渲染結果的殘差**，不是 3DGS 的參數空間。
  「高頻份額低」是必要條件不是充分條件 —— 還要那個頻段真的有可學的東西。

用法: python tools/spectrum_split.py agd2_b12 sched30_b12 --blk 12
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def radial_bands(h, w, nb):
    """回傳每個 FFT 格點所屬的頻段（0=最低頻），依歸一化半徑等分。"""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    r = r / r.max()
    b = np.clip((r * nb).astype(np.int32), 0, nb - 1)
    return b


def band_energy(a, b, nb):
    F = np.fft.fft2(a)
    P = (F.real ** 2 + F.imag ** 2)
    return np.bincount(b.ravel(), weights=P.ravel(), minlength=nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--nbands", type=int, default=6)
    args = ap.parse_args()

    nb = args.nbands
    d0 = final_test_dir(args.runs[0], args.blk)
    dirs = {r: final_test_dir(r, args.blk) for r in args.runs}
    fs = sorted(f for f in os.listdir(d0) if f.endswith(".png"))

    tot_gt = np.zeros(nb)
    tot_res = np.zeros(nb)
    fail_res = np.zeros(nb)
    succ_res = np.zeros(nb)
    fail_gt = np.zeros(nb)
    succ_gt = np.zeros(nb)
    n_img = n_ft = n_st = 0
    bfull = btile = None
    T = args.tile

    for f in fs:
        outs = {r: per_image(os.path.join(dirs[r], f), T, args.contrast_q) for r in args.runs}
        if any(o is None for o in outs.values()):
            continue
        keep, *_z, nx, ny = outs[args.runs[0]]
        ms = np.stack([outs[r][1] <= args.r_min for r in args.runs])
        allb, neverb = ms.all(0), ~ms.any(0)

        im = Image.open(os.path.join(d0, f))
        w = im.width // 2
        G = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
        R = np.asarray(im.crop((w, 0, im.width, im.height)).convert("L"), np.float32) / 255.
        E = G - R

        if bfull is None:
            bfull = radial_bands(*G.shape, nb)
            btile = radial_bands(T, T, nb)
        tot_gt += band_energy(G - G.mean(), bfull, nb)
        tot_res += band_energy(E, bfull, nb)

        Gt, Et = tiles(G, T), tiles(E, T)
        for idx, accR, accG, cnt in ((keep[allb], fail_res, fail_gt, "f"),
                                     (keep[neverb], succ_res, succ_gt, "s")):
            for t in idx:
                g2 = Gt[t].reshape(T, T)
                e2 = Et[t].reshape(T, T)
                # ⚠ 兩邊都要減去 tile 均值才對得起判準：`per_image` 的 corr 是
                #   減均值後算的（**不看 DC**）。只對 GT 減、對殘差不減，
                #   會把「整塊偏亮/偏暗」灌進殘差的最低頻段，做出假的低頻結論。
                accG += band_energy(g2 - g2.mean(), btile, nb)
                accR += band_energy(e2 - e2.mean(), btile, nb)
            if cnt == "f":
                n_ft += len(idx)
            else:
                n_st += len(idx)
        n_img += 1

    def share(x):
        return x / max(x.sum(), 1e-30)

    print(f"共 {n_img} 張圖；失敗 tile {n_ft:,} 個、成功 tile {n_st:,} 個；"
          f"徑向 {nb} 段（0=最低頻）\n")
    print(f"{'頻段':>6} {'GT 能量份額':>13} {'**殘差**份額':>14} {'殘差/GT':>10}")
    sg, sr = share(tot_gt), share(tot_res)
    for i in range(nb):
        print(f"{i:>6} {100*sg[i]:>12.3f}% {100*sr[i]:>13.3f}% {sr[i]/max(sg[i],1e-30):>10.3f}")
    hi = nb // 2
    print(f"\n  GT 的低頻佔比（段 0）    = {100*sg[0]:.2f}%   "
          f"=> 1/f^2 前提 {'✅ 成立' if sg[0] > 0.5 else '⚠ 不明顯'}")
    print(f"  **殘差**的低頻佔比（段 0）= {100*sr[0]:.2f}%")
    print(f"  殘差高頻份額（段 >= {hi}） = {100*sr[hi:].sum():.3f}%")
    print(f"  ★ 完全平衡後高頻的增益倍率 = **{(1.0/nb)/max(sr[hi:].sum()/(nb-hi), 1e-30):.2f}x**"
          f"   （這條線的天花板）")

    print(f"\n  ★★ 決定性的一欄：失敗 vs 成功 tile 的殘差，差異落在哪個頻段")
    print(f"{'頻段':>6} {'失敗殘差份額':>14} {'成功殘差份額':>14} {'比值':>9} "
          f"{'失敗GT份額':>12} {'成功GT份額':>12}")
    fr, sr2 = share(fail_res), share(succ_res)
    fg, sg2 = share(fail_gt), share(succ_gt)
    for i in range(nb):
        print(f"{i:>6} {100*fr[i]:>13.3f}% {100*sr2[i]:>13.3f}% "
              f"{fr[i]/max(sr2[i],1e-30):>9.3f} {100*fg[i]:>11.3f}% {100*sg2[i]:>11.3f}%")
    # 絕對能量比（不歸一化）—— 份額會被總量掩蓋
    print(f"\n{'頻段':>6} {'失敗/成功 殘差絕對能量比':>26} {'失敗/成功 GT 絕對能量比':>26}")
    for i in range(nb):
        print(f"{i:>6} {fail_res[i]/max(succ_res[i],1e-30):>25.3f} "
              f"{fail_gt[i]/max(succ_gt[i],1e-30):>25.3f}")

    print("""
判讀：
  失敗/成功的殘差比在**高頻**明顯較大 => 病是高頻欠擬合 => f-loss/FreGS 家族對症
                                        （舊記憶「我們的病是低頻」要修）
  在**低頻**較大或各頻段相近           => 舊記憶成立，頻率加權打錯頻段 => 收線
⚠ 同時看 GT 的絕對能量比：若失敗 tile 的 GT 高頻本來就多很多，
  那「殘差高頻多」只是因為**那裡本來就有比較多高頻可以錯**，不是欠擬合的證據。
  真正要看的是 **殘差比 / GT 比**（每單位可學內容的錯誤量）。""")


if __name__ == "__main__":
    main()
