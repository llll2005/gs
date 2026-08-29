#!/usr/bin/env python
"""疊影（半透明鬼影）偵測 —— 全套指標唯一抓不到的失效模式。

背景（2026-08-26，使用者目視 notrim2_b7 後提出，並提供標記樣本 1732/1756）：
  `_ctx.md` 明載「使用者目視發現的疊影，PSNR/LPIPS/建築低頻/floater% 全都沒抓到」。

⛔ 第一版（`ghost_blur_split.py`，梯度比值）**失敗**：假設「鬼影＝多出邊緣」，
   但實際看圖是「鬼影＝半透明白紗把底下的細節**洗淡**」=> 梯度是**下降**不是上升。
   實測 1732（使用者標為疊影）的 ghost% 只有 0.43%，幾乎最低。

✅ 正確的 signature 是 alpha blending 本身：
       render = (1-alpha)*GT + alpha*C
   => 逐 tile 對 GT 線性迴歸 render，得到
       增益 a = 1-alpha  < 1     偏移 b = alpha*C > 0     相關 r 不變（結構還在）
   這就把兩種病分開了：
       疊影  r 高（結構保留） + a 明顯 < 1（被洗淡） + b > 0（變亮）
       糊掉  r 低（結構根本消失）—— 1553 那種半張深灰虛空
       正常  r 高、a ≈ 1、b ≈ 0
   而 `1 - a` 直接是**鬼影不透明度的估計**，有物理意義。

用法:  python tools/veil_detect.py notrim2_b7:7 sched30_b7:7
       python tools/veil_detect.py notrim2_b7:7 --per-image
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image


def final_test_dir(run, blk):
    ds = glob.glob(f"outputs/{run}/blocks/block_{blk}/test/*/")
    if not ds:
        return None
    return max(ds, key=lambda d: int(re.search(r"step=(\d+)", d).group(1))
               if re.search(r"step=(\d+)", d) else -1)


def tiles(a, t):
    h, w = a.shape
    a = a[:h // t * t, :w // t * t]
    return a.reshape(h // t, t, w // t, t).transpose(0, 2, 1, 3).reshape(-1, t * t)


def analyse(path, t, r_min, veil_min, contrast_q):
    im = Image.open(path)
    w = im.width // 2
    g = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
    r = np.asarray(im.crop((w, 0, im.width, im.height)).convert("L"), np.float32) / 255.
    G, R = tiles(g, t), tiles(r, t)
    sg, sr = G.std(axis=1), R.std(axis=1)
    # 只看 GT 本身有對比的 tile：平坦處（天空/水面）的迴歸無意義
    keep = sg > np.quantile(sg, contrast_q)
    if keep.sum() < 8:
        return None
    G, R, sg, sr = G[keep], R[keep], sg[keep], sr[keep]
    gm, rm = G.mean(axis=1, keepdims=True), R.mean(axis=1, keepdims=True)
    cov = ((G - gm) * (R - rm)).mean(axis=1)
    a = cov / np.maximum(sg ** 2, 1e-8)                    # 增益
    corr = cov / np.maximum(sg * sr, 1e-8)                 # 相關
    b = rm.ravel() - a * gm.ravel()                        # 偏移
    veil = np.clip(1.0 - a, 0.0, 1.0)                      # 估計的鬼影不透明度
    is_veil = (corr > r_min) & (veil > veil_min) & (b > 0)  # 結構在、被洗淡、且變亮
    is_blur = corr <= r_min                                 # 結構消失
    return (float(is_veil.mean()), float(is_blur.mean()),
            float(np.median(veil[corr > r_min])) if (corr > r_min).any() else float("nan"),
            float(np.median(corr)), int(keep.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="run[:block]")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60, help="相關低於此 = 結構消失（糊）")
    ap.add_argument("--veil-min", type=float, default=0.25, help="1-a 高於此 = 明顯被洗淡")
    ap.add_argument("--contrast-q", type=float, default=0.40, help="只看 GT 對比高於此分位數的 tile")
    ap.add_argument("--per-image", action="store_true")
    args = ap.parse_args()

    print(f"tile={args.tile}  疊影判準: corr>{args.r_min} 且 1-a>{args.veil_min} 且 b>0  "
          f"糊掉判準: corr<={args.r_min}")
    print(f"{'run':>16} {'疊影%':>8} {'糊掉%':>8} {'鬼影alpha中位':>13} {'corr中位':>9} {'張數':>5}")
    for spec in args.specs:
        run, _, b = spec.partition(":")
        d = final_test_dir(run, b or "12")
        fs = sorted(glob.glob(d + "*.png")) if d else []
        rows = [(os.path.basename(p).split(".")[0],
                 analyse(p, args.tile, args.r_min, args.veil_min, args.contrast_q)) for p in fs]
        rows = [(n, s) for n, s in rows if s is not None]
        if not rows:
            print(f"{spec:>16}  無 test 圖")
            continue
        print(f"{spec:>16} {100*np.mean([s[0] for _,s in rows]):>7.2f}% "
              f"{100*np.mean([s[1] for _,s in rows]):>7.2f}% "
              f"{np.nanmedian([s[2] for _,s in rows]):>13.3f} "
              f"{np.median([s[3] for _,s in rows]):>9.3f} {len(rows):>5}")
        if args.per_image:
            for n, s in sorted(rows, key=lambda x: -x[1][0])[:12]:
                print(f"{'':>16}   {n:>6}  疊影 {100*s[0]:5.1f}%  糊掉 {100*s[1]:5.1f}%  "
                      f"alpha {s[2]:.3f}  corr {s[3]:.3f}")
    print("""
判讀：疊影% = 結構還在但被半透明層洗淡（corr 高、增益 < 1、變亮）
      糊掉% = 結構整個消失（corr 低）—— 那是尾巴那幾張災難視角
      兩者方向相反，全域紋理比會把它們平均掉 => 這正是它抓不到疊影的原因。""")


if __name__ == "__main__":
    main()
