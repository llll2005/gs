#!/usr/bin/env python
"""把「疊影」和「模糊」分開量 —— 全套指標唯一抓不到的失效模式。

背景（2026-08-26，使用者目視 notrim2_b7 後提出）：
  `_ctx.md` 明載「使用者目視發現的疊影，PSNR/LPIPS/建築低頻/floater% 全都沒抓到」。
  原因是**全域紋理比把兩種相反的錯誤平均掉了**：
    疊影 = 同一結構畫兩次、其中一份位置錯且半透明 => render 的梯度**多於** GT
    模糊 = 結構糊成一團                          => render 的梯度**少於** GT
  平均起來就是一個中間值（notrim2_b7 全域 0.7144），兩種病都看不見。

做法：把影像切成 tile，每個 tile 算 |grad(render)| / |grad(GT)|，然後分開統計兩尾：
  ghost%  = 比值 > `--ghost` 的 tile 佔比（render 憑空多出結構）
  blur%   = 比值 < `--blur`  的 tile 佔比（結構消失）
⚠ 只統計 GT 有內容的 tile（`|grad GT|` 高於全圖分位數 `--floor-q`），否則天空/水面的
  數值雜訊會把比值炸掉。

用法:  python tools/ghost_blur_split.py notrim2_b7:7 sched30_b7:7
       python tools/ghost_blur_split.py notrim2_b7:7 --per-image | head -20
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


def grad_mag(a):
    gx = np.abs(np.diff(a, axis=1, append=a[:, -1:]))
    gy = np.abs(np.diff(a, axis=0, append=a[-1:, :]))
    return gx + gy


def tile_mean(a, t):
    h, w = a.shape
    a = a[:h // t * t, :w // t * t]
    return a.reshape(h // t, t, w // t, t).mean(axis=(1, 3))


def score(path, tile, ghost, blur, floor_q):
    im = Image.open(path)
    w = im.width // 2
    g = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
    r = np.asarray(im.crop((w, 0, im.width, im.height)).convert("L"), np.float32) / 255.
    tg, tr = tile_mean(grad_mag(g), tile), tile_mean(grad_mag(r), tile)
    keep = tg > np.quantile(tg, floor_q)          # 只看 GT 有內容的 tile
    if keep.sum() < 8:
        return None
    ratio = tr[keep] / np.maximum(tg[keep], 1e-6)
    return (float((ratio > ghost).mean()), float((ratio < blur).mean()),
            float(np.median(ratio)), int(keep.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="run[:block]")
    ap.add_argument("--tile", type=int, default=32)
    ap.add_argument("--ghost", type=float, default=1.30, help="比值高於此 = 憑空多出結構")
    ap.add_argument("--blur", type=float, default=0.50, help="比值低於此 = 結構消失")
    ap.add_argument("--floor-q", type=float, default=0.50, help="只看 GT 梯度高於此分位數的 tile")
    ap.add_argument("--per-image", action="store_true", help="逐張列出（依 ghost% 排序）")
    args = ap.parse_args()

    print(f"tile={args.tile}  ghost>{args.ghost}  blur<{args.blur}  "
          f"只看 GT 梯度前 {100*(1-args.floor_q):.0f}% 的 tile")
    print(f"{'run':>16} {'ghost%':>8} {'blur%':>8} {'中位比值':>9} {'張數':>5}")
    for spec in args.specs:
        run, _, b = spec.partition(":")
        d = final_test_dir(run, b or "12")
        fs = sorted(glob.glob(d + "*.png")) if d else []
        rows = [(os.path.basename(p).split(".")[0], score(p, args.tile, args.ghost,
                                                          args.blur, args.floor_q))
                for p in fs]
        rows = [(n, s) for n, s in rows if s is not None]
        if not rows:
            print(f"{spec:>16}  無 test 圖")
            continue
        gh = np.mean([s[0] for _, s in rows])
        bl = np.mean([s[1] for _, s in rows])
        md = np.median([s[2] for _, s in rows])
        print(f"{spec:>16} {100*gh:>7.2f}% {100*bl:>7.2f}% {md:>9.3f} {len(rows):>5}")
        if args.per_image:
            for n, s in sorted(rows, key=lambda x: -x[1][0])[:12]:
                print(f"{'':>16}   {n:>6}  ghost {100*s[0]:5.2f}%  blur {100*s[1]:5.2f}%  "
                      f"中位 {s[2]:.3f}")
    print("""
判讀：ghost% 高 = render 憑空多出結構（疊影／半透明重複幾何）
      blur%  高 = 結構消失（灰色一坨）
      全域紋理比看不見這兩者，因為它們方向相反、會互相抵消。""")


if __name__ == "__main__":
    main()
