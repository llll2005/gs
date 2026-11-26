"""比對兩個 eval_official_test.py --save_dir 目錄的渲染圖（*_gt_vs_render.png 的右半＝渲染）。2026-11-26 新增。純 CPU。

python tools/compare_render_dirs.py <目錄 A> <目錄 B>
=> 每張：不同的像素數、最大差（0～255）、兩張渲染之間的 PSNR；最後印總結（逐位元相同幾張、最大差、最低 PSNR）。
用途：CITYGS_SH_KEEP_LOWDIM=1（不展開 SH）與原本展開版的渲染是不是看起來一樣。
"""
import glob
import os
import sys

import numpy as np
from PIL import Image


def render_half(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    return a[:, a.shape[1] // 2:]


def main(da, db):
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(da, "*_gt_vs_render.png")))
    if not names:
        raise SystemExit(f"⛔ {da} 沒有 *_gt_vs_render.png")
    same, tot_px, diff_px, maxd, psnrs, missing = 0, 0, 0, 0, [], 0
    worst = None
    for n in names:
        pb = os.path.join(db, n)
        if not os.path.exists(pb):
            missing += 1
            continue
        a, b = render_half(os.path.join(da, n)).astype(np.int16), render_half(pb).astype(np.int16)
        assert a.shape == b.shape, (n, a.shape, b.shape)
        d = np.abs(a - b)
        px = int(np.any(d > 0, axis=2).sum())
        tot_px += d.shape[0] * d.shape[1]
        diff_px += px
        m = int(d.max())
        maxd = max(maxd, m)
        mse = float((d.astype(np.float64) ** 2).mean())
        p = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
        psnrs.append(p)
        same += px == 0
        if worst is None or p < worst[1]:
            worst = (n, p, px, m)
    k = len(psnrs)
    finite = [p for p in psnrs if np.isfinite(p)]
    print(f"比對 {k} 張（B 缺 {missing} 張）：逐位元相同 {same} 張；不同的像素 {diff_px:,}／{tot_px:,}"
          f"（{100 * diff_px / max(tot_px, 1):.4f}%）；最大差 {maxd}／255")
    if finite:
        print(f"兩張渲染之間的 PSNR：最低 {min(finite):.2f} dB、中位 {np.median(finite):.2f} dB（有差的 {len(finite)} 張）；"
              f"最差那張 {worst[0]}：{worst[2]:,} 個像素不同、最大差 {worst[3]}")
    else:
        print("全部逐位元相同")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
