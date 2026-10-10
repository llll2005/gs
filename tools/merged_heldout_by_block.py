"""合併模型的 held-out 按 4x4 塊拆開（2026-10-10 使用者問：圖 9b 的 b12 單塊輸很多，合併後還輸嗎？）

單塊 held-out（task_full44.sh offheldout）比的是「單塊模型」，而官方單塊含塊外內容 => 偏向官方。
這裡改比**合併模型**在同一批塊視角（eval_official_test --block B --block_dim 4 4 的選法）上的逐幀 PSNR：
  我方   = eval_official_test --save_dir 存下的「左 GT／右渲染」PNG（uint8 量化，與浮點 PSNR 差約 0.01 dB）
  官方   = 我方評分工具對官方合併模型的逐幀 CSV（--per_image_csv；release 那份在 紀錄/實驗分析/data/evalgap/ours_release.csv）
純 CPU，不載模型。

用法：python tools/merged_heldout_by_block.py --ours_dir outputs/lab/full44_best/test_official \
        --official_csv 紀錄/實驗分析/data/evalgap/ours_release.csv [--blocks 12]
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", required=True)
    ap.add_argument("--official_csv", required=True)
    ap.add_argument("--blocks", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--block_dim", type=int, nargs=2, default=[4, 4])
    ap.add_argument("--train_dir", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--worst", type=int, default=8, help="每塊列出我方落後最多的幾幀")
    a = ap.parse_args()
    import eval_official_test as eot
    names, cams = eot.load_test_cameras(a.test_dir, 1.2)
    centres = np.array([(-cams.R[i].numpy().T @ cams.T[i].numpy()) for i in range(len(names))])
    off = {r["name"]: float(r["psnr"]) for r in csv.DictReader(open(a.official_csv))}
    cache = {}

    def ours_psnr(n):
        if n not in cache:
            im = np.asarray(Image.open(os.path.join(a.ours_dir, n.rsplit(".", 1)[0] + "_gt_vs_render.png")).convert("RGB"), np.float64) / 255.
            w = im.shape[1] // 2
            cache[n] = float(-10 * np.log10(max(((im[:, :w] - im[:, w:]) ** 2).mean(), 1e-12)))
        return cache[n]

    print(f"{'塊':>3} {'幀':>4} {'我方合併':>9} {'官方合併':>9} {'差':>7} {'我方贏的幀':>10}")
    for B in a.blocks:
        lo, hi, _ = eot.block_bounds(a.train_dir, B, a.block_dim)
        sel = [i for i in range(len(names)) if np.all(centres[i] >= lo) and np.all(centres[i] <= hi)]
        if not sel:
            print(f"{B:>3}    0"); continue
        o = np.array([ours_psnr(names[i]) for i in sel]); r = np.array([off[names[i]] for i in sel])
        print(f"{B:>3} {len(sel):>4} {o.mean():>9.3f} {r.mean():>9.3f} {o.mean() - r.mean():>+7.3f} {int((o > r).sum()):>5}/{len(sel)}")
        if len(a.blocks) <= 3:
            d = o - r; k = np.argsort(d)[:a.worst]
            print("     我方落後最多：" + "  ".join(f"{names[sel[j]]} {o[j]:.2f}/{r[j]:.2f}" for j in k))
            k = np.argsort(-d)[:4]
            print("     我方領先最多：" + "  ".join(f"{names[sel[j]]} {o[j]:.2f}/{r[j]:.2f}" for j in k))
    allo = np.array([ours_psnr(n) for n in names]); allr = np.array([off[n] for n in names])
    print(f"全部 {len(names)} 幀（PNG 量化版）：我方 {allo.mean():.3f}／官方 {allr.mean():.3f}")


if __name__ == "__main__":
    main()
