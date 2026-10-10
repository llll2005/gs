"""合併模型的 held-out 按 4x4 塊拆開（2026-10-10 使用者問：圖 9b 的 b12 單塊輸很多，合併後還輸嗎？）

單塊 held-out（task_full44.sh offheldout）比的是「單塊模型」，而官方單塊含塊外內容 => 偏向官方
（b12：我方單塊 22.55 vs 官方 26.39，合併後 27.82 vs 27.10）。
這裡改比**合併模型**在同一批塊視角（eval_official_test --block B --block_dim 4 4 的選法）上的逐幀數字：
  每條線一份逐幀 CSV（eval_official_test --per_image_csv；含 n_tiles／n_contrast／n_fail 時一併報失敗 tile）
  或 --ours_dir：eval_official_test --save_dir 存下的「左 GT／右渲染」PNG（只有 PSNR；uint8 量化，與浮點差約 0.01 dB）
第一條線是比較基準（差 = 其他線 − 第一條線）。純 CPU，不載模型。

用法：python tools/merged_heldout_by_block.py --csv 我方=logs/a.csv release=logs/b.csv [--blocks 12]
      python tools/merged_heldout_by_block.py --ours_dir outputs/lab/full44_best/test_official --csv release=logs/evalgap/ours_release.csv
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
    ap.add_argument("--csv", nargs="+", default=[], help="名稱=逐幀 CSV 路徑")
    ap.add_argument("--ours_dir", default=None, help="（可選）我方 PNG 目錄，當作名稱「我方」的第一條線")
    ap.add_argument("--blocks", type=int, nargs="+", default=list(range(16)))
    ap.add_argument("--block_dim", type=int, nargs=2, default=[4, 4])
    ap.add_argument("--train_dir", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--worst", type=int, default=8, help="只看 1~3 塊時，列出第二條線領先第一條最多的幾幀")
    a = ap.parse_args()
    import eval_official_test as eot
    names, cams = eot.load_test_cameras(a.test_dir, 1.2)
    centres = np.array([(-cams.R[i].numpy().T @ cams.T[i].numpy()) for i in range(len(names))])

    lines = {}   # 名稱 -> {幀名: (psnr, n_tiles, n_contrast, n_fail) }
    if a.ours_dir:
        d = {}
        for n in names:
            im = np.asarray(Image.open(os.path.join(a.ours_dir, n.rsplit(".", 1)[0] + "_gt_vs_render.png")).convert("RGB"), np.float64) / 255.
            w = im.shape[1] // 2
            d[n] = (float(-10 * np.log10(max(((im[:, :w] - im[:, w:]) ** 2).mean(), 1e-12))), np.nan, np.nan, np.nan)
        lines["我方"] = d
    for spec in a.csv:
        nm, path = spec.split("=", 1)
        d = {}
        for r in csv.DictReader(open(path)):
            g = lambda k: float(r[k]) if r.get(k) not in (None, "") else np.nan
            d[r["name"]] = (float(r["psnr"]), g("n_tiles"), g("n_contrast"), g("n_fail"))
        lines[nm] = d
    L = list(lines)
    has_fail = {k: not np.isnan(next(iter(v.values()))[3]) for k, v in lines.items()}

    hdr = f"{'塊':>3} {'幀':>4} " + " ".join(f"{k:>10}" for k in L) + " " + " ".join(f"{'差 ' + k:>10}" for k in L[1:])
    if any(has_fail.values()):
        hdr += "   失敗 tile %（全部 tile／過對比門檻）：" + "  ".join(k for k in L if has_fail[k])
    print(hdr)

    def row(tag, sel):
        P = {k: np.array([lines[k][names[i]][0] for i in sel]) for k in L}
        s = f"{tag:>3} {len(sel):>4} " + " ".join(f"{P[k].mean():>10.3f}" for k in L) + " " + \
            " ".join(f"{P[k].mean() - P[L[0]].mean():>+10.3f}" for k in L[1:])
        for k in L:
            if has_fail[k]:
                t = np.array([lines[k][names[i]][1:] for i in sel]).sum(0)
                s += f"   {k} {100 * t[2] / max(t[0], 1):.2f}／{100 * t[2] / max(t[1], 1):.2f}"
        print(s)
        return P

    for B in a.blocks:
        lo, hi, _ = eot.block_bounds(a.train_dir, B, a.block_dim)
        sel = [i for i in range(len(names)) if np.all(centres[i] >= lo) and np.all(centres[i] <= hi)]
        if not sel:
            print(f"{B:>3}    0"); continue
        P = row(str(B), sel)
        if len(a.blocks) <= 3 and len(L) >= 2:
            dd = P[L[1]] - P[L[0]]
            k = np.argsort(-dd)[:a.worst]
            print(f"     {L[1]} 領先 {L[0]} 最多：" + "  ".join(f"{names[sel[j]]} {P[L[0]][j]:.2f}/{P[L[1]][j]:.2f}" for j in k))
    row("全", list(range(len(names))))


if __name__ == "__main__":
    main()
