#!/usr/bin/env python
"""影像檔與相機的配對，在**指定編號範圍**上真的對嗎？（純 CPU，獨立訊號）

## 為什麼要再驗一次（使用者 2026-09-12 提問）

我方 dataparser 用**位置對應**：「SfM 名稱排序的第 k 個」配「檔名排序的第 k 個」
（`[dataparser] 檔名不同名，改用位置對應：0000.png -> 000001.png`）。
⇒ **若檔案編號相對於 SfM 名稱被重排過（例如 1..10 變成 1,3,5,7,9,2,4,6,8,10），
   位置對應會靜默吸收它、配成錯的對，而且不會報任何錯。**
這與 `zfill_gt_offset_bug` 是同一家族；那次的證據（GT 配對 corr 0.9997）只驗了少數幾張。
另外有一個一直沒解釋的現象：**SfM `image_id 1` 的名稱是 `0007.png`** ⇒ id 順序 != 名稱順序。

⚠ 已證無效的判準：**SIFT 特徵點落在高梯度像素**（2026-09-12 實測，正確配對 1.0x、
  故意配錯 +1/+2/-1 也都 1.0x ⇒ 合成城市紋理太密，特徵點不特別）。不要再用它。

## 本工具用的獨立訊號

`points3D.bin` 的每個 3D 點帶 **RGB**，那是 COLMAP 從**觀測它的影像**取的顏色。
把某張影像觀測到的點投影回去、比「點的 RGB」與「候選影像該像素的 RGB」：
```
正確的檔案  => 相關度高（點色就是從它身上來的）
錯的檔案    => 相關度掉下來
```
並且對每個候選偏移都算一次 => 看**勝出的是不是 0（dataparser 的選擇）**、margin 多大。
margin 太小 ⇒ 判準在該範圍沒有鑑別力，不可下結論（不是「證明對了」）。

用法:
  python tools/verify_image_pairing.py --from-name 2900 --to-name 3000
  python tools/verify_image_pairing.py --from-name 0 --to-name 100 --offsets -3 3
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--image-dir", default="input")
    ap.add_argument("--from-name", type=int, default=2900)
    ap.add_argument("--to-name", type=int, default=3000)
    ap.add_argument("--step", type=int, default=10, help="每隔幾張驗一張")
    ap.add_argument("--offsets", type=int, nargs=2, default=[-3, 3])
    ap.add_argument("--min-pts", type=int, default=200)
    args = ap.parse_args()

    import internal.utils.colmap as cu
    sd = os.path.join(args.data, "sparse", "0")
    imb = cu.read_images_binary(os.path.join(sd, "images.bin"))
    p3 = cu.read_points3D_binary(os.path.join(sd, "points3D.bin"))
    IN = os.path.join(args.data, args.image_dir)
    files = sorted(f for f in os.listdir(IN) if f.lower().endswith((".png", ".jpg")))
    names = sorted(v.name for v in imb.values())
    name2file = dict(zip(names, files))          # dataparser 的位置對應
    name2rec = {v.name: v for v in imb.values()}
    print(f"SfM {len(imb):,} 張／檔案 {len(files):,} 個")
    print(f"位置對應範例：{names[0]} -> {files[0]}   {names[-1]} -> {files[-1]}")
    bad_order = [ (k, imb[k].name) for k in sorted(imb)[:3] ]
    print(f"⚠ id 順序 != 名稱順序 的證據：前三個 id -> 名稱 = {bad_order}")
    print(f"\n驗證範圍：名稱 {args.from_name:04d}.png ~ {args.to_name:04d}.png，每 {args.step} 張取一張")
    print(f"{'相機名':>10} {'dataparser 檔案':>18} {'點數':>6} "
          + " ".join(f"{d:>+6d}" for d in range(args.offsets[0], args.offsets[1] + 1))
          + f" {'勝出':>6} {'margin':>8}")

    cache = {}
    def load(fn):
        if fn not in cache:
            if len(cache) > 24:
                cache.clear()
            cache[fn] = np.asarray(Image.open(os.path.join(IN, fn)).convert("RGB"), np.float32) / 255.
        return cache[fn]

    wins, margins = {}, []
    for num in range(args.from_name, args.to_name + 1, args.step):
        nm = f"{num:04d}.png"
        if nm not in name2rec or nm not in name2file:
            continue
        rec = name2rec[nm]
        h3 = rec.point3D_ids != -1
        xy = rec.xys[h3]
        pids = rec.point3D_ids[h3]
        keep = np.asarray([q in p3 for q in pids])
        xy, pids = xy[keep], pids[keep]
        if len(pids) < args.min_pts:
            continue
        rgb = np.stack([p3[q].rgb for q in pids]).astype(np.float32) / 255.
        base = name2file[nm]
        bnum = int(os.path.splitext(base)[0])
        w = len(os.path.splitext(base)[0])
        scores = []
        for d in range(args.offsets[0], args.offsets[1] + 1):
            fn = f"{bnum + d:0{w}d}{os.path.splitext(base)[1]}"
            if not os.path.exists(os.path.join(IN, fn)):
                scores.append(np.nan); continue
            g = load(fn)
            H, W = g.shape[:2]
            ok = (xy[:, 0] >= 0) & (xy[:, 0] < W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < H - 1)
            if ok.sum() < args.min_pts:
                scores.append(np.nan); continue
            px = g[np.round(xy[ok, 1]).astype(int), np.round(xy[ok, 0]).astype(int)]
            a, b = rgb[ok].reshape(-1), px.reshape(-1)
            scores.append(float(np.corrcoef(a, b)[0, 1]))
        s = np.asarray(scores)
        if np.all(np.isnan(s)):
            continue
        bi = int(np.nanargmax(s))
        bd = bi + args.offsets[0]
        srt = np.sort(s[~np.isnan(s)])[::-1]
        mg = float(srt[0] - srt[1]) if len(srt) > 1 else float("nan")
        wins[bd] = wins.get(bd, 0) + 1
        margins.append(mg)
        print(f"{nm:>10} {base:>18} {int(ok.sum()):>6,} "
              + " ".join(("  n/a" if np.isnan(v) else f"{v:>6.3f}") for v in s)
              + f" {bd:>+6d} {mg:>8.3f}")

    print(f"\n勝出偏移統計：{dict(sorted(wins.items()))}    margin 中位 {np.median(margins):.3f}")
    tot = sum(wins.values())
    if wins.get(0, 0) == tot:
        print(f"✅ 全部 {tot} 張都由偏移 **0** 勝出 ⇒ 此範圍的位置對應正確")
    else:
        print(f"⛔ 有 {tot - wins.get(0,0)}/{tot} 張不是偏移 0 勝出 ⇒ **此範圍的配對有問題**")
    if np.median(margins) < 0.05:
        print(f"⚠ margin 中位只有 {np.median(margins):.3f} ⇒ 判準在此範圍鑑別力不足，"
              f"上面的結論**不可採信**（不是「證明對了」也不是「證明錯了」）")


if __name__ == "__main__":
    main()
