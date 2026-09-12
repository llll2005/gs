#!/usr/bin/env python
"""每台相機真正對應的是哪個影像檔？——對**全部**影像做一次全域匹配（純 CPU）

## 為什麼需要全域掃描（使用者 2026-09-12 的提問）

dataparser 用**位置對應**（名稱排序第 k 個 配 檔名排序第 k 個）。若檔案編號相對於
SfM 名稱被**重排**過（1..10 -> 1,3,5,7,9,2,4,6,8,10 這種），位置對應會靜默吸收它。
局部偏移掃描（±3）已做過且無鑑別力，而**重排不是偏移** ⇒ 必須全域掃。

## 簽名怎麼來（不依賴任何我方程式碼）

```
檔案側：影像轉灰 -> 8x8 縮圖 -> 64 維向量
相機側：該相機的 SfM 特徵點（xys）帶著 points3D 的 RGB
        -> 按 xys 落在哪個 8x8 格，取該格的平均灰 -> 同樣 64 維
```
點的 RGB 是 COLMAP 從**觀測它的影像**取的（本資料集已驗證是真重建：重投影殘差
中位 0.41px、RGB 只有 0.54% 是灰）⇒ 正確的檔案應該明顯勝出。
單像素取樣已證無鑑別力（高頻紋理），8x8 聚合正是為了避開那個問題。

判讀：
```
最佳匹配 = dataparser 的指派，且 margin 明顯 => 位置對應正確
最佳匹配是別的檔案，且一致地偏移/重排        => **找到問題了**
margin 全都很小                              => 簽名沒鑑別力，不可下結論
```

用法: python tools/global_pairing_scan.py --grid 8 --sample 300
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
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--sample", type=int, default=300, help="驗幾台相機（檔案側一定全載）")
    ap.add_argument("--min-cells", type=int, default=40, help="相機簽名至少要填滿幾格")
    args = ap.parse_args()

    import internal.utils.colmap as cu
    sd = os.path.join(args.data, "sparse", "0")
    imb = cu.read_images_binary(os.path.join(sd, "images.bin"))
    p3 = cu.read_points3D_binary(os.path.join(sd, "points3D.bin"))
    IN = os.path.join(args.data, args.image_dir)
    files = sorted(f for f in os.listdir(IN) if f.lower().endswith((".png", ".jpg")))
    names = sorted(v.name for v in imb.values())
    assert len(files) == len(names), f"檔案 {len(files)} != SfM {len(names)}"
    pos = dict(zip(names, range(len(files))))       # dataparser 的指派（索引）
    G = args.grid

    print(f"載入 {len(files):,} 個檔案的 {G}x{G} 灰階簽名…", flush=True)
    F = np.zeros((len(files), G * G), np.float32)
    for i, f in enumerate(files):
        im = Image.open(os.path.join(IN, f))
        im.draft("L", (im.width // 8, im.height // 8))      # 解碼時就縮小，快很多
        F[i] = np.asarray(im.convert("L").resize((G, G), Image.BILINEAR), np.float32).reshape(-1) / 255.
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{len(files):,}", flush=True)
    Fz = F - F.mean(1, keepdims=True)
    Fz /= np.maximum(np.linalg.norm(Fz, axis=1, keepdims=True), 1e-9)

    recs = {v.name: v for v in imb.values()}
    rng = np.random.default_rng(0)
    picked = [names[i] for i in rng.choice(len(names), min(args.sample, len(names)), replace=False)]
    hits, offs, margins, nolabel = 0, {}, [], 0
    W = H = None
    cam0 = list(cu.read_cameras_binary(os.path.join(sd, "cameras.bin")).values())[0]
    W, H = cam0.width, cam0.height
    print(f"\n比對 {len(picked)} 台相機（影像 {W}x{H}）…")
    for nm in picked:
        rec = recs[nm]
        h3 = rec.point3D_ids != -1
        xy = rec.xys[h3]
        ids = rec.point3D_ids[h3]
        k = np.asarray([q in p3 for q in ids])
        xy, ids = xy[k], ids[k]
        if not len(ids):
            continue
        rgb = np.stack([p3[q].rgb for q in ids]).astype(np.float32) / 255.
        grey = rgb.mean(1)
        gx = np.clip((xy[:, 0] / W * G).astype(int), 0, G - 1)
        gy = np.clip((xy[:, 1] / H * G).astype(int), 0, G - 1)
        cell = gy * G + gx
        s = np.bincount(cell, weights=grey, minlength=G * G)
        c = np.bincount(cell, minlength=G * G)
        if (c > 0).sum() < args.min_cells:
            nolabel += 1
            continue
        sig = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        m = ~np.isnan(sig)
        v = sig[m] - sig[m].mean()
        v /= max(np.linalg.norm(v), 1e-9)
        # 只用有值的格子比（對每個候選檔案取同樣的格子）
        cand = Fz[:, m]
        cand = cand - cand.mean(1, keepdims=True)
        cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-9)
        sc = cand @ v
        best = int(np.argmax(sc))
        srt = np.sort(sc)[::-1]
        margins.append(float(srt[0] - srt[1]))
        d = best - pos[nm]
        offs[d] = offs.get(d, 0) + 1
        hits += int(d == 0)
    n = len(margins)
    print(f"\n可比對 {n} 台（{nolabel} 台格子填不滿被跳過）")
    print(f"最佳匹配 == dataparser 指派：**{hits}/{n} = {100*hits/max(n,1):.1f}%**")
    top = sorted(offs.items(), key=lambda x: -x[1])[:8]
    print(f"偏移分布（索引差 -> 幾台）前八：{top}")
    print(f"margin 中位 {np.median(margins):.4f}  平均 {np.mean(margins):.4f}")
    if np.median(margins) < 0.02:
        print("⚠ margin 太小 => 簽名沒鑑別力，**不可下任何結論**（要加大 grid 或換訊號）")
    elif hits / max(n, 1) > 0.95:
        print("✅ 位置對應正確（全域掃描，不只局部偏移）")
    else:
        print("⛔ 位置對應有問題 —— 看偏移分布是「一致偏移」還是「散亂＝重排」")


if __name__ == "__main__":
    main()
