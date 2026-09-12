#!/usr/bin/env python
"""糊掉的 tile 那裡，SfM 有點嗎？（純 CPU，幾秒）

## 為什麼這一題幾乎免費卻重要

記憶 `sfm_init_beats_depth_init` 記著一個**已寫下但從未驗過**的假說：
「A/C/D 被 floater 佔滿的地方 = B（SfM-init）的破洞 = SfM 無特徵區
 ⇒ 若與 40~46% 糊掉 tile 重合，糊掉就不是優化失敗而是那裡沒有可用的對應」
需要的東西全都在手上：糊掉遮罩（`blur_persistence` 的判準）＋ SfM 點（`points3D.bin`）。

⚠ 而 `coverage_needs_inbox_filter` 已經用**渲染覆蓋率**否證過一個相近的說法
  （「60k 時糊掉 tile 的覆蓋與沒糊的相同 ⇒ 那裡沒資訊不成立」）。
  這裡問的是**不同的量**：不是「模型有沒有把東西放在那」，而是
  **「SfM 有沒有在那裡成功三角化出對應」**。前者是模型的輸出，後者是輸入的品質。

## 量什麼（每張參考影像、每個 tile）

⚠ **第一版是錯的**（2026-09-12，已修）：它把**全場景 3.83M 點**投影進來數，
  於是每個 tile 數到的是「這根視線柱裡有幾個點」—— 含其他街區、場外區域、
  被遮住的背面。證據是深度相對散度 1.3~1.7（標準差是均值的 1.3 倍以上）＝
  柱子橫跨整個場景。那一版量到「糊掉 tile 點更多 33%」，量的是柱子不是表面。
  正解＝只用 **SfM 在這張影像上實際觀測到的特徵**（`images.bin` 的
  `xys` + `point3D_ids`）—— 可見性是 COLMAP 驗證過的，座標是原解析度的偵測位置。

```
n_obs        該 tile 內、SfM 在**這張影像**上觀測到且有 3D 對應的特徵數
track_med    那些點的 track 長度中位（被幾張影像看到 = 對應的可信度）
depth_iqr    那些點的深度四分位距 / 中位（大 = 該 tile 深度不確定/跨越邊緣）
```
判讀：
```
糊掉組 n_sfm 明顯低  => 那裡 SfM 匹配不出來 => **輸入端就沒有幾何約束**
糊掉組 n_sfm 相當    => 有對應卻擬合不出來 => 回到表示法/損失
糊掉組 track 低      => 有點但只被兩三張看到 => 對應弱（BA 可自由移動它，見 graded_init）
```

用法: python tools/tile_sfm_density.py --mask-run speed3_b12 --blk 12
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-run", default="speed3_b12")
    ap.add_argument("--blk", type=int, default=12)
    ap.add_argument("--block-dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--down-sample", type=float, default=1.2)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--r-good", type=float, default=0.85)
    ap.add_argument("--contrast-q", type=float, default=0.40)
    args = ap.parse_args()

    import torch
    import internal.utils.colmap as cu
    from internal.dataparsers.colmap_block_dataparser import ColmapBlock

    sd = "data/matrix_city/aerial/train/block_all/sparse/0"
    p3 = cu.read_points3D_binary(os.path.join(sd, "points3D.bin"))
    pid = np.fromiter(p3.keys(), dtype=np.int64)
    pxyz = np.stack([p3[k].xyz for k in pid]).astype(np.float32)
    ptrack = np.fromiter((len(p3[k].image_ids) for k in pid), dtype=np.int32)
    print(f"SfM 全場景 {len(pid):,} 點   track 長度 中位 {np.median(ptrack):.1f}")

    cfg = ColmapBlock(block_id=args.blk, block_dim=list(args.block_dim),
                      content_threshold=0.08, image_dir="input",
                      down_sample_factor=args.down_sample)
    out = cfg.instantiate(path="data/matrix_city/aerial/train/block_all",
                          output_path="/tmp", global_rank=0).get_outputs()
    ts = out.train_set
    cams = ts.cameras
    name2idx = {n: i for i, n in enumerate(ts.image_names)}
    imgs_bin = cu.read_images_binary(os.path.join(sd, "images.bin"))
    name2id = {v.name: k for k, v in imgs_bin.items()}

    d = final_test_dir(args.mask_run, args.blk)
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    if not files:
        raise SystemExit(f"找不到 {args.mask_run} 的 test 影像")

    agg = {"糊掉": [], "清楚": []}
    for p in files:
        fn = os.path.basename(p)
        key = fn[:-4] if fn.endswith(".png.png") else fn
        if key not in name2idx:
            continue
        i = name2idx[key]
        im = Image.open(p)
        w = im.width // 2
        g = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
        r = np.asarray(im.crop((w, 0, im.width, im.height)).convert("L"), np.float32) / 255.
        G, R = tiles(g, args.tile), tiles(r, args.tile)
        sg = G.std(axis=1)
        keep = np.where(sg > np.quantile(sg, args.contrast_q))[0]
        Gk, Rk = G[keep], R[keep]
        gm, rm = Gk.mean(1, keepdims=True), Rk.mean(1, keepdims=True)
        corr = ((Gk - gm) * (Rk - rm)).mean(1) / np.maximum(Gk.std(1) * Rk.std(1), 1e-8)
        nx = w // args.tile

        # SfM 在**這張影像**上觀測到的特徵（xys 是原解析度 => 除以 down_sample）
        iid = name2id.get(key)
        if iid is None:
            continue
        rec = imgs_bin[iid]
        has3d = rec.point3D_ids != -1
        xy = rec.xys[has3d] / args.down_sample
        pids = rec.point3D_ids[has3d]
        tr = np.asarray([len(p3[q].image_ids) if q in p3 else 0 for q in pids], np.float64)
        pz = np.asarray([p3[q].xyz for q in pids], np.float32) if len(pids) else np.zeros((0, 3), np.float32)
        zc = (torch.from_numpy(pz) @ cams.R[i].T + cams.T[i])[:, 2].numpy() if len(pids) else np.zeros(0)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < im.height)
        u, v = xy[ok, 0], xy[ok, 1]
        nt_total = nx * (im.height // args.tile)
        tix = (v // args.tile).astype(np.int64) * nx + (u // args.tile).astype(np.int64)
        tix = np.clip(tix, 0, nt_total - 1)
        n_t = np.bincount(tix, minlength=nt_total)
        tr_sum = np.bincount(tix, weights=tr[ok], minlength=nt_total)
        z_sum = np.bincount(tix, weights=zc[ok], minlength=nt_total)
        z_sq = np.bincount(tix, weights=zc[ok] ** 2, minlength=nt_total)

        for lab, sel in (("糊掉", keep[corr < args.r_min]), ("清楚", keep[corr > args.r_good])):
            for t in sel:
                n = int(n_t[t]) if t < len(n_t) else 0
                tr = (tr_sum[t] / n) if n else 0.0
                if n >= 2:
                    mu = z_sum[t] / n
                    sd = max(z_sq[t] / n - mu * mu, 0) ** 0.5
                    rel = sd / max(mu, 1e-6)
                else:
                    rel = np.nan
                agg[lab].append((n, tr, rel))

    print(f"\n遮罩來源 {args.mask_run}   {len(files)} 張參考影像   tile {args.tile}px")
    print(f"{'組':>6} {'tile 數':>8} {'n_obs 中位':>11} {'n_obs 平均':>11} "
          f"{'零點 tile':>10} {'track 中位':>11} {'深度相對散度':>13}")
    for lab in ("清楚", "糊掉"):
        a = np.asarray(agg[lab], dtype=np.float64)
        if not len(a):
            print(f"{lab:>6} (無樣本)")
            continue
        print(f"{lab:>6} {len(a):>8,} {np.median(a[:,0]):>11.1f} {a[:,0].mean():>11.2f} "
              f"{100*np.mean(a[:,0]==0):>9.1f}% {np.median(a[a[:,0]>0,1]):>11.2f} "
              f"{np.nanmedian(a[:,2]):>13.4f}")
    A, B = (np.asarray(agg["清楚"], dtype=np.float64), np.asarray(agg["糊掉"], dtype=np.float64))
    if len(A) and len(B):
        for k, nm in ((0, "n_obs"), (1, "track")):
            a, b = A[:, k], B[:, k]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            sd = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
            print(f"  {nm}: 清楚 {a.mean():.2f} vs 糊掉 {b.mean():.2f}  "
                  f"差 {a.mean()-b.mean():+.2f} ({(a.mean()-b.mean())/max(sd,1e-12):+.1f}sd)")
    print("""
判讀：
  糊掉組 n_obs 明顯低 / 零點比例明顯高 => **輸入端就沒有幾何約束**（SfM 在那裡匹配不出來）
  兩組相當                            => 有對應卻擬合不出來 => 病灶在表示法或損失
  糊掉組 track 低                     => 有點但只被兩三張看到 => 對應弱（見 graded_init）
⚠ 與 `coverage_needs_inbox_filter` 的否證**不衝突**：那支量的是模型的渲染覆蓋（輸出），
  這支量的是 SfM 的三角化密度（輸入）。""")


if __name__ == "__main__":
    main()
