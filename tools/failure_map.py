#!/usr/bin/env python
"""乾淨判準下，失敗區還是一個「區域」嗎？—— 把失敗 tile 映射到世界座標（純 CPU）

## 為什麼要重做

`blur_persistence` 家族用 `GT std > quantile(sg, 0.40)` 這個**相對**門檻保留 tile，
而空拍圖的 40 分位仍近乎平坦 ⇒ 母體混進大量「兩張近乎平坦的圖」，那種 corr 是噪音。
實測（2026-09-12，§16.10）：最低三分位的「糊掉率」77~99.9%，其中 33~76% 的 |corr|<0.2；
最平坦那格 render 的標準差還**大於** GT（1.31x）。而 `speed3_b12` 與 `agd2_b12`
**逐分位一致到 0.1%** ⇒ 那個「八配方都是 46%」的常數有一大部分是 GT 的性質。

⇒ 本工具改用：
```
絕對對比門檻   GT std >= --min-std（預設 0.10；0~1 灰階）
失敗判準       corr < --r-min  且同時報**紋理比** render_std/GT_std
                （紋理比是本專案唯一驗證過有鑑別力的尺，見 feedback_metric_resolution）
定位           每個 tile 的世界座標 = 該 tile 內 **SfM 在這張影像上觀測到**的點的中位
                （可見性經 COLMAP 驗證；並過濾離群：|z - 地表| > --z-clip 的點丟掉，
                 實測有點在地表下 20.1 單位）
```
**映射到世界座標的附加價值**：不同 `down_sample` 的跑次（例如 `normal_b12` 是全解析度、
四臂是 1.2）tile 網格不同、無法逐 tile 取交集（2026-09-12 踩過 ValueError），
但在世界座標上可以直接比。

用法:
  python tools/failure_map.py speed3_b12 agd2_b12 --blk 12
  python tools/failure_map.py normal_b12 --blk 12 --down-sample 1
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402

GROUND_Z = 0.26


def collect(run, blk, tile, ds, min_std, r_min, z_clip, p3, imb, name2id, ts, cams):
    import torch
    name2idx = {n: i for i, n in enumerate(ts.image_names)}
    recs = []
    files = sorted(glob.glob(os.path.join(final_test_dir(run, blk), "*.png")))
    for p in files:
        fn = os.path.basename(p)
        key = fn[:-4] if fn.endswith(".png.png") else fn
        if key not in name2idx or key not in name2id:
            continue
        i = name2idx[key]
        im = Image.open(p)
        w = im.width // 2
        g = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
        r = np.asarray(im.crop((w, 0, im.width, im.height)).convert("L"), np.float32) / 255.
        G, R = tiles(g, tile), tiles(r, tile)
        sg, sr = G.std(1), R.std(1)
        gm, rm = G.mean(1, keepdims=True), R.mean(1, keepdims=True)
        corr = ((G - gm) * (R - rm)).mean(1) / np.maximum(sg * sr, 1e-8)
        nx, ny = w // tile, im.height // tile

        rec = imb[name2id[key]]
        h3 = rec.point3D_ids != -1
        xy = rec.xys[h3] / ds
        pids = rec.point3D_ids[h3]
        if not len(pids):
            continue
        pts = np.asarray([p3[q].xyz for q in pids], np.float32)
        keepz = np.abs(pts[:, 2] - GROUND_Z) <= z_clip        # 離群過濾
        xy, pts = xy[keepz], pts[keepz]
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < im.height)
        xy, pts = xy[ok], pts[ok]
        tix = (xy[:, 1] // tile).astype(np.int64) * nx + (xy[:, 0] // tile).astype(np.int64)
        order = np.argsort(tix)
        tix, pts = tix[order], pts[order]
        bounds = np.searchsorted(tix, np.arange(nx * ny + 1))
        for t in range(nx * ny):
            if sg[t] < min_std:
                continue
            a, b = bounds[t], bounds[t + 1]
            if b - a < 3:
                continue
            xyz = np.median(pts[a:b], axis=0)
            recs.append((float(corr[t]), float(sr[t] / max(sg[t], 1e-8)),
                         float(xyz[0]), float(xyz[1]), float(xyz[2]), int(b - a)))
    return np.asarray(recs, dtype=np.float64), len(files)


def ascii_map(rec, r_min, n=12):
    """把失敗率畫成 XY 網格。0-9 = 失敗率 0~90%+，'.' = 樣本太少"""
    x, y, bad = rec[:, 2], rec[:, 3], rec[:, 0] < r_min
    xs = np.linspace(x.min(), x.max(), n + 1)
    ys = np.linspace(y.min(), y.max(), n + 1)
    print(f"    失敗率地圖（X {x.min():.2f}~{x.max():.2f} / Y {y.min():.2f}~{y.max():.2f}，"
          f"每格數字＝失敗率的十位數；. ＝ 樣本 <8）")
    for j in range(n - 1, -1, -1):
        row = ""
        for i in range(n):
            m = (x >= xs[i]) & (x < xs[i + 1]) & (y >= ys[j]) & (y < ys[j + 1])
            row += "." if m.sum() < 8 else str(min(9, int(10 * bad[m].mean())))
        print(f"      {row}")
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", type=int, default=12)
    ap.add_argument("--block-dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--down-sample", type=float, default=1.2)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--min-std", type=float, default=0.10,
                    help="**絕對**對比門檻（0~1 灰階）。相對門檻擋不住平坦 tile，見檔頭")
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--z-clip", type=float, default=5.0, help="|z-地表| 超過此值的 SfM 點視為離群")
    args = ap.parse_args()

    import internal.utils.colmap as cu
    from internal.dataparsers.colmap_block_dataparser import ColmapBlock
    sd = "data/matrix_city/aerial/train/block_all/sparse/0"
    p3 = cu.read_points3D_binary(os.path.join(sd, "points3D.bin"))
    imb = cu.read_images_binary(os.path.join(sd, "images.bin"))
    name2id = {v.name: k for k, v in imb.items()}
    out = ColmapBlock(block_id=args.blk, block_dim=list(args.block_dim), content_threshold=0.08,
                      image_dir="input", down_sample_factor=args.down_sample).instantiate(
        path="data/matrix_city/aerial/train/block_all", output_path="/tmp",
        global_rank=0).get_outputs()
    ts, cams = out.train_set, out.train_set.cameras

    print(f"判準：GT std >= {args.min_std}（絕對）／corr < {args.r_min}／"
          f"tile {args.tile}px／SfM 離群過濾 |z-{GROUND_Z}| <= {args.z_clip}")
    store = {}
    for run in args.runs:
        rec, nf = collect(run, args.blk, args.tile, args.down_sample, args.min_std,
                          args.r_min, args.z_clip, p3, imb, name2id, ts, cams)
        if not len(rec):
            print(f"\n{run}: 沒有通過門檻的 tile")
            continue
        store[run] = rec
        bad = rec[:, 0] < args.r_min
        print(f"\n=== {run} === {nf} 張影像／通過門檻 {len(rec):,} tile")
        print(f"    失敗率 **{100*bad.mean():.2f}%**   corr 中位 {np.median(rec[:,0]):.3f}   "
              f"紋理比 中位 {np.median(rec[:,1]):.3f}（失敗 {np.median(rec[bad,1]):.3f} / "
              f"通過 {np.median(rec[~bad,1]):.3f}）")
        print(f"    失敗 tile 的世界高度（地表 {GROUND_Z}）中位 {np.median(rec[bad,4]):.3f}"
              f"  vs 通過 {np.median(rec[~bad,4]):.3f}")
        ascii_map(rec, args.r_min)

    if len(store) >= 2:
        print(f"\n=== 跨配方一致性（世界座標 0.5 單位格）===")
        ks = list(store)
        cell = {}
        for k in ks:
            r = store[k]
            key = (np.floor(r[:, 2] / .5).astype(int), np.floor(r[:, 3] / .5).astype(int))
            d = {}
            for cx, cy, b in zip(key[0], key[1], r[:, 0] < args.r_min):
                d.setdefault((cx, cy), []).append(b)
            cell[k] = {c: float(np.mean(v)) for c, v in d.items() if len(v) >= 8}
        common = set.intersection(*[set(cell[k]) for k in ks])
        print(f"    兩邊都有 >=8 樣本的格子 {len(common)} 個")
        if len(common) >= 5:
            a = np.asarray([cell[ks[0]][c] for c in sorted(common)])
            b = np.asarray([cell[ks[1]][c] for c in sorted(common)])
            print(f"    {ks[0]} 平均失敗率 {a.mean():.3f} / {ks[1]} {b.mean():.3f}")
            print(f"    **逐格失敗率相關 r = {np.corrcoef(a,b)[0,1]:.4f}**")
            print("""    判讀：r 高 => 失敗位置確實跨配方固定（在乾淨判準下也成立）
          r 低 => 位置不固定 => 「病灶由輸入決定」要收回""")


if __name__ == "__main__":
    main()
