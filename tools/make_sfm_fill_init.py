#!/usr/bin/env python
"""SfM 點 + 空處補點 的初始化 PLY（使用者 2026-09-13 提案）—— 純 CPU。

## 這要解決什麼

`init_random` 徹底失敗：從 100,000 個隨機點出發，step 499 只剩 25,599（起始 trim 砍 74%），
整趟只長到 43~48k，**只有 cap 2.6M 的 1.7%**，比 SfM 臂少 30 倍。
使用者問「是不是起始太少」。量過之後：**是範圍的問題，不是點數的問題。**
`colmap_dataparser.py:532` 把隨機點撒在**半寬 3R 的立方體**裡，而實際內容是貼地薄層：
```
random 立方體每軸跨度 64.90
SfM 主體（2.5~97.5 百分位）  x 14.58 (22%)  y 11.91 (18%)  z **2.00 (3.1%)**
=> 主體只佔立方體 0.127%  =>  撒 100,000 點只有約 **127** 個落在內容區
   撒 1,000,000 也只有 1,272 個（SfM 有 3.83M）=> 要靠亂撒追上得撒到十億級
```
⇒ 使用者的第二個想法才對症：**SfM 有點的地方保留（可加倍），沒點的地方低密度補**。
   這也正是記憶 `graded_init` 設計過但沒實作的 Tier A / Tier B。

## 為什麼輸出「只有 xyz+rgb 的 PLY」而不是 Gaussian PLY

`--model.initialize_from <ply>` 載入的是**已存好的 Gaussian 模型**，scale/opacity/rotation 都由
檔案決定；而 `sfm` 臂走的是 `setup_from_pcd`，scale 是 `log(sqrt(distCUDA2(點雲)))`
——**從當下這組點算出來的**。兩者不同 ⇒ 用 Gaussian PLY 會多出「scale 初始化」這個變數。
⇒ 改走 `--data.parser.points_from ply --data.parser.ply_file <路徑>`，
  讓合併後的點雲經過**與 sfm 臂完全相同**的 `setup_from_pcd`，唯一變數就只有點集。
⚠ block parser 的 `ply` 分支**不做逐塊過濾**（`sfm` 會用 selected_image_ids 過濾）
  ⇒ 所以這支工具產生的是**逐塊**的 PLY。

## Tier A 取的是什麼

用 `ColmapDataParser.read_points3D_binary(..., selected_image_ids=...)` ——
**與 sfm 臂呼叫的是同一個函式、同一組相機** ⇒ Tier A 與 sfm 臂的點集**逐點相同**。
⇒ `--fill-ratio 0 --dup 1` 產生的 PLY 必須與 sfm 臂**點數相同**，那是本工具的自檢關卡。

用法:
  python tools/make_sfm_fill_init.py data/matrix_city/aerial/train/block_all \\
      --blocks 6 12 13 --fill-ratio 0.1 --fill-voxel 0.15
  python tools/make_sfm_fill_init.py <dataset> --blocks 6 --fill-ratio 0 --verify   # 關卡
"""
import argparse
import os
import sys

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_xyzrgb_ply(path, xyz, rgb):
    el = np.empty(len(xyz), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    el["x"], el["y"], el["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el["red"], el["green"], el["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--content_threshold", type=float, default=0.08)
    ap.add_argument("--out-dir", default=None, help="預設 <dataset>/sfmfill_init")
    ap.add_argument("--dup", type=int, default=1,
                    help="Tier A 複製幾份（>1 時額外份加抖動）。使用者說的「多複製幾個點」")
    ap.add_argument("--jitter", type=float, default=0.5,
                    help="複製份的抖動量，單位＝該塊 SfM 最近鄰間距的幾倍")
    ap.add_argument("--fill-ratio", type=float, default=0.1,
                    help="補點的密度是 SfM 區密度的幾倍（0 = 不補，用來跑自檢關卡）")
    ap.add_argument("--fill-voxel", type=float, default=0.15,
                    help="判斷「這裡有沒有 SfM 點」的體素邊長（場景單位）")
    ap.add_argument("--pct", type=float, default=2.5,
                    help="取範圍用的百分位（避免離群點把範圍撐爆；SfM 有 z 到 -461 的離群點）")
    ap.add_argument("--verify", action="store_true", help="只印統計不寫檔")
    args = ap.parse_args()

    from internal.utils.outlock import exclusive
    # ⚠ 2026-09-13：兩個本工具的行程同時寫 sfmfill_init/ ⇒ 產出的是**舊參數**的結果，
    #   而檔案大小一模一樣，差點被當成新的（沒有錯誤、沒有崩潰，只是內容不是你以為的那個）。
    #   ★ 鎖要在**載入任何東西之前**取 —— 放在讀 5,621 張 images.bin 之後的話，
    #     要等兩分鐘才拒絕，那就失去「快速失敗」的意義了（2026-09-13 實測）。
    out_dir = args.out_dir or os.path.join(args.dataset_dir, "sfmfill_init")
    _lock = None
    if not args.verify:
        _lock = exclusive(os.path.basename(out_dir), hint=f"目標目錄 {out_dir}")
        _lock.__enter__()

    from internal.dataparsers.colmap_dataparser import ColmapDataParser
    from internal.utils.colmap import read_images_binary

    sparse = os.path.join(args.dataset_dir, "sparse", "0")
    part = os.path.join(args.dataset_dir, "partition",
                        "partitions-dim_{}_{}_visibility_{}".format(
                            args.block_dim[0], args.block_dim[1], args.content_threshold))
    if not os.path.isdir(part):
        raise SystemExit(f"⛔ 找不到 partition：{part}\n  先跑 utils/partition_from_colmap.py")
    images = read_images_binary(os.path.join(sparse, "images.bin"))
    name_to_id = {im.name: i for i, im in images.items()}

    rng = np.random.default_rng(42)
    for blk in args.blocks:
        bx, by = blk % args.block_dim[0], blk // args.block_dim[0]
        pf = os.path.join(part, f"{bx:03d}_{by:03d}.txt")
        if not os.path.exists(pf):
            print(f"[block {blk}] ⛔ 缺 {pf}，跳過"); continue
        names = [ln.strip() for ln in open(pf) if ln.strip()]
        sel = [name_to_id[n] for n in names if n in name_to_id]
        # ★ 與 sfm 臂呼叫同一個函式、同一組相機 => Tier A 逐點相同
        xyz, rgb, _ = ColmapDataParser.read_points3D_binary(
            os.path.join(sparse, "points3D.bin"), selected_image_ids=sel)
        xyz = np.asarray(xyz, np.float64); rgb = np.asarray(rgb)
        nA = len(xyz)

        lo = np.percentile(xyz, args.pct, axis=0)
        hi = np.percentile(xyz, 100 - args.pct, axis=0)
        inb = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        vol = float(np.prod(hi - lo))
        dens = int(inb.sum()) / max(vol, 1e-9)          # SfM 區的點密度

        parts_xyz, parts_rgb = [xyz], [rgb]
        if args.dup > 1:
            sub = xyz[rng.choice(nA, size=min(20000, nA), replace=False)]
            d = np.sort(np.linalg.norm(sub[:2000, None] - sub[None], axis=-1), axis=1)[:, 1]
            nn = float(np.median(d))
            for _ in range(args.dup - 1):
                parts_xyz.append(xyz + rng.normal(0, nn * args.jitter, xyz.shape))
                parts_rgb.append(rgb)
        else:
            nn = float("nan")

        nB = 0
        if args.fill_ratio > 0:
            vs = args.fill_voxel
            key = np.floor((xyz - lo) / vs).astype(np.int64)
            dim = np.maximum(np.floor((hi - lo) / vs).astype(np.int64) + 1, 1)
            ok = np.all((key >= 0) & (key < dim), axis=1)
            occupied = set(map(int, (key[ok, 0] * dim[1] + key[ok, 1]) * dim[2] + key[ok, 2]))
            n_vox = int(np.prod(dim)); n_empty = n_vox - len(occupied)
            nB = int(dens * args.fill_ratio * n_empty * vs ** 3)
            if nB > 0 and n_empty > 0:
                # 在空體素裡撒點：先過取樣再丟掉落在已佔用體素的
                need, got = nB, []
                while need > 0 and len(got) < nB:
                    c = rng.random((max(need * 3, 10000), 3)) * (hi - lo) + lo
                    k = np.floor((c - lo) / vs).astype(np.int64)
                    k = np.clip(k, 0, dim - 1)
                    f = (k[:, 0] * dim[1] + k[:, 1]) * dim[2] + k[:, 2]
                    keep = np.fromiter((int(v) not in occupied for v in f), bool, len(f))
                    got.append(c[keep]); need = nB - sum(len(g) for g in got)
                fill = np.concatenate(got)[:nB]
                parts_xyz.append(fill)
                parts_rgb.append(np.tile(rgb.mean(0).astype(np.uint8), (len(fill), 1)))
                nB = len(fill)
            print(f"[block {blk}] 體素 {vs}：共 {n_vox:,} 格、已佔 {len(occupied):,}、空 {n_empty:,}")

        all_xyz = np.concatenate(parts_xyz).astype(np.float32)
        all_rgb = np.concatenate(parts_rgb).astype(np.uint8)
        print(f"[block {blk}] 相機 {len(sel)}／Tier A {nA:,}（x{args.dup}）"
              f"／Tier B {nB:,}／合計 **{len(all_xyz):,}**"
              f"   SfM 區密度 {dens:,.0f} 點/單位³   最近鄰 {nn:.4f}")
        if args.verify:
            print(f"           ★ 關卡：--fill-ratio 0 --dup 1 時應等於 sfm 臂的點數 {nA:,}")
            continue
        p = os.path.join(out_dir, f"block_{blk}.ply")
        write_xyzrgb_ply(p, all_xyz, all_rgb)
        rel = os.path.relpath(p, args.dataset_dir)
        print(f"           -> {p}\n"
              f"           用法：--data.parser.points_from ply --data.parser.ply_file {rel}")


if __name__ == "__main__":
    main()
