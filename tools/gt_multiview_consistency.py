#!/usr/bin/env python
"""失敗區的**GT 影像本身**，能不能被任何一個表面解釋？（不用模型、不用訓練）

## 為什麼是這一題

模型端已經用光了。八次密度控制介入全部無效，而糊掉 tile 的**位置跨配方固定**
（`blur_persistence`：八個配方的糊掉% 全落在 46 +- 0.8）。今天又加一筆：
**vanilla 2DGS、零機制、1500 步，左半邊一樣糊**（`normal_b12`）。
另外已排除的：
```
val ⊂ train 確認             => 是**擬合失敗**不是泛化落差   blur_survived_bug_hunt
60k 糊掉 tile 覆蓋 = 沒糊的  => 「那裡沒資訊」不成立         coverage_needs_inbox_filter
最佳位移只 +0.031            => 不是對位問題                 blur_survived_bug_hunt
Adam 對幅度尺度不變          => 「梯度小所以學得慢」推不出來 adam_normalizes_magnitude
淨位移在失敗區還大 1.32x     => 粒子在動，只是沒收斂          同上
init 來源 / 殼的厚薄 / off-by-one => 與失敗區都無關          §16.8
```
⇒ 剩下的唯一一類解釋是**資料本身**：那些像素要求的東西，不存在一個表面能同時滿足
   所有看到它的相機。**這一題完全不需要模型** —— 只要 GT 影像 + 姿態
   （姿態是 GT×1/100、對齊殘差 0.0px，見記憶 `data_poses_are_gt`）。

## 量什麼（平面掃描 / plane sweep）

對參考視角的一個 tile，把深度 d 掃過整個合理範圍；每個 d 把 tile 的像素反投影到 3D、
再投進 K 個鄰近相機取樣，算與參考 patch 的 NCC。取 d 上的最佳值：

```
best_ncc          「存不存在一個深度，讓所有視角看到的內容一致」的上界
                  失敗區明顯低 => **沒有任何表面能解釋那裡** => 46% 是資料性質
                  失敗區一樣高 => 資料可以被解釋 => 病灶回到表示法/損失
peak_sharpness    best_ncc 與「d 上的中位 ncc」之差
                  接近 0 => 深度不可辨（重複紋理/低訊號）=> 即使能擬合也沒有唯一解
n_valid_views     有幾個鄰居真的看得到（投影落在畫面內且 z>0）
```

⚠ 三個必須誠實的限制：
  1. 平面掃描假設 tile 內是**正對參考相機的平面**。空拍近正交、tile 48px 很小，
     這個近似可接受；但屋頂邊緣/陡facade 會被低估 => 對**兩組都一樣**地低估，
     所以組間比較仍有效（這是本工具的設計前提）。
  2. NCC 對亮度線性變化不變 => 曝光差不會混進來（若換成 L1 就會）。
  3. 失敗/清楚 tile 的**遮罩來自某個訓練好的模型**（`--mask-run`）=> 遮罩本身
     帶著那個模型的偏差。所以要用 60k 模型（46% 那個區分才存在），
     而且結論要看「兩組的差距」而非絕對值。

用法:
  python tools/gt_multiview_consistency.py --mask-run speed3_b12 --blk 12
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402


def load_image(path, W, H, dev):
    """載 GT 並縮到相機聲明的尺寸（相機的 W/H 已含 down_sample）=> 保證與姿態一致。"""
    im = Image.open(path).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H), Image.BILINEAR)
    a = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.
    return torch.from_numpy(a).to(dev)


def load_block_cameras(blk, block_dim, down_sample):
    from internal.dataparsers.colmap_block_dataparser import ColmapBlock
    cfg = ColmapBlock(block_id=blk, block_dim=list(block_dim), content_threshold=0.08,
                      image_dir="input", down_sample_factor=down_sample)
    dp = cfg.instantiate(path="data/matrix_city/aerial/train/block_all",
                         output_path="/tmp", global_rank=0)
    out = dp.get_outputs()
    return out.train_set, out.val_set, out.point_cloud


def tile_masks(mask_run, blk, tile, r_min, r_good, contrast_q):
    """回傳 {影像檔名: (失敗 tile 索引, 清楚 tile 索引, nx, ny)}；遮罩只依 GT 對比度保留。"""
    d = final_test_dir(mask_run, blk)
    res = {}
    for p in sorted(glob.glob(os.path.join(d, "*.png"))):
        im = Image.open(p)
        w = im.width // 2
        g = np.asarray(im.crop((0, 0, w, im.height)).convert("L"), np.float32) / 255.
        r = np.asarray(im.crop((w, 0, im.width, im.height)).convert("L"), np.float32) / 255.
        G, R = tiles(g, tile), tiles(r, tile)
        sg = G.std(axis=1)
        keep = np.where(sg > np.quantile(sg, contrast_q))[0]
        Gk, Rk = G[keep], R[keep]
        gm, rm = Gk.mean(1, keepdims=True), Rk.mean(1, keepdims=True)
        corr = ((Gk - gm) * (Rk - rm)).mean(1) / np.maximum(Gk.std(1) * Rk.std(1), 1e-8)
        res[os.path.basename(p)] = (keep[corr < r_min], keep[corr > r_good],
                                    w // tile, im.height // tile, w, im.height)
    return res


def sweep(ref_img, ref_cam, nbr_imgs, nbr_cams, ty, tx, tile, depths, dev):
    """一個 tile 的平面掃描（**所有深度一次算完**）。回傳 (best_ncc, median_ncc, n_valid)。

    向量化的理由：逐深度逐鄰居迴圈是 400x8 = 3,200 次 kernel launch/tile，
    24 個 tile 就 7.7 萬次 —— 全部時間花在啟動而非計算。改成每個鄰居一次
    `grid_sample`（batch = 深度數）後是 8 次/tile。
    """
    H, W = ref_img.shape[-2:]
    D = len(depths)
    ys = torch.arange(ty * tile, (ty + 1) * tile, device=dev, dtype=torch.float32)
    xs = torch.arange(tx * tile, (tx + 1) * tile, device=dev, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    ref_patch = ref_img[:, ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile].mean(0)
    rp = ref_patch - ref_patch.mean()
    rp_n = (rp / (rp.norm() + 1e-8)).reshape(-1)                       # (t*t,)

    fx, fy = ref_cam["fx"], ref_cam["fy"]
    dirx = ((xx + 0.5 - ref_cam["cx"]) / fx).reshape(-1)               # (t*t,)
    diry = ((yy + 0.5 - ref_cam["cy"]) / fy).reshape(-1)
    Rr, Tr = ref_cam["R"], ref_cam["T"]
    dd = torch.as_tensor(np.asarray(depths, np.float32), device=dev).reshape(D, 1)
    pc = torch.stack([dirx * dd, diry * dd, dd.expand(D, dirx.numel())], -1)   # (D,t*t,3)
    pw = (pc.reshape(-1, 3) - Tr) @ Rr                                  # x_w = R^T (x_c - T)

    acc = torch.zeros(D, device=dev)
    nval = 0
    for img, cam in zip(nbr_imgs, nbr_cams):
        p = pw @ cam["R"].T + cam["T"]
        z = p[:, 2]
        u = cam["fx"] * p[:, 0] / z.clamp(min=1e-6) + cam["cx"]
        v = cam["fy"] * p[:, 1] / z.clamp(min=1e-6) + cam["cy"]
        inb = ((z > 1e-3) & (u > 0) & (u < W - 1) & (v > 0) & (v < H - 1)
               ).reshape(D, -1).float().mean(1)
        if float(inb.max()) < 0.95:            # 沒有任何深度讓這個鄰居看全這塊
            continue
        grid = torch.stack([(u / (W - 1)) * 2 - 1, (v / (H - 1)) * 2 - 1], -1
                           ).reshape(D, tile, tile, 2)
        smp = F.grid_sample(img.mean(0)[None, None].expand(D, 1, H, W), grid,
                            align_corners=True, mode="bilinear", padding_mode="border")
        smp = smp.reshape(D, -1)
        smp = smp - smp.mean(1, keepdim=True)
        smp = smp / (smp.norm(dim=1, keepdim=True) + 1e-8)
        ncc = smp @ rp_n                                                # (D,)
        # 該深度下這個鄰居沒看全 => 不讓它貢獻（給 -1 會把最大值壓掉，改成遮成 nan 再忽略）
        ncc = torch.where(inb >= 0.95, ncc, torch.full_like(ncc, float("nan")))
        acc = acc + torch.nan_to_num(ncc, nan=0.0)
        nval += 1
    if nval == 0:
        return -1.0, -1.0, 0
    m = (acc / nval).cpu().numpy()
    return float(np.nanmax(m)), float(np.nanmedian(m)), nval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-run", default="speed3_b12", help="用哪個跑次的渲染定義失敗/清楚 tile")
    ap.add_argument("--blk", type=int, default=12)
    ap.add_argument("--block-dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--down-sample", type=float, default=1.2)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60, help="corr < 此值 = 糊掉")
    ap.add_argument("--r-good", type=float, default=0.85, help="corr > 此值 = 清楚")
    ap.add_argument("--contrast-q", type=float, default=0.40)
    ap.add_argument("--n-images", type=int, default=6, help="用幾張參考影像")
    ap.add_argument("--n-tiles", type=int, default=40, help="每張每組抽幾個 tile")
    ap.add_argument("--n-nbr", type=int, default=8, help="每個參考視角取幾個最近鄰相機")
    ap.add_argument("--n-depth", type=int, default=400,
                    help="深度格數。太粗會讓**兩組一起去相關** => 尺失去鑑別力："
                         "基線 b、深度 z、焦距 f 下，格距 dz 造成的視差誤差是 f*b*dz/z^2；"
                         "48px 的 tile 需要 ~1px => 400 格才夠（48 格時清楚組只有 0.27）")
    ap.add_argument("--min-baseline", type=float, default=0.05,
                    help="鄰居最小基線。**零基線在任何深度都給高 NCC** => 會把兩組一起灌水，"
                         "而且峰銳度恆為 0（本塊確實有距離 0.00 的重複機位）")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = tile_masks(args.mask_run, args.blk, args.tile, args.r_min, args.r_good,
                       args.contrast_q)
    if not masks:
        raise SystemExit(f"找不到 {args.mask_run} 的 test 影像（先跑 main.py test --save_val）")
    train_set, val_set, pcd = load_block_cameras(args.blk, args.block_dim, args.down_sample)
    # 存檔名是 `<gt 檔名>.png`（例：0242.png.png）=> 去掉多的那層
    tmap = {k[:-4] if k.endswith(".png.png") else k: v for k, v in masks.items()}
    name2idx = {n: i for i, n in enumerate(train_set.image_names)}
    cams = train_set.cameras
    centers = cams.camera_center.numpy()
    xyz = torch.from_numpy(np.asarray(pcd.xyz, np.float32))

    def cam_dict(i):
        return {"R": cams.R[i].to(dev), "T": cams.T[i].to(dev),
                "fx": float(cams.fx[i]), "fy": float(cams.fy[i]),
                "cx": float(cams.cx[i]), "cy": float(cams.cy[i])}

    print(f"裝置 {dev}   遮罩來源 {args.mask_run}   tile {args.tile}px   "
          f"可用參考影像 {len(tmap)} 張（取 {args.n_images}）   SfM 點 {len(xyz):,}")

    rng = np.random.default_rng(0)
    rows = {"糊掉": [], "清楚": []}
    picked = 0
    for fn in sorted(tmap):
        if picked >= args.n_images:
            break
        if fn not in name2idx:
            print(f"  ⚠ {fn} 不在 block 影像清單裡，跳過")
            continue
        idx = name2idx[fn]
        bad, good, nx, ny, W, H = tmap[fn]
        assert W == int(cams.width[idx]) and H == int(cams.height[idx]), \
            f"存檔尺寸 {W}x{H} != 相機 {int(cams.width[idx])}x{int(cams.height[idx])}"
        rc = cam_dict(idx)
        ref_img = load_image(train_set.image_paths[idx], W, H, dev)
        # 深度掃描範圍 = SfM 點在這台相機下的深度 1~99%（不猜尺度）
        pc = xyz @ cams.R[idx].T + cams.T[idx]
        z = pc[:, 2]
        z = z[z > 1e-3].numpy()
        if len(z) < 100:
            print(f"  ⚠ {fn} 前方 SfM 點太少，跳過")
            continue
        lo, hi = np.percentile(z, [5, 95])
        depths = np.linspace(lo * 0.9, hi * 1.1, args.n_depth)
        _dz = (depths[-1] - depths[0]) / (args.n_depth - 1)
        dist = np.linalg.norm(centers - centers[idx], axis=1)
        ok_b = np.where(dist >= args.min_baseline)[0]
        nbr = ok_b[np.argsort(dist[ok_b])[:args.n_nbr]]
        if len(nbr) < 3:
            print(f"  ⚠ {fn} 基線 >= {args.min_baseline} 的鄰居只有 {len(nbr)} 個，跳過")
            continue
        nbr_imgs = [load_image(train_set.image_paths[int(j)], int(cams.width[int(j)]),
                               int(cams.height[int(j)]), dev) for j in nbr]
        nbr_cams = [cam_dict(int(j)) for j in nbr]
        print(f"  {fn}: 糊掉 {len(bad)} / 清楚 {len(good)} tile   "
              f"深度掃描 {depths[0]:.2f}~{depths[-1]:.2f}（{args.n_depth} 格，格距 {_dz:.4f}）  "
              f"鄰居基線 {dist[nbr].min():.2f}~{dist[nbr].max():.2f}  "
              f"最粗視差誤差 {float(cams.fx[idx])*dist[nbr].min()*_dz/max(depths[0],1e-6)**2:.2f}px")
        for lab, arr in (("糊掉", bad), ("清楚", good)):
            sel = rng.choice(arr, size=min(args.n_tiles, len(arr)), replace=False) if len(arr) else []
            for t in sel:
                ty, tx = int(t) // nx, int(t) % nx
                if (ty + 1) * args.tile > H or (tx + 1) * args.tile > W:
                    continue
                b, m, nv = sweep(ref_img, rc, nbr_imgs, nbr_cams, ty, tx, args.tile, depths, dev)
                if nv >= 3:
                    rows[lab].append((b, b - m, nv))
        picked += 1

    print(f"\n{'組':>6} {'n':>6} {'best_ncc 中位':>14} {'best_ncc 平均':>14} "
          f"{'峰銳度 中位':>13} {'有效鄰居':>9}")
    for lab in ("清楚", "糊掉"):
        v = np.asarray(rows[lab], dtype=np.float64)
        if not len(v):
            print(f"{lab:>6} {'(無樣本)':>6}")
            continue
        print(f"{lab:>6} {len(v):>6} {np.median(v[:,0]):>14.4f} {v[:,0].mean():>14.4f} "
              f"{np.median(v[:,1]):>13.4f} {np.median(v[:,2]):>9.1f}")

    if len(rows["糊掉"]) and len(rows["清楚"]):
        a = np.asarray(rows["清楚"], dtype=np.float64)[:, 0]
        b = np.asarray(rows["糊掉"], dtype=np.float64)[:, 0]
        sd = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"\n  差距 best_ncc(清楚) - best_ncc(糊掉) = {a.mean()-b.mean():+.4f}"
              f"  ({(a.mean()-b.mean())/max(sd,1e-12):+.1f}sd)")
    if len(rows["清楚"]):
        m = float(np.median(np.asarray(rows["清楚"], dtype=np.float64)[:, 0]))
        if m < 0.6:
            print(f"\n⛔ **正控制不過**：清楚 tile 的 best_ncc 中位只有 {m:.3f}。"
                  f"\n   清楚區的表面是確定存在且一致的，量不到高 NCC 表示**尺本身不夠利**"
                  f"（深度格距太粗／基線不對／tile 太大），"
                  f"\n   這時候「糊掉組較低」也可能只是同一個雜訊 => **不可下結論**。"
                  f"\n   先加 --n-depth、縮 --tile，或檢查 --min-baseline。")
        else:
            print(f"\n✅ 正控制通過：清楚 tile best_ncc 中位 {m:.3f} >= 0.6 => 尺有鑑別力")
    print("""
判讀：
  糊掉組 best_ncc **明顯低** => GT 本身在那裡多視角不一致
                              => **沒有任何表面能解釋它** => 46% 是資料性質，不是優化失敗
  兩組 best_ncc **相當**     => 資料可以被一個表面解釋 => 病灶在表示法或損失，回頭查那邊
  糊掉組 峰銳度接近 0        => 深度不可辨（重複紋理）=> 即使擬合得出也沒有唯一解
⚠ 平面掃描假設 tile 內正對參考相機的平面 => 對兩組**一樣**低估，組間比較仍有效。""")


if __name__ == "__main__":
    main()
