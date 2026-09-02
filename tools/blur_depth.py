#!/usr/bin/env python
"""持續糊掉的 tile 是不是遠場？順便看它們的 SfM 點多不多。（純 CPU，零 GPU）

## 為什麼不用渲染深度

訓練跑次佔著 5.67/6.1 GB，再開一次渲染會把它 OOM。而且要回答的問題是
「**那裡的內容遠不遠**」——用 SfM 點雲投影比用渲染深度更直接（後者還混入重建誤差）。

## 量什麼（每個 48px tile）

  depth   該 tile 內 SfM 點的**中位深度**（相機座標 z）
  n_pts   落在該 tile 的 SfM 點數  <- 免費的第二個判別量

## 判讀

  持續糊掉的 depth **明顯大**  => 遠場確立（§11.72 的縱向位置訊號得到獨立確認）
  持續糊掉的 n_pts **明顯小**  => 那裡 SfM 本來就稀疏 => 是**輸入覆蓋**問題，
                                  不是密度控制問題（與八次密度介入全無效一致）
  兩者都不明顯                  => 遠場假說垮掉，回頭重想

⚠ SfM 點密度本身與內容複雜度相關（幾何複雜處點多）=> `n_pts` 小**不等於**「沒東西」，
  但它與 depth 一起看仍有鑑別力：遠場 + 點稀疏 是「內容在塊外/取樣不足」的簽名。
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.colmap import read_images_binary, read_cameras_binary  # noqa: E402
from tools.veil_detect import final_test_dir, tiles  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="至少兩個配方，取糊掉遮罩的交集")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--down-sample", type=float, default=1.2)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85,
                    help="只看紋理最強的 tile（§11.72：q=0.85 時仍有 39.8% 糊掉，非假影）")
    args = ap.parse_args()

    sp = os.path.join(args.data, "sparse/0")
    images = read_images_binary(os.path.join(sp, "images.bin"))
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    cam0 = list(cams.values())[0]
    fx = float(cam0.params[0]) / args.down_sample
    W = int(round(cam0.width / args.down_sample))
    H = int(round(cam0.height / args.down_sample))
    by_name = {im.name: im for im in images.values()}

    # SfM 點雲（xyz）
    from internal.utils.colmap import read_points3D_binary
    pts = read_points3D_binary(os.path.join(sp, "points3D.bin"))
    XYZ = np.stack([p.xyz for p in pts.values()]).astype(np.float64)
    print(f"SfM 點 {len(XYZ):,}   影像 {W}x{H}  fx {fx:.1f}")

    dirs = {r: final_test_dir(r, args.blk) for r in args.runs}
    if any(v is None for v in dirs.values()):
        raise SystemExit("有配方找不到 test 圖")
    fs = sorted(f for f in os.listdir(dirs[args.runs[0]]) if f.endswith(".png"))

    P, N, miss = [], [], 0
    for f in fs:
        outs = {r: per_image(os.path.join(dirs[r], f), args.tile, args.contrast_q)
                for r in args.runs}
        if any(o is None for o in outs.values()):
            continue
        keep, corr, *_rest, nx, ny = outs[args.runs[0]]
        ms = np.stack([outs[r][1] <= args.r_min for r in args.runs])
        allb, neverb = ms.all(0), ~ms.any(0)

        # test 圖檔名是 `<原名>.png`（多一層副檔名）
        name = f[:-4] if f.endswith(".png.png") else f
        im = by_name.get(name)
        if im is None:
            miss += 1
            continue
        R = im.qvec2rotmat()
        T = im.tvec
        pc = XYZ @ R.T + T
        z = pc[:, 2]
        vis = z > 0.2
        u = fx * pc[vis, 0] / z[vis] + W / 2
        v = fx * pc[vis, 1] / z[vis] + H / 2
        zz = z[vis]
        inb = (u >= 0) & (u < nx * args.tile) & (v >= 0) & (v < ny * args.tile)
        ti = (v[inb] // args.tile).astype(int) * nx + (u[inb] // args.tile).astype(int)
        zb = zz[inb]

        # 每個 tile 的中位深度與點數
        order = np.argsort(ti, kind="stable")
        ti_s, zb_s = ti[order], zb[order]
        uniq, start = np.unique(ti_s, return_index=True)
        med = np.full(nx * ny, np.nan)
        cnt = np.zeros(nx * ny)
        spread = np.full(nx * ny, np.nan)      # tile 內深度離散度 p90/p10
        ends = np.append(start[1:], len(ti_s))
        for t_, s_, e_ in zip(uniq, start, ends):
            seg = zb_s[s_:e_]
            med[t_] = np.median(seg)
            cnt[t_] = e_ - s_
            if len(seg) >= 8:
                lo = np.percentile(seg, 10)
                spread[t_] = np.percentile(seg, 90) / max(lo, 1e-6)

        for m, acc in ((allb, P), (neverb, N)):
            if m.sum():
                acc.append(np.stack([med[keep[m]], cnt[keep[m]], spread[keep[m]]], 1))

    if miss:
        print(f"⚠ {miss} 張 test 圖在 COLMAP 裡找不到同名影像（已略過）")
    P = np.concatenate(P) if P else np.zeros((0, 3))
    N = np.concatenate(N) if N else np.zeros((0, 3))
    print(f"\n{'':>12} {'深度中位':>10} {'深度 p25':>10} {'深度 p75':>10} "
          f"{'SfM點數中位':>12} {'深度離散p90/p10':>16} {'n':>7}")
    for nm, A in (("持續糊掉", P), ("從不糊掉", N)):
        if len(A) == 0:
            continue
        d = A[:, 0]
        ok = ~np.isnan(d)
        sp_ = A[:, 2][~np.isnan(A[:, 2])]
        print(f"{nm:>12} {np.median(d[ok]):>10.2f} {np.percentile(d[ok],25):>10.2f} "
              f"{np.percentile(d[ok],75):>10.2f} {np.median(A[:,1]):>12.1f} "
              f"{np.median(sp_) if len(sp_) else float('nan'):>16.3f} {len(A):>7,}")
    if len(P) and len(N):
        dp = np.median(P[~np.isnan(P[:, 0]), 0])
        dn = np.median(N[~np.isnan(N[:, 0]), 0])
        sp_p = np.median(P[:, 2][~np.isnan(P[:, 2])])
        sp_n = np.median(N[:, 2][~np.isnan(N[:, 2])])
        print(f"\n  深度比（糊/不糊）= {dp/max(dn,1e-9):.2f}x"
              f"   點數比 = {np.median(P[:,1])/max(np.median(N[:,1]),1e-9):.2f}x"
              f"   **深度離散比 = {sp_p/max(sp_n,1e-9):.2f}x**")
    print("""
判讀：
  深度比 >> 1        => 遠場確立（獨立確認 §11.72 的「集中畫面上半」）
  點數比 << 1        => 那裡 SfM 本來就稀疏 => **輸入覆蓋**問題，非密度控制問題
  兩者都接近 1       => 遠場假說垮掉，回頭重想""")


if __name__ == "__main__":
    main()
