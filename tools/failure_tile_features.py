"""失敗 tile 的共通特徵（2026-10-12 使用者：合併模型多出來的失敗 tile 集中在格子中央，找失敗 tile 與區塊的共通特徵）

輸入：eval_official_test --save_dir 存的「左 GT／右渲染」PNG（合併模型、官方 741 幀）＋ test 集相機（sparse/0）＋ 4x4 分區。純 CPU。
每個 tile（48px）：失敗判定同 failure_map（GT 灰階 std >= 0.10 且 corr < 0.6）；特徵：
  影像   GT 紋理（灰階 std）、亮度、飽和度、色相分類（綠／藍／灰白／暖色）、紋理比（渲染 std / GT std；<1 偏糊）、亮度偏差
  位置   畫面中的位置（中央／邊緣）、相機俯仰（射線與地面的夾角）、tile 中心沿射線打到地面（分區座標系 z = --ground）
         => 落在哪一塊、離接縫多遠（內部格距；同 tools/seam_failure.py）
⚠ 地面交點對建築屋頂會有偏差（屋頂比地面高）=> 位置只當近似；lab 的 seam_failure 用渲染深度，較準。
輸出：每個特徵分組的失敗率（過對比門檻的 tile 中）＋失敗最嚴重的 tile 拼圖（GT｜渲染）。
用法：python tools/failure_tile_features.py --png_dir <test_official> --out <輸出目錄>
"""
import argparse
import colorsys
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png_dir", required=True)
    ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
    ap.add_argument("--partition", default="data/matrix_city/aerial/train/block_all/partition/partitions-dim_4_4_visibility_0.08/partitions.pt")
    ap.add_argument("--ground", type=float, default=0.24, help="分區座標系的地面高度（seam_failure 自檢量到反推點 z 中位 0.236）")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_montage", type=int, default=48)
    a = ap.parse_args()
    import eval_official_test as eot
    import seam_failure as sf
    os.makedirs(a.out, exist_ok=True)
    names, cams = eot.load_test_cameras(a.test_dir, 1.2)
    parts = torch.load(a.partition)
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].float()
    mn, mx, internal, unit = sf.seam_frame(parts)
    t = a.tile
    rows = []        # 每個過門檻 tile 一列
    crops = []
    for i, n in enumerate(names):
        p = os.path.join(a.png_dir, n.rsplit(".", 1)[0] + "_gt_vs_render.png")
        if not os.path.exists(p):
            continue
        im = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
        w = im.shape[1] // 2
        gt, rd = im[:, :w], im[:, w:]
        H, W = gt.shape[:2]
        ny, nx = H // t, W // t
        def tiles(x):
            return x[:ny * t, :nx * t].reshape(ny, t, nx, t, -1).transpose(0, 2, 1, 3, 4).reshape(ny, nx, t * t, -1)
        lum = lambda x: np.floor((np.round(x * 255) @ np.array([299, 587, 114])) / 1000) / 255.
        G, R = tiles(lum(gt)[..., None])[..., 0], tiles(lum(rd)[..., None])[..., 0]
        Gc, Rc = tiles(gt), tiles(rd)
        sg, sr = G.std(-1), R.std(-1)
        corr = ((G - G.mean(-1, keepdims=True)) * (R - R.mean(-1, keepdims=True))).mean(-1) / np.maximum(sg * sr, 1e-8)
        keep = sg >= 0.10
        fail = keep & (corr < 0.60)
        # 地面交點
        Rc_, Tc_ = cams.R[i].numpy(), cams.T[i].numpy()
        C = -Rc_.T @ Tc_
        vv, uu = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        u = (uu + 0.5) * t; v = (vv + 0.5) * t
        fx, fy, cx, cy = float(cams.fx[i]), float(cams.fy[i]), float(cams.cx[i]), float(cams.cy[i])
        d_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1).reshape(-1, 3)
        d_w = d_cam @ Rc_                                  # R^T d（列向量版）
        Rpn = Rp.numpy()
        Cp, dp = Rpn @ C, d_w @ Rpn.T
        s = (a.ground - Cp[2]) / np.where(np.abs(dp[:, 2]) < 1e-9, -1e-9, dp[:, 2])
        xy = Cp[None, :2] + s[:, None] * dp[:, :2]
        ok = s > 0
        dist = sf.seam_distance(torch.from_numpy(xy).float(), mn, mx, internal, unit).numpy()
        inside = ((torch.from_numpy(xy).float()[:, None, :] >= mn[None]) & (torch.from_numpy(xy).float()[:, None, :] <= mx[None])).all(-1)
        blk = np.where(inside.any(1).numpy(), inside.float().argmax(1).numpy(), -1)
        pitch = np.degrees(np.arcsin(np.clip(-dp[:, 2] / np.linalg.norm(dp, axis=1), -1, 1)))   # 射線向下的角度
        gm = Gc.mean(2)                                    # tile 平均顏色 [ny,nx,3]
        rm = Rc.mean(2)
        for k in np.flatnonzero(keep.reshape(-1)):
            yy, xx = divmod(k, nx)
            r_, g_, b_ = gm[yy, xx]
            hh, ss_, vv_ = colorsys.rgb_to_hsv(r_, g_, b_)
            if ss_ < 0.12:
                hue = "灰白"
            elif 0.17 < hh < 0.45:
                hue = "綠"
            elif 0.45 <= hh < 0.75:
                hue = "藍"
            else:
                hue = "暖色"
            ec = min(xx + 0.5, nx - xx - 0.5) / nx, min(yy + 0.5, ny - yy - 0.5) / ny
            rows.append(dict(frame=i, name=n, fail=bool(fail.reshape(-1)[k]), corr=float(corr.reshape(-1)[k]), gstd=float(sg.reshape(-1)[k]),
                             tex=float(sr.reshape(-1)[k] / max(sg.reshape(-1)[k], 1e-6)), bright=float(vv_), sat=float(ss_), hue=hue,
                             dbright=float(rm[yy, xx].mean() - gm[yy, xx].mean()),
                             edge=float(min(ec)), pitch=float(pitch[k]), blk=int(blk[k]) if ok[k] else -1,
                             dist=float(dist[k]) if ok[k] else float("nan")))
            if fail.reshape(-1)[k]:
                crops.append((float(corr.reshape(-1)[k]), i, yy, xx))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(names)}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "tiles.csv"), index=False)
    print(f"過對比門檻 tile {len(df):,}、失敗 {int(df.fail.sum()):,}（{100 * df.fail.mean():.2f}%）")

    def grp(col, bins=None, labels=None):
        x = df[col] if bins is None else pd.cut(df[col], bins, labels=labels)
        g = df.groupby(x, observed=True).fail.agg(["mean", "size", "sum"])
        print(f"\n== 依 {col} ==")
        for idx, r in g.iterrows():
            print(f"  {str(idx):<16} 失敗率 {100 * r['mean']:6.2f}%   tile {int(r['size']):>8,}   失敗 {int(r['sum']):>6,}")
    grp("dist", [-1, 0.05, 0.1, 0.2, 0.3, 0.5, 100])
    grp("blk")
    grp("gstd", [0.1, 0.13, 0.16, 0.2, 0.25, 1])
    grp("tex", [0, 0.5, 0.7, 0.9, 1.1, 10])
    grp("bright", [0, 0.2, 0.35, 0.5, 0.7, 1.01])
    grp("hue")
    grp("dbright", [-1, -0.08, -0.03, 0.03, 0.08, 1])
    grp("edge", [0, 0.1, 0.2, 0.35, 0.51], ["畫面邊 0~0.1", "0.1~0.2", "0.2~0.35", "中央"])
    grp("pitch", [0, 30, 40, 50, 60, 70, 90])
    # 每幀失敗數：失敗是不是集中在少數幀
    ff = df.groupby("frame").fail.sum().sort_values(ascending=False)
    top = ff.head(10)
    print(f"\n== 失敗集中度：前 10 幀佔全部失敗 {100 * top.sum() / max(ff.sum(), 1):.1f}%（共 {int((ff > 0).sum())} 幀有失敗）")
    for fr, c in top.items():
        print(f"  {df[df.frame == fr].name.iloc[0]}  失敗 {int(c)}")
    # 拼圖：失敗最嚴重（corr 最低）的 tile
    crops.sort()
    sel = crops[:a.n_montage]
    cols = 8
    tiles_img = []
    cache = {}
    for _, fi, yy, xx in sel:
        if fi not in cache:
            cache = {fi: np.asarray(Image.open(os.path.join(a.png_dir, names[fi].rsplit(".", 1)[0] + "_gt_vs_render.png")).convert("RGB"))}
        im = cache[fi]; w = im.shape[1] // 2
        g = im[yy * t:(yy + 1) * t, xx * t:(xx + 1) * t]; r = im[yy * t:(yy + 1) * t, w + xx * t:w + (xx + 1) * t]
        tiles_img.append(np.concatenate([g, np.full((t, 2, 3), 255, np.uint8), r], 1))
    if tiles_img:
        rows_img = [np.concatenate(tiles_img[j:j + cols] + [np.zeros_like(tiles_img[0])] * (cols - len(tiles_img[j:j + cols])), 1)
                    for j in range(0, len(tiles_img), cols)]
        mont = np.concatenate([np.concatenate([r_, np.full((4, r_.shape[1], 3), 255, np.uint8)], 0) for r_ in rows_img], 0)
        Image.fromarray(mont).resize((mont.shape[1] * 2, mont.shape[0] * 2), Image.NEAREST).save(os.path.join(a.out, "worst_tiles.png"))
        print(f"\n拼圖：{os.path.join(a.out, 'worst_tiles.png')}（每格左 GT｜右渲染，corr 最低的 {len(tiles_img)} 個）")


if __name__ == "__main__":
    main()
