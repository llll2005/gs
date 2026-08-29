#!/usr/bin/env python
"""每個視角拿到多少「觀測資源」，以及它跟該視角好壞的關係。

要回答的問題（2026-08-25，使用者看到 errunlock_b12 的 3422 建築整片糊）：
  糊掉是**視角不足**（=> few-shot 類方法才對症），還是視角夠但別的機制壞掉
  （=> 排序衝突 StopThePop / 梯度碰撞 AbsGS）？

量法（全部來自 SfM，零 GPU）：對每個評測視角 v，取它觀測到的 3D 點，統計
  n_pts       v 觀測到的 SfM 點數
  track       這些點的 track 長度中位數 = 「這塊內容被幾台相機看過」  <- 視角資源
  parallax    觀測同一點的相機方向向量夾角中位數（度）               <- 視差/排序難度
  baseline    觀測相機兩兩距離 / 到該點距離 的中位數
然後跟 tools/tail_analysis.py 的逐視角 PSNR 對起來。

判讀：
  壞視角 track 明顯低      => 真的是觀測不足，few-shot 先驗才有意義
  track 相同但 parallax 高 => 排序衝突（不同視角要求互相矛盾的深度序）
  兩者都相同               => 病灶在密度控制端（大足跡在既有訊號下隱形）
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.colmap import read_images_binary, read_points3D_binary  # noqa: E402


def per_view_psnr(run, blk):
    """逐視角 PSNR。test 圖是 GT|render 併排在同一張，要對半切（同 tools/tail_analysis.py）。
    只取步數最大的 test 目錄 —— earlyckpt 診斷會在 test/ 留下多個 checkpoint 的圖。"""
    import re
    from PIL import Image
    ds = glob.glob(f"outputs/{run}/blocks/block_{blk}/test/*/")
    if not ds:
        return {}
    d = max(ds, key=lambda p: int(re.search(r"step=(\d+)", p).group(1))
            if re.search(r"step=(\d+)", p) else -1)
    out = {}
    for p in sorted(glob.glob(d + "*.png")):
        im = Image.open(p)
        w = im.width // 2
        g = np.asarray(im.crop((0, 0, w, im.height)), np.float32) / 255.0
        x = np.asarray(im.crop((w, 0, im.width, im.height)), np.float32) / 255.0
        mse = float(((g - x) ** 2).mean())
            # GT 的梯度能量 = 內容難度代理。b12 有一半平坦水面（均勻色塊天生 32-40 dB），
        # 不控制它的話「觀測多寡 vs PSNR」的相關會整條被「水 vs 建築」解釋掉。
        gg = g.mean(axis=2) if g.ndim == 3 else g
        tex = float(np.abs(np.diff(gg, axis=0)).mean() + np.abs(np.diff(gg, axis=1)).mean())
        out[os.path.basename(p).split(".")[0]] = (
            10 * np.log10(1.0 / max(mse, 1e-12)), tex)
    return out


def partial_corr(x, y, z):
    """控制 z 之後 x 與 y 的偏相關"""
    x, y, z = map(np.asarray, (x, y, z))
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[m], y[m], z[m]
    if len(x) < 4:
        return float("nan")
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    d = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / d if d > 1e-9 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--run", default="sched30_b12")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--max-pts-per-view", type=int, default=4000,
                    help="每個視角抽樣多少點做視差統計（全算太慢）")
    ap.add_argument("--block-list", default=None,
                    help="該塊的影像清單（partition 的 XXX_YYY.txt）。"
                         "⚠ 一定要給：track 預設會數全部 5,621 台相機，但只有塊內的"
                         "約 284 台真的參與訓練，不限制的話量到的不是這個模型的視角資源。")
    args = ap.parse_args()

    sp = os.path.join(args.data, "sparse/0")
    print(f"讀 COLMAP: {sp}")
    images = read_images_binary(os.path.join(sp, "images.bin"))
    points = read_points3D_binary(os.path.join(sp, "points3D.bin"))
    print(f"  {len(images)} 影像 / {len(points)} 點")

    # 相機中心 C = -R^T t
    centers, name2id = {}, {}
    for iid, im in images.items():
        R = im.qvec2rotmat()
        centers[iid] = -R.T @ im.tvec
        name2id[os.path.splitext(im.name)[0]] = iid

    blk_ids = None
    if args.block_list:
        want = {l.strip() for l in open(args.block_list) if l.strip()}
        blk_ids = {iid for iid, im in images.items() if im.name in want}
        print(f"  塊內相機 {len(blk_ids)} / 清單 {len(want)} 台"
              f"{'  ⚠ 對不上，檢查清單' if len(blk_ids) != len(want) else ''}")

    psnr = per_view_psnr(args.run, args.blk)
    if not psnr:
        print(f"⚠ 找不到 {args.run} 的 test 圖，只印統計不做相關")
    print(f"  評測視角 {len(psnr)} 個")

    rng = np.random.default_rng(0)
    rows = []
    for vname, (pv, tex) in sorted(psnr.items(), key=lambda kv: kv[1][0]):
        iid = name2id.get(vname)
        if iid is None:
            continue
        im = images[iid]
        pids = np.array([p for p in im.point3D_ids if p != -1])
        if len(pids) == 0:
            continue
        if len(pids) > args.max_pts_per_view:
            pids = rng.choice(pids, args.max_pts_per_view, replace=False)
        tracks, paras, bases = [], [], []
        for pid in pids:
            pt = points.get(pid)
            if pt is None:
                continue
            obs = np.unique(pt.image_ids)
            if blk_ids is not None:
                obs = np.array([o for o in obs if o in blk_ids])
                if len(obs) == 0:
                    continue
            tracks.append(len(obs))
            if len(obs) < 2:
                continue
            C = np.array([centers[o] for o in obs if o in centers])
            if len(C) < 2:
                continue
            d = C - pt.xyz                      # 點 -> 相機
            n = np.linalg.norm(d, axis=1, keepdims=True)
            u = d / np.maximum(n, 1e-9)
            # 兩兩夾角的最大值（視差角）
            cosm = np.clip(u @ u.T, -1, 1)
            paras.append(np.degrees(np.arccos(cosm.min())))
            # 最大基線 / 中位距離
            dist = np.linalg.norm(C[:, None] - C[None], axis=-1)
            bases.append(dist.max() / max(np.median(n), 1e-9))
        if not tracks:
            continue
        # ★ 2026-08-28（使用者目視帶出）：最糟的視角（2857 等）GT 是**斜看高樓立面**，
        # 而 37~40dB 的視角都是**正俯視平面**（水面/屋頂）=> 假說：難度來自視野內的高聳結構。
        # 深度離散度是它的直接代理：正俯視平面 -> 深度幾乎一致；斜看高樓 -> 深度跨度大。
        dz = np.array([float(np.linalg.norm(points[p].xyz - centers[iid]))
                       for p in pids if p in points])
        depth_spread = float(np.percentile(dz, 90) / max(np.percentile(dz, 10), 1e-6)) if len(dz) > 8 else np.nan
        rows.append((vname, pv, len(im.point3D_ids[im.point3D_ids != -1]),
                     float(np.median(tracks)), float(np.median(paras)) if paras else np.nan,
                     float(np.median(bases)) if bases else np.nan, tex, depth_spread))

    if not rows:
        print("沒有可用資料")
        return

    def show(rs, title):
        print(f"\n===== {title} =====")
        print(f"{'視角':>8} {'PSNR':>7} {'紋理':>7} {'深度跨度':>9} {'n_pts':>7} {'track':>7} {'視差°':>7}")
        for r in rs:
            print(f"{r[0]:>8} {r[1]:>7.2f} {r[6]:>7.4f} {r[7]:>9.2f} {r[2]:>7d} {r[3]:>7.1f} {r[4]:>7.2f}")

    show(rows[:8], "最差 8 個")
    show(rows[-8:], "最好 8 個")

    # ⚠ 2026-08-28 修正索引錯位：原本 names 的第 5 項對到 a[:,5]＝tex（不是深度跨度），
    # 結果「深度跨度」印出的是 tex 自己的相關（-0.642）、偏相關 nan。
    # 現在把深度跨度排在 tex 之前，讓 enumerate(names, 1) 的位置全部對上。
    a = np.array([[r[1], r[2], r[3], r[4], r[5], r[7], r[6]] for r in rows], dtype=float)
    names = ["n_pts", "track", "視差°", "基線比", "深度跨度"]
    tex = a[:, 6]
    print(f"\n===== 與 PSNR 的相關（n={len(rows)}）=====")
    print(f"  {'':>8} {'原始 r':>9} {'控制紋理後':>11}")
    print(f"  {'紋理':>8} {np.corrcoef(a[:,0], tex)[0,1]:>+9.3f} {'—':>11}"
          "    <- 內容難度本身")
    for i, nm in enumerate(names, start=1):
        m = ~np.isnan(a[:, i])
        if m.sum() < 4:
            continue
        r = np.corrcoef(a[m, 0], a[m, i])[0, 1]
        pr = partial_corr(a[:, 0], a[:, i], tex)
        print(f"  {nm:>8} {r:>+9.3f} {pr:>+11.3f}")

    # 只看建築（紋理高於中位）的那一半，避免水面把關係整條解釋掉
    med = float(np.median(tex))
    sub = [r for r in rows if r[6] > med]
    if len(sub) >= 6:
        b = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in sub], dtype=float)
        print(f"\n===== 只看高紋理（建築）那一半，n={len(sub)}，"
              f"PSNR {b[:,0].min():.1f}~{b[:,0].max():.1f} =====")
        for i, nm in enumerate(names, start=1):
            m = ~np.isnan(b[:, i])
            if m.sum() < 4:
                continue
            print(f"  {nm:>8}: r = {np.corrcoef(b[m,0], b[m,i])[0,1]:+.3f}")

    print("""
判讀（⚠ n 很小，看**方向**不要看強度）：
  視角不足假說預測 track 與 PSNR **正相關**（看得少 => 重建差）。
  量到負相關 => 方向就相反，這個推翻不受效應量小的影響 => few-shot 類方法不對症。
  視差在壞視角沒有偏高 => 排序衝突（StopThePop）也沒有支持。
  ⚠ 但「觀測多 => 差」不可讀成因果：track 與 n_pts 同時在量 SfM 密度，
    而 SfM 密度高的地方就是幾何複雜的地方 —— 最可能的解讀是「觀測多 = 內容更難」。""")


if __name__ == "__main__":
    main()
