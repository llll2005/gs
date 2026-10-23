#!/usr/bin/env python
"""漂浮物顆粒在本塊訓練視角裡被遮罩分成哪一類（10-23 遮罩全場景診斷第二步；接 tools/blob_owner.py --save_pts）。

## 問的問題
元兇塊的漂浮物（壞區顆粒）在它自己的訓練影像上投影到哪些像素、那些像素的三分類（A 塊內／B 塊外且空／C 不計分）是什麼：
- 多半是 A => 遮罩說「這裡是本塊表面」，本塊就得用盒內顆粒去畫它（假說：單目深度在反光面判錯）
- 多半是 C => 沒有任何梯度去清掉它們（不計分區）
- 多半是 B => 遮罩其實要求透明，漂浮物不該活下來（就要往別處找原因）
並報同一批像素上單目深度反投影的第一表面：高度（相對地面）、是否落在本塊盒內 —— 直接檢查「深度判錯 => 誤標 A」。
對照組＝同塊隨機顆粒（blob_owner 一起存）。

可見性只用單目深度粗判：顆粒相機深度 <= 單目深度 x 1.1 才算「看得到」（被前面表面擋住的投影不算）。

用法（lab，CPU）：python tools/blob_mask_check.py --pts outputs/lab/blob_owner_1023/km25_pts.npz [--overlay 3 --save_dir ...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from internal.utils.colmap import read_cameras_binary, read_images_binary, qvec2rotmat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", required=True)
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--masks", default="block_masks44")
    ap.add_argument("--depth_dir", default="estimated_depths")
    ap.add_argument("--vis_tol", type=float, default=1.1)
    ap.add_argument("--overlay", type=int, default=3, help="每塊存幾張「漂浮物投影最多」的訓練視角疊圖")
    ap.add_argument("--save_dir", default=None)
    a = ap.parse_args()
    D = a.data
    z = np.load(a.pts)
    parts = torch.load(os.path.join(D, a.part, "partitions.pt"))
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    ids = parts["partition_coordinates"]["id"].numpy(); xys = parts["partition_coordinates"]["xy"].double().numpy()
    size = parts["scene_config"]["partition_size"].double().numpy()
    sp = os.path.join(D, "sparse/0")
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin")); imgs = read_images_binary(os.path.join(sp, "images.bin"))
    by_name = {v.name: v for v in imgs.values()}
    scales = json.load(open(os.path.join(D, "estimated_depth_scales.json")))
    # 地面：與 block_mask44 同定義（SfM 點 z 5 百分位）—— 這裡用顆粒對照組近似即可，只報相對高度
    zg = np.percentile(np.concatenate([z["rand_xyz"], z["xyz"]]) @ Rp.T, 5, axis=0)[2]
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)

    for b in sorted(set(z["block"].tolist())):
        k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0])
        lo, hi = xys[k], xys[k] + size[k]
        groups = {"漂浮物": z["xyz"][z["block"] == b].astype(np.float64), "隨機對照": z["rand_xyz"][z["rand_block"] == b].astype(np.float64)}
        names = [l.strip() for l in open(os.path.join(D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip()]
        md = os.path.join(D, a.masks, f"block_{b}")
        acc = {g: {"lab": np.zeros(3), "lab_vis": np.zeros(3), "front": 0, "n": 0,
                   "ps_h": [], "ps_inbox": 0, "ps_n": 0} for g in groups}
        per_view = []
        for nm in names:
            im = by_name.get(nm); mp = os.path.join(md, nm); npy = os.path.join(D, a.depth_dir, nm + ".npy")
            if im is None or not os.path.exists(mp) or not os.path.exists(npy) or nm not in scales:
                continue
            cam = cams[im.camera_id]; fx, fy, cx, cy = cam.params[:4]
            cls = np.asarray(Image.open(mp)); H, W = cls.shape
            sx, sy = W / cam.width, H / cam.height
            inv = np.load(npy).astype(np.float64) * scales[nm]["scale"] + scales[nm]["offset"]
            R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec); C = -R.T @ t
            nv = 0
            for g, X in groups.items():
                Xc = X @ R.T + t
                ok = Xc[:, 2] > 1e-6
                u = (fx * Xc[:, 0] / np.where(ok, Xc[:, 2], 1) + cx) * sx; v = (fy * Xc[:, 1] / np.where(ok, Xc[:, 2], 1) + cy) * sy
                ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if not ok.any():
                    continue
                ui, vi = u[ok].astype(int), v[ok].astype(int)
                lab = cls[vi, ui]; ivd = inv[vi, ui]
                zm = np.where(ivd > 1e-6, 1 / np.maximum(ivd, 1e-6), np.inf)
                vis = Xc[ok, 2] <= zm * a.vis_tol
                A_ = acc[g]
                A_["lab"] += np.bincount(lab, minlength=3)[:3]; A_["lab_vis"] += np.bincount(lab[vis], minlength=3)[:3]
                A_["front"] += int((Xc[ok, 2] < zm / a.vis_tol).sum()); A_["n"] += int(ok.sum())
                # 單目第一表面（看得到的投影）：高度與是否在本塊盒內
                if vis.any():
                    dc = np.stack([(ui[vis] / sx - cx) / fx, (vi[vis] / sy - cy) / fy, np.ones(int(vis.sum()))], -1)
                    ps = (C + (zm[vis])[:, None] * (dc @ R)) @ Rp.T
                    A_["ps_h"].append(ps[:, 2] - zg); A_["ps_n"] += len(ps)
                    A_["ps_inbox"] += int(((ps[:, 0] >= lo[0]) & (ps[:, 0] <= hi[0]) & (ps[:, 1] >= lo[1]) & (ps[:, 1] <= hi[1])).sum())
                if g == "漂浮物":
                    nv = int((vis & (lab == 1)).sum())
                    per_view.append((nv, nm, ui[vis], vi[vis], lab[vis]))
        print(f"\n== block {b}（{len(names)} 張訓練影像；地面 z≈{zg:.3f}）")
        for g, A_ in acc.items():
            if A_["n"] == 0:
                print(f"  {g}: 沒有投影"); continue
            l, lv = A_["lab"] / max(A_["lab"].sum(), 1), A_["lab_vis"] / max(A_["lab_vis"].sum(), 1)
            h = np.concatenate(A_["ps_h"]) if A_["ps_h"] else np.array([np.nan])
            print(f"  {g}（{len(groups[g]):,} 顆、投影 {A_['n']:,} 次）：全部投影 C/A/B {100 * l[0]:.1f}/{100 * l[1]:.1f}/{100 * l[2]:.1f}%  "
                  f"單目可見 {int(A_['lab_vis'].sum()):,} 次 C/A/B {100 * lv[0]:.1f}/{100 * lv[1]:.1f}/{100 * lv[2]:.1f}%  "
                  f"在單目表面前方 {100 * A_['front'] / A_['n']:.1f}%")
            print(f"      可見投影處的單目第一表面：高度 中位 {np.nanmedian(h):+.3f} p10 {np.nanpercentile(h, 10):+.3f} p90 {np.nanpercentile(h, 90):+.3f}；"
                  f"低於地面 0.05 的 {100 * np.mean(h < -0.05):.1f}%；落在本塊盒內 {100 * A_['ps_inbox'] / max(A_['ps_n'], 1):.1f}%")
        if a.save_dir and a.overlay:
            per_view.sort(key=lambda x: -x[0])
            for nv, nm, ui, vi, lab in per_view[:a.overlay]:
                cls = np.asarray(Image.open(os.path.join(md, nm))); H, W = cls.shape
                g = np.asarray(Image.open(os.path.join(D, "input", nm)).convert("RGB").resize((W, H))).astype(np.float32)
                col = np.array([[128, 128, 128], [0, 255, 0], [255, 0, 0]], dtype=np.float32)[cls]
                ov = (0.6 * g + 0.4 * col)
                ov[np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1)] = [255, 255, 0]
                Image.fromarray(np.concatenate([g, ov.clip(0, 255)], 1).astype(np.uint8)).save(
                    os.path.join(a.save_dir, f"b{b}_{nm[:-4]}_A{nv}.jpg"), quality=85)
            print(f"  疊圖（左原圖／右：綠 A、紅 B、灰 C、黃＝漂浮物投影）：" + " ".join(f"{nm}(A 可見 {nv})" for nv, nm, *_ in per_view[:a.overlay]))


if __name__ == "__main__":
    main()
