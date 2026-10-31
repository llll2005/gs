#!/usr/bin/env python
"""塊外遮罩前處理（使用者 10-17「作」）：每塊每張訓練影像一張三值遮罩，深度＝Depth Anything（只當前處理，不進訓練）。

## 為什麼
單塊模型 73~81% 的顆粒在分區盒外（合併時全丟），因為訓練影像約一半是鄰塊內容、光度 loss 要求它被解釋。
=> 訓練前就把每個像素分成三類，訓練端照類別處理：
```
A 塊內        第一個表面（深度反投影）落在本塊盒內（外擴 m_in）          正常光度 loss
B 塊外且空    第一個表面在塊外，且射線過了它到地面為止不再進入本塊盒（外擴 m_c）  目標＝背景色（=> 本塊在這格必須透明；塊外生東西就扣分）
C 不計分      第一個表面在塊外但射線之後會進入本塊（鄰塊高樓擋住本塊街道）、
              深度無效、A/B 交界 band px 內                                 權重 0
```
B 的判定刻意保守（m_c > m_in，交界帶歸 C）：B 判錯會刪掉本塊真的內容，A 判錯只是多學一點背景。

## 座標
- 深度：`estimated_depths/<COLMAP 名>.npy` 是**反深度**，`*scale+offset`（estimated_depth_scales.json，對 SfM 擬合；
  utils/get_depth_scales.py 抄自 hierarchical-3dgs）後 = 1/z_cam。⚠ 本機那份 estimated_depths 是 09-13 前的舊命名，只能在 lab 跑。
- 分區：partitions.pt 的 rotation_transform[:3,:3]（與 utils/merge_citygs_ckpts.py 同一個轉換）後取 xy；
  盒＝[xy_b, xy_b + size_b]，外擴量用**絕對值**（內側塊尺寸中位 x 比例），外圈塊本來就延伸到場景邊。
- 上方向：相機中心與 SfM 點在轉換後 z 的相對位置決定；地面高度＝SfM 點 z 的 5 百分位（0.5 百分位是低於地面的雜訊點）（下方那一側）。

## 檢查（--check）
每張影像的 SfM 觀測點（images.bin 的 xys／point3D_ids，重投影誤差 < 1.5px）是**真的第一個表面**：
它的 3D 位置在本塊盒內 => 期望 A；在外擴 m_c 盒外 => 期望非 A。報「盒內點被標 B」（危險）與「盒外點被標 A」（漏）。

用法（lab，CPU）：
  python tools/block_mask44.py --blocks 6 12 --check --overlay 4
輸出：<data>/block_masks44/block_<B>/<名>.png（0=C、1=A、2=B；與深度圖同解析度）＋ stats.json ＋ overlay_*.jpg
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from internal.utils.colmap import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat  # noqa: E402


def seg_hits_box(p0, p1, lo, hi):
    """2D 線段 p0->p1（[...,2]）是否與軸對齊盒 [lo,hi] 相交（Liang–Barsky，向量化）。"""
    d = p1 - p0
    t0 = np.zeros(p0.shape[:-1]); t1 = np.ones(p0.shape[:-1])
    ok = np.ones(p0.shape[:-1], dtype=bool)
    for k in range(2):
        for p, q in ((-d[..., k], p0[..., k] - lo[k]), (d[..., k], hi[k] - p0[..., k])):
            par = np.abs(p) < 1e-12
            ok &= ~(par & (q < 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                r = q / p
            neg = (~par) & (p < 0); pos = (~par) & (p > 0)
            t0 = np.where(neg, np.maximum(t0, r), t0)
            t1 = np.where(pos, np.minimum(t1, r), t1)
    return ok & (t0 <= t1)


def in_box(xy, lo, hi):
    return (xy[..., 0] >= lo[0]) & (xy[..., 0] <= hi[0]) & (xy[..., 1] >= lo[1]) & (xy[..., 1] <= hi[1])


def dilate(m, r):
    if r <= 0:
        return m
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, iterations=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--m_in", type=float, default=0.05, help="A 的盒外擴（x 內側塊尺寸中位）")
    ap.add_argument("--m_c", type=float, default=0.15, help="B/C 判定用的盒外擴（要 >= m_in；越大 B 越保守）")
    ap.add_argument("--band", type=int, default=2, help="A/非A 交界 band px 歸 C（深度圖解析度）")
    ap.add_argument("--far", type=float, default=50.0, help="射線不打地面時的延伸距離（x 第一表面深度）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--depth_dir", default="estimated_depths")
    ap.add_argument("--depth_raw", action="store_true",
                    help="★ 10-31 深度檔已是場景單位的逆深度（例：已訓練模型渲染的 1/z，0＝無效），不套 estimated_depth_scales.json")
    ap.add_argument("--max_views", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--overlay", type=int, default=0, help="存幾張疊圖")
    ap.add_argument("--crop_ply", default="", help="另把 <data>/<此目錄>/dup*/block_<B>.ply 裁到本塊盒（外擴 m_c），寫到 <此目錄>_crop/（初始化用）")
    ap.add_argument("--skip_masks", action="store_true")
    a = ap.parse_args()
    D = a.data
    out_root = a.out or os.path.join(D, "block_masks44")
    parts = torch.load(os.path.join(D, a.part, "partitions.pt"))
    Rp = parts["extra_data"]["rotation_transform"][:3, :3].double().numpy()
    ids = parts["partition_coordinates"]["id"].numpy()
    xys = parts["partition_coordinates"]["xy"].double().numpy()
    size = parts["scene_config"]["partition_size"].double().numpy()
    assert not parts["scene_config"]["contract"], "contract 模式未支援"
    inner = np.median(np.minimum(size[:, 0], size[:, 1]))
    ref = np.median(size[(size[:, 0] < 10 * inner) & (size[:, 1] < 10 * inner)], axis=0)   # 內側塊的典型尺寸
    mi, mc = a.m_in * ref.min(), a.m_c * ref.min()
    print(f"內側塊尺寸 ≈ {ref[0]:.3f} x {ref[1]:.3f}；外擴 A {mi:.3f}、B/C {mc:.3f}（場景單位）")

    sp = os.path.join(D, "sparse/0")
    cams = read_cameras_binary(os.path.join(sp, "cameras.bin"))
    imgs = read_images_binary(os.path.join(sp, "images.bin"))
    pts = read_points3D_binary(os.path.join(sp, "points3D.bin"))
    by_name = {v.name: v for v in imgs.values()}
    pid = np.array(list(pts.keys())); pxyz = np.stack([pts[k].xyz for k in pid]); perr = np.array([pts[k].error for k in pid])
    lut_xyz = np.zeros((pid.max() + 1, 3)); lut_xyz[pid] = pxyz
    lut_err = np.full(pid.max() + 1, 1e9); lut_err[pid] = perr
    scales = json.load(open(os.path.join(D, "estimated_depth_scales.json")))

    # 上方向與地面
    P_rot = pxyz @ Rp.T
    Cs = np.stack([-qvec2rotmat(v.qvec).T @ v.tvec for v in imgs.values()]) @ Rp.T
    up = 1.0 if np.median(Cs[:, 2]) > np.median(P_rot[:, 2]) else -1.0
    zg = np.percentile(P_rot[:, 2], 5) if up > 0 else np.percentile(P_rot[:, 2], 95)   # 0.5 百分位是雜訊（-10.97），5 百分位 ≈ 地面 0.005
    print(f"上方向 = {'+' if up > 0 else '-'}z；地面高度 {zg:.3f}（SfM 5 百分位）；相機高度中位 {np.median(Cs[:, 2]):.3f}")

    summary = {}
    for b in a.blocks:
        k = int(np.where((ids[:, 0] == b % 4) & (ids[:, 1] == b // 4))[0][0])
        lo, hi = xys[k], xys[k] + size[k]
        lo_a, hi_a = lo - mi, hi + mi
        lo_c, hi_c = lo - mc, hi + mc
        if a.crop_ply:
            from plyfile import PlyData, PlyElement
            src = os.path.join(D, a.crop_ply)
            for dk in sorted(os.listdir(src)):
                f = os.path.join(src, dk, f"block_{b}.ply")
                if not os.path.isfile(f):
                    continue
                pd = PlyData.read(f); v = pd["vertex"].data
                xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64) @ Rp.T
                keep = in_box(xyz[:, :2], lo_c, hi_c)
                of = os.path.join(src.rstrip("/") + "_crop", dk, f"block_{b}.ply"); os.makedirs(os.path.dirname(of), exist_ok=True)
                PlyData([PlyElement.describe(v[keep], "vertex")], text=False).write(of)
                print(f"  crop {dk}/block_{b}.ply：{len(v):,} -> {int(keep.sum()):,}（{100 * keep.mean():.1f}%）=> {of}", flush=True)
        if a.skip_masks:
            continue
        names = [l.strip() for l in open(os.path.join(D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip()]
        if a.max_views > 0:
            names = names[:: max(1, len(names) // a.max_views)][: a.max_views]
        od = os.path.join(out_root, f"block_{b}"); os.makedirs(od, exist_ok=True)
        fr = []; chk = np.zeros((2, 3), dtype=np.int64)   # [盒內點, 盒外點] x [C, A, B]
        t_start = time.time(); n_ov = 0; miss = 0
        for i, nm in enumerate(names):
            im = by_name.get(nm); npy = os.path.join(D, a.depth_dir, nm + ".npy")
            if im is None or (nm not in scales and not a.depth_raw) or not os.path.exists(npy):
                miss += 1; continue
            cam = cams[im.camera_id]; fx, fy, cx, cy = cam.params[:4]
            inv = np.load(npy).astype(np.float64)
            if not a.depth_raw:
                inv = inv * scales[nm]["scale"] + scales[nm]["offset"]
            H, W = inv.shape
            sx, sy = W / cam.width, H / cam.height          # 深度圖解析度 vs COLMAP 相機
            R = qvec2rotmat(im.qvec); C = -R.T @ im.tvec
            u = (np.arange(W) + 0.5) / sx; v = (np.arange(H) + 0.5) / sy
            uu, vv = np.meshgrid(u, v)
            dc = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1)   # 相機座標、z=1
            Dw = (dc @ R) @ Rp.T                              # 世界方向（轉到分區座標；z_cam 每單位）
            Cr = Rp @ C
            valid = inv > 1e-6
            z = np.where(valid, 1.0 / np.maximum(inv, 1e-6), 0.0)
            p_s = Cr + z[..., None] * Dw                      # 第一個表面
            A = valid & in_box(p_s[..., :2], lo_a, hi_a)
            # 射線過了第一表面之後：延伸到地面（往下）或 far x z
            dz = Dw[..., 2] * up
            with np.errstate(divide="ignore", invalid="ignore"):
                s_g = (zg - Cr[2]) / Dw[..., 2]
            s_end = np.where(dz < 0, s_g, a.far * z)
            s_end = np.minimum(np.maximum(s_end, z), a.far * np.maximum(z, 1e-6))
            p_e = Cr + s_end[..., None] * Dw
            behind = seg_hits_box(p_s[..., :2], p_e[..., :2], lo_c, hi_c)
            Bm = valid & ~A & ~behind
            band = dilate(A, a.band) & dilate(~A, a.band)
            cls = np.zeros((H, W), dtype=np.uint8)
            cls[A & ~band] = 1
            cls[Bm & ~band] = 2
            Image.fromarray(cls).save(os.path.join(od, nm))
            fr.append([(cls == 1).mean(), (cls == 2).mean(), (cls == 0).mean()])
            if a.check:
                pi = im.point3D_ids; ok = (pi >= 0) & (pi < len(lut_err))
                ok[ok] &= lut_err[pi[ok]] < 1.5
                if ok.any():
                    q = lut_xyz[pi[ok]] @ Rp.T
                    xy = im.xys[ok]
                    ui = np.clip((xy[:, 0] * sx).astype(int), 0, W - 1); vi = np.clip((xy[:, 1] * sy).astype(int), 0, H - 1)
                    lab = cls[vi, ui]
                    t_in = in_box(q[:, :2], lo, hi); t_out = ~in_box(q[:, :2], lo_c, hi_c)
                    for r, m in ((0, t_in), (1, t_out)):
                        chk[r] += np.bincount(lab[m], minlength=3)[:3]
            if n_ov < a.overlay and i % max(1, len(names) // max(a.overlay, 1)) == 0:
                ip = os.path.join(D, "input", nm)
                if os.path.exists(ip):
                    g = np.asarray(Image.open(ip).convert("RGB").resize((W, H))).astype(np.float32)
                    col = np.array([[128, 128, 128], [0, 255, 0], [255, 0, 0]], dtype=np.float32)[cls]
                    ov = (0.6 * g + 0.4 * col).clip(0, 255).astype(np.uint8)
                    Image.fromarray(ov).resize((W // 2, H // 2)).save(os.path.join(od, f"overlay_{nm[:-4]}.jpg"), quality=85)
                    n_ov += 1
            if (i + 1) % 100 == 0:
                print(f"  block {b}：{i + 1}/{len(names)}（{time.time() - t_start:.0f}s）", flush=True)
        fr = np.array(fr)
        st = {"views": int(len(fr)), "missing": miss, "A": float(fr[:, 0].mean()), "B": float(fr[:, 1].mean()), "C": float(fr[:, 2].mean()),
              "A_p10_p90": [float(np.percentile(fr[:, 0], 10)), float(np.percentile(fr[:, 0], 90))],
              "m_in": a.m_in, "m_c": a.m_c, "band": a.band}
        if a.check:
            st["sfm_in_box_CAB"] = chk[0].tolist(); st["sfm_out_box_CAB"] = chk[1].tolist()
            ti, to = chk[0].sum(), chk[1].sum()
            st["sfm_in_labeled_B"] = float(chk[0, 2] / max(ti, 1)); st["sfm_in_labeled_A"] = float(chk[0, 1] / max(ti, 1))
            st["sfm_out_labeled_A"] = float(chk[1, 1] / max(to, 1)); st["sfm_out_labeled_B"] = float(chk[1, 2] / max(to, 1))
        json.dump(st, open(os.path.join(od, "stats.json"), "w"), indent=1)
        summary[b] = st
        print(f"block {b}：{len(fr)} 張（缺 {miss}）  A {100 * st['A']:.1f}%  B {100 * st['B']:.1f}%  C {100 * st['C']:.1f}%"
              f"（A 的 p10~p90 {100 * st['A_p10_p90'][0]:.0f}~{100 * st['A_p10_p90'][1]:.0f}%）")
        if a.check:
            print(f"   SfM 檢查：盒內點 -> A {100 * st['sfm_in_labeled_A']:.1f}% ／ B（危險）{100 * st['sfm_in_labeled_B']:.2f}%；"
                  f"外擴盒外點 -> A（漏）{100 * st['sfm_out_labeled_A']:.2f}% ／ B {100 * st['sfm_out_labeled_B']:.1f}%")
    if summary:
        os.makedirs(out_root, exist_ok=True)
        json.dump(summary, open(os.path.join(out_root, f"stats_{'_'.join(map(str, a.blocks))}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
