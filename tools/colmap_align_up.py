#!/usr/bin/env python
"""★ 2026-11-22 把 COLMAP 模型轉成「世界 +z 朝上」（Mill-19 等上方不是 +z 的資料）。

為什麼：我們的分區（partition_from_colmap 預設不轉）、塊外遮罩、只罰躺平（|n_z| > 0.8）、顆數分配、起點裁盒都假設世界 +z 朝上、
分區旋轉是單位矩陣。CityGS 提供的 Mill-19 colmap_results 上方是 −y（相機平均上方 (0.01, −1.00, −0.04)）=> 11-21 第一版的 4x4
是按「x × 高度」切的，全部作廢。在資料準備最前面把模型轉正，下游工具就不用改。

做法：up = −mean(相機 y 軸)（--ref 模型的相機；影像 y 朝下）；R = 把 up 轉到 +z 的最小旋轉（與 partition_from_colmap --reorient 同式）。
每個模型：X' = R X；R_cw' = R_cw Rᵀ、t' = t。=> 相機中心 C' = R C；任一點在相機座標下的值不變
=> 深度、單目尺度、SfM 三角形校正深度都不受影響（資料準備第 1～3 步不用重跑）。
寫 <sparse>/ALIGNED_UP.json（R、原 up、ref）；已經有就跳過（不會轉兩次）。ref 已轉過時，R 從 ref 的 ALIGNED_UP.json 讀（訓練與 test 一定用同一個 R）。
用法：python tools/colmap_align_up.py --ref <train>/sparse/0 <train>/sparse/0 <test>/sparse/0
"""
import argparse, dataclasses, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from internal.utils.colmap import (qvec2rotmat, read_images_binary, read_points3D_binary, rotmat2qvec,
                                   write_images_binary, write_points3D_binary)

MARK = "ALIGNED_UP.json"


def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        raise SystemExit("⛔ 上方剛好是 −z（180°），這裡沒處理")
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def cam_up(ims):
    R = np.stack([qvec2rotmat(im.qvec) for im in ims.values()])
    up = -R[:, 1, :].mean(0)
    return up / np.linalg.norm(up)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="決定上方的模型（訓練集的 sparse/0）")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    ref_mark = os.path.join(a.ref, MARK)
    if os.path.exists(ref_mark):
        R = np.array(json.load(open(ref_mark))["R"]); print(f"[align-up] ref 已轉過，R 讀自 {ref_mark}")
    else:
        up = cam_up(read_images_binary(os.path.join(a.ref, "images.bin")))
        R = rot_a_to_b(up, np.array([0.0, 0.0, 1.0]))
        print(f"[align-up] ref 相機平均上方 {np.round(up, 4)}（與 +z 夾 {np.degrees(np.arccos(np.clip(up[2], -1, 1))):.1f}°）")
    for d in a.models:
        mk = os.path.join(d, MARK)
        if os.path.exists(mk):
            print(f"[align-up] {d}：已有 {MARK}，跳過"); continue
        ims = read_images_binary(os.path.join(d, "images.bin")); pts = read_points3D_binary(os.path.join(d, "points3D.bin"))
        up0 = cam_up(ims)
        # 自檢用：轉之前幾個點在幾台相機下的相機座標
        pid = list(pts.keys())[: min(200, len(pts))]; iid = list(ims.keys())[: min(20, len(ims))]
        def camxyz(ims_, pts_):
            X = np.stack([pts_[p].xyz for p in pid])
            return np.stack([X @ qvec2rotmat(ims_[i].qvec).T + ims_[i].tvec for i in iid])
        before = camxyz(ims, pts)
        ims2 = {k: im._replace(qvec=rotmat2qvec(qvec2rotmat(im.qvec) @ R.T)) for k, im in ims.items()}
        pts2 = {k: dataclasses.replace(p, xyz=R @ p.xyz) for k, p in pts.items()}   # Point3D 是 dataclass
        after = camxyz(ims2, pts2)
        err = float(np.abs(after - before).max() / max(1e-9, np.abs(before).max()))
        up1 = cam_up(ims2)
        if err > 1e-6 or not np.allclose(up1, R @ up0, atol=1e-9):
            raise SystemExit(f"⛔ {d}：自檢失敗（相機座標相對差 {err:.2e}、轉後上方 {np.round(up1, 4)}）")
        tmp_i, tmp_p = os.path.join(d, "images.bin.tmp"), os.path.join(d, "points3D.bin.tmp")
        write_images_binary(ims2, tmp_i); write_points3D_binary(pts2, tmp_p)
        os.replace(tmp_i, os.path.join(d, "images.bin")); os.replace(tmp_p, os.path.join(d, "points3D.bin"))
        for c in ("points3D.bin.cache.npz",):     # colmap_fast 的快取（以大小＋mtime 判斷，保險起見直接刪）
            if os.path.exists(os.path.join(d, c)):
                os.remove(os.path.join(d, c))
        json.dump({"R": R.tolist(), "up_before": up0.tolist(), "up_after": up1.tolist(), "ref": os.path.abspath(a.ref)},
                  open(mk, "w"), indent=1)
        print(f"[align-up] ✅ {d}：{len(ims):,} 台相機、{len(pts):,} 點｜上方 {np.round(up0, 4)} -> {np.round(up1, 4)}｜相機座標相對差 {err:.1e}")


if __name__ == "__main__":
    main()
