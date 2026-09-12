#!/usr/bin/env python
"""還原「`input/00000N.png` 來自哪個 block 的哪一幀」—— 用姿態比對，不猜順序（純 CPU）

## 為什麼需要

官方 `data_proc_mc.sh` 吃 `pose/block_all/transforms_train.json`，但 **MatrixCity 官方
只出 `transforms_test.json`**（HF 上實際確認：`pose/block_all/` 只有 test 那份，741 幀）
⇒ train 版是使用者自己串接 10 個 block 來的，而**串接順序決定 `0000.png..5620.png` 的命名**，
  預算 COLMAP 的 sparse 認的就是那組名字。
⇒ 本機 `pose/block_all/transforms_train.json` 只有 **3932 幀**（少了 block_9 的 1689）
  —— 那是當初 block_9 的 json 不在磁碟上時串出來的殘缺版（`eval_official_test.py` 檔頭記過）。
  本機真正在用、且已被幾何驗證過的是 `train/block_all/transforms.json`（5621 幀）。

本工具把那份**已驗證**的 5621 幀，對回 10 個 block 的原始幀（`frame_index`），
輸出 `全域索引 -> (block, frame_index)` 的映射 ⇒ 任何機器都能從原始 tar 重建出
**位元相同**的 `input/`，不必上傳 20 GB，也不必猜串接順序。

## 為什麼可信

比對用**相機位置**（1e-7 級）＋**朝向**；per-block 的 `rot_mat` 與 block_all 的
`transform_matrix` 差一個固定尺度（MatrixCity README 自述 pose 版本「rotation matrix
needs to be multiplied by 100」），所以朝向比對前先正規化。
⚠ 每個航點約 3 個朝向（5,621 幀只有 1,879 個不同位置）⇒ **只比位置會配錯朝向**，
  兩者都比才是一對一。

用法:
  python tools/build_global_frame_map.py --out data/matrix_city/aerial/global_frame_map.json
"""
import argparse
import json
import os
import sys

import numpy as np


def norm_rot(R):
    R = np.asarray(R, float)[:3, :3]
    s = np.linalg.norm(R, axis=0).mean()
    return R / max(s, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matrix_city/aerial")
    ap.add_argument("--out", default="data/matrix_city/aerial/global_frame_map.json")
    ap.add_argument("--tol-pos", type=float, default=1e-4)
    ap.add_argument("--tol-rot", type=float, default=1e-3)
    args = ap.parse_args()

    gj = os.path.join(args.data, "train", "block_all", "transforms.json")
    g = json.load(open(gj))["frames"]
    GP = np.stack([np.asarray(f["transform_matrix"], float)[:3, 3] for f in g])
    GR = np.stack([norm_rot(f["transform_matrix"]) for f in g])
    gname = [os.path.basename(f["file_path"]) for f in g]
    print(f"已驗證的全域 json：{len(g):,} 幀（{gj}）")

    src = []      # (block, frame_index, pos, rot)
    for n in range(1, 11):
        p = os.path.join(args.data, "train", f"block_{n}", "transforms.json")
        if not os.path.exists(p):
            print(f"  ⚠ block_{n} 沒有 transforms.json —— 從 HF 抓："
                  f"huggingface.co/datasets/BoDai/MatrixCity/resolve/main/"
                  f"small_city/aerial/train/block_{n}/transforms.json")
            continue
        fr = json.load(open(p))["frames"]
        for f in fr:
            M = np.asarray(f["rot_mat"], float)
            src.append((n, int(f["frame_index"]), M[:3, 3], norm_rot(M)))
        print(f"  block_{n}: {len(fr):,} 幀")
    print(f"  合計 {len(src):,} 幀（全域 json 是 {len(g):,}）")
    if not src:
        raise SystemExit("⛔ 沒有任何 per-block transforms.json")

    SP = np.stack([s[2] for s in src])
    SR = np.stack([s[3] for s in src])
    # 位置的尺度：全域 json 的平移是 100x（見檔頭）；用中位比值自動對齊，不寫死
    sc = float(np.median(np.linalg.norm(GP, axis=1)) / max(np.median(np.linalg.norm(SP, axis=1)), 1e-12))
    print(f"  位置尺度自動對齊：全域 / per-block = {sc:.6g}")
    SPs = SP * sc

    dist = np.linalg.norm(GP[:, None, :] - SPs[None, :, :], axis=2)
    out, bad = {}, 0
    used = set()
    for i in range(len(g)):
        cand = np.where(dist[i] < args.tol_pos * max(1.0, np.abs(GP).max()))[0]
        if len(cand) == 0:
            cand = np.array([int(dist[i].argmin())])
        e = np.array([np.abs(SR[j] - GR[i]).max() for j in cand])
        k = int(cand[int(e.argmin())])
        if e.min() > args.tol_rot or dist[i, k] > args.tol_pos * max(1.0, np.abs(GP).max()):
            bad += 1
        used.add(k)
        b, fi, _, _ = src[k]
        out[gname[i]] = {"block": b, "frame_index": fi,
                         "pos_res": float(dist[i, k]), "rot_res": float(e.min())}
    pr = np.array([v["pos_res"] for v in out.values()])
    rr = np.array([v["rot_res"] for v in out.values()])
    print(f"\n位置殘差 中位 {np.median(pr):.2e} 最大 {pr.max():.2e}")
    print(f"朝向殘差 中位 {np.median(rr):.2e} 最大 {rr.max():.2e}")
    print(f"一對一？ 用到 {len(used):,} 個不同的來源幀（該是 {len(g):,}）")
    print(f"超過容許值的 {bad} 筆")
    if bad == 0 and len(used) == len(g):
        print("✅ 映射完整且一對一")
    else:
        print("⛔ 映射不完整 —— 不要拿去重建 input/")
    json.dump(out, open(args.out, "w"), indent=0)
    print(f"\n已寫出 {args.out}（{len(out):,} 筆）")
    # 順手報「每個 block 貢獻的全域索引區間」—— 這就是當初的串接順序
    import collections
    seg = collections.OrderedDict()
    for nm in gname:
        b = out[nm]["block"]
        seg.setdefault(b, [nm, nm])[1] = nm
    print("\n串接順序（每個 block 佔的全域檔名區間）：")
    for b, (a, z) in seg.items():
        print(f"  block_{b:<2}  {a} ~ {z}")


if __name__ == "__main__":
    sys.exit(main())
