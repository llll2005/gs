#!/usr/bin/env python
"""影像↔相機配對的**模型無關**驗證（純 CPU，幾秒）—— 資料準備完成後的必過關卡。

## 為什麼需要這支（2026-09-12 的教訓）

dataparser 在「SfM 名稱 != 檔名」時走**位置對應**（名稱排序第 k 個 配 檔名排序第 k 個）。
若兩邊的集合/順序不一致，它會**靜默**配成錯的對。而錯配對長得正好像我們一直在追的症狀
（地面大致對得上、細節永遠收斂不了）⇒ 必須有一個不經過模型的驗證。

⛔ **不可以用「模型重建得好」來驗配對**：全專案是 `split_mode=reconstruction`（val ⊂ train），
   模型可以把錯的影像背下來。使用者 2026-09-12 指出這個循環論證，記在研究總覽 §16.11。
⛔ 已證**無鑑別力**、不要再用的判準：
   · SIFT 特徵點落在高梯度像素（正確配對 1.0x、故意配錯 +1/+2/-1 也都 1.0x）
   · 單像素顏色相關（|r| < 0.18、margin 0.005）
   · points3D.rgb binned 成 16x16 vs 影像縮圖（**我用它誤判過一次**，差點讓使用者刪掉資料）

## 本支用的兩件事

```
A 純 metadata／幾何（不用影像）
  COLMAP sparse 的每台相機 vs transforms.json 的對應幀，比**位置與朝向**
  慣例（已驗證，非配出）：R_w2c = diag(1,-1,-1) @ (100*R_json)^T
    · MatrixCity README 自述 pose 版本「rotation matrix needs to be multiplied by 100」
    · 實測 Q=100*R_json 的正交性誤差 9.9e-08
  判準：位置殘差 < 1e-4 且**朝向殘差 < 1e-4**，全部相機都要過
  ★ 朝向那一半不可省：5,621 幀只有 1,879 個不同位置（每航點約 3 個朝向）
    ⇒ 只比位置會漏掉「同一點但朝向配錯」，而那正是最傷的錯法
B 對極幾何（用影像、不用模型）—— 抽樣覆核
  由已知相對姿態算 F，ORB 匹配看是否落在對極線上；指派檔應明顯勝過鄰近幀
```

用法:
  python tools/verify_pairing_geometric.py --data data/matrix_city/aerial/train/block_all
  python tools/verify_pairing_geometric.py --data ... --skip-epipolar     # 只跑 A（純 CPU 幾秒）
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def qvec2R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--tol", type=float, default=1e-4, help="位置與朝向殘差的容許值")
    ap.add_argument("--skip-epipolar", action="store_true")
    ap.add_argument("--n-epipolar", type=int, default=4)
    args = ap.parse_args()

    import json
    import internal.utils.colmap as cu
    sd = os.path.join(args.data, "sparse", "0")
    imb = cu.read_images_binary(os.path.join(sd, "images.bin"))
    names, Cc, Rw2c = [], [], []
    for _, v in sorted(imb.items(), key=lambda kv: kv[1].name):
        R = qvec2R(v.qvec)
        names.append(v.name); Cc.append(-R.T @ v.tvec); Rw2c.append(R)
    Cc = np.asarray(Cc); Rw2c = np.asarray(Rw2c)

    tj = os.path.join(args.data, "transforms.json")
    if not os.path.exists(tj):
        raise SystemExit(f"⛔ 找不到 {tj}（資料準備沒跑完？）")
    fr = json.load(open(tj))["frames"]
    if len(fr) != len(names):
        raise SystemExit(f"⛔ 幀數不符：sparse {len(names)} vs transforms.json {len(fr)}")
    M = np.stack([np.asarray(f["transform_matrix"], float) for f in fr])
    Pj = M[:, :3, 3] * 0.01
    Q = M[:, :3, :3] * 100.0
    ortho = np.abs(Q @ np.transpose(Q, (0, 2, 1)) - np.eye(3)).max()
    FL = np.diag([1., -1., -1.])

    files = sorted(f for f in os.listdir(os.path.join(args.data, "input"))
                   if f.lower().endswith((".png", ".jpg")))
    bn = [os.path.basename(f["file_path"]) for f in fr]
    print(f"相機 {len(names):,}  幀 {len(fr):,}  影像檔 {len(files):,}")
    print(f"Q=100*R_json 的正交性誤差 {ortho:.2e}"
          + ("  ✅（慣例成立）" if ortho < 1e-5 else "  ⛔（慣例不成立，下面的數字不可信）"))
    print(f"transforms.json 的 file_path 順序 == 檔名排序：{bn == files}")

    ec = np.linalg.norm(Cc - Pj, axis=1)
    er = np.array([np.abs(Rw2c[i] - FL @ Q[i].T).max() for i in range(len(names))])
    print(f"\nA 幾何（不用影像）")
    print(f"  位置殘差 中位 {np.median(ec):.2e}  最大 {ec.max():.2e}")
    print(f"  朝向殘差 中位 {np.median(er):.2e}  最大 {er.max():.2e}")
    uniq = int(np.unique(np.round(Pj, 5), axis=0).shape[0])
    print(f"  （不同位置只有 {uniq:,} 個 / {len(fr):,} 幀 ⇒ 每航點約 {len(fr)/max(uniq,1):.1f} 個朝向，"
          f"所以朝向那一半不可省）")
    nbad = int(((ec > args.tol) | (er > args.tol)).sum())
    if nbad:
        idx = np.argsort(-(er + ec))[:10]
        print(f"  ⛔ 有 **{nbad}** 台超過容許值 {args.tol:g}：")
        for i in idx:
            print(f"     {names[i]:>12} 位置 {ec[i]:.2e} 朝向 {er[i]:.2e}  指派檔 {bn[i]}")
    else:
        print(f"  ✅ 全部 {len(names):,} 台的位置與朝向都在 {args.tol:g} 內")

    rc = 0 if nbad == 0 else 1
    if args.skip_epipolar:
        return rc

    print(f"\nB 對極幾何（用影像、不用模型）抽樣 {args.n_epipolar} 組")
    try:
        import cv2
    except Exception:
        print("  （沒有 cv2，跳過）")
        return rc
    cams = cu.read_cameras_binary(os.path.join(sd, "cameras.bin"))
    c0 = list(cams.values())[0]
    p = c0.params
    fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if len(p) == 3 else (p[0], p[1], p[2], p[3])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    orb = cv2.ORB_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    IN = os.path.join(args.data, "input")
    idx = {n: i for i, n in enumerate(names)}

    def skew(t):
        return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

    def score(i, j, fB):
        RA, RB = Rw2c[i], Rw2c[j]
        tA = -RA @ Cc[i]; tB = -RB @ Cc[j]
        R = RB @ RA.T; t = tB - R @ tA
        F = Ki.T @ skew(t) @ R @ Ki
        a = cv2.imread(os.path.join(IN, bn[i]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(os.path.join(IN, fB), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return None
        ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None:
            return None
        ms = bf.match(da, db)
        if len(ms) < 30:
            return None
        pa = np.array([ka[m.queryIdx].pt for m in ms]); pb = np.array([kb[m.trainIdx].pt for m in ms])
        ha = np.c_[pa, np.ones(len(pa))]; hb = np.c_[pb, np.ones(len(pb))]
        l = (F @ ha.T).T
        d = np.abs((l * hb).sum(1)) / np.maximum(np.linalg.norm(l[:, :2], axis=1), 1e-9)
        return len(ms), float(np.median(d)), float(np.mean(d < 3))

    step = max(1, len(names) // (args.n_epipolar + 1))
    print(f"{'相機對':>24} {'B 用的檔案':>16} {'匹配':>6} {'對極距中位':>11} {'<3px':>7}")
    ok_cnt = tot = 0
    for k in range(1, args.n_epipolar + 1):
        i = min(k * step, len(names) - 2)
        j = i + 1
        rows = []
        base = os.path.splitext(bn[j])[0]
        try:
            num = int(base); w = len(base)
        except ValueError:
            continue
        for d, tag in ((0, "指派"), (1, "+1"), (11, "+11")):
            fB = f"{num + d:0{w}d}{os.path.splitext(bn[j])[1]}"
            if not os.path.exists(os.path.join(IN, fB)):
                continue
            r = score(i, j, fB)
            if r:
                rows.append((tag, fB, r))
        if len(rows) < 2:
            continue
        tot += 1
        best = max(rows, key=lambda x: x[2][2])
        if best[0] == "指派":
            ok_cnt += 1
        for tag, fB, (n, md, f3) in rows:
            print(f"{names[i] + '->' + names[j]:>24} {fB + ' (' + tag + ')':>16} {n:>6} "
                  f"{md:>11.2f} {100 * f3:>6.1f}%" + ("  ★" if tag == "指派" else ""))
    if tot:
        print(f"  指派檔勝出 {ok_cnt}/{tot} 組"
              + ("  ✅" if ok_cnt == tot else "  ⚠ 有組別不是指派檔勝出，看是不是重疊太少"))
        if ok_cnt < tot:
            rc = max(rc, 0)      # 不當成硬失敗（重疊太少的配對兩邊都差）
    return rc


if __name__ == "__main__":
    sys.exit(main())
