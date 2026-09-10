#!/usr/bin/env python
"""後半 30,000 步，粒子**真的**移動了多少？（純 CPU，比對兩個 ckpt）

## 為什麼需要它

`tools/adam_state.py` 量到每步位移 1.81e-07，乘以 30,000 步得「3.20 個像素」。
**但那個乘法假設方向一致，而實測 SNR 只有 0.052** —— 也就是說更新方向高度互相矛盾，
真實淨位移會遠小於那個上界。⇒ 「3.20 像素」是**上界**，不是實際值。

本工具直接比 `step=29,999`（densify 停止點）與 `step=60,000` 的位置：
```
淨位移 / 上界(每步位移 x 步數)  =  **方向一致度**
```
```
一致度接近 1   => 粒子確實在朝一個方向前進 => 它們「在走，只是走不到」
一致度接近 0   => 純粹原地抖動 => **卡住是真的**，只是原因不是 Adam 步長小
```

## ⚠ 對應關係怎麼建（這是本工具唯一的技術風險）

`densify_until=30,000` 之後不再增生，但**週期性 trim 仍在剪**
（N 從 2,600,000 -> 2,340,000，正好 cap -> 0.9xcap）⇒ **索引不保留**。
⇒ 用 `cKDTree` 從 60k 的每顆去找 29,999 裡最近的一顆。
**有效性檢查（工具會印出來）**：
```
配對距離中位  <<  29,999 自身的最近鄰間距中位   => 配對可信
兩者同量級                                     => 配對不可信，結果作廢
```
⚠ 這個偏誤是**單向**的：若真實位移大於間距，配對會系統性**低估**位移
  ⇒ 「量到位移很小」需要上面的檢查才算數；「量到位移很大」則本來就安全。

用法: python tools/net_motion.py agd2_b12 sched30_b12 --blk 12
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.blur_persistence import per_image  # noqa: E402
from tools.veil_detect import final_test_dir  # noqa: E402


def load_means(run, step):
    ck = sorted(glob.glob(f"outputs/{run}/**/*step={step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {run} 的 step={step} ckpt")
    c = torch.load(ck[0], map_location="cpu")
    return c, c["state_dict"]["gaussian_model.gaussians.means"].numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--model-run", default=None)
    ap.add_argument("--from-step", type=int, default=29999)
    ap.add_argument("--to-step", type=int, default=60000)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--max-cam", type=int, default=36)
    ap.add_argument("--min-views", type=int, default=4)
    ap.add_argument("--step-len", type=float, default=1.8111e-07,
                    help="每步 Adam 位移（adam_state.py 量到的失敗組中位）")
    args = ap.parse_args()

    mr = args.model_run or args.runs[0]
    cA, A = load_means(mr, args.from_step)
    cB, B = load_means(mr, args.to_step)
    print(f"{mr}: step {args.from_step} N={A.shape[0]:,}  ->  step {args.to_step} N={B.shape[0]:,}")

    tree = cKDTree(A)
    d, j = tree.query(B, k=1, workers=-1)
    # 有效性：29,999 自身的最近鄰間距（取樣 20 萬顆，k=2 的第二個才是鄰居）
    sub = A[np.random.default_rng(0).choice(A.shape[0], 200_000, replace=False)]
    dd, _ = cKDTree(A).query(sub, k=2, workers=-1)
    spacing = float(np.median(dd[:, 1]))
    print(f"  配對距離中位 = {np.median(d):.4e}   29,999 自身最近鄰間距中位 = {spacing:.4e}"
          f"   比值 = {np.median(d)/spacing:.3f}")
    print(f"  {'✅ 配對可信（位移 << 間距）' if np.median(d) < 0.5*spacing else '⚠ 配對存疑，結果只能當下界'}")

    # ── 粒子歸屬（用 60k 的位置投影）──
    dmh = cB["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(
        path=dmh["path"],
        output_path=os.path.dirname(os.path.dirname(
            sorted(glob.glob(f"outputs/{mr}/**/*step={args.to_step}.ckpt", recursive=True))[0])),
        global_rank=0)
    vset = dp.get_outputs().val_set
    dirs = {r: final_test_dir(r, args.blk) for r in args.runs}
    fs = sorted(f for f in os.listdir(dirs[args.runs[0]]) if f.endswith(".png"))
    masks = {}
    for f in fs:
        outs = {r: per_image(os.path.join(dirs[r], f), args.tile, args.contrast_q)
                for r in args.runs}
        if any(o is None for o in outs.values()):
            continue
        keep, *_z, nx, ny = outs[args.runs[0]]
        ms = np.stack([outs[r][1] <= args.r_min for r in args.runs])
        name = f[:-4] if f.endswith(".png.png") else f
        masks[name] = (keep, ms.all(0), ~ms.any(0), nx, ny)

    N = B.shape[0]
    vote_f = np.zeros(N, np.int32); vote_s = np.zeros(N, np.int32)
    seen = np.zeros(N, np.int32); pxw = np.zeros(N); pxn = np.zeros(N, np.int32)
    used = 0
    for i in range(min(len(vset), args.max_cam)):
        name, _p, _mk, cam, _e = vset[i]
        name = os.path.basename(str(name))
        if name not in masks:
            continue
        keep, allb, neverb, nx, ny = masks[name]
        R = np.asarray(cam.R.cpu() if torch.is_tensor(cam.R) else cam.R, np.float64)
        Tv = np.asarray(cam.T.cpu() if torch.is_tensor(cam.T) else cam.T, np.float64)
        fx = float(cam.fx); W, H = int(cam.width), int(cam.height)
        pc = B @ R.T + Tv
        z = pc[:, 2]; zc = np.clip(z, 0.2, None)
        u = fx * pc[:, 0] / zc + W / 2
        v2 = fx * pc[:, 1] / zc + H / 2
        inb = (z > 0.2) & (u >= 0) & (u < nx * args.tile) & (v2 >= 0) & (v2 < ny * args.tile)
        lut = np.zeros(nx * ny, np.int8); lut[keep[allb]] = 1; lut[keep[neverb]] = 2
        ti = (v2[inb] // args.tile).astype(np.int64) * nx + (u[inb] // args.tile).astype(np.int64)
        cls = lut[ti]; idx = np.nonzero(inb)[0]
        np.add.at(vote_f, idx, (cls == 1).astype(np.int32))
        np.add.at(vote_s, idx, (cls == 2).astype(np.int32))
        np.add.at(seen, idx, 1)
        np.add.at(pxw, idx, zc[inb] / fx)
        np.add.at(pxn, idx, 1)
        used += 1

    pw = pxw / np.maximum(pxn, 1)
    ok = seen >= args.min_views
    gf = ok & (vote_f > vote_s) & (vote_f > 0)
    gs = ok & (vote_s > vote_f) & (vote_s > 0)
    nsteps = args.to_step - args.from_step
    upper = args.step_len * nsteps

    print(f"\n  使用 {used} 台相機；失敗組 {gf.sum():,} 顆 / 成功組 {gs.sum():,} 顆")
    print(f"  上界（每步 {args.step_len:.3e} x {nsteps:,} 步，假設方向一致）= {upper:.4e}\n")
    print(f"{'':>6} {'淨位移中位':>13} {'淨位移(像素)':>14} {'**方向一致度**':>16} {'p90 位移(像素)':>15}")
    out = {}
    for lab, gm in (("失敗", gf), ("成功", gs)):
        if gm.sum() == 0:
            continue
        dm = float(np.median(d[gm]))
        dpx = dm / max(float(np.median(pw[gm])), 1e-30)
        p90 = float(np.percentile(d[gm], 90)) / max(float(np.median(pw[gm])), 1e-30)
        out[lab] = (dm, dpx, dm / upper, p90)
        print(f"{lab:>6} {dm:>13.4e} {dpx:>14.3f} {dm/upper:>16.4f} {p90:>15.3f}")

    if len(out) == 2:
        f_, s_ = out["失敗"], out["成功"]
        print(f"\n  淨位移比（失敗/成功）= {f_[0]/max(s_[0],1e-30):.3f}x")

    print("""
判讀：
  方向一致度 ~ 1   => 粒子朝一個方向穩定前進 => 「在走，只是走不到」
  方向一致度 << 1  => 原地抖動 => **卡住是真的**（但成因不是 Adam 步長，§11.89 已排除）
  兩組一致度相近   => 這個量也不具鑑別力 => 差異在別的地方
⚠ 配對距離若逼近最近鄰間距，位移是**被低估**的，此時只有「位移大」的結論安全。""")


if __name__ == "__main__":
    main()
