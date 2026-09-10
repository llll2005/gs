#!/usr/bin/env python
"""訓練當下**每一步真正走多遠**，和「需要走多遠」差幾個數量級？（純 CPU，讀 ckpt 的 Adam 真值）

## 為什麼這個量測取代了前面的估計

`tools/grad_snr.py` 想用 36 台相機**估** Adam 的 m 與 v。但 ckpt 裡就存著**真值**：
```
optimizer_states[0]  group name=means  lr=1.7669857788085934e-06
  state[0]: step / exp_avg (N,3) / exp_avg_sq (N,3)
```
⇒ 不必估，直接讀訓練在 step 60,000 當下的 `m` 與 `v`。

## 為什麼這是關鍵的量

§11.80 說失敗區「每單位殘差的 |g| 少 4.6 倍」，並據此說優化器看不見錯誤。
**但 Adam 的更新對梯度幅度是尺度不變的**：梯度全部乘 0.4 ⇒ m 與 sqrt(v) 同乘 0.4
⇒ **步長不變**。所以「幅度差」本身不解釋學不動 —— 要看的是 Adam 實際的步長：
```
update = (lr / bc1) * exp_avg / (sqrt(exp_avg_sq / bc2) + eps)
```
再把它和「**需要**移動多遠」比。後者有一個自然的尺度：**一個像素**。
一顆粒子在距離 z、焦距 fx 下，移動一像素對應的世界距離是 `z / fx`。
```
每步走的距離 / 一像素的世界距離  =>  乘上剩餘步數  =>  它**走得完**嗎？
```
```
失敗組走得完（>> 1 像素）  => 步長不是瓶頸 => 是**方向**錯或**目標**本身矛盾
失敗組走不完（<< 1 像素）  => **步長就是瓶頸**，而 lr 是一個 config 數字
```

## ⚠ 邊界

- 只看 `means`（位置）。scales/opacities 有各自的 lr 與狀態，本工具另外報但不做尺度比較。
- `exp_avg` 是 beta1=0.9 的 EMA（等效窗約 10 步 = 10 台相機），
  `exp_avg_sq` 是 beta2=0.999（約 1000 步）⇒ **兩者的平均窗差 100 倍**，
  這正是「跨視角矛盾」會壓 m 而不壓 v 的原因，也是本量測的重點。
- 粒子歸屬用投影中心多數決（與 `grad_snr.py` 相同），純 numpy。

用法: python tools/adam_state.py agd2_b12 sched30_b12 --blk 12
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.blur_persistence import per_image  # noqa: E402
from tools.veil_detect import final_test_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--model-run", default=None)
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--max-cam", type=int, default=36)
    ap.add_argument("--min-views", type=int, default=4)
    ap.add_argument("--remaining", type=int, default=30000,
                    help="拓撲凍結後還剩幾步（sched30：densify 停在 30k，共 60k）")
    args = ap.parse_args()

    mr = args.model_run or args.runs[0]
    ck = sorted(glob.glob(f"outputs/{mr}/**/*step={args.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {mr} 的 step={args.step} ckpt")
    c = torch.load(ck[0], map_location="cpu")

    sd = c["state_dict"]
    means = sd["gaussian_model.gaussians.means"].numpy()
    scales = sd["gaussian_model.gaussians.scales"].numpy()
    N = means.shape[0]

    # ── Adam 真值 ──
    opt0 = c["optimizer_states"][0]
    g0 = opt0["param_groups"][0]
    assert g0.get("name") == "means", f"opt[0] 不是 means 而是 {g0.get('name')}"
    lr = float(g0["lr"])
    b1, b2 = [float(x) for x in g0.get("betas", (0.9, 0.999))]
    eps = float(g0.get("eps", 1e-8))
    st = opt0["state"][0]
    t = float(st["step"]) if torch.is_tensor(st["step"]) else float(st["step"])
    m = st["exp_avg"].numpy()
    v = st["exp_avg_sq"].numpy()
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    upd = (lr / bc1) * m / (np.sqrt(v / bc2) + eps)          # 每步的世界座標位移
    step_len = np.linalg.norm(upd, axis=1)
    snr = np.abs(m) / (np.sqrt(v) + 1e-30)

    print(f"{mr} @ step={args.step}   N={N:,}")
    print(f"  Adam(means): lr={lr:.4e}  betas=({b1},{b2})  eps={eps:g}  step={t:.0f}")

    # ── 相機與糊掉遮罩 ──
    dmh = c["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
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

    vote_f = np.zeros(N, np.int32)
    vote_s = np.zeros(N, np.int32)
    seen = np.zeros(N, np.int32)
    px_world = np.zeros(N, np.float64)     # 一像素對應的世界距離，跨視角平均
    px_n = np.zeros(N, np.int32)
    used = 0
    for i in range(min(len(vset), args.max_cam)):
        name, _p, _mk, cam, _e = vset[i]
        name = os.path.basename(str(name))
        if name not in masks:
            continue
        keep, allb, neverb, nx, ny = masks[name]
        R = np.asarray(cam.R.cpu() if torch.is_tensor(cam.R) else cam.R, np.float64)
        Tv = np.asarray(cam.T.cpu() if torch.is_tensor(cam.T) else cam.T, np.float64)
        fx = float(cam.fx)
        W, H = int(cam.width), int(cam.height)
        pc = means.astype(np.float64) @ R.T + Tv
        z = pc[:, 2]
        zc = np.clip(z, 0.2, None)
        u = fx * pc[:, 0] / zc + W / 2
        v2 = fx * pc[:, 1] / zc + H / 2
        inb = (z > 0.2) & (u >= 0) & (u < nx * args.tile) & (v2 >= 0) & (v2 < ny * args.tile)
        lut = np.zeros(nx * ny, np.int8)
        lut[keep[allb]] = 1
        lut[keep[neverb]] = 2
        ti = (v2[inb] // args.tile).astype(np.int64) * nx + (u[inb] // args.tile).astype(np.int64)
        cls = lut[ti]
        idx = np.nonzero(inb)[0]
        np.add.at(vote_f, idx, (cls == 1).astype(np.int32))
        np.add.at(vote_s, idx, (cls == 2).astype(np.int32))
        np.add.at(seen, idx, 1)
        np.add.at(px_world, idx, zc[inb] / fx)
        np.add.at(px_n, idx, 1)
        used += 1

    pw = px_world / np.maximum(px_n, 1)
    ok = seen >= args.min_views
    gf = ok & (vote_f > vote_s) & (vote_f > 0)
    gs = ok & (vote_s > vote_f) & (vote_s > 0)

    print(f"  使用 {used} 台相機；失敗組 {gf.sum():,} 顆 / 成功組 {gs.sum():,} 顆\n")
    print(f"{'':>6} {'|m| 中位':>12} {'sqrt(v) 中位':>14} {'SNR=|m|/√v':>13} "
          f"{'每步位移':>12} {'每步(像素)':>12} {'x{:,} 步(像素)'.format(args.remaining):>18}")
    res = {}
    for lab, gm in (("失敗", gf), ("成功", gs)):
        if gm.sum() == 0:
            continue
        mm = float(np.median(np.abs(m[gm]).mean(1)))
        vv = float(np.median(np.sqrt(v[gm]).mean(1)))
        sn = float(np.median(snr[gm].mean(1)))
        sl = float(np.median(step_len[gm]))
        px = sl / max(float(np.median(pw[gm])), 1e-30)
        res[lab] = (mm, vv, sn, sl, px)
        print(f"{lab:>6} {mm:>12.4e} {vv:>14.4e} {sn:>13.4f} {sl:>12.4e} "
              f"{px:>12.3e} {px*args.remaining:>18.3f}")

    if len(res) == 2:
        f_, s_ = res["失敗"], res["成功"]
        print(f"\n  |m| 比（失敗/成功）    = {f_[0]/max(s_[0],1e-30):.3f}x")
        print(f"  sqrt(v) 比             = {f_[1]/max(s_[1],1e-30):.3f}x")
        print(f"  **SNR 比**             = **{f_[2]/max(s_[2],1e-30):.3f}x**   "
              f"（§11.80 的 |g| 比是 0.406x，那個沒經過 Adam 歸一化）")
        print(f"  每步位移比             = {f_[3]/max(s_[3],1e-30):.3f}x")
        print(f"\n  ★ 失敗組在剩下的 {args.remaining:,} 步內總共能走 "
              f"**{f_[4]*args.remaining:.2f} 個像素**（成功組 {s_[4]*args.remaining:.2f}）")

    print(f"""
判讀：
  ① SNR 比 ~ 1     => Adam 已把幅度差歸一化掉 => **§11.80 的推論有缺口**
     SNR 比 << 1    => 盲區在 Adam 之下仍成立，機制是「訊號矛盾」不是「訊號小」
  ② 剩餘步數走得動的像素數 << 1  => **步長就是瓶頸**（而 lr 只是 config 數字）
     >> 1                        => 步長夠，問題在方向/目標
⚠ 「一像素」只是自然尺度，不是嚴格的需求量；真正要移動多遠取決於誤差的空間結構。
  但**數量級**足以分辨「走不動」與「走得動」。""")


if __name__ == "__main__":
    main()
