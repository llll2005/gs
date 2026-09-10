#!/usr/bin/env python
"""失敗是不是「footprint 跨過一個閾值」造成的？—— 決定 `ac_shrink` 這條線的生死。

## 為什麼要問

§11.76 實測：失敗區投影半徑 **20.73px** vs 成功區 **17.51px = 1.18x**。
這個幅度與當初**正確預測 `egd` 失敗**的 1.22x（DC 誤差天花板）同一個形狀
⇒ 直覺是「半徑不是鑑別變數，縮小沒用」。

但 §11.82 第 5 條留了活口：**雙峰而非連續退化**（早期夠小的沒掉進盲區、稍大的永久卡住）。
若 corr 對半徑存在**懸崖**，那 18% 的平均差就可能是「跨過閾值的比例差」，
而不是「效果只有 18%」——兩者對介入的含意完全相反：

```
連續平滑下降  => 半徑不是鑑別變數 => ac_shrink 是對 1.18x 的變數做 10x 介入 => 收線
有懸崖        => 閾值就是 target_px，且能算出「縮到閾值以下」能救回多少 tile
```

## 同時量第二件事：盲區是不是 footprint 驅動的

「空間梯度積分抵消」假說（§11.80，已得 4.6x 支持）說抵消程度是
**footprint 相對於殘差空間頻率**的函數 ⇒ 它預測 `|g|/AC` 應**隨半徑單調下降**。

```
|g|/AC 隨半徑下降  => 機制自洽，footprint 確實是那個旋鈕
|g|/AC 與半徑無關  => **抵消的驅動量不是 footprint** => 縮小打不到成因，
                      要改找真正的驅動量（殘差頻率？重疊層數？）
```
⚠ 這是本工具最有價值的輸出：它能**否證掉整個「縮 footprint」家族**，
   而不是再花 9.6 小時測一個點（§11.86 的教訓：診斷能拒絕家族，介入一次只能測一個點）。

## 實作與 `grad_blindspot.py` 保持一致

- `|g|` 同樣取自 `viewspace_points.grad[:, 2]`（ABSGRAD=1 累加的 `|dL_ds|`，逐位元不變）。
- loss 同樣是訓練時的 `(1-lambda)*L1 + lambda*(1-SSIM)`。
- primitive 依**投影中心**分到 tile —— 與 grad_blindspot 相同，兩邊才可互比。
  ⚠ 這會低估大 footprint 的跨 tile 影響；但本工具問的是「同一批 tile 的半徑 vs 品質」，
    center 分派在兩組間是無偏的。

用法: python tools/radius_cliff.py agd2_b12 sched30_b12 --blk 12
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.veil_detect import final_test_dir, tiles  # noqa: E402
from tools.blur_persistence import per_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--model-run", default=None)
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--lambda-dssim", type=float, default=0.2)
    ap.add_argument("--max-cam", type=int, default=36)
    ap.add_argument("--nbin", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    mr = args.model_run or args.runs[0]
    ck = sorted(glob.glob(f"outputs/{mr}/**/*step={args.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {mr} 的 step={args.step} ckpt")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    from PIL import Image as _Im
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck[0], device=dev, eval_mode=False, pre_activate=False)
    print(f"模型 {mr} @ step={args.step}   N = {model.n_gaussians:,}")

    ckpt = torch.load(ck[0], map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
                                   global_rank=0)
    vset = dp.get_outputs().val_set

    # 每個 tile 的品質（corr）—— 取跨配方的**平均**，避免綁單一配方
    dirs = {r: final_test_dir(r, args.blk) for r in args.runs}
    fs = sorted(f for f in os.listdir(dirs[args.runs[0]]) if f.endswith(".png"))
    qual = {}
    for f in fs:
        outs = {r: per_image(os.path.join(dirs[r], f), args.tile, args.contrast_q)
                for r in args.runs}
        if any(o is None for o in outs.values()):
            continue
        keep, *_z, nx, ny = outs[args.runs[0]]
        rmean = np.mean([outs[r][1] for r in args.runs], axis=0)
        name = f[:-4] if f.endswith(".png.png") else f
        qual[name] = (keep, rmean, nx, ny)

    bg = torch.zeros((3,), device=dev)
    rows = []          # (radius, corr, |g|, AC)
    used = 0
    for i in range(min(len(vset), args.max_cam)):
        name, img_path, _mask, cam, _extra = vset[i]
        name = os.path.basename(str(name))
        if name not in qual:
            continue
        keep, rmean, nx, ny = qual[name]
        cam = cam.to_device(dev)
        W, H = int(cam.width), int(cam.height)
        _a = np.asarray(_Im.open(img_path).convert("RGB").resize((W, H), _Im.BILINEAR),
                        np.float32) / 255.
        gt = torch.from_numpy(_a).permute(2, 0, 1).to(dev)

        model.zero_grad(set_to_none=True)
        o = renderer(cam, model, bg_color=bg)
        img = o["render"]
        loss = (1.0 - args.lambda_dssim) * torch.abs(img - gt).mean() \
            + args.lambda_dssim * (1.0 - ssim_fn(img, gt))
        loss.backward()
        vp = o.get("viewspace_points")
        if vp is None or vp.grad is None or vp.grad.shape[-1] < 3:
            raise SystemExit("拿不到 viewspace_points.grad[:,2] —— 光柵器不是 ABSGRAD=1 編的")
        ag = vp.grad[:, 2].detach().abs()

        with torch.no_grad():
            R = (cam.R if torch.is_tensor(cam.R) else torch.tensor(cam.R, device=dev)).to(dev).float()
            Tv = (cam.T if torch.is_tensor(cam.T) else torch.tensor(cam.T, device=dev)).to(dev).float()
            pc = model.get_xyz.detach() @ R.T + Tv
            z = pc[:, 2]
            fx = float(cam.fx)
            vis = z > 0.2
            u = fx * pc[vis, 0] / z[vis] + W / 2
            v = fx * pc[vis, 1] / z[vis] + H / 2
            rad = 3.0 * fx * model.get_scales().detach().max(dim=1).values[vis] / z[vis]
            ok = (u >= 0) & (u < nx * args.tile) & (v >= 0) & (v < ny * args.tile)
            ti = ((v[ok] // args.tile).long() * nx + (u[ok] // args.tile).long()).cpu().numpy()
            gv = ag[vis][ok].cpu().numpy()
            rv = rad[ok].cpu().numpy()

            e = gt.mean(0).cpu().numpy() - img.detach().mean(0).cpu().numpy()
            ac = tiles(e[:ny * args.tile, :nx * args.tile], args.tile).std(axis=1)

        nt = nx * ny
        med_g = np.full(nt, np.nan)
        med_r = np.full(nt, np.nan)
        ordr = np.argsort(ti, kind="stable")
        ts, gs, rs = ti[ordr], gv[ordr], rv[ordr]
        uq, st = np.unique(ts, return_index=True)
        en = np.append(st[1:], len(ts))
        for t_, a_, b_ in zip(uq, st, en):
            med_g[t_] = np.median(gs[a_:b_])
            med_r[t_] = np.median(rs[a_:b_])
        # ⚠ 索引語意不同，混用會靜默取到別的 tile：
        #   `rmean`(=per_image 的 corr) 按 **keep 內的位置** 編號（長度 len(keep)）
        #   `med_r`/`med_g`/`ac`        按 **tile 編號** 編號（長度 nx*ny）
        # 2026-09-04 第一版就是拿 tile 編號去索引 corr 才 IndexError。
        assert len(rmean) == len(keep), f"corr 長度 {len(rmean)} != keep {len(keep)}"
        for j, k in enumerate(keep):
            if k < len(ac) and not (np.isnan(med_r[k]) or np.isnan(med_g[k])):
                rows.append((med_r[k], rmean[j], med_g[k], ac[k]))
        used += 1

    A = np.array(rows)
    print(f"實際使用 {used} 台相機，{len(A):,} 個高對比 tile\n")
    if len(A) < 100:
        raise SystemExit("樣本太少")

    # 依半徑分位分箱（等樣本數），避免長尾把箱子拉空
    qs = np.quantile(A[:, 0], np.linspace(0, 1, args.nbin + 1))
    qs[-1] += 1e-6
    print(f"{'半徑區間 px':>18} {'n':>7} {'corr 中位':>10} {'糊掉%':>8} {'|g|/AC':>12}")
    cur, ratio = [], []
    for j in range(args.nbin):
        m = (A[:, 0] >= qs[j]) & (A[:, 0] < qs[j + 1])
        if m.sum() < 10:
            continue
        c = np.median(A[m, 1])
        blur = 100.0 * np.mean(A[m, 1] <= 0.60)
        rr = np.median(A[m, 2]) / max(np.median(A[m, 3]), 1e-12)
        cur.append((0.5 * (qs[j] + qs[j + 1]), c, blur))
        ratio.append(rr)
        print(f"{qs[j]:>8.2f}~{qs[j+1]:<9.2f} {m.sum():>7,} {c:>10.3f} {blur:>8.1f} {rr:>12.4e}")

    cur = np.array(cur)
    # 懸崖 = 相鄰箱之間 corr 的最大跌幅，對照「整體全距」
    d = np.diff(cur[:, 1])
    span = cur[:, 1].max() - cur[:, 1].min()
    k = int(np.argmin(d))
    print(f"\n  corr 全距（首箱-末箱）  = {cur[0,1] - cur[-1,1]:+.3f}")
    print(f"  最大單箱跌幅            = {d[k]:+.3f}  在 {cur[k,0]:.1f} -> {cur[k+1,0]:.1f} px")
    print(f"  最大跌幅 / 全距         = **{abs(d[k]) / max(span, 1e-9):.1%}**"
          f"   （均勻下降的期望值 = {100.0/max(len(d),1):.0f}%）")
    rho = np.corrcoef(np.log(A[:, 0].clip(1e-6)), A[:, 1])[0, 1]
    per_tile_ratio = A[:, 2] / np.maximum(A[:, 3], 1e-12)
    ok = per_tile_ratio > 0
    rg = np.corrcoef(np.log(A[ok, 0].clip(1e-6)), np.log(per_tile_ratio[ok]))[0, 1]
    print(f"  corr(log 半徑, tile corr)  = {rho:+.3f}")
    print(f"  corr(log 半徑, log |g|/AC) = **{rg:+.3f}**")

    print(f"""
判讀：
  ① 懸崖？ 最大跌幅/全距 >> {100.0/max(len(d),1):.0f}%  => **有閾值**，該處半徑就是 ac_shrink 的 target_px
            ~= {100.0/max(len(d),1):.0f}%              => 連續平滑 => 半徑不是鑑別變數
                                                          => **ac_shrink 收線**（對 1.18x 的變數做 10x 介入）
  ② 機制？ corr(log 半徑, log |g|/AC) 顯著為負 => 抵消確實由 footprint 驅動，縮小打得到成因
            ~0                                => **抵消的驅動量不是 footprint**
                                                 => 整個「縮 footprint」家族被否證，
                                                    要改找真正的驅動量（殘差頻率／重疊層數）
⚠ primitive 依投影中心分派（與 grad_blindspot 一致），會低估大 footprint 的跨 tile 影響；
  本工具比的是同一批 tile 的相對關係，此偏差對兩組同向。""")


if __name__ == "__main__":
    main()
