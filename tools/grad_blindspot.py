#!/usr/bin/env python
"""失敗區的位置梯度是不是「死的」？—— 直接檢驗「空間梯度積分抵消」假說。

## 假說（外部諮詢第二輪，Gemini）

```
dL/dmu_i = sum_p (C(p) - I_GT(p)) · T_i(p) · d(alpha_i(p))/d(mu_i) · c_i
```
DC 擬合完之後，殘差是**零均值的高頻振盪**；而 `d(alpha)/d(mu)` 在 footprint 上是
**平滑的奇函數** ⇒ 兩者在 footprint 上求和**正負抵消 ≈ 0**
⇒ **殘差很大而位置梯度為零** ⇒ `|g_mu| > tau` 永不觸發 ⇒ 該區永久卡在低頻態。

## 它預測什麼（本工具就是量這個）

**失敗 tile 的「梯度 / 殘差」比值應該遠低於成功 tile。**
```
量 R_t = (該 tile 內 primitive 的 |g| 中位) / (該 tile 的高頻殘差能量)
失敗/成功 的 R 比值 << 1  => 假說成立：殘差大而梯度死
失敗/成功 的 R 比值 ~ 1   => 假說否證，梯度並不特別低 => 要改看
                             「梯度夠大但方向不對」（GPT 的 <J,e> 餘弦那一支）
```

## 實作要點

- `|g|` 來源：光柵器以 `ABSGRAD=1` 編譯，`|dL_ds|` 被累加進**從未被讀取的**
  `dL_dmean2D.z`（渲染與梯度逐位元不變）⇒ `viewspace_points.grad[:, 2]`。
- loss 必須**與訓練時同構**（`vanilla_metrics.py:68`）：
  `(1-lambda_dssim)·L1 + lambda_dssim·(1-SSIM)`，預設 `lambda_dssim=0.2`。
  ⚠ 用錯 loss 量到的梯度就不是訓練時的梯度。
- **一台相機一次 backward**，逐相機把 `|g|` 投影到 tile 後累加；
  殘差同樣逐相機算，最後才比中位數。

用法: python tools/grad_blindspot.py agd2_b12 sched30_b12 --blk 12
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
    ap.add_argument("runs", nargs="+", help="至少兩個配方（取糊掉遮罩交集）")
    ap.add_argument("--blk", default="12")
    ap.add_argument("--model-run", default=None, help="取哪個跑次的模型（預設第一個）")
    ap.add_argument("--tile", type=int, default=48)
    ap.add_argument("--r-min", type=float, default=0.60)
    ap.add_argument("--contrast-q", type=float, default=0.85)
    ap.add_argument("--lambda-dssim", type=float, default=0.2)
    ap.add_argument("--max-cam", type=int, default=36)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    mr = args.model_run or args.runs[0]
    ck = sorted(glob.glob(f"outputs/{mr}/**/*step=60000.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {mr} 的 60k ckpt")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.ssim import ssim as ssim_fn
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck[0], device=dev, eval_mode=False, pre_activate=False)
    print(f"模型 {mr}  N = {model.n_gaussians:,}")

    ckpt = torch.load(ck[0], map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                  output_path=os.path.dirname(os.path.dirname(ck[0])),
                                  global_rank=0)
    out = dp.get_outputs()
    vset = out.val_set
    print(f"驗證相機 {len(vset)} 台")

    # 糊掉遮罩（跨配方交集），鍵是 test 圖檔名
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

    bg = torch.zeros((3,), device=dev)
    P, N_ = [], []
    used = 0
    # ⚠ ImageSet 的一筆是 5 元素：(COLMAP名, 磁碟路徑, mask, Camera, extra)
    #   影像**不會**被回傳，要自己從路徑讀，並縮到相機的解析度（已含 down_sample）。
    from PIL import Image as _Im
    for i in range(min(len(vset), args.max_cam)):
        name, img_path, _mask, cam, _extra = vset[i]
        name = os.path.basename(str(name))
        if name not in masks:
            continue
        keep, allb, neverb, nx, ny = masks[name]
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
        g = None if vp is None else vp.grad
        if g is None or g.shape[-1] < 3:
            raise SystemExit("拿不到 viewspace_points.grad[:,2] —— 光柵器可能不是以 ABSGRAD=1 編譯的")
        ag = g[:, 2].detach().abs()

        # 逐 tile：|g| 中位（把 primitive 依投影中心分到 tile）
        with torch.no_grad():
            R = cam.R if torch.is_tensor(cam.R) else torch.tensor(cam.R, device=dev)
            T = cam.T if torch.is_tensor(cam.T) else torch.tensor(cam.T, device=dev)
            R, T = R.to(dev).float(), T.to(dev).float()
            pc = model.get_xyz.detach() @ R.T + T
            z = pc[:, 2]
            fx = float(cam.fx)
            vis = z > 0.2
            u = fx * pc[vis, 0] / z[vis] + W / 2
            v = fx * pc[vis, 1] / z[vis] + H / 2
            ok = (u >= 0) & (u < nx * args.tile) & (v >= 0) & (v < ny * args.tile)
            ti = ((v[ok] // args.tile).long() * nx + (u[ok] // args.tile).long()).cpu().numpy()
            gv = ag[vis][ok].cpu().numpy()

            # 高頻殘差能量（tile 內，灰階）
            gg = gt.mean(0).cpu().numpy()
            rr = img.detach().mean(0).cpu().numpy()
            e = gg - rr
            Ge = tiles(e[:ny * args.tile, :nx * args.tile], args.tile)
            ac = Ge.std(axis=1)                      # 零均值高頻能量

        nt = nx * ny
        med_g = np.full(nt, np.nan)
        ordr = np.argsort(ti, kind="stable")
        ts, gs = ti[ordr], gv[ordr]
        uq, st = np.unique(ts, return_index=True)
        en = np.append(st[1:], len(ts))
        for t_, a_, b_ in zip(uq, st, en):
            med_g[t_] = np.median(gs[a_:b_])
        for m, acc in ((allb, P), (neverb, N_)):
            if m.sum():
                k = keep[m]
                acc.append(np.stack([med_g[k], ac[k]], 1))
        used += 1

    print(f"實際使用 {used} 台相機\n")
    P = np.concatenate(P) if P else np.zeros((0, 2))
    N_ = np.concatenate(N_) if N_ else np.zeros((0, 2))
    print(f"{'':>8} {'|g| 中位':>14} {'高頻殘差':>12} {'比值 |g|/殘差':>16} {'n':>8}")
    med = {}
    for nm, A in (("失敗", P), ("成功", N_)):
        if not len(A):
            continue
        g_, a_ = np.nanmedian(A[:, 0]), np.nanmedian(A[:, 1])
        med[nm] = (g_, a_)
        print(f"{nm:>8} {g_:>14.4e} {a_:>12.5f} {g_ / max(a_, 1e-12):>16.4e} {len(A):>8,}")
    if len(med) == 2:
        gp, ap_ = med["失敗"]; gn, an = med["成功"]
        print(f"\n  |g| 比（失敗/成功）      = **{gp / max(gn, 1e-30):.3f}x**")
        print(f"  高頻殘差比               = {ap_ / max(an, 1e-12):.3f}x")
        print(f"  **梯度/殘差 比值的比**   = **{(gp / max(ap_, 1e-12)) / max(gn / max(an, 1e-12), 1e-30):.3f}x**")
    print("""
判讀：
  梯度/殘差 比值的比 << 1  => **假說成立**：失敗區殘差大而位置梯度死（空間積分抵消）
                              ⇒ 解法不是調密度排程，而是**不依賴 |g| 的顯性高頻觸發**
  ~1 或 >1                 => **假說否證**：梯度並不特別低
                              ⇒ 改查「梯度夠大但方向不對」（GPT 的 <J,e> 餘弦）""")


if __name__ == "__main__":
    main()
