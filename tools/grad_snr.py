#!/usr/bin/env python
"""Adam 真正看見的是 `m/sqrt(v)`，不是 `|g|` —— 盲區在 Adam 之下還站得住嗎？

## 為什麼必須量這個（它可能推翻 §11.80）

§11.80 量到失敗區「每單位殘差的 |g| 少 4.6 倍」，並據此說優化器看不見自己的錯誤。
**但 Adam 的更新是 `lr * m / (sqrt(v) + eps)`，對梯度幅度是尺度不變的**：
```
某顆粒子的梯度全部乘 0.4  =>  m 乘 0.4、sqrt(v) 也乘 0.4  =>  **步長不變**
```
⇒ **「幅度少 4.6 倍」本身不會讓它學得慢。** §11.80 的推論有缺口。

會讓 Adam 步長變小的是**訊噪比**，而空間抵消恰好只壓一邊：
```
抵消壓低 m（跨視角/跨步的**平均**）
但不壓低 v（每個視角自己的梯度都很大，只是方向互相相反）
=> m/sqrt(v) 崩掉  => Adam 的有效步長崩掉
```
⇒ **正確的可量量是 `|m| / sqrt(v)`（跨相機），不是 `|g|`。**

## 判準

```
失敗區 SNR << 成功區  => 盲區在 Adam 之下**仍成立**，而且機制被講精確了：
                         不是「訊號小」而是「訊號互相矛盾」=> 出口是**降低視角間的衝突**
                         （例如逐視角/逐區域的權重、或先在子集相機上收斂）
失敗區 SNR ~ 成功區   => **§11.80 的推論不成立**：Adam 已經把幅度差歸一化掉了，
                         失敗另有原因 => 回頭查 ①步長 ④Adam 狀態 ⑤損失形狀
```

## 實作要點

- 量的是 **`means.grad`（世界座標 3D）** —— 那才是 Adam 實際更新的張量。
  （§11.80 量的 `dL_dmean2D.z` 是光柵器裡累加的 `|dL_ds|`，是**視空間、取絕對值**的量，
   不能拿來算跨視角的**帶號**平均。）
- **一台相機一次 backward**，把帶號梯度累進 `m_sum` / `v_sum`，最後才算 SNR。
  這與 Adam 的行為同構：Adam 的 v 累積的正是跨步（＝跨相機）的平方。
- loss 與訓練同構：`(1-lambda)*L1 + lambda*(1-SSIM)`。
- 粒子歸屬：在每個視角依投影中心落入的 tile 投票，取**多數決**決定它屬於失敗區還是成功區。
  ⚠ 只納入「在至少 `--min-views` 個視角被看到」的粒子，否則單視角粒子會製造假訊號。

用法: python tools/grad_snr.py agd2_b12 sched30_b12 --blk 12
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
    ap.add_argument("--lambda-dssim", type=float, default=0.2)
    ap.add_argument("--max-cam", type=int, default=36)
    ap.add_argument("--min-views", type=int, default=4)
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
    N = model.n_gaussians
    print(f"模型 {mr} @ step={args.step}   N = {N:,}")

    ckpt = torch.load(ck[0], map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
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

    means = model.gaussians["means"]
    means.requires_grad_(True)
    m_sum = torch.zeros((N, 3), device=dev)
    v_sum = torch.zeros((N, 3), device=dev)
    vote_f = torch.zeros(N, device=dev)      # 落入失敗 tile 的次數
    vote_s = torch.zeros(N, device=dev)      # 落入成功 tile 的次數
    seen = torch.zeros(N, device=dev)
    bg = torch.zeros((3,), device=dev)
    used = 0

    for i in range(min(len(vset), args.max_cam)):
        name, img_path, _mask, cam, _extra = vset[i]
        name = os.path.basename(str(name))
        if name not in masks:
            continue
        keep, allb, neverb, nx, ny = masks[name]
        cam = cam.to_device(dev)
        W, H = int(cam.width), int(cam.height)
        gt = torch.from_numpy(
            np.asarray(_Im.open(img_path).convert("RGB").resize((W, H), _Im.BILINEAR),
                       np.float32) / 255.).permute(2, 0, 1).to(dev)

        model.zero_grad(set_to_none=True)
        img = renderer(cam, model, bg_color=bg)["render"]
        loss = (1.0 - args.lambda_dssim) * torch.abs(img - gt).mean() \
            + args.lambda_dssim * (1.0 - ssim_fn(img, gt))
        loss.backward()
        g = means.grad
        if g is None:
            raise SystemExit("means.grad 是 None")
        m_sum += g.detach()
        v_sum += g.detach() ** 2

        with torch.no_grad():
            R = (cam.R if torch.is_tensor(cam.R) else torch.tensor(cam.R, device=dev)).to(dev).float()
            Tv = (cam.T if torch.is_tensor(cam.T) else torch.tensor(cam.T, device=dev)).to(dev).float()
            pc = model.get_xyz.detach() @ R.T + Tv
            z = pc[:, 2]
            fx = float(cam.fx)
            vis = z > 0.2
            u = fx * pc[:, 0] / z.clamp_min(0.2) + W / 2
            v = fx * pc[:, 1] / z.clamp_min(0.2) + H / 2
            inb = vis & (u >= 0) & (u < nx * args.tile) & (v >= 0) & (v < ny * args.tile)
            ti = (v[inb] // args.tile).long() * nx + (u[inb] // args.tile).long()
            # tile 分類查表（0=不管、1=失敗、2=成功）
            lut = torch.zeros(nx * ny, device=dev)
            lut[torch.as_tensor(keep[allb], device=dev, dtype=torch.long)] = 1.0
            lut[torch.as_tensor(keep[neverb], device=dev, dtype=torch.long)] = 2.0
            cls = lut[ti]
            idx = torch.nonzero(inb, as_tuple=True)[0]
            vote_f.index_add_(0, idx, (cls == 1).float())
            vote_s.index_add_(0, idx, (cls == 2).float())
            seen.index_add_(0, idx, torch.ones_like(cls))
        used += 1

    print(f"實際使用 {used} 台相機\n")
    m = m_sum / used
    v = v_sum / used
    # Adam 的有效步長 ∝ |m| / sqrt(v)，逐軸算再取三軸平均
    snr = (m.abs() / (v.sqrt() + 1e-30)).mean(dim=1)

    ok = seen >= args.min_views
    grp_f = ok & (vote_f > vote_s) & (vote_f > 0)
    grp_s = ok & (vote_s > vote_f) & (vote_s > 0)
    print(f"{'':>8} {'顆數':>10} {'|m| 中位':>13} {'sqrt(v) 中位':>15} "
          f"{'**SNR=|m|/sqrt(v)**':>22}")
    out = {}
    for lab, gmask in (("失敗", grp_f), ("成功", grp_s)):
        if int(gmask.sum()) == 0:
            continue
        mm = float(m[gmask].abs().mean(dim=1).median())
        vv = float(v[gmask].sqrt().mean(dim=1).median())
        ss = float(snr[gmask].median())
        out[lab] = (mm, vv, ss)
        print(f"{lab:>8} {int(gmask.sum()):>10,} {mm:>13.4e} {vv:>15.4e} {ss:>22.4e}")

    if len(out) == 2:
        (mf, vf, sf), (ms_, vs, ss_) = out["失敗"], out["成功"]
        print(f"\n  |m| 比（失敗/成功）        = {mf/max(ms_,1e-30):.3f}x")
        print(f"  sqrt(v) 比                 = {vf/max(vs,1e-30):.3f}x")
        print(f"  **SNR 比（Adam 實際感受）** = **{sf/max(ss_,1e-30):.3f}x**")
        print(f"  對照：§11.80 的 |g|/殘差 比值的比 = 0.218x（那個量沒有經過 Adam 歸一化）")

    print("""
判讀：
  SNR 比 << 1  => 盲區在 Adam 之下**仍成立**，且機制更精確：不是「訊號小」而是
                  「訊號互相矛盾」（m 被壓、v 沒被壓）=> 出口是**降低視角間的衝突**
  SNR 比 ~ 1   => **§11.80 的推論有缺口**：Adam 已把幅度差歸一化掉，
                  失敗另有原因 => 回頭查 步長 / Adam 狀態 / 損失形狀
⚠ 本工具用 36 台相機的**單步**梯度估 m 與 v；Adam 是 beta=0.9/0.999 的指數移動平均，
  等效視窗約 10 與 1000 步，所以這是 v 的**下界估計**（真實的 v 累積得更久）。
  比值仍可用，因為兩組用同一組相機。""")


if __name__ == "__main__":
    main()
