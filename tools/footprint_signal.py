#!/usr/bin/env python
"""足跡當增生訊號值不值得跑？（純 CPU，量天花板與重疊率）

## 這條線從哪來

§11.88 量到 `|g|` 隨投影面積成長（末箱/首箱 37x）—— `|g|` 是**外延量**（對覆蓋像素求和）。
⇒ **`absgrad_densify` 之所以有效，可能不是因為 `|g|` 是誤差訊號，
   而是因為它 ∝ 足跡，等於一條粗糙的「優先分裂大顆」規則。**
而 §11.88 的懸崖說「投影半徑 >12px 的 tile 品質崩掉」⇒ 大顆就是該被分裂的。

若如此，**直接用足跡當權重**應該同樣有效甚至更好，而且它是
`cost_aware_densify` 的**負權重**方向（大足跡 -> 增生更多），
＝ **Taming 3DGS 的符號**（他們 `+0.1 * c^i_g`，c = 該高斯在該視角覆蓋的像素數）。
⚠ Taming 的權重是 0.1，而 `∇g` 是 50 ⇒ 該項只佔分數約 **0.2%**
  ⇒ **他們等於沒測過這個方向**（研究總覽 §3.1 已記）。

## 判準（沿用 §11.30 驗證過的方法：先量天花板再決定花不花 GPU）

```
ceiling(top5%)/mean     我方 DC 誤差 1.22x（失敗）／AC 2.83x（弱）／|g| 16.68x（成功）
                        => 足跡若 >> 1.22 才有空間
top10% 與 opacity 的重疊  ~10%（隨機）=> 換訊號會換掉增生位置，值得跑
                          高      => 與現況取樣一樣，白跑
```

用法: python tools/footprint_signal.py --run agd2_b12
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ceiling(x, q=0.05):
    k = max(1, int(len(x) * q))
    return float(np.sort(x)[-k:].mean() / max(x.mean(), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="agd2_b12")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--max-cam", type=int, default=36)
    ap.add_argument("--with-grad", action="store_true",
                    help="★ 需要 GPU：同時算 |g|（ABSGRAD 累加在 grad[:,2]），"
                         "回答**現行最佳配方的增益到底來自哪個訊號**：\n"
                         "  |g| 與足跡的 top10% 高度重疊 => absgrad 其實是「優先分裂大顆」，"
                         "誤差導向的說法要改；\n"
                         "  重疊低 => 兩者互補，可疊加")
    args = ap.parse_args()

    ck = sorted(glob.glob(f"outputs/{args.run}/**/*step={args.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {args.run} 的 step={args.step} ckpt")
    c = torch.load(ck[0], map_location="cpu")
    sd = c["state_dict"]
    means = sd["gaussian_model.gaussians.means"].numpy().astype(np.float64)
    scales = np.exp(sd["gaussian_model.gaussians.scales"].numpy().astype(np.float64))
    opac = torch.sigmoid(sd["gaussian_model.gaussians.opacities"]).numpy().ravel()
    N = means.shape[0]
    smax = scales.max(axis=1)

    dmh = c["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
                                   global_rank=0)
    cams = dp.get_outputs().train_set.cameras

    area = np.zeros(N)          # sum over views of pi*r^2（＝真正的「渲染成本」外延量）
    seen = np.zeros(N, np.int32)
    rad_sum = np.zeros(N)
    n = min(args.max_cam, len(cams))
    for i in range(n):
        cam = cams[i]
        R = np.asarray(cam.R.cpu() if torch.is_tensor(cam.R) else cam.R, np.float64)
        Tv = np.asarray(cam.T.cpu() if torch.is_tensor(cam.T) else cam.T, np.float64)
        fx = float(cam.fx); W, H = int(cam.width), int(cam.height)
        pc = means @ R.T + Tv
        z = pc[:, 2]; zc = np.clip(z, 0.2, None)
        u = fx * pc[:, 0] / zc + W / 2
        v = fx * pc[:, 1] / zc + H / 2
        r = 3.0 * fx * smax / zc
        vis = (z > 0.2) & (u > -r) & (u < W + r) & (v > -r) & (v < H + r)
        area[vis] += np.pi * r[vis] ** 2
        rad_sum[vis] += r[vis]
        seen[vis] += 1

    rad = rad_sum / np.maximum(seen, 1)
    print(f"{args.run} @ {args.step}   N={N:,}   用 {n} 台訓練相機\n")
    print(f"{'訊號':>22} {'ceiling(top5%)/mean':>22} {'中位':>12} {'p99':>12}")
    sigs = {
        "足跡面積 Σπr²（外延）": area,
        "平均投影半徑 r": rad,
        "r²（強度量）": rad ** 2,
        "opacity（現況取樣）": opac,
        "opacity x 足跡": opac * area,
    }
    for k, v in sigs.items():
        print(f"{k:>22} {ceiling(v):>22.2f}x {np.median(v):>12.4e} "
              f"{np.percentile(v,99):>12.4e}")

    print(f"\n  對照天花板：我方 DC 誤差 **1.22x**（實測失敗）／AC **2.83x**（弱）"
          f"／|g| **16.68x**（實測成功）")

    if args.with_grad:
        # ★ 現行最佳配方 absgrad_densify 用的就是這個 |g|。它跟足跡是不是同一批？
        import torch as _t
        from internal.utils.gaussian_model_loader import GaussianModelLoader
        from internal.utils.ssim import ssim as ssim_fn
        from PIL import Image as _Im
        dev = _t.device("cuda")
        model, renderer, _ = GaussianModelLoader.\
            initialize_model_and_renderer_from_checkpoint_file(ck[0], device=dev,
                                                               eval_mode=False, pre_activate=False)
        vset = dp.get_outputs().val_set
        gacc = _t.zeros(N, device=dev)
        bg = _t.zeros((3,), device=dev)
        for i in range(min(len(vset), args.max_cam)):
            _n2, ip, _m2, cam2, _e2 = vset[i]
            cam2 = cam2.to_device(dev)
            W2, H2 = int(cam2.width), int(cam2.height)
            gt = _t.from_numpy(np.asarray(_Im.open(ip).convert("RGB").resize((W2, H2),
                               _Im.BILINEAR), np.float32) / 255.).permute(2, 0, 1).to(dev)
            model.zero_grad(set_to_none=True)
            o = renderer(cam2, model, bg_color=bg)
            img = o["render"]
            (0.8 * _t.abs(img - gt).mean() + 0.2 * (1 - ssim_fn(img, gt))).backward()
            vp = o.get("viewspace_points")
            if vp is None or vp.grad is None or vp.grad.shape[-1] < 3:
                raise SystemExit("拿不到 grad[:,2] —— 光柵器不是 ABSGRAD=1 編的")
            gacc += vp.grad[:, 2].detach().abs()
        g = (gacc / min(len(vset), args.max_cam)).cpu().numpy()
        sigs["|g|（absgrad_densify 用的）"] = g
        print(f"\n  ★ |g| 實測 ceiling = **{ceiling(g):.2f}x**（記錄值 16.68x）")

    # top10% 重疊
    k = int(0.1 * N)
    top = {name: set(np.argpartition(-v, k)[:k].tolist()) for name, v in sigs.items()}
    print(f"\n  top10% 集合重疊率（相對隨機 10%）")
    base = top["opacity（現況取樣）"]
    for name in ("足跡面積 Σπr²（外延）", "平均投影半徑 r", "opacity x 足跡"):
        ov = len(top[name] & base) / k
        print(f"    {name:>22} vs opacity : {100*ov:>6.2f}%  （隨機 = 10%）")
    if args.with_grad:
        gk = "|g|（absgrad_densify 用的）"
        print(f"\n  ★★ 決定性的一組：現行最佳配方的訊號 vs 足跡")
        for name in ("足跡面積 Σπr²（外延）", "平均投影半徑 r", "opacity（現況取樣）"):
            ov = len(top[gk] & top[name]) / k
            print(f"    {gk} vs {name:>22} : {100*ov:>6.2f}%")
        print("""    判讀：|g| vs 足跡 >> 10%  => **absgrad 的增益主要來自「優先分裂大顆」**
           => 「誤差導向增生」的說法要改寫，且足跡是更直接、更便宜的同一件事
           ~10%              => 兩者互補 => 值得疊加（absgrad + 足跡權重）""")

    print(f"""
判讀：
  足跡天花板 >> 1.22x 且與 opacity 重疊接近 10%
    => 換成足跡權重會**換掉增生位置**且有空間 => 值得跑 `cost_aware_densify` 的**負權重**
       （大足跡 -> 增生更多；程式已支援負值，`!= 0` 的守衛 2026-09-04 改過）
  天花板 ~1.2x 或重疊很高 => 白跑，收線
⚠ 這條若成立，論文敘事要誠實處理：它是 **Taming 3DGS 的符號方向**（+c），
  與我方「除以 c」的定價方向相反。我方的貢獻仍在「c 綁預算」那一半，
  但「c 該乘還是該除」必須以實測為準，不能為了敘事挑符號。""")


if __name__ == "__main__":
    main()
