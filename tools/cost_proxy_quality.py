#!/usr/bin/env python
"""代換 1 的前置：我們一直在用的成本**代理**，跟光柵器算出的**精確** c_i 差多少？

## 為什麼重要（不只是省時）

`cost_aware_densify` 用的是 `_max_radii2D ** 2`：
  「150 步視窗內、**單相機**光柵器半徑的最大值」再平方 —— 足跡面積的**代理**。

而 trim pass（每 500 步、**全部 284 台**相機）已經在算 `num_covered_pixels`，
其原始碼註解自述：*"What it IS, is a per-primitive render COST"* —— 那才是命題定義的 `c_i`。
**現行配方下它被算出來就丟掉**（唯一消費者 `blur_split_budget` = 0）。

⇒ 代換的收益有兩層：
```
① 省成本   代理是免費的，精確量也是免費的（已在算）=> 這一層是平手
② 升精度   從近似換成精確，且是**論文命題定義的那個量**
```
⚠ **而且這個量測有獨立的價值**：若代理與精確 c 的相關度不高，
   那麼**先前所有用代理得到的成本結論都要打折**（§11.102 的 82% 目標重疊、
   §11.28 的 rho(v,c)、退化定理的前提檢驗，全都建立在代理上）。

## 量什麼

```
Spearman(代理, 精確)         排序一致性 —— 取樣權重只看排序，這是最相關的統計
top10% 重疊                  換訊號會不會換掉增生位置
ceiling(top5%)/mean          兩者的動態範圍（決定加權的強度）
```
⚠ 兩者的**歸約方式**也不同：`_max_radii2D` 是**跨視角取 max**，
  而 `num_covered_pixels` 累積時可取 sum（總渲染成本）或 max。兩種都報。

用法: python tools/cost_proxy_quality.py --run agd2_b12 --step 60000
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def ceiling(x, q=0.05):
    k = max(1, int(len(x) * q))
    return float(np.sort(x)[-k:].mean() / max(x.mean(), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="agd2_b12")
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--max-cam", type=int, default=284)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU（要跑光柵器的 record_coverage 路徑）")
    dev = torch.device("cuda")
    ck = sorted(glob.glob(f"outputs/{args.run}/**/*step={args.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {args.run} 的 step={args.step} ckpt")

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck[0], device=dev, eval_mode=False, pre_activate=False)
    N = model.n_gaussians
    ckpt = torch.load(ck[0], map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck[0])),
                                   global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    n = min(args.max_cam, len(cams))
    print(f"{args.run} @ {args.step}   N={N:,}   相機 {n}/{len(cams)}")

    bg = torch.zeros((3,), device=dev)
    cov_sum = torch.zeros(N, device=dev)
    cov_max = torch.zeros(N, device=dev)
    rad_max = torch.zeros(N, device=dev)
    with torch.no_grad():
        for i in range(n):
            cam = cams[i].to_device(dev)
            out = renderer(cam, model, bg_color=bg,
                           record_transmittance=True, record_coverage=True)
            # renderer 在 record_coverage 時回傳 (per-pixel mean, covered)
            mean_t, covered = out if isinstance(out, tuple) else (out, None)
            if covered is None:
                raise SystemExit("拿不到 covered —— record_coverage 路徑沒生效")
            c = covered.float()
            cov_sum += c
            cov_max = torch.maximum(cov_max, c)
            # 代理：同一批相機的光柵器半徑取 max（＝ `_max_radii2D` 的定義）
            o2 = renderer(cam, model, bg_color=bg)
            r = o2.get("radii")
            if r is not None:
                rad_max = torch.maximum(rad_max, r.float())

    cs = cov_sum.cpu().numpy()
    cm = cov_max.cpu().numpy()
    pr = (rad_max.float() ** 2).cpu().numpy()          # 代理 = max_radii^2
    seen = cs > 0
    print(f"  被任一相機看到的粒子 {seen.sum():,} / {N:,}\n")

    print(f"{'配對':>34} {'Spearman':>10} {'top10% 重疊':>13}")
    k = int(0.1 * seen.sum())
    def top(x):
        v = x[seen]
        return set(np.argpartition(-v, k)[:k].tolist())
    tp, ts, tm = top(pr), top(cs), top(cm)
    print(f"{'代理 radii² vs 精確 c（sum 視角）':>34} {spearman(pr[seen], cs[seen]):>10.4f} "
          f"{100*len(tp & ts)/k:>12.2f}%")
    print(f"{'代理 radii² vs 精確 c（max 視角）':>34} {spearman(pr[seen], cm[seen]):>10.4f} "
          f"{100*len(tp & tm)/k:>12.2f}%")
    print(f"{'精確 c：sum vs max':>34} {spearman(cs[seen], cm[seen]):>10.4f} "
          f"{100*len(ts & tm)/k:>12.2f}%")

    print(f"\n{'訊號':>34} {'ceiling(top5%)/mean':>21} {'中位':>12}")
    for lab, v in (("代理 radii²", pr[seen]), ("精確 c（sum 視角）", cs[seen]),
                   ("精確 c（max 視角）", cm[seen])):
        print(f"{lab:>34} {ceiling(v):>20.2f}x {np.median(v):>12.4e}")

    # ── 代換 2 的前置：加法式該用多大的 w ──
    # absgrad（已證有效）：`1 + w*ĝ`，w=2、ceiling(ĝ)=12.4 => 動態範圍 1 ~ **25.8x**
    # cost_aware（兩向皆輸）：冪次式，實測權重範圍 0.083~25.0 => 跨度 **300x**
    # ⇒ 把加法式的 w 校到與 absgrad 相同的動態範圍，避免重蹈「加權過猛」。
    print(f"\n  ★★ 代換 2：加法式 `1 + w*ŝ` 的 w 該設多少")
    for lab, sig in (("1/c（偏好便宜，我方方向）", 1.0 / np.maximum(cs[seen], 1.0)),
                     ("c（偏好貴，Taming 方向）", cs[seen])):
        sh = sig / sig.mean()
        ceil = ceiling(sh)
        w_match = (25.8 - 1.0) / max(ceil, 1e-9)     # 令 1+w*ceiling = 25.8（absgrad 的範圍）
        print(f"    {lab:>26}  ceiling(top5%)/mean = {ceil:>7.2f}x"
              f"   => 匹配 absgrad 的 w = **{w_match:.3f}**")
    print(f"    （對照 absgrad：ceiling(|g|)=12.4x、w=2.0 => 1~25.8x；"
          f"冪次式實測 0.083~25.0 = 跨度 300x）")

    print("""
判讀：
  Spearman > 0.95 且 重疊 > 90%  => 代理夠好 => 代換只是「精度升級」，
                                    且**先前用代理得到的成本結論仍成立**
  Spearman < 0.8 或 重疊 < 70%   => 代理不夠好 => **代換有實質意義**，
                                    且 §11.102 / §11.28 等建立在代理上的結論要重新檢視
⚠ 代理是「150 步視窗、單相機」累積的，本工具用同一批相機的 max 來近似它
  ⇒ 這是**對代理有利**的比法（同時刻、同相機集合）；真實代理只會更差。""")


if __name__ == "__main__":
    main()
