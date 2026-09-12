#!/usr/bin/env python
"""被拉長的退化高斯，在我們的模型裡到底有多少、佔多少成本？（純 CPU，先算天花板）

## 為什麼問這個

CityGaussian V2 有 **Elongation Filter**：訓練中剔除被拉成極長片的高斯，避免單顆覆蓋
成千上萬個像素，從源頭降低 backward 的 atomicAdd 衝突與 overdraw。
而我們**把整個 density controller 換成 MCMC 了**（`MCMC2DGSDensityController`），
elongation filter 是 `CityGSV2DensityController` 的一部分 ⇒ **我們沒有它**。
（記憶 `paper_audit` 記著同一件事：「怪物是 3DGS 命名的問題，而 **MCMC 把解藥刪掉了**」。）

我們的替代品是兩個**間接**機制：
```
screen_size_prune_px 300   看螢幕半徑（實測觸發 193 次/跑次、回收 cap 的 7.29%）
scale_reg 0.007            壓**平均**尺度（關掉 -18sd）
```
兩者都不直接看「長寬比」。而訊號其實**已經實作**在
`internal/metrics/scale_regularization_metrics.py:58`（`scale_ratios = max/mid`），
只是那個 mixin 掛在 `VanillaMetrics` 上，我們用 `MCMCCityGSV2Metrics` ⇒ **接不到**。

## 先算天花板，不要先寫機制

記憶 `feedback_experiment_methodology`：「借方法先查對不對症」；
`feedback_measure_before_explaining`：讀碼猜的三個減計算候選**全沒進前 40 名**。
⇒ 若過長粒子只佔 0.01% 又很便宜，加 filter 沒有東西可贏。

## 量什麼（全部只讀 ckpt）

```
長寬比分布            2DGS 只有兩個 scale => elong = s_max / s_min
各門檻的佔比           >2 / >5 / >10 / >20 / >50
它們佔多少「面積」      sum(s_max*s_min) 的比例 —— 螢幕足跡 ∝ 世界面積（r=0.977，§11.x）
它們佔多少不透明度質量   sum(opacity) 的比例 —— 決定它們真的在混色裡有多少份量
最極端的幾顆長什麼樣    尺度、opacity
```
判準：
```
>10 的佔比 < 0.1% 且面積佔比 < 1%   => 沒東西可贏，這條線關掉
面積或質量佔比 > 5%                => 值得做，而且先做 (b) 融入 MCMC 死亡判準
```
用法: python tools/elongation_ceiling.py --runs speed3_b12 sfminit2_b12
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["speed3_b12", "sfminit2_b12"])
    ap.add_argument("--step", type=int, default=60000)
    ap.add_argument("--ths", type=float, nargs="+", default=[2, 5, 10, 20, 50])
    a = ap.parse_args()

    print("\n判準：>10 的佔比 < 0.1% 且**面積**佔比 < 1% => 沒東西可贏，關掉這條線")
    print("      面積或不透明度質量佔比 > 5% => 值得做\n")
    for run in a.runs:
        ck = sorted(glob.glob(f"outputs/{run}/**/*step={a.step}.ckpt", recursive=True))
        if not ck:
            print(f"{run}: 找不到 ckpt"); continue
        sd = torch.load(ck[0], map_location="cpu")["state_dict"]
        s = torch.exp(sd["gaussian_model.gaussians.scales"]).double()   # [N, 2]（2DGS）
        o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"]).squeeze(-1).double()
        if s.shape[1] != 2:
            print(f"{run}: scales 有 {s.shape[1]} 軸，不是 2DGS，跳過"); continue
        smax = s.max(1).values
        smin = s.min(1).values
        elong = smax / smin.clamp_min(1e-12)
        area = smax * smin                       # 世界面積 ∝ 螢幕足跡（空拍深度範圍窄）
        N = elong.numel()
        A = float(area.sum()); O = float(o.sum())

        print(f"=== {run} @ {a.step}   N={N:,} ===")
        q = [50, 90, 99, 99.9, 100]
        print("  長寬比分位：" + "  ".join(
            f"p{x}={float(np.percentile(elong.numpy(), x)):.2f}" for x in q))
        print(f"  {'門檻':>8} {'顆數':>10} {'顆數佔比':>9} {'面積佔比':>9} "
              f"{'不透明度質量佔比':>13} {'平均 s_max':>11}")
        for t in a.ths:
            m = elong > t
            n = int(m.sum())
            if n == 0:
                print(f"  {'>'+str(int(t)):>8} {0:>10} {'0.00%':>9} {'0.00%':>9} {'0.00%':>13}")
                continue
            print(f"  {'>'+str(int(t)):>8} {n:>10,} {100*n/N:>8.3f}% "
                  f"{100*float(area[m].sum())/A:>8.3f}% "
                  f"{100*float(o[m].sum())/O:>12.3f}% {float(smax[m].mean()):>11.5f}")
        # 最極端的五顆
        idx = torch.topk(elong, 5).indices
        print("  最極端 5 顆：" + "  ".join(
            f"[{float(elong[i]):.0f}x o={float(o[i]):.2f} s={float(smax[i]):.4f}/{float(smin[i]):.5f}]"
            for i in idx))
        print()
    print("""⚠ 「面積」用世界空間的 s_max*s_min 當螢幕足跡的代理 —— 空拍相機高度接近、
   深度範圍窄，實測 corr(log 世界尺度, log 螢幕足跡) = **r=0.977**，所以這個代理是站得住的。
⚠ 這裡不量「拉長的粒子是不是壞的」——只量「有沒有份量」。有份量才值得花 GPU 驗。""")


if __name__ == "__main__":
    main()
