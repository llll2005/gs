#!/usr/bin/env python
"""週期 trim 只用 1/n 的相機，剪掉的還是同一批粒子嗎？—— 在**訓練後**的幾何上量。

## 為什麼要做

`tools/` 的逐段計時（§13.12）量到：週期 trim = 一個 7.3 h 跑次裡的 **49 分鐘（11.2%）**，
它每 500 步把**全部 284 台相機**重新渲染一遍算逐顆貢獻。
`TRIM_SUBSAMPLE=n` 這個旋鈕早就實作好（`sep_depth_trim_2dgs_renderer.py:after_training_step`），
判準也寫好了：`底部 prune_ratio 的遮罩重疊 > 95%` => 可取樣。

## 為什麼既有的探針不夠

`TRIM_SUBSAMPLE_PROBE` 住在 `before_training_step`，而那裡 **`if step != 1: return`**
=> 它只量過 **step 1**，也就是 depth-init 那層**還沒訓練過的均勻殼**。
程式碼註解自己寫著：「⚠ 但只在 step=1 量過一個時點，而一次跑次有約 120 次 trim
⇒ 誤差可能累積 ⇒ **必須端到端驗分數才可採用**」。

而風險是**結構性**的，不是隨機的：`contribution_accumulator` 預設 `reduce="max"`
（跨視角取**最大**，見 `internal/utils/topk_contribution.py` 的實測理由）
=> **少看幾台相機只可能讓最大值變小**，所以「最佳視角剛好被抽掉」的粒子會往底部掉、被誤剪。
step 1 的殼粒子尺度全同、視角可互換；訓練後粒子會對視角特化 => 那個時點測不到這件事。

⇒ 本工具直接在**訓練後的 ckpt** 上量，而且不用重跑訓練。

## 量什麼

```
基準  stride=1（全部 284 台）的 contribution，取底部 prune_ratio 當「真的會被剪的集合」
對照  stride=2/4/8 的同一個集合
主判準  底部遮罩重疊率（trim 只用到 `contribution <= quantile(prune_ratio)`）
副判準  Spearman（全排序；⚠ 沒有任何程式碼讀它，只當背景）
後果    誤剪 = 基準會留、取樣會剪的粒子數（這些是**真的會被刪掉**的）
```
判準（沿用程式碼裡既有的）：**重疊 > 95%** => 可取樣。

用法: python tools/trim_subsample_probe.py --run speed3_b12 --steps 14999 29999
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from internal.utils.topk_contribution import contribution_accumulator  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def contribution_at(renderer, model, cams, dev, stride, K):
    push, gather = contribution_accumulator(K)
    t0 = time.perf_counter()
    used = 0
    with torch.no_grad():
        for i in range(0, len(cams), stride):
            cam = cams[i].to_device(dev)
            mean, covered = renderer(cam, model, bg_color=torch.zeros(3, device=dev),
                                     record_transmittance=True, record_coverage=True)
            push(mean)
            used += 1
            del mean, covered
    torch.cuda.synchronize()
    return gather(), used, (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="speed3_b12")
    ap.add_argument("--steps", type=int, nargs="+", default=[14999, 29999])
    ap.add_argument("--strides", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--prune-ratio", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=5)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    from internal.utils.gaussian_model_loader import GaussianModelLoader

    print(f"\n判準（沿用程式碼裡既有的）：**底部 {100*a.prune_ratio:.0f}% 遮罩重疊 > 95%** => 可取樣")
    print("⚠ contribution 是跨視角取 max => 抽樣只會讓值變小 => 風險是**單向**的（誤剪，不會誤留）\n")

    for step in a.steps:
        ck = sorted(glob.glob(f"outputs/{a.run}/**/*step={step}.ckpt", recursive=True))
        if not ck:
            print(f"step={step}: 找不到 ckpt"); continue
        model, renderer, _ = GaussianModelLoader.\
            initialize_model_and_renderer_from_checkpoint_file(ck[0], device=dev,
                                                               eval_mode=True, pre_activate=False)
        c = torch.load(ck[0], map_location="cpu")
        dmh = c["datamodule_hyper_parameters"]
        dp = dmh["parser"].instantiate(path=dmh["path"],
                                       output_path=os.path.dirname(os.path.dirname(ck[0])),
                                       global_rank=0)
        cams = dp.get_outputs().train_set.cameras
        del c
        N = model.n_gaussians

        base, used0, t0 = contribution_at(renderer, model, cams, dev, 1, a.K)
        k = max(1, int(N * a.prune_ratio))
        base_mask = torch.zeros(N, dtype=torch.bool, device=base.device)
        base_mask[base.topk(k, largest=False).indices] = True
        bnp = base.detach().cpu().numpy()

        print(f"=== {a.run} @ step {step}   N={N:,}   相機 {len(cams)} 台 ===")
        print(f"  基準 stride=1：{used0} 台，{t0:.1f} s，底部 {100*a.prune_ratio:.0f}% = {k:,} 顆")
        print(f"  {'stride':>7} {'相機':>6} {'秒':>7} {'省時':>7} "
              f"{'底部遮罩重疊':>13} {'Spearman':>9} {'誤剪顆數':>10} {'判定':>6}")
        for st in a.strides:
            sub, used, t = contribution_at(renderer, model, cams, dev, st, a.K)
            m = torch.zeros(N, dtype=torch.bool, device=sub.device)
            m[sub.topk(k, largest=False).indices] = True
            ov = float((base_mask & m).sum()) / k
            # 誤剪＝基準會留（不在底部）但取樣會剪（在底部）=> 這些粒子真的會被刪掉
            wrong = int((m & ~base_mask).sum())
            rho = spearman(bnp, sub.detach().cpu().numpy())
            ok = "✅ 過" if ov > 0.95 else "❌ 不過"
            print(f"  {st:>7} {used:>6} {t:>7.1f} {100*(1-t/t0):>6.1f}% "
                  f"{100*ov:>12.2f}% {rho:>9.4f} {wrong:>10,} {ok:>6}")
            del sub, m
        del model, renderer, base, base_mask
        torch.cuda.empty_cache()
        print()

    print("""判讀：
  重疊 > 95%  => 取樣不改變「誰被剪」=> 可採用，但**仍須端到端驗分數**：
                一次跑次約 58 次 trim，單次的小誤差會累積（程式碼註解已標明這一點）
  重疊 < 95%  => 直接否決，不必再花 9 小時做端到端
⚠ 這裡量的是**單一時點的靜態一致性**，不是訓練動態。誤剪的粒子會被真的刪掉，
  而 MCMC 會用 relocate 補回別的地方 => 端到端的影響可能比重疊率暗示的大或小。""")


if __name__ == "__main__":
    main()
