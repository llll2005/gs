#!/usr/bin/env python
"""背包有沒有空間？—— 直接在真實模型上模擬一次，不用訓練（單次前向掃相機）。

## 要答的

論文命題的約束端是 `max Q s.t. Σ c_i <= B`。它有沒有空間，取決於**價值 v 與成本 c 是否解耦**：
```
v_i = kappa * c_i（同漲同跌） => v/c 是常數 => 選擇沒有自由度 => 背包退化成「挑最便宜的」
v 與 c 解耦                   => 按 v/c 排序會選出與按 v 排序**不同**的集合 => 有空間
```
⚠ 目前有**兩個互相矛盾的舊量測，而且都在錯位資料上做的**：
```
mcmc_2dgs_density_controller.py:529   rho(v,c)=+0.127 => 幾乎不相關 => **有自由度**
記憶 degeneracy_and_blur_split         v_i ~ c_i（三個實測）=> **沒有自由度**
```
⇒ 兩個都不算數，要在修正後的資料上重量。

## 量什麼（前三個是診斷，第四個是判決）

```
1 rho(v,c)              Pearson 與 Spearman
2 v/c 的離散度          ceiling(top5%)/mean；≈1 表示沒有訊號
3 top10% 重疊           「按 v 選」與「按 v/c 選」選到的是不是同一批
4 ★ 背包增益            同樣的成本預算下，貪婪 by v/c 比貪婪 by v 多拿到多少總價值
                        —— 這是命題能拿到的**上界**（真實訓練只會更少）
```
判準：
```
第 4 項 < 5%   => 就算實作完美也賺不到 => 約束端可以直接收掉，不必跑 60k
第 4 項 顯著   => 值得標定 B 再跑介入
```
⚠ v 取自 trim pass 的 per-primitive contribution（`reduce="max"`，見
  `internal/utils/topk_contribution.py`：實測 max 比 topk_mean 好，因為邊緣只被少數相機看到）。
⚠ c 取自同一個 pass 的 `num_covered_pixels` —— **兩者在同一次前向裡算出來**，沒有對齊問題。

用法: python tools/rho_value_cost.py --ckpt outputs/gate15000/blocks/block_12/checkpoints/*step=15000.ckpt --max-cam 200
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-cam", type=int, default=200)
    ap.add_argument("--budget-frac", type=float, nargs="+",
                    default=[0.1, 0.25, 0.5, 0.75],
                    help="成本預算取總成本的幾成（背包模擬用）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    cks = sorted(glob.glob(args.ckpt))
    if len(cks) != 1:
        raise SystemExit(f"--ckpt 要展開成唯一一個檔，目前 {len(cks)} 個")
    ck = cks[0]

    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from internal.utils.topk_contribution import contribution_accumulator
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ck, device=dev, eval_mode=False, pre_activate=False)
    N = model.n_gaussians
    ckpt = torch.load(ck, map_location="cpu")
    dmh = ckpt["datamodule_hyper_parameters"]
    dp = dmh["parser"].instantiate(path=dmh["path"],
                                   output_path=os.path.dirname(os.path.dirname(ck)),
                                   global_rank=0)
    cams = dp.get_outputs().train_set.cameras
    n = min(args.max_cam, len(cams))
    print(f"來源 {ck}\n  N = {N:,}   相機 {n}/{len(cams)}")

    bg = torch.zeros((3,), device=dev)
    K = getattr(renderer, "K", 5)
    push, gather = contribution_accumulator(K, reduce="max")
    cov_max = torch.zeros(N, device=dev)
    with torch.no_grad():
        for i in range(n):
            cam = cams[i].to_device(dev)
            out = renderer(cam, model, bg_color=bg,
                           record_transmittance=True, record_coverage=True)
            if isinstance(out, tuple):
                out, covered = out
            else:
                covered = out.get("covered") if isinstance(out, dict) else None
            push(out)
            if covered is not None:
                cov_max = torch.maximum(cov_max, covered.float())
    v = gather()
    v = (v[0] if isinstance(v, tuple) else v).float().cpu().numpy()
    c = cov_max.cpu().numpy()

    seen = (c > 0) & np.isfinite(v) & np.isfinite(c)
    v, c = v[seen], c[seen]
    print(f"  被看到且訊號有效的粒子 {seen.sum():,} / {N:,}\n")
    if len(v) < 1000:
        raise SystemExit("⛔ 有效粒子太少，量測不可信")

    print("1 相關性")
    print(f"    Pearson  rho(v,c) = {float(np.corrcoef(v, c)[0,1]):+.4f}")
    print(f"    Spearman rho(v,c) = {spearman(v, c):+.4f}")
    r = v / np.maximum(c, 1.0)
    print("\n2 訊號的離散度（ceiling(top5%)/mean；≈1 = 沒有訊號可用）")
    for lab, x in (("v（價值）", v), ("c（成本）", c), ("v/c（每單位成本的價值）", r)):
        print(f"    {lab:>26} {ceiling(x):>7.2f}x")

    k = max(1, int(0.1 * len(v)))
    top_v = set(np.argpartition(-v, k)[:k].tolist())
    top_r = set(np.argpartition(-r, k)[:k].tolist())
    print(f"\n3 top10% 重疊：按 v 選 vs 按 v/c 選 = **{100*len(top_v & top_r)/k:.2f}%**"
          f"（隨機是 10%；100% 表示兩種排序選到同一批）")

    print("\n4 ★ 背包增益：同樣成本預算下，貪婪 by v/c 比貪婪 by v 多拿到多少總價值")
    tot_c, tot_v = c.sum(), v.sum()
    ord_v = np.argsort(-v)
    ord_r = np.argsort(-r)
    print(f"{'預算(佔總成本)':>16} {'by v 的價值':>14} {'by v/c 的價值':>14} {'增益':>9}")
    for f in args.budget_frac:
        B = tot_c * f
        out = []
        for od in (ord_v, ord_r):
            cs = np.cumsum(c[od])
            m = np.searchsorted(cs, B)
            out.append(v[od][:m].sum())
        gain = (out[1] - out[0]) / max(out[0], 1e-30)
        print(f"{f*100:>14.0f}% {out[0]/tot_v:>13.4f} {out[1]/tot_v:>13.4f} {100*gain:>+8.2f}%")

    print("""
判讀：
  第 4 項 < 5%  => 就算實作完美也賺不到 => **約束端可以收掉**，不必標定 B、不必跑 60k
  第 4 項顯著   => 值得往下走（標定 B -> 介入）
⚠ 這是**上界**：這裡是對已訓練好的模型做離線選擇，知道每顆的真實 v；
  訓練期只能用代理訊號，實際能拿到的一定更少。""")


if __name__ == "__main__":
    main()
