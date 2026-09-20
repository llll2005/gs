#!/usr/bin/env python
"""標定 `cost_budget` 的單位：現在的操作點，Load 是多少？（單次前向，不用訓練）

## 為什麼需要標定

`cost_budget` 把生長條件從 `N <= cap_max` 換成 `max_view Σ_i (2r_i/16)^2 <= cost_budget`。
旗標的 docstring 明文警告：**不可沿用 `tools/cost_budget_probe.py` 的 23.1M**
——那支用解析投影，而生效的是**光柵器 radii**，同定義不同實作路徑。

原本的標定方式是跑一次 60k 訓練、開 `cost_budget_report`。但 Load 只跟**模型與相機**有關，
不需要訓練：拿已訓練好的 ckpt、用**同一個 renderer** 取 radii 算一次就好。
⇒ 15 小時的 lab 訓練換成本機幾分鐘。

## 為什麼要知道這個數

2026-09-13 `tools/rho_value_cost.py` 量到：背包的天花板是**預算鬆緊**的函數
（預算佔總成本 10% => +57%，75% => +1~2%，兩塊一致）。
而被否證的三個成本感知變體跑在 `cap_max 2.6M` ⇒ 落在**寬鬆端**。
⇒ 介入要排在**緊預算**那一側，而「現在落在哪一格」就是這支工具要回答的。

用法: python tools/cost_budget_calibrate.py --ckpt 'outputs/.../*step=15000.ckpt' --max-cam 150
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-cam", type=int, default=150)
    ap.add_argument("--tile", type=int, default=16, help="(2r/tile)^2 的 tile 邊長")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 GPU")
    dev = torch.device("cuda")
    cks = sorted(glob.glob(args.ckpt))
    if len(cks) != 1:
        raise SystemExit(f"--ckpt 要展開成唯一一個檔，目前 {len(cks)} 個")
    ck = cks[0]

    from internal.utils.gaussian_model_loader import GaussianModelLoader
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
    print(f"來源 {ck}\n  N = {N:,}   相機 {n}/{len(cams)}   tile = {args.tile}")

    bg = torch.zeros((3,), device=dev)
    loads = []
    with torch.no_grad():
        for i in range(n):
            out = renderer(cams[i].to_device(dev), model, bg_color=bg)
            r = out.get("radii")
            if r is None:
                raise SystemExit("⛔ renderer 沒回傳 radii，拿不到 Load")
            loads.append(float((((2.0 * r.float()) / args.tile) ** 2).sum()))
    loads = np.array(loads)
    L = loads.max()
    print(f"\n  Load（每視角 Σ (2r/{args.tile})^2）")
    print(f"    max   **{L:,.0f}**   <- 這就是 cost_budget 的單位與當前操作點")
    print(f"    p95   {np.percentile(loads,95):,.0f}")
    print(f"    中位   {np.median(loads):,.0f}")
    print(f"    min   {loads.min():,.0f}")
    # 2026-09-14 補（使用者問「有總成本嗎」）：中位數只代表典型視角；
    #   訓練每步隨機渲染一台相機 => 期望每步 binning 成本＝**平均**；全部相機加總＝走一遍所有視角的總成本。
    #   ⚠ 這仍是「終點模型」的成本，不是整趟訓練的積分（族群在訓練中會變）。
    print(f"    平均   {loads.mean():,.0f}   <- 期望每步成本（每步隨機一台相機）")
    print(f"    總和   {loads.sum():,.0f}   （{len(loads)} 台相機加總）")
    print(f"    B/N   {L/N:.2f}  （每顆平均攤到多少 Load）")
    print(f"\n  ★ 緊預算的候選設定（rho_value_cost.py 量到天花板隨預算收緊而跳）")
    for f in (0.75, 0.5, 0.25, 0.10):
        print(f"    {f*100:>5.0f}% B0  =>  --model.density.init_args.cost_budget {L*f:,.0f}")
    print("""
⚠ 保留：
  ① 這是**已訓練好的模型**的 Load；訓練期 Load 隨 N 成長，約束是在成長途中生效的
     => 設 B < B0 會讓族群停在更早的地方，那正是「緊預算」的意思
  ② `cap_max` 仍要保留當安全閥（VRAM = 逐顆儲存只看 N ＋ binning 只看成本）
  ③ 相機只取了一部分；max 是對這些視角取的，全部相機的 max 只會更大""")


if __name__ == "__main__":
    main()
