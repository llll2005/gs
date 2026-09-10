#!/usr/bin/env python
"""代換 3：給「即將被 relocate」的粒子加噪音，是不是白算的？（純 CPU）

## 結構觀察（2026-09-07）

```
MCMC 噪音的閘門  gate(o) = sigmoid(100 * (0.005 - o))    中心 = **0.005**
relocation 死線  min_opacity                                    = **0.005**
```
**兩者是同一個門檻** ⇒ 拿到最多噪音的粒子，正是下一個 densify 事件會被 relocate、
**位置被整個覆寫**的那批。若如此，那部分噪音是白算的。

## ⚠ 為什麼不能用「外推 Adam 看它逃不逃得掉」來測

噪音的**目的**就是改變梯度。拿當前的 opacity 動量外推 150 步，等於假設噪音沒有效果
—— 那是循環論證。所以本工具改測三件**不依賴該假設**的量：

```
A 噪音預算怎麼分配   多少比例的噪音計算量落在 o < 死線 的粒子上
B 位移的相對量級     噪音在一個 densify 週期內能走多遠 vs relocation 把它搬多遠
                     若 噪音位移 << 搬移距離，則「它在原地走了多少」對結局無關緊要
C 逃脫的必要條件     要在 150 步內把 o 從死線拉上來，需要多大的 opacity 梯度；
                     對照 opacity 的 Adam 實際步長，看那是不是一個現實的要求
```
判準：
```
A 高（>1/3）且 B << 1  => 噪音的大部分投在「位置即將被覆寫」的粒子上 => 可跳過
B ~ 1 或更大           => 噪音位移與搬移同量級 => 它可能真的改變了去留 => 不可跳過
```

用法: python tools/noise_vs_relocate.py --run agd2_b12 --step 29999
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
    ap.add_argument("--run", default="agd2_b12")
    ap.add_argument("--step", type=int, default=29999)
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--interval", type=int, default=150, help="densify 間隔（步）")
    ap.add_argument("--noise-lr", type=float, default=500000.0)
    args = ap.parse_args()

    ck = sorted(glob.glob(f"outputs/{args.run}/**/*step={args.step}.ckpt", recursive=True))
    if not ck:
        raise SystemExit(f"找不到 {args.run} 的 step={args.step} ckpt")
    c = torch.load(ck[0], map_location="cpu")
    sd = c["state_dict"]
    mu = sd["gaussian_model.gaussians.means"].double()
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"]).squeeze(-1).double()
    s = torch.exp(sd["gaussian_model.gaussians.scales"]).double().max(dim=1).values
    N = o.numel()

    # means 的 lr（噪音直接乘它）
    g0 = c["optimizer_states"][0]["param_groups"][0]
    assert g0.get("name") == "means"
    lr_m = float(g0["lr"])
    gate = 1.0 / (1.0 + torch.exp(-100.0 * (args.min_opacity - o)))
    dead = o < args.min_opacity

    print(f"{args.run} @ {args.step}   N={N:,}   means lr={lr_m:.4e}   "
          f"densify 間隔={args.interval} 步\n")

    # ── A 噪音預算的分配 ──
    tot = float(gate.sum())
    print("A 噪音預算怎麼分配（權重 = gate，因為位移 ∝ gate）")
    for lab, m in (("o < 死線 0.005（將被 relocate）", dead),
                   ("死線 ~ 0.05", (o >= args.min_opacity) & (o < 0.05)),
                   ("o >= 0.05", o >= 0.05)):
        print(f"    {lab:>34} 顆數 {int(m.sum()):>9,} ({100*float(m.float().mean()):>5.2f}%)"
              f"   佔噪音預算 **{100*float(gate[m].sum())/tot:>5.2f}%**")

    # ── B 位移量級 ──
    # 每步 |noise| ~ scale^2 * noise_lr * lr * gate（cov 的最大特徵值 = max(scale)^2）
    step_disp = (s ** 2) * args.noise_lr * lr_m * gate
    walk = step_disp * np.sqrt(args.interval)          # 隨機遊走，一個 densify 週期
    # relocation 把死粒子搬到「依 probs 抽到的宿主」=> 搬移距離的尺度 = 族群的空間分布
    sub = mu[torch.randperm(N)[:20000]]
    pdist = torch.cdist(sub[:2000], sub).median()      # 族群內的典型距離
    nn = torch.cdist(sub[:2000], sub).kthvalue(2, dim=1).values.median()   # 最近鄰間距
    print(f"\nB 位移的相對量級（一個 densify 週期 = {args.interval} 步）")
    print(f"    死粒子的噪音遊走 中位     = {float(walk[dead].median()):.4e}")
    print(f"    最近鄰間距 中位           = {float(nn):.4e}   => 比值 "
          f"**{float(walk[dead].median())/float(nn):.4f}**")
    print(f"    族群內典型距離 中位       = {float(pdist):.4e}   => 比值 "
          f"**{float(walk[dead].median())/float(pdist):.6f}**")
    print(f"    （relocation 把它搬到抽中的宿主 ⇒ 搬移距離是後者的量級）")

    # ── C 逃脫的必要條件 ──
    op = c["optimizer_states"][1]
    gi = [i for i, g in enumerate(op["param_groups"]) if g.get("name") == "opacities"][0]
    lr_o = float(op["param_groups"][gi]["lr"])
    st = op["state"][gi]
    t = float(st["step"]); b1, b2 = 0.9, 0.999
    upd = (lr_o / (1 - b1 ** t)) * st["exp_avg"].double() \
        / (torch.sqrt(st["exp_avg_sq"].double() / (1 - b2 ** t)) + 1e-15)
    upd = upd.squeeze(-1)
    raw = sd["gaussian_model.gaussians.opacities"].squeeze(-1).double()
    raw_need = float(np.log(args.min_opacity / (1 - args.min_opacity)))   # logit(死線)
    gap = raw_need - raw[dead]                                            # 要往上走多少（raw 空間）
    reach = upd[dead].abs() * args.interval
    print(f"\nC 逃脫的必要條件（opacity lr={lr_o:g}，raw=logit 空間）")
    print(f"    距死線的缺口 中位         = {float(gap.median()):.4e}")
    print(f"    {args.interval} 步能走的距離 中位 = {float(reach.median()):.4e}"
          f"   => 覆蓋率 **{float(reach.median())/max(float(gap.median()),1e-30):.3f}x**")
    up = (upd[dead] > 0).double().mean()
    print(f"    死粒子中 opacity 動量**向上**的比例 = {100*float(up):.1f}%")
    print(f"    {args.interval} 步內夠得到死線的比例       = "
          f"**{100*float(((upd[dead] > 0) & (reach > gap)).double().mean()):.2f}%**")

    print("""
判讀：
  A 高 且 B << 1  => 噪音大部分投在「位置即將被覆寫」的粒子 => **可跳過**
  B ~ 1 或更大    => 噪音位移與搬移同量級 => 它可能真的改變去留 => **不可跳過**
  C 只是輔證：逃脫比例高 => 死線附近確實是「還有救」的族群，跳過噪音要更謹慎
⚠ C 用當前 Adam 動量外推，**假設噪音不改變梯度** —— 那正是噪音想做的事
  ⇒ C 只能當**下界**（真實逃脫率只會更高），不可拿來單獨下結論。""")


if __name__ == "__main__":
    main()
