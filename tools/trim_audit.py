#!/usr/bin/env python
"""超剪稽核：既有跑次的每一次 trim 實際剪掉幾 %？

背景（研究總覽 §11.20）：`prune_mask = (contribution <= quantile(contribution, prune_ratio))`
用 `<=`，所以實際剪除率 = max(prune_ratio, 零貢獻比例)，是**狀態相依**的，
不是破平衡公式假設的常數 10%。實測 `probe_ctrl`（= noprior_b12 的前 2,000 步）：
step 1000 剪 28.58%（門檻恰好 0、剪掉數恰好等於零貢獻數），step 1500/2000 才是 10.00%。

本工具不需要重跑：`train/gaussians_count` 每 100 步記一次，而 trim 只在
`contribution_prune_interval` 的倍數發生，且記錄的是**剪枝後**的值（已對照診斷確認）。
所以每個 trim 邊界的 N(s)/N(s-100) 就是「這 100 步裡 densify 加的 − trim 剪的」。

    N(s) = N(s-100) x 1.05^(此窗內 densify 事件數) x (1 - 實際剪除率)

反解實際剪除率。⚠ densify 事件數用 interval 推算，若 interval 不整除 100 會有相位誤差，
所以同時印出非 trim 邊界的窗當作對照（那些窗的推估剪除率應該 ≈ 0）。
"""
import argparse
import glob
import re

import numpy as np
import yaml
from tensorboard.backend.event_processing import event_accumulator as ea


def load(run):
    evs = sorted(glob.glob(f"outputs/{run}/**/events.out.tfevents.*", recursive=True))
    cfgs = sorted(glob.glob(f"outputs/{run}/**/config.yaml", recursive=True))
    if not evs or not cfgs:
        return None
    a = ea.EventAccumulator(evs[-1], size_guidance={ea.SCALARS: 0})
    a.Reload()
    if "train/gaussians_count" not in a.Tags()["scalars"]:
        return None
    n = {s.step: s.value for s in a.Scalars("train/gaussians_count")}
    c = yaml.safe_load(open(cfgs[-1]))
    r = c["model"]["renderer"]["init_args"]
    d = c["model"]["density"]["init_args"]
    return dict(n=n, ci=r.get("contribution_prune_interval", 500),
                cfrom=r.get("contribution_prune_from_iter", 1000),
                pr=r.get("prune_ratio", 0.1), notrim=r.get("diable_trimming", False),
                di=d.get("densification_interval", 150),
                dfrom=d.get("densify_from_iter", 500),
                duntil=d.get("densify_until_iter", 42000),
                cap=d.get("cap_max", 0), add=d.get("add_ratio", 1.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=None)
    args = ap.parse_args()
    runs = args.runs or [r.split("/")[1] for r in sorted(glob.glob("outputs/*/"))]

    print(f"{'run':>18} {'trim事件':>8} {'超剪次數':>9} {'超剪佔比':>9} "
          f"{'最大剪除率':>10} {'中位剪除率':>10} {'名目':>6}")
    for run in runs:
        d = load(run)
        if d is None or d["notrim"]:
            continue
        n, ci, di = d["n"], d["ci"], d["di"]
        steps = sorted(n)
        if len(steps) < 10:
            continue
        gap = steps[1] - steps[0]
        rates = []
        for s in steps:
            # 程式碼是 `if step < from_iter: return`，所以 step == from_iter 會觸發，
            # 而那**正是唯一會超剪的那一次**（§11.20），不可用 `<=` 排除掉。
            if s < d["cfrom"] or s % ci != 0 or s > d["duntil"]:
                continue
            prev = s - gap
            if prev not in n or n[prev] <= 0:
                continue
            # 此窗內的 densify 事件數
            ev = sum(1 for k in range(prev + 1, s + 1)
                     if k % di == 0 and d["dfrom"] < k < d["duntil"])
            grown = n[prev] * (d["add"] ** ev)
            if d["cap"] and grown > d["cap"]:
                grown = d["cap"]           # 撞 cap 後 densify 不再加
            cut = 1.0 - n[s] / max(grown, 1e-9)
            if -0.05 < cut < 0.95:
                rates.append((s, cut))
        if not rates:
            continue
        cuts = np.array([c for _, c in rates])
        over = cuts > d["pr"] * 1.5
        print(f"{run:>18} {len(rates):>8} {int(over.sum()):>9} "
              f"{100*over.mean():>8.0f}% {100*cuts.max():>9.1f}% "
              f"{100*np.median(cuts):>9.1f}% {100*d['pr']:>5.0f}%")
        worst = sorted(rates, key=lambda x: -x[1])[:3]
        print(f"{'':>18} 最嚴重: " + "  ".join(f"step {s}: {100*c:.1f}%" for s, c in worst))

    print("""
判讀：中位剪除率應該 ≈ 名目。明顯高於名目的事件 = 該次的零貢獻比例超過 prune_ratio，
      `<=` 把零貢獻粒子全部剪光。⚠ 這是**推估**（densify 事件數由 interval 推算，
      有相位誤差），量級可信、單次數值不可引用；要精確值請看訓練 log 的
      `Trimming done. step=... 剪=...` 診斷輸出。""")


if __name__ == "__main__":
    main()
