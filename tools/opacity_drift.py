#!/usr/bin/env python
"""收割期 opacity 分布的漂移（純 CPU，讀 ckpt）。

要回答的問題（§11.61/11.62）：收割期（`densify_until` 之後）到底發生什麼？

免費量測（`agd2_b12` 六個 ckpt）已經推翻了「L1 把大家壓死」的直覺說法：
```
step        N        判死%    o中位     o 0.005~0.05   o>0.5
29,999  2,600,000   3.77%   0.1083      22.26%       9.12%
41,999  2,340,000   9.81%   0.1371      13.79%      19.48%
60,000  2,340,000  15.03%   0.1539       8.99%      23.22%
```
⇒ 收割期在做**兩極化**：中間帶排乾、兩端都長。中位數**升**而非降。
⇒ 往「確信」的遷移（o>0.5 從 9%% 翻到 23%%）很可能就是收割期 +1 dB 的來源。

所以介入時**必須分開**這兩件事：
  「移除 L1 的固定下壓力」（想要）vs「凍結 opacity 學習」（會殺掉兩極化，不想要）

用法：`python tools/opacity_drift.py run[@step][:標籤] ...`
"""
import glob
import os
import sys

import torch

# ckpt 的 pickle 參照 `internal.*` 的類別，沒有這行會 ModuleNotFoundError
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BANDS = [(0.0, 0.005, "判死<0.005"), (0.005, 0.05, "中間帶"),
         (0.05, 0.5, "0.05~0.5"), (0.5, 1.01, "確信>0.5")]


def load(spec):
    label = None
    if ":" in spec:
        spec, label = spec.split(":", 1)
    step = None
    if "@" in spec:
        spec, w = spec.split("@", 1)
        step = int(w)
    cks = sorted(glob.glob(f"outputs/{spec}/**/*.ckpt", recursive=True),
                 key=lambda p: int(p.split("step=")[-1].split(".")[0]))
    if not cks:
        return None, None, None
    if step is not None:
        cks = [p for p in cks if int(p.split("step=")[-1].split(".")[0]) == step] or cks[-1:]
    got = int(cks[-1].split("step=")[-1].split(".")[0])
    sd = torch.load(cks[-1], map_location="cpu")["state_dict"]
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).squeeze(-1)
    return o, got, (label or spec)


def main():
    specs = sys.argv[1:]
    if not specs:
        print(__doc__)
        return
    rows = []
    for sp in specs:
        o, step, label = load(sp)
        if o is None:
            print(f"⚠ 找不到 {sp}")
            continue
        n = o.numel()
        frac = [float(((o >= lo) & (o < hi)).float().mean()) for lo, hi, _ in BANDS]
        rows.append((label, step, n, float(o.median()), frac))

    print(f"\n{'臂':>22} {'step':>7} {'N':>10} {'o中位':>8} "
          + " ".join(f"{nm:>10}" for *_, nm in BANDS))
    for label, step, n, med, frac in rows:
        print(f"{label:>22} {step:>7,} {n:>10,} {med:>8.4f} "
              + " ".join(f"{f:>9.2%}" for f in frac))

    if len(rows) >= 2:
        base = rows[0]
        same_step = all(r[1] == base[1] for r in rows)
        if same_step:
            # 同步數的臂間比較：速率沒有意義（Δ步=0），只印絕對差
            print(f"\n相對「{base[0]}」的差（同一步數，直接比絕對值）：")
            for label, step, n, med, frac in rows[1:]:
                print(f"  {label:>20}  判死 {frac[0] - base[4][0]:>+7.2%}  "
                      f"中間帶 {frac[1] - base[4][1]:>+7.2%}  "
                      f"0.05~0.5 {frac[2] - base[4][2]:>+7.2%}  "
                      f"確信 {frac[3] - base[4][3]:>+7.2%}  中位 {med - base[3]:>+8.4f}")
            return
        print(f"\n相對「{base[0]}」的變化（每 1000 步）：")
        for label, step, n, med, frac in rows[1:]:
            d = max(step - base[1], 1) / 1000.0
            print(f"  {label:>20}  Δ步={step - base[1]:>6,}  "
                  f"判死 {(frac[0] - base[4][0]) / d:>+7.3%}/k  "
                  f"確信 {(frac[3] - base[4][3]) / d:>+7.3%}/k  "
                  f"中位 {(med - base[3]) / d:>+8.5f}/k  "
                  f"N {n - base[2]:>+9,}")
        print("""
判讀：
  介入臂的「判死/k」明顯小於對照臂 => L1 確實是收割期損耗的主因（想要的效果）
  介入臂的「確信/k」**與對照臂相當**  => 兩極化沒被破壞（關鍵前提，必須成立）
  兩者同時成立才值得投 9.6h 完整跑次；若確信/k 也一起變緩 => L1 是兩極化的驅動力之一，收線。""")


if __name__ == "__main__":
    main()
