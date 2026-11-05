"""定價自決容量的離線檢查（2026-11-05）：同一塊的多個訓練時點，價格規則會給多少容量、什麼時候收斂。

對每個 ckpt 掃訓練相機（同 tools/price_alloc.py 的逐顆 V/C：Σ T·α ÷ 覆蓋像素），對每個 λ 算
  cap(λ) = #{V/C >= λ} / 0.9                         （訓練中的規則另外把「剛出生／沒被看到」算成合格，這裡沒有出生資訊）
⚠ ckpt 是在固定 cap 下訓練的 => cap(λ) 不會超過當時的 N/0.9；它回答的是「這個價格下，現有顆粒有多少值得留」與收斂時點，
  不是「放開 cap 會長到多少」。

用法：python tools/price_cap_probe.py --ckpts <ckpt...> [--lams 0.0021 0.00415 0.0083] [--max-cam 300]
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--lams", nargs="+", type=float, default=[0.0021, 0.00415, 0.0083])
    ap.add_argument("--max-cam", type=int, default=300)
    a = ap.parse_args()
    from tools.price_alloc import block_scores
    dev = torch.device("cuda")
    rows = []
    for ck in sorted(a.ckpts, key=lambda p: int(re.search(r"step=(\d+)", p).group(1))):
        s, used, tot = block_scores(ck, a.max_cam, dev)
        st = int(re.search(r"step=(\d+)", ck).group(1))
        seen = s >= 0
        caps = [int((s >= lam).sum() / 0.9) for lam in a.lams]
        q = np.quantile(s[seen], [0.1, 0.5, 0.9]) if seen.any() else [0, 0, 0]
        rows.append((st, len(s), int(seen.sum()), caps, q))
        print(f"step {st:>6}：N {len(s):>9,}  看到 {int(seen.sum()):>9,}（相機 {used}/{tot}）  V/C p10/p50/p90 {q[0]:.2e}/{q[1]:.2e}/{q[2]:.2e}  "
              + "  ".join(f"cap(λ={lam:.2e}) {c:>9,}" for lam, c in zip(a.lams, caps)), flush=True)
    print("\n相鄰時點的 cap 相對變動（價格規則的收斂判準看這個）：")
    for i in range(1, len(rows)):
        print(f"  {rows[i - 1][0]:>6} -> {rows[i][0]:>6}：" + "  ".join(
            f"λ={lam:.2e} {abs(c1 - c0) / max(c0, 1):6.1%}" for lam, c0, c1 in zip(a.lams, rows[i - 1][3], rows[i][3])))


if __name__ == "__main__":
    main()
