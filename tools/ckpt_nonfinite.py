"""ckpt 裡參數非有限的顆粒計數（2026-10-24：MCMC relocation 在 o 恰好 1.0 時算出 NaN scale，症狀是 train_status 的 `loss nan`）。

用法（在 repo 根目錄）：
    python tools/ckpt_nonfinite.py outputs/lab/<跑次>/blocks/block_1 [更多跑次目錄或 .ckpt ...]
目錄會取 checkpoints/ 底下排序最後的 .ckpt。每個印：N、非有限顆數（+inf／-inf／nan）、那些顆的 sigmoid(opacity) 中位、means 是否有限。
"""
import glob
import os
import sys

import torch


def _ckpt(p):
    if p.endswith(".ckpt"):
        return p
    ps = sorted(glob.glob(os.path.join(p, "checkpoints", "*.ckpt")))
    return ps[-1] if ps else None


def main(paths):
    for r in paths:
        p = _ckpt(r)
        if p is None:
            print(f"{r}: 無 ckpt")
            continue
        sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        g = "gaussian_model.gaussians."
        s, o, m = sd[g + "scales"], sd[g + "opacities"], sd[g + "means"]
        bad = ~(torch.isfinite(s).all(1) & torch.isfinite(o).all(1) & torch.isfinite(m).all(1))
        nb = int(bad.sum())
        if nb == 0:
            print(f"{r}: N {s.shape[0]:,}，參數全有限")
            continue
        sb = s[bad]
        print(f"{r}: N {s.shape[0]:,}，非有限 {nb:,} 顆（scale +inf {int((sb == float('inf')).any(1).sum())}／"
              f"-inf {int((sb == -float('inf')).any(1).sum())}／nan {int(sb.isnan().any(1).sum())}）；"
              f"這些顆的 sigmoid(o) 中位 {float(torch.sigmoid(o[bad]).median()):.3f}、means 有限 {int(torch.isfinite(m[bad]).all(1).sum()):,}")


if __name__ == "__main__":
    main(sys.argv[1:])
