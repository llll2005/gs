"""Two-sided check on the depth-loss coverage gate.

The gate (`CityGSV2Metrics.depth_coverage_eps`) must satisfy two things at once, and only one of
them is about fixing the bug:

  FIRES on the run it was written for. `cap80k_60k_b12` pinned the count at 80,000, `d_reg` spiked
  to 1.1e6 against a healthy 0.002-0.17, and val collapsed 20.34 -> 17.06. If the gate does not
  find uncovered pixels there, the diagnosis is wrong.

  SILENT on the runs already in the results table. `aggr24k_b12`, `graded_k4_24k_b12`,
  `oreg_0p002_b12` and `b12_cap2m_reg000_exact` all held `d_reg` inside 0.0009-0.0055 for their
  whole schedule, so a change in their loss would invalidate every comparison drawn against them.
  A single 1e8 pixel would have lifted the mean over 1.44M pixels to ~69, so those runs cannot have
  had any -- the gate should report 0.00%.

Reports the fraction of pixels below the alpha threshold, and what the depth term would have been
with and without the gate, so "no-op" is measured rather than asserted.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader

DATA = "data/matrix_city/aerial/train/block_all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="名稱=路徑")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--eps", type=float, default=1e-4)
    a = ap.parse_args()

    dev = "cuda"
    names, cams = load_test_cameras(a.data, 1.2)
    by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
    want = {l.strip() for l in open(os.path.join(
        a.data, "partition", f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08",
        f"{bx:03d}_{by:03d}.txt")) if l.strip()}
    pool = [i for i, n in enumerate(names) if n in want]
    probe = [pool[j] for j in np.linspace(0, len(pool) - 1, min(a.views, len(pool))).astype(int)]
    print(f"[視角] block {a.block} 取樣 {len(probe)} 台   eps={a.eps}")

    print(f"\n{'模型':<34}{'N':>10}{'未覆蓋像素':>12}{'1/(d+1e-8) 最大':>17}{'判定':>10}")
    for spec in a.ckpts:
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"  {name}: 找不到，跳過")
            continue
        model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            path, dev, eval_mode=True)
        bg = torch.zeros(3, device=dev)
        frac, mx = [], []
        with torch.no_grad():
            for i in probe:
                o = rend(cams[i].to_device(dev), model, bg_color=bg)
                al = o["rend_alpha"].squeeze()
                inv = 1.0 / (o["surf_depth"].clamp_min(0.).squeeze() + 1e-8)
                frac.append(float((al <= a.eps).float().mean()))
                mx.append(float(inv.max()))
        f = 100 * float(np.mean(frac))
        m = float(np.max(mx))
        verdict = "✅ 靜默" if f < 1e-6 else ("⚠ 會觸發" if f < 1.0 else "❗大量觸發")
        print(f"{name:<34}{model.get_xyz.shape[0]:>10,}{f:>11.4f}%{m:>17.3g}{verdict:>10}")
        del model, rend
        torch.cuda.empty_cache()

    print(f"\n  未覆蓋像素 = rend_alpha <= eps 的比例（gate 會把這些像素的預測換成 GT）")
    print(f"  1/(d+1e-8) 最大 = 沒有 gate 時深度項看到的最大值；1e8 表示該像素完全沒東西")
    print(f"  ✅ 靜默 = 這個 run 的損失不會被 gate 改變，既有比較仍然有效")


if __name__ == "__main__":
    main()
