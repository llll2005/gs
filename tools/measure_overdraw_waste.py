"""How much of the blending work buys nothing? The single number the cost-aware thesis rests on.

The claim is that prior methods price primitives by COUNT and therefore spend part of their budget
on primitives that are rasterised and blended but do not change the image. If that wasted fraction
w is large, a cost-aware method reaches the same quality on a smaller budget and w is the paper's
first figure. If w is small, the ceiling is low and we should know BEFORE building anything.

2026-08-07 measurements that motivate it: VRAM is not explained by N (R^2 = 0.066 over 20 runs)
but tracks blending depth -- opacity_reg 0 -> 0.002 at a fixed 900k points cost 1.81 GB, and a
uniform volume fill at 0.5M points cost the same as a surface cloud at 3.6M. The mechanism is
transmittance: T = (1-a)^n, so blended layers per pixel ~ ln(255)/a = 5.54/a.

TWO INDEPENDENT ESTIMATES, because each rests on a different assumption
----------------------------------------------------------------------
W IS THE WORK ACTUALLY PERFORMED (verified in forward.cu:406/452/455/459, 2026-08-07)
`num_covered_pixels` increments before the `alpha < 1/255` skip and before the `T < 1e-4` early-out,
but the loop is gated by `!done`, so anything behind an already-saturated pixel is never reached and
never counted. sum_i c_i is therefore the number of (primitive, pixel) evaluations the kernel ran --
not a frustum-membership count, and not an upper bound. That is what makes it a cost measure.

1. GEOMETRIC.  W = sum_i c_i is the exact number of (primitive, pixel) blend operations;
   P is the number of pixels that received any. W/P is the mean blended layers per pixel. A 2D
   surfel representation of an opaque scene ideally needs ~1 layer per pixel, so
       w_geom = 1 - ideal / (W/P)
   Rests on the "ideal is ~1 layer" argument, which is arguable at silhouettes and for genuinely
   translucent content -- hence `--ideal_layers`.

2. MATERIAL.  The rasteriser also returns each primitive's MEAN per-pixel deposit T*alpha. A
   primitive whose mean deposit is below a materiality threshold changed nothing it touched, yet
   paid c_i blend operations. Their share of W is a waste estimate that needs no ideal-layer
   argument at all.
       w_material = sum{ c_i : deposit_i < thresh } / W
   It is a LOWER bound: a primitive can be material in a few pixels and wasted in the rest, and
   this test, working on the mean, credits it as fully material.

Agreement between two estimates built on different assumptions is the evidence; a single number
would not be.
"""
import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader


def latest_ckpt(run, block):
    """Numeric sort on step. Sorting the filenames as STRINGS puts epoch=105 before epoch=52 and
    silently returns a mid-training checkpoint."""
    f = glob.glob(f"outputs/{run}/blocks/block_{block}/checkpoints/*.ckpt")
    if not f:
        return None
    f.sort(key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
    return f[-1]


def block_views(data, block, block_dim, names):
    """Views from the block's PARTITION LIST, never an AABB over cameras -- the AABB is a superset
    and feeds a block model cameras it never trained on."""
    by, bx = block // block_dim[0], block % block_dim[0]
    p = os.path.join(data, "partition",
                     f"partitions-dim_{block_dim[0]}_{block_dim[1]}_visibility_0.08",
                     f"{bx:03d}_{by:03d}.txt")
    want = {l.strip() for l in open(p) if l.strip()}
    return [i for i, n in enumerate(names) if n in want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--views", type=int, default=16)
    ap.add_argument("--ideal_layers", type=float, default=1.0,
                    help="layers per pixel an ideal surface representation would need")
    ap.add_argument("--material_thresh", type=float, default=0.01,
                    help="mean per-pixel deposit T*alpha below which a primitive changed nothing")
    ap.add_argument("--alpha_eps", type=float, default=1e-4, help="a pixel counts as covered above this")
    a = ap.parse_args()

    dev = "cuda"
    names, cams = load_test_cameras(a.data, 1.2)
    idx = block_views(a.data, a.block, a.block_dim, names)
    step = max(1, len(idx) // a.views)
    picked = idx[::step][:a.views]
    print(f"[視角] block {a.block} 分區 {len(idx)} 張，取樣 {len(picked)} 張")

    print(f"\n{'run':<30}{'N':>10}{'混合/像素':>11}{'w_geom':>9}{'w_material':>12}")
    for run in a.runs:
        ckpt = latest_ckpt(run, a.block)
        if ckpt is None:
            print(f"{run:<30}  無 ckpt")
            continue
        model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
            ckpt, dev, eval_mode=True)
        bg = torch.zeros(3, device=dev)
        n_pts = model.get_xyz.shape[0]

        W = P = 0.0
        wasted_W = 0.0
        with torch.no_grad():
            for i in picked:
                cam = cams[i].to_device(dev)
                # pass 1: per-primitive mean deposit + covered-pixel count (no image produced)
                deposit, covered = rend(cam, model, bg_color=bg,
                                        record_transmittance=True, record_coverage=True)
                W += float(covered.sum())
                wasted_W += float(covered[deposit < a.material_thresh].sum())
                # pass 2: the alpha map, to count pixels that received anything
                out = rend(cam, model, bg_color=bg)
                P += float((out["rend_alpha"] > a.alpha_eps).sum())
                del out
                torch.cuda.empty_cache()

        layers = W / max(P, 1.0)
        w_geom = 1.0 - a.ideal_layers / layers if layers > 0 else float("nan")
        w_mat = wasted_W / max(W, 1.0)
        print(f"{run[:29]:<30}{n_pts:>10,}{layers:>11.2f}{100 * w_geom:>8.1f}%{100 * w_mat:>11.1f}%")
        del model, rend
        torch.cuda.empty_cache()

    print(f"\n  混合/像素 = W/P，W=sum_i c_i 是精確的（粒子,像素）混合次數，P=收到任何貢獻的像素數")
    print(f"  w_geom     = 1 - {a.ideal_layers}/(W/P)   ← 靠「理想是 {a.ideal_layers} 層」這個論證")
    print(f"  w_material = 平均沉積 < {a.material_thresh} 的粒子佔 W 的比例   ← 不靠該論證，且是下界")
    print(f"  ⚠ 兩個估計基於不同假設；它們一致才是證據，單一個數字不是。")


if __name__ == "__main__":
    main()
