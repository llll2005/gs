"""How much peak VRAM does the split backward cost, per point?

`B_RENDER = 1550` bytes/point in `internal/utils/strip_cameras.py` is labelled "per-point backward
activation, cut by K" and is the dominant splittable term -- 3.1 GB at N=2M. It was calibrated
while every densify step ran the SPLIT backward with `retain_graph=True`, so the autograd graph
survived across two backwards and the second allocated its workspace on top.

MCMC never read the viewspace gradient that split existed to produce, and it is now skipped, so the
graph is freed after one backward. If B_RENDER is over-stated for that path, then
`predict_num_strips` picks a larger K than needed and `calibrate_block_caps` caps below what the
card holds -- and "b12@2M hits a ~1.5M wall" may be largely this.

Measures the SLOPE (bytes per point) rather than one absolute number, so it can run at small N
alongside a training job instead of waiting for the card. Two N values, split on and off, same
camera, same model.

`torch.cuda.max_memory_allocated` is per-process, so a concurrent job does not pollute the reading
-- but it does share the card, so keep N small enough that the total stays under budget.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader

GB = 2 ** 30


def peak_for(ckpt, n_points, camera, split, dev="cuda"):
    """Peak allocated bytes for one forward+backward at `n_points`, with/without the split."""
    model, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ckpt, dev, eval_mode=False)
    keep = torch.randperm(model.get_xyz.shape[0], device=dev)[:n_points]
    for k, v in list(model.gaussians.items()):
        model.gaussians[k] = torch.nn.Parameter(v[keep].detach().clone(), requires_grad=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bg = torch.zeros(3, device=dev)
    out = rend(camera.to_device(dev), model, bg_color=bg)
    img = out["render"]
    # Two terms so the split has something to split, mirroring the metric's
    # loss = lambda*(1-ssim) + ... and extra_loss = (1-lambda)*rgb decomposition.
    a = (img ** 2).mean()
    b = img.abs().mean()
    if split:
        a.backward(retain_graph=True)
        b.backward()
    else:
        (a + b).backward()
    peak = torch.cuda.max_memory_allocated()
    del model, rend, out, img, a, b
    torch.cuda.empty_cache()
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--n", type=int, nargs="+", default=[250_000, 500_000])
    a = ap.parse_args()

    names, cams = load_test_cameras(a.data, 1.2)
    by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
    want = {l.strip() for l in open(os.path.join(
        a.data, "partition", f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08",
        f"{bx:03d}_{by:03d}.txt")) if l.strip()}
    idx = [i for i, n in enumerate(names) if n in want]
    camera = cams[idx[len(idx) // 2]]        # a mid-block view, same one for every measurement

    print(f"{'N':>10}{'分裂 peak':>13}{'合併 peak':>13}{'差':>11}{'差/點':>11}")
    pts, split_peaks, merged_peaks = [], [], []
    for n in sorted(a.n):
        ps = peak_for(a.ckpt, n, camera, split=True)
        pm = peak_for(a.ckpt, n, camera, split=False)
        pts.append(n); split_peaks.append(ps); merged_peaks.append(pm)
        print(f"{n:>10,}{ps / GB:>12.3f}G{pm / GB:>12.3f}G"
              f"{(ps - pm) / GB:>10.3f}G{(ps - pm) / n:>10.0f}B")

    if len(pts) >= 2:
        dn = pts[-1] - pts[0]
        b_split = (split_peaks[-1] - split_peaks[0]) / dn
        b_merged = (merged_peaks[-1] - merged_peaks[0]) / dn
        print(f"\n[斜率 bytes/point]  分裂 {b_split:.0f}   合併 {b_merged:.0f}"
              f"   降低 {100 * (1 - b_merged / max(b_split, 1e-9)):.1f}%")
        print(f"[對照] strip_cameras.py 的 B_RENDER = 1550 B/pt（在分裂還存在時標定）")
        print(f"[含意] N=2M 時，斜率差 {(b_split - b_merged) * 2e6 / GB:.2f}G 是預測器沒用到的餘裕")
    print(f"\n  ⚠ 斜率量的是「峰值隨 N 的變化」，含 backward activation 與 render 中可分攤的部分；")
    print(f"    不等於 B_RENDER 的定義（B_RENDER 只算可被 K 分攤的那一項）。當作量級估計。")


if __name__ == "__main__":
    main()
