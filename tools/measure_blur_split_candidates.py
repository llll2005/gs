"""Does Mini-Splatting's blur split have anything to do on our models? Premise test, no training.

Mini-Splatting (arXiv 2403.14166) Eq. 2 splits Gaussians whose MAXIMUM CONTRIBUTION AREA
    S_i = #{ pixels where i is the argmax-weight Gaussian }
exceeds theta_blur * H * W (theta_blur = 2e-4). Large S_i = one Gaussian alone explains a big
patch = 'under-reconstruction', which is exactly the failure measured here on 2026-08-07: in
high-texture cells the error is invariant to a 4.1x density increase.

We cannot get the hard argmax without a CUDA change, but the rasteriser already accumulates
    raw_i = sum_p T_i(p) * alpha_i(p)
the SOFT version of the same quantity: sum_i raw_i = sum_p (1 - T_final(p)) ~= P, and sum_i S_i = P
too (one argmax per pixel), so the two are on the same scale and the paper's threshold transfers.
raw_i is arguably the better statistic -- a Gaussian that is a strong second everywhere is doing
the same damage and the hard argmax misses it.
"""
import argparse, glob, os, re, sys
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--block", type=int, default=12)
ap.add_argument("--views", type=int, default=8)
ap.add_argument("--theta", type=float, default=2e-4)
a = ap.parse_args()

names, cams = load_test_cameras("data/matrix_city/aerial/train/block_all", 1.2)
want = {l.strip() for l in open("data/matrix_city/aerial/train/block_all/partition/"
        "partitions-dim_5_5_visibility_0.08/002_002.txt") if l.strip()}
idx = [i for i, n in enumerate(names) if n in want]
picked = idx[:: max(1, len(idx) // a.views)][: a.views]
H, W = int(cams.height[0]), int(cams.width[0])
thr = a.theta * H * W
print(f"[門檻] theta*H*W = {a.theta} * {H} * {W} = {thr:.0f} 像素\n")
print(f"{'run':<24}{'N':>11}{'超標顆數':>11}{'佔比':>8}{'其佔總權重':>11}{'最大 raw':>10}{'p99':>8}")

for run in a.runs:
    ck = glob.glob(f"outputs/{run}/blocks/block_{a.block}/checkpoints/*.ckpt")
    if not ck: print(f"{run:<24} 無 ckpt"); continue
    ck.sort(key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
    m, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck[-1], "cuda", eval_mode=True)
    N = m.get_xyz.shape[0]
    tot = torch.zeros(N, device="cuda")
    with torch.no_grad():
        for i in picked:
            mean, cov = rend(cams[i].to_device("cuda"), m, bg_color=torch.zeros(3, device="cuda"),
                             record_transmittance=True, record_coverage=True)
            tot += mean * cov.float()          # raw_i summed over views
            torch.cuda.empty_cache()
    raw = (tot / len(picked)).cpu().numpy()    # per-view average
    over = raw > thr
    print(f"{run[:23]:<24}{N:>11,}{int(over.sum()):>11,}{100*over.mean():>7.2f}%"
          f"{100*raw[over].sum()/max(raw.sum(),1e-9):>10.1f}%{raw.max():>10.0f}{np.percentile(raw,99):>8.1f}")
    del m, rend; torch.cuda.empty_cache()
print("\n  超標顆數少 ⇒ blur-split 在我方場景無事可做；佔總權重高 ⇒ 少數大團主宰畫面，值得分裂")
