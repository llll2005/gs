# -*- coding: utf-8 -*-
"""Measure peak VRAM for the three orthogonal levers on b12 death-era geometry:
  (a) K-strip alone       (K=1,2,4)
  (b) gradient checkpoint alone
  (c) both together (K + checkpoint)
Single fwd+bwd per config (isolated, like calib_peak) — predicts whether checkpoint
helps the trim CUDA rasterizer (which manages its own buffers) and whether K+ckpt
synergy is real, WITHOUT committing a multi-hour training run.

The rasterizer is CUDA-only so this needs a free GPU. Run when dynk finishes.

Usage: python tools/measure_vram_levers.py
"""
import sys, os, glob
sys.path.insert(0, "."); sys.path.insert(0, "probes/p5_cost_aware")
import torch
from train_p5 import build_sets
from internal.models.gaussian_2d_sb import Gaussian2DSB
from internal.renderers.sep_depth_trim_2dgs_sb_renderer import SepDepthTrim2DGSSBRenderer
from internal.utils.strip_cameras import make_strip_camera, strip_bounds
import torch.utils.checkpoint as ckpt
GB = 2 ** 30
N_TEST = 1_450_000  # death-era point count
VIEW = 223          # a high-load view


def build(N):
    train_set, _ = build_sets("configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml",
                              "data/matrix_city/aerial/train/block_all", 12, "/tmp/mvl")
    ck = sorted(glob.glob("outputs/*sb*b12*/blocks/block_12/checkpoints/*.ckpt"),
                key=os.path.getmtime)[-1]
    sd = torch.load(ck, map_location="cpu")["state_dict"]
    props = {k.split("gaussians.")[-1]: v for k, v in sd.items() if "gaussian_model.gaussians." in k}
    N0 = props["means"].shape[0]; rep = int(N / N0) + 1
    for k in props:
        props[k] = props[k].repeat(rep, *([1] * (props[k].dim() - 1)))[:N].clone()
    m = Gaussian2DSB(sh_degree=0, sb_number=2).instantiate(); m.setup_from_number(N)
    for k, v in props.items():
        m.gaussians[k].data = v.cuda().float()
    m.active_sh_degree = 0; m.to("cuda")
    for p in m.gaussians.values():
        p.requires_grad_(True)
    return m, train_set


def one_step(m, cam, r, bg, K, use_ckpt):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    H = int(cam.height)
    for v0, v1 in strip_bounds(H, K):
        cs = make_strip_camera(cam, v0, v1 - v0)
        def _fwd():
            return r(cs, m, bg_color=bg)
        out = ckpt.checkpoint(_fwd, use_reentrant=False) if use_ckpt else _fwd()
        w = (v1 - v0) / H
        (out["render"].mean() * w + (out["rend_normal"] * out["surf_normal"]).sum(0).mean() * w).backward()
        for p in m.gaussians.values():
            p.grad = None
    return torch.cuda.max_memory_allocated() / GB


def main():
    m, ts = build(N_TEST)
    cam = ts.cameras[VIEW].to_device("cuda")
    r = SepDepthTrim2DGSSBRenderer(depth_ratio=1.0); bg = torch.zeros(3, device="cuda")
    pp = N_TEST * 25 * 4 / GB
    print("b12 SB N=%.2fM view %d | params-only %.2fG" % (N_TEST / 1e6, VIEW, pp))
    print("config                    peak(G)  vs K=1")
    base = None
    for K, ck_on, label in [(1, False, "K=1"), (2, False, "K=2"), (4, False, "K=4"),
                            (1, True, "ckpt only"), (2, True, "K=2 + ckpt"), (4, True, "K=4 + ckpt")]:
        try:
            pk = one_step(m, cam, r, bg, K, ck_on)
            if base is None:
                base = pk
            print("  %-22s  %.3f    %+.1f%%" % (label, pk, 100 * (pk - base) / base))
        except Exception as e:
            print("  %-22s  OOM/err (%s)" % (label, str(e)[:40]))


if __name__ == "__main__":
    main()
