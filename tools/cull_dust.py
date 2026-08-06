# -*- coding: utf-8 -*-
"""Deployment dust-cull: strip the zombie points (o<0.05 AND sub-pixel scale)
from a trained ckpt. Verified lossless on b12 (render_dust_ablation: removing
86.5% of points changed the render by -0.000 dB, images bit-identical — the
rasterizer never draws them).

Writes <out>.ckpt (loadable via --model.initialize_from, e.g. for merge /
consolidation / further finetune) and optionally a summary of the size cut.

Usage:
  python tools/cull_dust.py <ckpt_in> <ckpt_out> [--opacity 0.05] [--px 0.00155]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_in")
    ap.add_argument("ckpt_out")
    ap.add_argument("--opacity", type=float, default=0.05, help="cull below this activated opacity ...")
    ap.add_argument("--px", type=float, default=0.00155, help="... AND with max tangent scale under this (1px in scene units)")
    ap.add_argument("--min_view_dist", type=float, default=None,
                    help="conservative mode: tighten the px threshold so culled points stay "
                         "sub-pixel even at this closest viewer distance (default calibration "
                         "assumes ~3.0 units); e.g. 0.3 tightens the threshold 10x")
    ap.add_argument("--ply", default=None, help="also export the culled model as a .ply (for eyeballing in a viewer)")
    args = ap.parse_args()

    if args.min_view_dist is not None:
        args.px = args.px * args.min_view_dist / 3.0
        print("conservative px threshold: %.6f scene units (sub-pixel down to dist %.2f)" % (args.px, args.min_view_dist))

    c = torch.load(args.ckpt_in, map_location="cpu")
    sd = c["state_dict"]
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float().squeeze())
    sc = torch.exp(sd["gaussian_model.gaussians.scales"].float())
    dust = (o < args.opacity) & (sc.max(1).values < args.px)
    keep = ~dust

    n, k = len(o), int(keep.sum())
    bytes_before = sum(v.numel() * 4 for kk, v in sd.items() if "gaussian_model.gaussians." in kk)
    for kk in list(sd.keys()):
        if "gaussian_model.gaussians." in kk:
            sd[kk] = sd[kk][keep]
    bytes_after = sum(v.numel() * 4 for kk, v in sd.items() if "gaussian_model.gaussians." in kk)
    c.pop("optimizer_states", None)
    c.pop("lr_schedulers", None)
    torch.save(c, args.ckpt_out)
    print("culled %d -> %d points (%.1f%% dust) | params %.1f MB -> %.1f MB (%.1fx)" % (
        n, k, 100 * (1 - k / n), bytes_before / 2**20, bytes_after / 2**20, bytes_before / max(bytes_after, 1)))
    print("wrote", args.ckpt_out)

    if args.ply:
        from internal.utils.gaussian_utils import GaussianPlyUtils
        GaussianPlyUtils.load_from_state_dict(sd).to_ply_format().save_to_ply(args.ply)
        print("wrote", args.ply)


if __name__ == "__main__":
    main()
