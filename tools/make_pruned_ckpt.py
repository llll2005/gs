# -*- coding: utf-8 -*-
"""Surgical prune for the condensation-attribution experiment (Gemini round-6):
take a mid-training ckpt, keep only the top-X% points by render-mass proxy
(o * pixel coverage), and write a pruned ckpt usable as --model.initialize_from.

The experiment: prune the 42k ckpt to top-20% (simulating the harvest phase's
winner-take-all), then finetune a few k steps with ONLY color/SB learnable
(means/opacity/scale/rotation lr = 0). If PSNR recovers to ~21.99, the +2.75dB
harvest jump was appearance convergence (condensation = epiphenomenon); if it
collapses, condensation's continuous mass-transfer is the real engine.

Usage:
  python tools/make_pruned_ckpt.py <ckpt_in> <ckpt_out> --keep_frac 0.2 [--px 0.00155]
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_in")
    ap.add_argument("ckpt_out")
    ap.add_argument("--keep_frac", type=float, default=0.2)
    ap.add_argument("--px", type=float, default=0.00155)
    args = ap.parse_args()

    c = torch.load(args.ckpt_in, map_location="cpu")
    sd = c["state_dict"]
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float().squeeze())
    sc = torch.exp(sd["gaussian_model.gaussians.scales"].float())
    area = np.pi * sc[:, 0] * sc[:, 1]
    cover = torch.clamp(area / (args.px ** 2), min=1.0)
    mass = o * cover

    N = len(o)
    n_keep = int(N * args.keep_frac)
    keep = torch.topk(mass, n_keep).indices
    keep, _ = torch.sort(keep)

    kept_mass = float(mass[keep].sum() / mass.sum())
    print("N %d -> %d (keep %.0f%%), retained render mass %.2f%%" % (N, n_keep, 100 * args.keep_frac, 100 * kept_mass))

    for k in list(sd.keys()):
        if "gaussian_model.gaussians." in k:
            sd[k] = sd[k][keep]
    # optimizer states in the ckpt are shape-stale now; initialize_from ignores them
    # (fresh warm-start), but strip them to avoid accidental --ckpt_path resume.
    c.pop("optimizer_states", None)
    c.pop("lr_schedulers", None)
    torch.save(c, args.ckpt_out)
    print("wrote", args.ckpt_out)


if __name__ == "__main__":
    main()
