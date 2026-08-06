# -*- coding: utf-8 -*-
"""Zombie-economy audit of a trained ckpt: opacity/scale histograms, render-mass
share by population, and working-set size (min #points carrying 50%/95% of mass).

Born 2026-07-19 from the b12 discovery: 96.6% of points (o<0.05, mostly sub-pixel
after scale-L1 crushing) carry 0.1% of render mass — parked above the min_opacity
death line, so MCMC never recycles them ("zombies"). Use this to (a) reinterpret
count-ablation arms (did cap growth grow the working set or the zombie pool?) and
(b) verify the min_opacity annealing fix.

Usage: python tools/audit_working_set.py <ckpt> [--px 0.00155]
       (--px = pixel footprint in scene units at typical view distance)
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--px", type=float, default=0.00155, help="pixel footprint (scene units) at typical distance")
    args = ap.parse_args()

    c = torch.load(args.ckpt, map_location="cpu")
    sd = c["state_dict"]
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float().squeeze()).numpy()
    sc = np.exp(sd["gaussian_model.gaussians.scales"].float().numpy())
    N = len(o)
    px_area = args.px * args.px
    area = np.pi * sc[:, 0] * sc[:, 1]
    cover = np.maximum(area / px_area, 1.0)
    mass = o * cover
    tot = mass.sum()
    sub_px = sc.max(1) < args.px

    print("===== WORKING-SET AUDIT: %s (N=%d) =====" % (os.path.basename(args.ckpt), N))
    print("opacity: <0.05 %.1f%% | 0.05-0.2 %.1f%% | 0.2-0.8 %.1f%% | >0.8 %.1f%% | >0.9 %.1f%%" % (
        100 * (o < 0.05).mean(), 100 * ((o >= 0.05) & (o < 0.2)).mean(),
        100 * ((o >= 0.2) & (o < 0.8)).mean(), 100 * (o >= 0.8).mean(), 100 * (o > 0.9).mean()))
    print("scale(px): P50 %.2f P90 %.1f P99 %.1f | sub-pixel %.1f%%" % (
        *np.percentile(sc.mean(1) / args.px, [50, 90, 99]), 100 * sub_px.mean()))
    for name, m in [("anchors o>0.9", o > 0.9), ("mid 0.2-0.8", (o > 0.2) & (o < 0.8)),
                    ("fog o<0.05", o < 0.05), ("zombie (o<0.05 & sub-px)", (o < 0.05) & sub_px)]:
        print("  %-26s n=%8d (%5.1f%%)  mass %5.1f%%" % (name, m.sum(), 100 * m.mean(), 100 * mass[m].sum() / tot))
    idx = np.argsort(mass)[::-1]
    cum = np.cumsum(mass[idx]) / tot
    n50 = int(np.searchsorted(cum, 0.50)) + 1
    n95 = int(np.searchsorted(cum, 0.95)) + 1
    print("working set: 50%% mass in %d pts (%.2f%%) | 95%% in %d pts (%.2f%%)" % (
        n50, 100 * n50 / N, n95, 100 * n95 / N))


if __name__ == "__main__":
    main()
