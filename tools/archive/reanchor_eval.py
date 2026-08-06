# -*- coding: utf-8 -*-
"""Re-anchor evaluation after the rasterizer rebuild (2026-07-19).

Background: the March-2026 installed trim-surfel binary renders ~0.4dB differently
from a fresh build of the pinned commit (source of the delta unexplained; symbol
sets identical; March wheel preserved in the pip cache). All historical numbers
(22.45 / 21.99 / harvest arms ...) were measured on the March binary. Going
forward the project standard is the reproducible pinned+pp_shifty build, so the
key checkpoints must be re-measured once on the new scale.

Renders the FULL val set of a block and reports mean PSNR.

Usage: python tools/reanchor_eval.py <ckpt> --block 12 [--sb]
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "probes", "p5_cost_aware"))

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--sb", action="store_true")
    ap.add_argument("--config", default="configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml")
    ap.add_argument("--data_path", default="data/matrix_city/aerial/train/block_all")
    args = ap.parse_args()

    from train_p5 import build_sets, load_gt
    from tools.render_dust_ablation import build_model
    if args.sb:
        from internal.renderers.sep_depth_trim_2dgs_sb_renderer import SepDepthTrim2DGSSBRenderer as R
    else:
        from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer as R

    device = "cuda"
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    model = build_model(sd, args.sb, device)
    renderer = R(depth_ratio=1.0)
    bg = torch.zeros(3, device=device)

    _, val_set = build_sets(args.config, args.data_path, args.block, "/tmp/reanchor")
    psnrs = []
    with torch.no_grad():
        for i in range(len(val_set.cameras)):
            cam = val_set.cameras[i].to_device(device)
            render = renderer(cam, model, bg_color=bg)["render"]
            gt = load_gt(val_set.image_paths[i], int(cam.width), int(cam.height), device)
            gt = gt.permute(2, 0, 1) if gt.dim() == 3 and gt.shape[-1] == 3 else gt
            psnrs.append(-10 * torch.log10(((render - gt) ** 2).mean()).item())
    print("REANCHOR %s | block %d | n_val %d | PSNR %.3f" % (
        os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.ckpt)))) or args.ckpt,
        args.block, len(psnrs), float(np.mean(psnrs))))


if __name__ == "__main__":
    main()
