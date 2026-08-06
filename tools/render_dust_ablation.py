# -*- coding: utf-8 -*-
"""Decisive zombie test: render val views with the FULL point set vs the point set
with dust removed (keep only o>0.05 or scale>1px), and report PSNR full-vs-GT,
filtered-vs-GT, and full-vs-filtered.

Predictions on record (2026-07-19): Gemini <0.1dB drop (maybe +0.05 gain);
zombie hypothesis says dust carries ~0.1% of render mass -> visually nothing.

Usage (needs GPU ~1-2G, run when training is idle):
  python tools/render_dust_ablation.py outputs/mcmc_sb_60k_aggr17_b12_cap1m/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt --block 12 --sb --n_views 6
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "probes", "p5_cost_aware"))

import numpy as np
import torch


def build_model(sd, sb, device, keep=None):
    props = {k.split("gaussians.")[-1]: v for k, v in sd.items() if "gaussian_model.gaussians." in k}
    if sb:
        from internal.models.gaussian_2d_sb import Gaussian2DSB
        cfg = Gaussian2DSB(sh_degree=0, sb_number=props["sb_params"].shape[1])
    else:
        from internal.models.gaussian_2d import Gaussian2D
        k = props["shs_rest"].shape[1]
        cfg = Gaussian2D(sh_degree={0: 0, 3: 1, 8: 2, 15: 3}[k])
    model = cfg.instantiate()
    n = props["means"].shape[0] if keep is None else int(keep.sum())
    model.setup_from_number(n)
    with torch.no_grad():
        for name, v in props.items():
            t = v if keep is None else v[keep]
            model.gaussians[name].data = t.to(device).float()
    model.active_sh_degree = model.max_sh_degree
    model.to(device)
    return model


def psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--sb", action="store_true")
    ap.add_argument("--n_views", type=int, default=6)
    ap.add_argument("--config", default="configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml")
    ap.add_argument("--data_path", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--px", type=float, default=0.00155)
    args = ap.parse_args()

    device = "cuda"
    from train_p5 import build_sets, load_gt
    if args.sb:
        from internal.renderers.sep_depth_trim_2dgs_sb_renderer import SepDepthTrim2DGSSBRenderer as R
    else:
        from internal.renderers.sep_depth_trim_2dgs_renderer import SepDepthTrim2DGSRenderer as R
    renderer = R(depth_ratio=1.0)

    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float().squeeze())
    sc = torch.exp(sd["gaussian_model.gaussians.scales"].float())
    keep = (o > 0.05) | (sc.max(1).values > args.px)
    print("full N=%d -> filtered N=%d (removed %.1f%% dust)" % (
        len(o), int(keep.sum()), 100 * (1 - keep.float().mean())))

    _, val_set = build_sets(args.config, args.data_path, args.block, "/tmp/dust_ablation")
    bg = torch.zeros(3, device=device)
    idxs = np.linspace(0, len(val_set.cameras) - 1, args.n_views).astype(int)

    rows = []
    for label, k in [("full", None), ("filtered", keep)]:
        model = build_model(sd, args.sb, device, k)
        renders = []
        with torch.no_grad():
            for i in idxs:
                cam = val_set.cameras[int(i)].to_device(device)
                renders.append(renderer(cam, model, bg_color=bg)["render"].cpu())
        rows.append((label, renders))
        del model
        torch.cuda.empty_cache()

    full_r, filt_r = rows[0][1], rows[1][1]
    p_full, p_filt, p_cross = [], [], []
    for j, i in enumerate(idxs):
        cam = val_set.cameras[int(i)]
        gt = load_gt(val_set.image_paths[int(i)], int(cam.width), int(cam.height), "cpu")
        gt = gt.permute(2, 0, 1) if gt.dim() == 3 and gt.shape[-1] == 3 else gt
        p_full.append(psnr(full_r[j], gt))
        p_filt.append(psnr(filt_r[j], gt))
        p_cross.append(psnr(filt_r[j], full_r[j]))
    print("PSNR vs GT:  full %.2f | filtered %.2f | delta %+.3f dB" % (
        np.mean(p_full), np.mean(p_filt), np.mean(p_filt) - np.mean(p_full)))
    print("filtered vs full (image agreement): %.1f dB" % np.mean(p_cross))


if __name__ == "__main__":
    main()
