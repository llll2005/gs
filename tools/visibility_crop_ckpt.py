"""
Faithfully reproduce CityGaussian's per-block step-1 start-trimming OFFLINE, so a single block
can be fine-tuned on 6GB without the OOM that the in-training step-1 backward causes on the full
global coarse (the original runs this on 8xA100/80GB).

The original (sep_depth_trim_2dgs_renderer.py:178-205) at step 1: for every block camera, render
with record_transmittance, keep the top-K(=5) transmittance per Gaussian, contribution = mean of
top-K, prune contribution <= quantile(contribution, start_prune_ratio=0.0) (i.e. drop the
~zero-contribution Gaussians not seen by any block camera). block_id only selects cameras; it does
NOT crop Gaussians — only this trimming does. We replicate exactly that, no_grad/forward-only so it
fits 6GB, then save a cropped ckpt (full SH preserved) for `--model.initialize_from`.

Unlike tools/crop_ckpt_to_block.py (a crude AABB box, which can drop Gaussians an aerial camera
sees beyond the block footprint and thus UNDER-include / underestimate CityGSV2), this keeps exactly
the visibility set the paper's trimming keeps.

Usage:
  python tools/visibility_crop_ckpt.py --ckpt <global_coarse.ckpt> --config <block_cfg.yaml> \
      --block 7 --K 5 --out <cropped.ckpt> --render-downscale 1.0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from internal.utils.gaussian_model_loader import GaussianModelLoader
from tools.diag_multiview_consistency import build_train_set

GAUSS_PREFIX = "gaussian_model.gaussians."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="global coarse checkpoint")
    ap.add_argument("--config", required=True, help="config with block data settings (block_dim, down_sample, path)")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--K", type=int, default=5, help="top-K views per Gaussian (renderer default 5)")
    ap.add_argument("--start-prune-ratio", type=float, default=0.0, help="renderer default 0.0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp_out = os.path.join(os.path.dirname(args.out) or ".", "_viscrop_tmp")

    print(f"[load] coarse model+renderer from {args.ckpt}")
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        args.ckpt, device, eval_mode=True)
    n0 = model.get_xyz.shape[0]
    print(f"[load] global gaussians = {n0:,}")

    train_set = build_train_set(args.config, args.block, tmp_out)
    cams = train_set.cameras
    n_cam = len(cams)
    print(f"[data] block {args.block}: {n_cam} cameras (visibility trim source)")

    bg = torch.zeros(3, device=device)
    topk = torch.zeros(n0, args.K, device=device)  # K largest transmittances per Gaussian so far
    with torch.no_grad():
        for i in range(n_cam):
            cam = cams[i].to_device(device)
            trans = renderer(cam, model, bg_color=bg, record_transmittance=True).float()  # [N]
            topk = torch.topk(torch.cat([topk, trans[:, None]], dim=1), args.K, dim=1).values
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{n_cam}] cameras")
    contribution = topk.mean(dim=1)                       # = renderer's top-K mean
    tile = torch.quantile(contribution, args.start_prune_ratio)
    keep = contribution > tile                            # prune contribution <= tile (the ~0 ones)
    n1 = int(keep.sum())
    print(f"[trim] keep {n1:,}/{n0:,} ({n1/n0:.1%})  "
          f"(contribution: min={contribution.min():.4g} max={contribution.max():.4g} "
          f"zeros={(contribution<=tile).sum():,})")
    if n1 == 0:
        raise SystemExit("0 kept — check cameras/bg")

    # crop the ckpt's per-gaussian tensors by `keep`, preserve full SH, patch renderer to fine stage
    print(f"[save] cropping ckpt -> {args.out}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    keep_cpu = keep.cpu()
    cropped = 0
    for k in list(sd.keys()):
        if k.startswith(GAUSS_PREFIX) and torch.is_tensor(sd[k]) and sd[k].shape[:1] == (n0,):
            sd[k] = sd[k][keep_cpu].clone()
            cropped += 1
    for k in list(sd.keys()):
        if not k.startswith(GAUSS_PREFIX) and not k.startswith("renderer.") and torch.is_tensor(sd[k]) \
                and sd[k].dim() >= 1 and sd[k].shape[0] == n0:
            del sd[k]
    r = ckpt["hyper_parameters"]["renderer"]
    r.diable_trimming = False      # fine stage trims on
    r.prune_ratio = 0.05
    print(f"[save] cropped {cropped} per-gaussian tensors; renderer patched (trim on, prune_ratio 0.05)")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[done] {n1:,} gaussians (full SH) -> {args.out}")


if __name__ == "__main__":
    main()
