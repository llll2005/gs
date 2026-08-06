"""
Crop a GLOBAL coarse checkpoint to one block's AABB, preserving full SH (unlike the .ply
init path which zeros shs_rest). Produces a small .ckpt that `--model.initialize_from` loads
via the .ckpt branch (full gaussian params kept, fresh optimizer/density buffers rebuilt).

Why: faithfully re-running CityGSV2 on a single block on 6GB. The global coarse (e.g. 4.39M
gaussians) OOMs if loaded whole (step-1 backward runs before visibility trimming). Cropping to
the block AABB (+dilation) gives the per-block coarse seed the official partition flow uses,
keeping full sh2 color.

Usage:
  python tools/crop_ckpt_to_block.py --ckpt <global.ckpt> --partitions <partitions.pt> \
      --block 7 --dilation 0.15 --out <cropped.ckpt>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from internal.utils.boundary_graph import load_partition_aabbs

GAUSS_PREFIX = "gaussian_model.gaussians."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--partitions", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--dilation", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    means_key = GAUSS_PREFIX + "means"
    if means_key not in sd:
        raise SystemExit(f"{means_key} not in state_dict; keys e.g. {list(sd)[:5]}")
    means = sd[means_key]
    n0 = means.shape[0]
    print(f"[load] global gaussians = {n0:,}")

    aabbs = load_partition_aabbs(args.partitions)
    min_xy, max_xy = aabbs[args.block]
    size = max_xy - min_xy
    min_xy = min_xy - args.dilation * size
    max_xy = max_xy + args.dilation * size
    x, y = means[:, 0].numpy(), means[:, 1].numpy()
    mask = torch.from_numpy(
        (x >= min_xy[0]) & (x <= max_xy[0]) & (y >= min_xy[1]) & (y <= max_xy[1])
    )
    n1 = int(mask.sum())
    print(f"[crop] block {args.block} AABB(+{args.dilation:.0%}): kept {n1:,}/{n0:,} ({n1/n0:.1%})")
    if n1 == 0:
        raise SystemExit("0 kept — AABB/coord frame mismatch")

    # crop every per-gaussian tensor (dim0 == n0) under the gaussian prefix
    cropped = 0
    for k in list(sd.keys()):
        if k.startswith(GAUSS_PREFIX) and torch.is_tensor(sd[k]) and sd[k].shape[:1] == (n0,):
            sd[k] = sd[k][mask].clone()
            cropped += 1
    print(f"[crop] cropped {cropped} per-gaussian tensors under '{GAUSS_PREFIX}'")

    # drop non-gaussian per-N buffers that would now mismatch (density controller etc.);
    # the init loader ignores them anyway, but strip to keep the ckpt clean/loadable.
    for k in list(sd.keys()):
        if not k.startswith(GAUSS_PREFIX) and not k.startswith("renderer.") and torch.is_tensor(sd[k]) \
                and sd[k].dim() >= 1 and sd[k].shape[0] == n0:
            del sd[k]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"[save] -> {args.out}  ({n1:,} gaussians, full SH preserved)")


if __name__ == "__main__":
    main()
