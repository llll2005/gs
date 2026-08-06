"""
Export a per-block initialization PLY by cropping the GLOBAL coarse model to one
block's partition AABB. Used for experiment A: coarse-vs-depth init A/B in the new
aggressive regime, isolating the single variable "global-consistent geometry seed"
vs "per-block depth-projected seed".

The crop uses the canonical partition AABB (XY, +dilation) so the exported seed
covers the same spatial region as the depth-init block_*.ply. Saved as a full PLY;
the training loader's .ply path keeps DC color only (shs_rest re-learned), which
matches depth-init (sh0) — so both arms start DC-only + same renderer (trim on),
differing ONLY in the geometry seed.

Usage:
  python tools/export_coarse_block_init.py \
    --ckpt outputs/RTG_mc_aerial_coarse_sh2/checkpoints/epoch=6-step=30000.ckpt \
    --partitions data/matrix_city/aerial/train/block_all/partition/partitions-dim_5_5_visibility_0.08/partitions.pt \
    --block 7 --dilation 0.05 \
    --out data/matrix_city/aerial/train/block_all/depth_init/coarse_block_7_crop.ply
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.boundary_graph import load_partition_aabbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--partitions", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--dilation", type=float, default=0.05, help="fraction of AABB size to expand on each side")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[load] ckpt={args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    ply = GaussianPlyUtils.load_from_state_dict(sd)  # tensors, [n,...]
    xyz = ply.xyz if isinstance(ply.xyz, torch.Tensor) else torch.tensor(ply.xyz)
    n0 = xyz.shape[0]
    print(f"[load] global coarse gaussians = {n0:,}  sh_degree={ply.sh_degrees}")

    aabbs = load_partition_aabbs(args.partitions)
    if args.block not in aabbs:
        raise SystemExit(f"block {args.block} not in partitions (have {sorted(aabbs)})")
    min_xy, max_xy = aabbs[args.block]
    size = max_xy - min_xy
    min_xy = min_xy - args.dilation * size
    max_xy = max_xy + args.dilation * size
    print(f"[aabb] block {args.block}: x[{min_xy[0]:.2f},{max_xy[0]:.2f}] "
          f"y[{min_xy[1]:.2f},{max_xy[1]:.2f}]  (+{args.dilation:.0%} dilation)")

    x = xyz[:, 0].numpy()
    y = xyz[:, 1].numpy()
    mask = (x >= min_xy[0]) & (x <= max_xy[0]) & (y >= min_xy[1]) & (y <= max_xy[1])
    idx = torch.from_numpy(np.where(mask)[0])
    n1 = idx.shape[0]
    print(f"[crop] kept {n1:,} / {n0:,} ({n1/n0:.1%}) within block-{args.block} AABB")
    if n1 == 0:
        raise SystemExit("crop produced 0 gaussians — AABB/coords mismatch, check partitions.pt frame")

    cropped = GaussianPlyUtils(
        sh_degrees=ply.sh_degrees,
        xyz=ply.xyz[idx],
        opacities=ply.opacities[idx],
        features_dc=ply.features_dc[idx],
        features_rest=ply.features_rest[idx],
        scales=ply.scales[idx],
        rotations=ply.rotations[idx],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cropped.to_ply_format().save_to_ply(args.out)
    print(f"[save] -> {args.out}  ({n1:,} gaussians)")


if __name__ == "__main__":
    main()
