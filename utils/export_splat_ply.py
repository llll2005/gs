"""
Export a trained Gaussian checkpoint to standard 3DGS PLY for web viewers (Spark, SuperSplat, etc.).

Handles 2DGS (scale dim=2) by padding a near-zero scale_2 so standard viewers work.

Usage:
  # Single block:
  python utils/export_splat_ply.py \
    outputs/RTG_s10_mc_no_freeze_no_ADDM_aerial_sh2_trim/blocks/block_9/checkpoints/epoch=136-step=30000.ckpt \
    web_demo/block_9.ply

  # Merged model:
  python utils/export_splat_ply.py \
    outputs/RTG_s10_mc_no_freeze_no_ADDM_aerial_sh2_trim/checkpoints/merged.ckpt \
    web_demo/merged.ply
"""

import argparse
import sys
import os
import numpy as np
import struct
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_gaussians(ckpt_path: str) -> dict:
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    # support both naming conventions
    prefix = "gaussian_model.gaussians."
    means     = sd[prefix + "means"]           # [N, 3]
    shs_dc    = sd[prefix + "shs_dc"]          # [N, 1, 3]
    shs_rest  = sd[prefix + "shs_rest"]        # [N, K, 3]
    opacities = sd[prefix + "opacities"]       # [N, 1]
    scales    = sd[prefix + "scales"]          # [N, 2] or [N, 3]
    rotations = sd[prefix + "rotations"]       # [N, 4]

    print(f"  N={means.shape[0]:,}  SH_rest={shs_rest.shape[1]}  scale_dim={scales.shape[1]}")
    return dict(means=means, shs_dc=shs_dc, shs_rest=shs_rest,
                opacities=opacities, scales=scales, rotations=rotations)


def save_3dgs_ply(path: str, g: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    import torch, torch.nn.functional as F

    means     = g["means"].numpy().astype(np.float32)          # [N, 3]
    shs_dc    = g["shs_dc"].numpy().astype(np.float32)         # [N, 1, 3]
    shs_rest  = g["shs_rest"].numpy().astype(np.float32)       # [N, K, 3]
    opacities = g["opacities"].numpy().astype(np.float32)      # [N, 1]
    scales    = g["scales"].numpy().astype(np.float32)         # [N, 2 or 3]

    # normalize quaternions — raw model params are NOT unit quaternions
    rotations = F.normalize(g["rotations"].float(), dim=-1).numpy().astype(np.float32)

    N = means.shape[0]

    # 2DGS: pad scale_2 using the smaller of scale_0/scale_1 so Gaussians stay
    # visible from all angles (scale_2=-10 makes them invisible when viewed face-on)
    if scales.shape[1] == 2:
        scale_2 = np.minimum(scales[:, 0:1], scales[:, 1:2])  # [N, 1]
        scales = np.concatenate([scales, scale_2], axis=1)    # [N, 3]

    # flatten SH: [N, 1, 3] -> [N, 3],  [N, K, 3] -> [N, K*3]
    f_dc   = shs_dc.reshape(N, -1)                            # [N, 3]
    f_rest = shs_rest.reshape(N, -1)                          # [N, K*3]
    n_rest = f_rest.shape[1]

    # build property list
    props = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(n_rest)]
        + ["opacity"]
        + ["scale_0", "scale_1", "scale_2"]
        + ["rot_0", "rot_1", "rot_2", "rot_3"]
    )

    # write header
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
        )
        for p in props:
            header += f"property float {p}\n"
        header += "end_header\n"
        f.write(header.encode("ascii"))

        # build row data
        normals = np.zeros((N, 3), dtype=np.float32)
        rows = np.concatenate([
            means, normals, f_dc, f_rest, opacities, scales, rotations
        ], axis=1)                                             # [N, total_props]
        f.write(rows.astype(np.float32).tobytes())

    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved {N:,} Gaussians → {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", help="Path to .ckpt checkpoint")
    parser.add_argument("output_ply", help="Output .ply path")
    args = parser.parse_args()

    g = load_gaussians(args.ckpt_path)
    save_3dgs_ply(args.output_ply, g)
