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

# 2026-10-14 修正（使用者反映 viewer 偏紫／偏綠、細節糊）；檔名帶這個標記，web_demo 的轉檔快取才不會拿到舊的錯誤檔
FORMAT_TAG = "v2"
# 2DGS 沒有法向厚度：補 scale_2 = min(s0, s1) × THICKNESS_RATIO。
#   gsplat（標準 3DGS 光柵器）模擬 viewer、只用 SH0、與我方 2DGS 渲染器比（b12、實體像素）：
#   比例 1（舊版）19.3 dB、0.01 26.3 dB；全 SH 6 視角 0.3／0.1／0.01 => 20.0／20.4／20.4 => 0.1 已到平台，
#   不取更薄是因為舊註解記錄 scale_2 = -10（約 e^-10）時 Spark 在某些角度會看不到。
THICKNESS_RATIO = 0.1
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


def save_3dgs_ply(path: str, g: dict, thickness_ratio: float = THICKNESS_RATIO):
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

    # 2DGS: pad scale_2 (the normal axis). It used to be min(scale_0, scale_1), which turns flat surfels into
    # near-spheres => blurry in viewers (measured, see THICKNESS_RATIO). Now a fraction of it (log space: + log(ratio)).
    if scales.shape[1] == 2:
        scale_2 = np.minimum(scales[:, 0:1], scales[:, 1:2]) + np.float32(np.log(thickness_ratio))  # [N, 1]
        scales = np.concatenate([scales, scale_2], axis=1)    # [N, 3]

    # flatten SH: [N, 1, 3] -> [N, 3]
    f_dc   = shs_dc.reshape(N, -1)                            # [N, 3]
    # ⚠ 2026-10-14 修正：標準 3DGS PLY 的 f_rest 是「通道優先」（R 的 K 個係數、G 的、B 的；原版 save_ply 是
    #   features_rest.transpose(1, 2).flatten）。舊版直接 reshape(N, K*3) 是「係數優先」，viewer 讀回的高階 SH
    #   全部錯位 => 顏色隨視角偏紫／偏綠（同一批顆粒、隨機方向，與 3DGS 原版 eval_sh 的顏色最大差 1.65；修正後 2e-7）。
    f_rest = shs_rest.transpose(0, 2, 1).reshape(N, -1)       # [N, 3*K]
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
    parser.add_argument("--thickness", type=float, default=THICKNESS_RATIO, help="2DGS 補的法向厚度 = min(s0,s1) × 這個比例")
    args = parser.parse_args()

    g = load_gaussians(args.ckpt_path)
    save_3dgs_ply(args.output_ply, g, args.thickness)
