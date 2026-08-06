# -*- coding: utf-8 -*-
"""Export a single CityGS block as a standalone COLMAP scene for kernel probes
(beta/triangle/convex splatting official repos).

- images resized to the CityGS eval resolution (down_sample 1.2 -> 1600x900),
  intrinsics rescaled accordingly, so probe PSNR is on the same yardstick.
- val_names.txt = the EXACT val split of our parser (global every-8th ∩ block),
  consumed by a 5-line patch in each repo's readColmapSceneInfo (llffhold would
  split differently -> val contamination).

Usage: python tools/export_probe_scene.py --block_id 7 --out data/probe_b7
"""
import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import yaml
from PIL import Image

from internal.utils import colmap as colmap_utils
from internal.dataparsers.estimated_depth_colmap_block_dataparser import EstimatedDepthBlockColmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block_id", type=int, required=True)
    ap.add_argument("--config", default="configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml")
    ap.add_argument("--data_path", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--down", type=float, default=1.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ia = dict(cfg["data"]["parser"].get("init_args", {}))
    ia["block_id"] = args.block_id
    valid = {f.name for f in dataclasses.fields(EstimatedDepthBlockColmap)}
    ia = {k: v for k, v in ia.items() if k in valid}
    out = EstimatedDepthBlockColmap(**ia).instantiate(
        path=args.data_path, output_path=args.out + "_tmp", global_rank=0).get_outputs()
    train_names = list(out.train_set.image_names)
    val_names = list(out.val_set.image_names)
    name_to_path = {n: p for n, p in zip(train_names + val_names,
                                         list(out.train_set.image_paths) + list(out.val_set.image_paths))}
    keep = set(train_names) | set(val_names)
    print(f"block {args.block_id}: {len(train_names)} train + {len(val_names)} val")

    sparse_dir = os.path.join(args.data_path, "sparse")
    if os.path.isdir(os.path.join(sparse_dir, "0")):
        sparse_dir = os.path.join(sparse_dir, "0")
    cameras = colmap_utils.read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = colmap_utils.read_images_binary(os.path.join(sparse_dir, "images.bin"))
    points = colmap_utils.read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "sparse", "0"), exist_ok=True)

    new_images = {}
    used_cam_ids = set()
    for iid, im in images.items():
        if im.name not in keep:
            continue
        new_images[iid] = im
        used_cam_ids.add(im.camera_id)
        src = name_to_path[im.name]
        dst = os.path.join(args.out, "images", im.name)
        if not os.path.exists(dst):
            img = Image.open(src).convert("RGB")
            W, H = img.size
            img.resize((round(W / args.down), round(H / args.down)), Image.BILINEAR).save(dst)
    print(f"exported {len(new_images)} images")

    new_cams = {}
    for cid in used_cam_ids:
        c = cameras[cid]
        params = list(c.params)
        # PINHOLE/SIMPLE_PINHOLE-style params are all in pixels -> divide uniformly
        params = [p / args.down for p in params]
        new_cams[cid] = colmap_utils.Camera(
            id=c.id, model=c.model,
            width=round(c.width / args.down), height=round(c.height / args.down),
            params=type(c.params)(params) if not hasattr(c.params, "dtype") else c.params / args.down)

    # keep only points observed by at least one kept image — 3DGS-family trainers
    # INITIALIZE from points3D, so an unfiltered city-wide cloud OOMs them at init
    kept_ids = set(new_images.keys())
    new_points = {pid: p for pid, p in points.items()
                  if any(int(i) in kept_ids for i in p.image_ids)}
    print(f"points3D: {len(points)} -> {len(new_points)} (observed by kept images)")

    colmap_utils.write_cameras_binary(new_cams, os.path.join(args.out, "sparse", "0", "cameras.bin"))
    colmap_utils.write_images_binary(new_images, os.path.join(args.out, "sparse", "0", "images.bin"))
    colmap_utils.write_points3D_binary(new_points, os.path.join(args.out, "sparse", "0", "points3D.bin"))
    ply_cache = os.path.join(args.out, "sparse", "0", "points3D.ply")
    if os.path.exists(ply_cache):
        os.remove(ply_cache)  # 3DGS loaders cache a ply conversion; stale = full cloud

    # 3DGS-family loaders use image_name WITHOUT extension
    with open(os.path.join(args.out, "val_names.txt"), "w") as f:
        for n in sorted(val_names):
            f.write(os.path.splitext(n)[0] + "\n")
    print(f"wrote {args.out} (val_names.txt: {len(val_names)})")


if __name__ == "__main__":
    main()
