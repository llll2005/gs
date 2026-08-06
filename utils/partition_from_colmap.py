"""
COLMAP-based Scene Partitioner  (no coarse Gaussian model required)

Replaces utils/partition_citygs.py for use in the depth-prior initialization
pipeline. Instead of a trained Gaussian model, uses:
  - COLMAP sparse 3D points    → partition layout + "point cloud"
  - COLMAP track visibility     → camera-to-block assignment (replaces rendering)

Usage:
    python utils/partition_from_colmap.py <dataset_dir> [options]

Output format is identical to partition_citygs.py:
    <dataset_dir>/partition/partitions-dim_BX_BY_visibility_T/
        {bx:03d}_{by:03d}.txt    — image name list per block
        cameras-*.json           — camera metadata for visualization
        partitions.pt            — torch tensor bundle
        points.ply               — subsampled point cloud for visualization
"""
import add_pypath
import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from internal.utils.colmap import read_model, qvec2rotmat
from internal.utils.citygs_partitioning_utils import (
    CityGSSceneConfig, CityGSPartitionableScene, CityGSPartitioning,
)
from internal.dataparsers.colmap_dataparser import ColmapDataParser
from internal.utils.graphics_utils import store_ply

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", help="e.g. data/matrix_city/aerial/train/block_all")
parser.add_argument("--block_dim", type=int, nargs=2, default=[5, 5], metavar=("BX", "BY"))
parser.add_argument("--content_threshold", type=float, default=0.08,
                    help="fraction of COLMAP tracks in a block for secondary camera assignment")
parser.add_argument("--aabb", type=float, nargs="+", default=None,
                    help="manual scene AABB: xmin ymin xmax ymax")
parser.add_argument("--reorient", action="store_true",
                    help="align scene so that camera up = world Z (aerial scenes usually don't need this)")
parser.add_argument("--force", action="store_true")
parser.add_argument(
    '--origin',
    type=lambda v: "auto" if v.lower() == "auto" else [float(x) for x in v.split(',')],
    default=None,
)
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Load COLMAP
# ─────────────────────────────────────────────────────────────────────────────

sparse_dir = os.path.join(args.dataset_dir, "sparse")
if not os.path.exists(os.path.join(sparse_dir, "images.bin")):
    sparse_dir = os.path.join(sparse_dir, "0")

cameras_colmap, images_colmap, points3d_colmap = read_model(sparse_dir)

# Sort images by key for stable ordering (matches ColmapDataParser)
images_sorted = dict(sorted(images_colmap.items()))
image_names = [images_sorted[k].name for k in images_sorted]
n_cameras = len(image_names)
print(f"Loaded {n_cameras} cameras, {len(points3d_colmap)} 3D points from COLMAP")

# ─────────────────────────────────────────────────────────────────────────────
# Build camera-to-world (c2w) matrices
# ─────────────────────────────────────────────────────────────────────────────

c2w_list = []
for k in images_sorted:
    img = images_sorted[k]
    R_cw = qvec2rotmat(img.qvec)      # world-to-camera rotation
    t = img.tvec                       # world-to-camera translation
    # c2w = inv(T_cw)
    R_wc = R_cw.T
    t_wc = -R_cw.T @ t
    mat = np.eye(4)
    mat[:3, :3] = R_wc
    mat[:3,  3] = t_wc
    c2w_list.append(mat)

c2w = torch.tensor(np.stack(c2w_list), dtype=torch.float32)   # [N, 4, 4]
camera_centers = c2w[:, :3, 3]                                  # [N, 3]

# ─────────────────────────────────────────────────────────────────────────────
# Build sparse point cloud from COLMAP
# ─────────────────────────────────────────────────────────────────────────────

pts_list, rgb_list = [], []
for k in points3d_colmap:
    p = points3d_colmap[k]
    pts_list.append(p.xyz)
    rgb_list.append(p.rgb)

pts_xyz = torch.tensor(np.stack(pts_list), dtype=torch.float32)    # [M, 3]
pts_rgb = torch.tensor(np.stack(rgb_list), dtype=torch.float32)    # [M, 3]
print(f"Sparse point cloud: {len(pts_xyz):,} points")

# ─────────────────────────────────────────────────────────────────────────────
# Optional reorientation (usually not needed for aerial scenes)
# ─────────────────────────────────────────────────────────────────────────────

if args.reorient:
    up = -torch.mean(c2w[:, :3, 1], dim=0)
    up = up / torch.linalg.norm(up)
    rotation = ColmapDataParser.rotation_matrix(up, torch.tensor([0., 0., 1.]))
    rotation_transform = torch.eye(4)
    rotation_transform[:3, :3] = rotation
    reoriented_camera_centers = camera_centers @ rotation_transform[:3, :3].T
    reoriented_pts_xyz = pts_xyz @ rotation_transform[:3, :3].T
else:
    up = torch.tensor([0., 0., 1.])
    rotation_transform = torch.eye(4)
    reoriented_camera_centers = camera_centers
    reoriented_pts_xyz = pts_xyz

# ─────────────────────────────────────────────────────────────────────────────
# Build partition
# ─────────────────────────────────────────────────────────────────────────────

scene_config = CityGSSceneConfig(
    origin=torch.tensor([0., 0.]),
    block_dim=args.block_dim,
    aabb=args.aabb,
    contract=False,
    content_threshold=args.content_threshold,
)
scene = CityGSPartitionableScene(
    scene_config,
    reoriented_camera_centers[:, :2],
    reoriented_points=reoriented_pts_xyz[:, :2],
)

scene.get_bounding_box_by_points()
if args.origin:
    if args.origin == 'auto':
        scene_config.origin = (
            scene.point_based_bounding_box.min + scene.point_based_bounding_box.max
        ) / 2
    else:
        scene_config.origin = torch.tensor(args.origin)
scene.get_scene_bounding_box()
scene.build_partition_coordinates()

print(f"Camera center based partition assignment:")
is_cam_in_partition = scene.camera_center_based_partition_assignment()
print(f"  {is_cam_in_partition.sum(-1).tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# COLMAP-track based secondary assignment  (replaces Gaussian rendering)
#
# For each camera C not already in block B:
#   compute fraction of C's COLMAP tracks that land in B's AABB
#   if fraction > content_threshold → assign C to B
# ─────────────────────────────────────────────────────────────────────────────

n_blocks = args.block_dim[0] * args.block_dim[1]
device = torch.device("cpu")

# Build point-id → reoriented 2D xy lookup (ordered list)
pt_ids = np.array([points3d_colmap[k].id for k in points3d_colmap])
pt_xy = reoriented_pts_xyz[:, :2]   # [M, 2]
id_to_idx = {int(pt_ids[i]): i for i in range(len(pt_ids))}

# Per-block AABB (2D xy)
partition_bboxes = scene.partition_coordinates.get_bounding_boxes(
    scene.scene_config.partition_size, enlarge=0.0
).to(device)  # [N_blocks, 2, 2]: [[xmin,ymin],[xmax,ymax]]

# is_in_block[b, m] = True if point m is in block b (2D AABB test)
is_in_block = CityGSPartitioning.is_in_bounding_boxes(
    bounding_boxes=partition_bboxes,
    coordinates=pt_xy.to(device),
)  # [N_blocks, M]

# For each camera, get set of point indices it observes (via COLMAP tracks)
print("Computing COLMAP-track based visibility...")
cam_track_indices = []
for img_name in image_names:
    # find the image key
    img_key = next(k for k in images_sorted if images_sorted[k].name == img_name)
    img = images_sorted[img_key]
    pt3d_ids = img.point3D_ids                    # [n_keypoints], -1 = unmatched
    valid = pt3d_ids >= 0
    indices = [id_to_idx[int(pid)] for pid in pt3d_ids[valid] if int(pid) in id_to_idx]
    cam_track_indices.append(indices)

is_partitions_visible = torch.zeros(n_blocks, n_cameras, dtype=torch.bool)
for cam_idx, indices in enumerate(tqdm(cam_track_indices, desc="  track visibility")):
    if len(indices) == 0:
        continue
    track_in_block = is_in_block[:, indices]  # [N_blocks, n_tracks]
    frac = track_in_block.float().mean(dim=-1)  # [N_blocks]
    vis = frac > args.content_threshold
    # Only assign cameras NOT already in the block via center-based assignment
    not_already = ~is_cam_in_partition[:, cam_idx]
    is_partitions_visible[:, cam_idx] = vis & not_already

scene.is_partitions_visible_to_cameras = is_partitions_visible
print(f"Track-based visibility assignment: {is_partitions_visible.sum(-1).tolist()}")

# camera_visibilities: fraction of block's COLMAP points visible from each camera [N_blocks, N_cameras]
# point_getter(cam_idx) returns 3D positions of COLMAP tracks from that camera
def point_getter(cam_idx):
    indices = cam_track_indices[cam_idx]
    if len(indices) == 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    return reoriented_pts_xyz[indices]

scene.camera_visibilities = CityGSPartitioning.cameras_point_based_visibilities_calculation(
    partition_coordinates=scene.partition_coordinates,
    size=scene.scene_config.partition_size,
    n_cameras=n_cameras,
    point_getter=point_getter,
    device=device,
)

# ─────────────────────────────────────────────────────────────────────────────
# Write output (same format as partition_citygs.py)
# ─────────────────────────────────────────────────────────────────────────────

output_path = os.path.join(args.dataset_dir, scene.build_output_dirname())
if not args.force and os.path.exists(output_path):
    if os.path.exists(os.path.join(output_path, "partitions.pt")):
        print(f"Partition already exists at {output_path}. Use --force to overwrite.")
        sys.exit(0)
os.makedirs(output_path, exist_ok=True)
print(f"Output path: {output_path}")

scene.save(output_path, extra_data={"up": up, "rotation_transform": rotation_transform})
scene.save_plot(scene.plot_partitions, os.path.join(output_path, "partitions.png"), notebook=False)

is_assigned = torch.logical_or(is_cam_in_partition, is_partitions_visible)
print(f"Overall cameras assigned to partitions: {is_assigned.sum(-1).tolist()}")

written_idx_list = []
for partition_idx in tqdm(range(n_blocks), desc="Writing partition files"):
    assigned_cam_indices = is_assigned[partition_idx].nonzero().squeeze(-1).tolist()
    if len(assigned_cam_indices) == 0:
        continue
    written_idx_list.append(partition_idx)

    id_str = scene.partition_coordinates.get_str_id(partition_idx)
    camera_list = []
    with open(os.path.join(output_path, f"{id_str}.txt"), "w") as f:
        for cam_idx in assigned_cam_indices:
            f.write(image_names[cam_idx] + "\n")
            color = [255, 0, 0] if is_partitions_visible[partition_idx][cam_idx] else [0, 0, 255]
            cam_c2w = c2w[cam_idx]
            camera_list.append({
                "id": cam_idx,
                "img_name": image_names[cam_idx],
                "width": 1920, "height": 1080,
                "position": cam_c2w[:3, 3].numpy().tolist(),
                "rotation": cam_c2w[:3, :3].numpy().tolist(),
                "fx": 1600, "fy": 1600,
                "color": color,
            })
    with open(os.path.join(output_path, f"cameras-{id_str}.json"), "w") as f:
        json.dump(camera_list, f, indent=4, ensure_ascii=False)

# Save subsampled point cloud for visualization
max_store = 512_000
step = max(len(pts_xyz) // max_store, 1)
store_ply(
    os.path.join(output_path, "points.ply"),
    pts_xyz[::step].numpy(),
    pts_rgb[::step].numpy(),
)

print(f"\nDone. Wrote {len(written_idx_list)} / {n_blocks} partition files to:")
print(f"  {output_path}")
print("\nTo start training with depth-init:")
print(f"  python utils/depth_init_blocks.py {args.dataset_dir} --block_dim {args.block_dim[0]} {args.block_dim[1]}")
print(f"  python utils/train_citygs_partitions.py -n citygsv2_mc_aerial_sh2_trim \\")
print(f"      --depth_init_dir {args.dataset_dir}/depth_init")
