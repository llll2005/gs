"""
Depth-Prior Per-Block Initialization (RTG-SLAM style, adapted for offline batch use)
Generates per-block 2DGS PLY from DA2 depth maps + COLMAP poses.

  Point 1 — Semi-transparent seed Gaussians (alpha=0.1) for gradient flow
  Point 2 — Surface normals from depth map → proper surfel rotation
  Point 3 — M_s mask (valid depth + grazing angle < 60°); 100% kept, voxel decides density
  Point 4 — No color/depth loss here; opacity set for downstream 3DGS optimisation
  Point 5 — eta/t fields written to PLY for future stable/unstable controller
  Point 6 — Voxel deduplication replaces online fusion for offline case

Usage:
    python utils/depth_init_blocks.py <dataset_dir> [options]

Requires partition to have been run first (partition_citygs.py / partition_from_colmap.py).
Output: <dataset_dir>/depth_init/block_<id>.ply  (one 2DGS-compatible PLY per block)
"""
import add_pypath
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from internal.utils.colmap import read_model, qvec2rotmat

try:
    import open3d as o3d
except ImportError:
    raise RuntimeError("open3d not installed — run: pip install open3d")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", help="e.g. data/matrix_city/aerial/train/block_all")
parser.add_argument("--block_dim", type=int, nargs=2, default=[5, 5], metavar=("BX", "BY"))
parser.add_argument("--content_threshold", type=float, default=0.08)
parser.add_argument("--depth_dir", type=str, default=None,
                    help="default: <dataset_dir>/estimated_depths")
parser.add_argument("--depth_scales", type=str, default=None,
                    help="default: <dataset_dir>/estimated_depth_scales.json")
parser.add_argument("--output_dir", "-o", type=str, default=None,
                    help="default: <dataset_dir>/depth_init")
parser.add_argument("--sample_ratio", type=float, default=1.0,
                    help="Fraction of M_s pixels kept per frame; 1.0 = all (voxel handles density)")
parser.add_argument("--scale_factor", type=float, default=1.0,
                    help="Multiplier on depth-based surfel radius")
parser.add_argument("--voxel_alpha", type=float, default=0.003,
                    help="voxel_size = voxel_alpha * scene_scale (fallback if depth unavailable)")
parser.add_argument("--voxel_min", type=float, default=0.015,
                    help="Minimum voxel size in scene units; 0.015 ≈ target 500K Gaussians/block (0.02→300K)")
parser.add_argument("--voxel_max", type=float, default=0.5,
                    help="Maximum voxel size in scene units (caps far-range Gaussians)")
parser.add_argument("--chunk_size", type=int, default=50,
                    help="Frames per intermediate voxel downsample; caps peak RAM to ~chunk_size frames")
parser.add_argument("--depth_near", type=float, default=0.5)
parser.add_argument("--depth_far", type=float, default=1000.0)
args = parser.parse_args()

if args.depth_dir is None:
    args.depth_dir = os.path.join(args.dataset_dir, "estimated_depths")
if args.depth_scales is None:
    args.depth_scales = os.path.join(args.dataset_dir, "estimated_depth_scales.json")
if args.output_dir is None:
    args.output_dir = os.path.join(args.dataset_dir, "depth_init")

os.makedirs(args.output_dir, exist_ok=True)
rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# Load COLMAP + depth scales
# ─────────────────────────────────────────────────────────────────────────────

sparse_dir = os.path.join(args.dataset_dir, "sparse")
if not os.path.exists(os.path.join(sparse_dir, "images.bin")):
    sparse_dir = os.path.join(sparse_dir, "0")
cameras, images, _ = read_model(sparse_dir)
name_to_image = {images[k].name: images[k] for k in images}

with open(args.depth_scales) as f:
    depth_scales_dict = json.load(f)

partition_dir = os.path.join(
    args.dataset_dir, "partition",
    "partitions-dim_{}_{}_visibility_{}".format(
        args.block_dim[0], args.block_dim[1], args.content_threshold
    ),
)
assert os.path.exists(partition_dir), (
    f"Partition directory not found:\n  {partition_dir}\n"
    "Run utils/partition_from_colmap.py first."
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_intrinsics(cam, depth_h, depth_w):
    """Return (fx, fy, cx, cy) scaled to the depth map resolution."""
    s_h = depth_h / cam.height
    s_w = depth_w / cam.width
    p = cam.params
    if cam.model in ("PINHOLE",):
        fx, fy, cx, cy = p[0]*s_w, p[1]*s_h, p[2]*s_w, p[3]*s_h
    elif cam.model in ("SIMPLE_PINHOLE",):
        fx = fy = p[0] * s_h
        cx, cy = p[1]*s_w, p[2]*s_h
    elif cam.model in ("SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0] * s_h
        cx, cy = p[1]*s_w, p[2]*s_h
    elif cam.model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0]*s_w, p[1]*s_h, p[2]*s_w, p[3]*s_h
    else:
        raise NotImplementedError(f"Camera model {cam.model} not handled")
    return fx, fy, cx, cy


def npy_path_for(img_name):
    """Return the .npy path, trying both unpadded and zero-padded names."""
    p1 = os.path.join(args.depth_dir, f"{img_name}.npy")
    if os.path.exists(p1):
        return p1
    base, ext = img_name.rsplit('.', 1)
    p2 = os.path.join(args.depth_dir, f"{base.zfill(6)}.{ext}.npy")
    return p2 if os.path.exists(p2) else None


def estimate_normals_cam(depth, fx, fy, cx, cy):
    """
    Compute per-pixel surface normals in camera space from a depth map.
    Uses finite differences on the vertex map (RTG-SLAM Sec. 3.1).

    Returns:
        normals  [H, W, 3]  unit normals pointing toward camera (n_z < 0)
        valid    [H, W]     bool mask: True where normal is reliable
    """
    H, W = depth.shape
    uu = (np.arange(W, dtype=np.float32)[None, :] - cx) / fx
    vv = (np.arange(H, dtype=np.float32)[:, None] - cy) / fy

    # Vertex map in camera space
    Xc = uu * depth          # [H, W]
    Yc = vv * depth          # [H, W]
    V = np.stack([Xc, Yc, depth], axis=-1)  # [H, W, 3]

    # Tangent vectors via forward difference
    dVdu = np.zeros_like(V)
    dVdv = np.zeros_like(V)
    dVdu[:, :-1] = V[:, 1:] - V[:, :-1]
    dVdv[:-1, :] = V[1:, :] - V[:-1, :]

    n = np.cross(dVdu, dVdv)          # [H, W, 3]
    nlen = np.linalg.norm(n, axis=-1)  # [H, W]
    valid = nlen > 1e-6
    n[valid] /= nlen[valid, None]

    # Flip so normals point toward camera (cam-space z component should be negative
    # for surfaces facing the camera, since depth is positive-z).
    flip = n[..., 2] > 0
    n[flip] *= -1

    return n, valid


def normals_cam_to_world(n_cam, R_cw):
    """
    Rotate normals from camera to world space.
    COLMAP: p_cam = R_cw @ p_world + t  →  n_world = R_cw.T @ n_cam
    For row vectors [N, 3]:  n_world = n_cam @ R_cw
    """
    return n_cam @ R_cw  # equivalent to (R_cw.T @ n_cam.T).T


def normals_to_quats(normals):
    """
    Vectorized: quaternion [w,x,y,z] rotating z-axis [0,0,1] to each normal.
    Used to align 2DGS surfel discs with the estimated surface plane.

    normals: [N, 3] float32, will be normalised internally.
    """
    n = normals.astype(np.float64)
    nlen = np.linalg.norm(n, axis=-1, keepdims=True)
    n /= (nlen + 1e-8)

    # cross(z=[0,0,1], n) = [-ny, nx, 0]
    cross_x = -n[:, 1]
    cross_y =  n[:, 0]
    cross_len = np.sqrt(cross_x**2 + cross_y**2)  # sin of angle

    dot = n[:, 2]                                  # cos of angle
    angle = np.arctan2(cross_len, dot)             # [N]

    cos_half = np.cos(angle / 2)
    sin_half = np.sin(angle / 2)

    # Rotation axis (undefined when n ≈ ±z; fall back to x-axis for n≈-z case)
    valid = cross_len > 1e-6
    ax = np.where(valid, cross_x / (cross_len + 1e-8), 1.0)
    ay = np.where(valid, cross_y / (cross_len + 1e-8), 0.0)

    quats = np.stack([
        cos_half,
        ax * sin_half,
        ay * sin_half,
        np.zeros(len(normals)),
    ], axis=-1).astype(np.float32)
    return quats


def save_2dgs_ply(path, xyz, log_scales_2d, rots, opacities_raw, normals):
    """
    Write a 2DGS-compatible PLY (sh_degree=0).

    xyz          : [N,3]  float32 — Gaussian centres
    log_scales_2d: [N,2]  float32 — log-space surfel scales
    rots         : [N,4]  float32 — quaternion [w,x,y,z]
    opacities_raw: [N]    float32 — logit-space opacity
    normals      : [N,3]  float32 — unit surface normals (world space)
    """
    N = len(xyz)
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    el = np.empty(N, dtype=dtype_full)
    el['x'],  el['y'],  el['z']  = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    el['nx'], el['ny'], el['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    el['f_dc_0'] = el['f_dc_1'] = el['f_dc_2'] = 0.0
    el['opacity']  = opacities_raw
    el['scale_0'], el['scale_1'] = log_scales_2d[:, 0], log_scales_2d[:, 1]
    el['rot_0'], el['rot_1'], el['rot_2'], el['rot_3'] = (
        rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PlyData([PlyElement.describe(el, 'vertex')]).write(path)


# RTG-SLAM B1 (Sec 3.1): M_s surface Gaussians are initialised as opaque (α=0.99).
# "Each Gaussian is determined once after being added to be opaque (α=0.99) for fitting
# the 3D surface and dominant color."
# Opacity lr=0 (full freeze) is handled in RTGStableDensityController.freeze_opacity.
INIT_ALPHA = 0.99
OPAQUE_LOGIT = float(np.log(INIT_ALPHA / (1.0 - INIT_ALPHA)))

# RTG-SLAM Point 3: grazing angle threshold (60°) for M_s mask
COS_ANGLE_THRESH = 0.5   # cos(60°)

# RTG-SLAM Point 3: delta_T (transmission threshold for "newly observed")
# In offline init all regions are newly observed; we use the angle filter instead.

# ─────────────────────────────────────────────────────────────────────────────
# Per-block loop
# ─────────────────────────────────────────────────────────────────────────────

num_blocks = args.block_dim[0] * args.block_dim[1]

for block_id in range(num_blocks):
    bx = block_id % args.block_dim[0]
    by = block_id // args.block_dim[0]

    img_list_path = os.path.join(partition_dir, f"{bx:03d}_{by:03d}.txt")
    if not os.path.exists(img_list_path):
        print(f"[Block {block_id:2d} ({bx},{by})] partition file missing, skipping")
        continue

    with open(img_list_path) as f:
        block_imgs = [ln.strip() for ln in f if ln.strip()]

    print(f"[Block {block_id:2d} ({bx},{by})] {len(block_imgs)} cameras")

    all_xyz      = []   # intermediate-downsampled chunks (small)
    all_normals  = []
    all_cam_pos  = []   # camera world positions for depth-adaptive scale
    all_focal    = []   # sqrt(fx*fy) per frame
    chunk_xyz    = []   # raw pts within current chunk (cleared every chunk_size frames)
    chunk_normals = []

    def _flush_chunk(chunk_xyz, chunk_normals, voxel_size):
        """Intermediate voxel downsample to cap peak RAM."""
        if not chunk_xyz:
            return
        cxyz = np.concatenate(chunk_xyz, axis=0)
        cnrm = np.concatenate(chunk_normals, axis=0)
        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points  = o3d.utility.Vector3dVector(cxyz.astype(np.float64))
        pcd_c.normals = o3d.utility.Vector3dVector(cnrm.astype(np.float64))
        pcd_c_down    = pcd_c.voxel_down_sample(voxel_size)
        all_xyz.append(np.asarray(pcd_c_down.points,  dtype=np.float32))
        all_normals.append(np.asarray(pcd_c_down.normals, dtype=np.float32))

    for img_name in tqdm(block_imgs, desc=f"  block {block_id}", leave=False):
        img_meta = name_to_image.get(img_name)
        if img_meta is None:
            continue

        npy = npy_path_for(img_name)
        if npy is None:
            continue

        sc_info = depth_scales_dict.get(img_name)
        if sc_info is None or sc_info["scale"] == 0:
            continue
        sc, off = sc_info["scale"], sc_info["offset"]

        # ── Load + align depth ────────────────────────────────────────────────
        invdepth = np.load(npy).astype(np.float32)
        dh, dw = invdepth.shape
        inv_aligned = sc * invdepth + off
        valid_depth = inv_aligned > 1e-4
        depth = np.where(valid_depth, 1.0 / inv_aligned, 0.0)

        cam_intr = cameras[img_meta.camera_id]
        fx, fy, cx, cy = get_intrinsics(cam_intr, dh, dw)

        # ── RTG-SLAM Point 2: surface normals in camera space ─────────────────
        normals_cam, valid_n = estimate_normals_cam(depth, fx, fy, cx, cy)

        # ── RTG-SLAM Point 3: M_s mask ────────────────────────────────────────
        # Pixel ray direction (normalised) in camera space
        uu_g = (np.arange(dw, dtype=np.float32)[None, :] - cx) / fx  # [1, W]
        vv_g = (np.arange(dh, dtype=np.float32)[:, None] - cy) / fy  # [H, 1]
        ray = np.stack([
            np.broadcast_to(uu_g, (dh, dw)),
            np.broadcast_to(vv_g, (dh, dw)),
            np.ones((dh, dw), dtype=np.float32),
        ], axis=-1)                                                    # [H, W, 3]
        ray_len = np.linalg.norm(ray, axis=-1, keepdims=True)
        ray /= (ray_len + 1e-8)

        # |n · r| >= cos(60°) = 0.5  →  surface not edge-on to camera
        cos_ang = np.abs(np.einsum('hwc,hwc->hw', normals_cam, ray))  # [H, W]

        depth_range = (depth >= args.depth_near) & (depth <= args.depth_far)
        ms_mask = valid_depth & valid_n & depth_range & (cos_ang >= COS_ANGLE_THRESH)

        # ── Keep all M_s pixels; voxel_down_sample handles density ──────────────
        ms_idx = np.argwhere(ms_mask)          # [K, 2] (row, col)
        if len(ms_idx) == 0:
            continue
        if args.sample_ratio < 1.0:
            n_sample = max(1, int(len(ms_idx) * args.sample_ratio))
            chosen = rng.choice(len(ms_idx), size=n_sample, replace=False)
            ms_idx = ms_idx[chosen]
        vs_s = ms_idx[:, 0]
        us_s = ms_idx[:, 1]
        d_s  = depth[vs_s, us_s]

        # ── Backproject to camera space ───────────────────────────────────────
        Xc = (us_s - cx) / fx * d_s
        Yc = (vs_s - cy) / fy * d_s
        pts_cam = np.stack([Xc, Yc, d_s], axis=-1)  # [n_sample, 3]

        # ── Camera → World (COLMAP: p_cam = R_cw @ p_world + t_cw) ──────────
        R_cw = qvec2rotmat(img_meta.qvec)
        t_cw = img_meta.tvec
        pts_world = (pts_cam - t_cw) @ R_cw  # R_cw.T @ (p_cam - t)

        # ── Normals: camera → world ───────────────────────────────────────────
        n_cam_s   = normals_cam[vs_s, us_s]  # [n_sample, 3]
        n_world_s = normals_cam_to_world(n_cam_s, R_cw)
        n_world_s /= (np.linalg.norm(n_world_s, axis=-1, keepdims=True) + 1e-8)

        chunk_xyz.append(pts_world.astype(np.float32))
        chunk_normals.append(n_world_s.astype(np.float32))

        # ── RTG-SLAM Sec. 3.1: collect camera pos + focal for adaptive scale ──
        cam_pos_world = -(R_cw.T @ t_cw)           # camera centre in world
        all_cam_pos.append(cam_pos_world.astype(np.float32))
        all_focal.append(float(np.sqrt(fx * fy)))

        # ── Intermediate voxel downsample every chunk_size frames ─────────────
        # Caps peak RAM: at most chunk_size frames × ~1.28M pts in memory at once
        if len(chunk_xyz) >= args.chunk_size:
            _flush_chunk(chunk_xyz, chunk_normals, args.voxel_min)
            chunk_xyz.clear()
            chunk_normals.clear()

    # Flush remaining frames (< chunk_size)
    _flush_chunk(chunk_xyz, chunk_normals, args.voxel_min)
    chunk_xyz.clear()
    chunk_normals.clear()

    if not all_xyz:
        print(f"  [Block {block_id}] no valid points, skipping")
        continue

    xyz_all = np.concatenate(all_xyz,     axis=0)  # [M, 3] — chunk-deduped
    nrm_all = np.concatenate(all_normals, axis=0)  # [M, 3]
    print(f"  pts after chunk-dedup ({args.chunk_size} fr/chunk): {len(xyz_all):,}")

    # ── RTG-SLAM Sec. 3.1: depth-adaptive voxel size (pixel footprint) ──────
    # "cover the scene as much as possible with little overlapping"
    # voxel_size = pixel footprint at median depth = depth / focal
    if all_cam_pos and all_focal:
        cam_centroid = np.mean(all_cam_pos, axis=0)          # [3]
        focal_mean   = float(np.mean(all_focal))
        depths_from_cam = np.linalg.norm(xyz_all - cam_centroid, axis=-1)
        depth_median = float(np.median(depths_from_cam))
        pixel_footprint = depth_median / focal_mean
        voxel_size = float(np.clip(pixel_footprint, args.voxel_min, args.voxel_max))
        print(f"  depth-adaptive: median_depth={depth_median:.1f}m focal={focal_mean:.1f}px "
              f"→ pixel_footprint={pixel_footprint:.3f}m → voxel={voxel_size:.3f}m")
    else:
        scene_scale = np.percentile(
            np.linalg.norm(xyz_all - xyz_all.mean(0), axis=-1), 75
        )
        voxel_size = max(args.voxel_alpha * scene_scale, args.voxel_min)
        cam_centroid = xyz_all.mean(0)
        focal_mean = None

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz_all.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(nrm_all.astype(np.float64))
    pcd_down = pcd.voxel_down_sample(voxel_size)

    xyz_d = np.asarray(pcd_down.points,  dtype=np.float32)
    nrm_d = np.asarray(pcd_down.normals, dtype=np.float32)
    nrm_d /= (np.linalg.norm(nrm_d, axis=-1, keepdims=True) + 1e-8)
    print(f"  after voxel (size={voxel_size:.3f}m): {len(xyz_d):,}")

    if len(xyz_d) < 10:
        print(f"  [Block {block_id}] too few points after dedup, skipping")
        continue

    # ── RTG-SLAM Sec. 3.1: per-voxel depth-adaptive scale ───────────────────
    # scale_i = depth_i / focal  →  each Gaussian covers exactly its pixel footprint
    if focal_mean is not None:
        voxel_depths = np.linalg.norm(xyz_d - cam_centroid, axis=-1)  # [N_voxels]
        scales_v = np.clip(
            voxel_depths / focal_mean * args.scale_factor,
            args.voxel_min, args.voxel_max
        )
        log_scales_2d = np.stack([
            np.log(scales_v).astype(np.float32),
            np.log(scales_v).astype(np.float32),
        ], axis=-1)
    else:
        log_s_d = float(np.log(max(voxel_size * args.scale_factor, 1e-6)))
        log_scales_2d = np.full((len(xyz_d), 2), log_s_d, dtype=np.float32)

    # ── RTG-SLAM Point 2: rotation from surface normal ────────────────────────
    rots = normals_to_quats(nrm_d)

    # ── Initial opacity: OPAQUE_LOGIT = logit(0.99) (RTG-SLAM B1, opaque surface) ───────────
    opacities_raw = np.full(len(xyz_d), OPAQUE_LOGIT, dtype=np.float32)

    out_path = os.path.join(args.output_dir, f"block_{block_id}.ply")
    save_2dgs_ply(out_path, xyz_d, log_scales_2d, rots, opacities_raw, nrm_d)
    print(f"  saved → {out_path}  ({len(xyz_d):,} surfels)")

print("\nDone. PLY files in:", args.output_dir)
