"""Strip-camera forging for K-strip tiled training (route-2 camera crop).

Renders a horizontal band [v0, v0+h) of a full image as its own camera by
building an off-center (principal-point-shifted) projection. Derivation, using
the repo's centered convention (pixel = (ndc+1)/2 * S, y-down, cy_eff = H/2):

    ndc_y(strip) = (2*fy/h) * y/z + ((H - 2*v0)/h - 1)

so relative to the full-frame P (untransposed): P'[1,1] = P[1,1] * H/h and
P'[1,2] = (H - 2*v0)/h - 1. The x row and z rows are unchanged. fov_y is forged
so the rasterizer's focal_y = h / (2*tan(fov_y'/2)) stays equal to fy.

depth_to_normal & co. keep working: they rebuild intrinsics from full_projection
and view.height, both of which are consistently modified here.

Validated on the gsplat probe line (verify_tiling.py): strip composition equals
the full render to MSE ~1e-14 and the intersection buffers scale ~1/K.
"""
import dataclasses
import math

import torch


def make_strip_camera(camera, v0: int, h: int):
    """Return a Camera viewing rows [v0, v0+h) of `camera`'s image."""
    H = int(camera.height)

    P = camera.projection.T.clone()          # un-transpose to the OpenGL-style P
    P[1, 1] = P[1, 1] * H / h
    P[1, 2] = (H - 2 * v0) / h - 1.0
    projection = P.T.contiguous()
    full_projection = camera.world_to_camera @ projection

    fov_y = 2.0 * math.atan(math.tan(float(camera.fov_y) * 0.5) * h / H)

    cam = dataclasses.replace(
        camera,
        height=torch.tensor(h, dtype=camera.height.dtype, device=camera.height.device),
        cy=camera.cy - v0,
        fov_y=torch.tensor(fov_y, dtype=camera.fov_y.dtype, device=camera.fov_y.device),
        projection=projection,
        full_projection=full_projection,
    )
    # The trim surfel rasterizer builds its pixel mapping from intrins with a
    # HARDCODED center (W/2, H/2) — the shifted projmatrix above is only used by
    # depth_to_normal etc. The rebuilt rasterizer accepts pp_shifty so its cy
    # becomes h/2 + shift = H/2 - v0, matching the projmatrix-derived mapping.
    cam.strip_pp_shifty = (H / 2.0 - v0) - h / 2.0
    return cam


def strip_bounds(H: int, K: int):
    """K near-equal horizontal bands covering [0, H)."""
    edges = [round(H * k / K) for k in range(K + 1)]
    return [(edges[k], edges[k + 1]) for k in range(K)]


TILE = 16
BYTES_PER_ISECT = 24.0  # key(8)+value(4) doubled for CUB radix ping-pong


@torch.no_grad()

def projected_radius(means: torch.Tensor, scales: torch.Tensor, camera,
                     near: float = 0.2) -> torch.Tensor:
    """Per-point screen radius (px) for `camera`, from projection only — the single source
    of truth for "what does this primitive cost to draw". Facing-worst-case tangent extent
    times the perspective Jacobian stretch; see estimate_render_load for the derivation of
    both terms.

    Unlike the rasterizer's `radii` output this is defined for EVERY point, visible or not,
    which is what a cost signal needs: the rasterizer window is a running max that resets at
    each densify event, so points missed by recent views read 0 and the cost signal becomes a
    150-step sawtooth (measured on the b12 DAR smoke: median ĉ collapsed to the storage floor
    right after each event). Behind the camera (z <= near) is clamped, not zeroed, since such
    a point still costs model state and can swing back into frame.
    """
    pc = means @ camera.R.T + camera.T
    zc = pc[:, 2].clamp_min(near)
    s_max = scales[:, :2].max(dim=1).values
    stretch = torch.sqrt(1.0 + (pc[:, 0] / zc) ** 2 + (pc[:, 1] / zc) ** 2)
    return 3.0 * float(camera.fx) * s_max / zc * stretch


def estimate_render_load(means: torch.Tensor, scales: torch.Tensor, camera, near: float = 0.2,
                         r_crit: float = 1e18) -> float:  # r_crit disabled (see note below)
    """Conservative upper bound on the binning intersections a full-frame render
    of `camera` would produce, from projection only (no rasterization). Used by
    predictive dynamic-K to size the number of strips BEFORE committing the render.

    Per point: screen radius r = 3*fx*s_max/z (facing-worst-case), tiles ~
    (2r/TILE)^2 capped at the whole frame. No AABB frame-clip (that only reduces
    load) -> the estimate is an over-estimate, i.e. errs toward MORE strips = safe.

    means [N,3], scales [N,>=2] (activated, tangent). camera: internal Camera.

    ★Monster early-exit (2026-07-23): returns +inf if ANY visible point has screen
    radius > r_crit. Rationale: the AABB-tile sum caps a giant splat at frame_tiles,
    so it UNDER-counts near-camera monsters (z->0 makes r explode but binning produces
    orders of magnitude more intersections than one frame of tiles) — this is exactly
    what killed dynk at 1.55M (single-strip 4.61 GiB). The radius r is computed BEFORE
    any frame-clip, so `r > r_crit` detects a monster by its intrinsic size regardless
    of whether its center is on-frame — sidestepping the clip blindness entirely. An
    L_inf (max radius) trigger, not L1 (tile sum): a single giant splat alone OOMs, so
    we don't need the sum. predict_num_strips maps +inf -> K_max (full defense). Cheap:
    one boolean reduction, no gather; early-exits before the AABB compute.
    """
    W, H = int(camera.width), int(camera.height)
    fx, fy = float(camera.fx), float(camera.fy)
    cx, cy = float(camera.cx), float(camera.cy)
    R = camera.R  # world->camera rotation ([3,3]); world_to_camera[:3,:3] = R (cameras.py)
    T = camera.T
    pc = means @ R.T + T                              # [N,3] camera-space
    z = pc[:, 2]
    s_max = scales[:, :2].max(dim=1).values
    vis = z > near
    zc = z.clamp_min(near)
    # Jacobian perspective elongation (2026-07-23): the paraxial r = 3*fx*s/z misses
    # the off-axis stretch of the projection Jacobian — an off-centre splat's 2D
    # bbox is amplified by sqrt(1 + (x/z)^2 + (y/z)^2). This is why the AABB load
    # under-counts multiplicatively (1.35x in-frame, up to ~10x for extreme off-axis
    # near points that stretch into a long ray sweeping the frame). Physics term, not
    # a magic constant: fixes both the average bias and the tail spike at once.
    stretch = torch.sqrt(1.0 + (pc[:, 0] / zc) ** 2 + (pc[:, 1] / zc) ** 2)
    r = 3.0 * fx * s_max / zc * stretch               # screen radius (px), pre-clip
    # NOTE (2026-07-23): the radius-based monster early-exit is DISABLED (r_crit huge).
    # Measurement on b12 death-era ckpt killed it: r_faceon>15000 in ALL views (orientation
    # bias is pervasive, not a tail) AND uncorrelated with real danger (safest view had
    # HIGHER r_faceon). b12 = spread overdraw, no single giant monster (real radii ≤1556px).
    # The right fix is accurate LOAD (AABB undercounts real by ~1.35x), not radius detection.
    if bool(((r > r_crit) & vis).any()):
        return float("inf")
    u = fx * pc[:, 0] / zc + cx                       # projected center (px)
    v = fy * pc[:, 1] / zc + cy
    # AABB in tiles, CLIPPED to the frame (an off-frame near-camera splat -> 0 tiles)
    x0 = torch.clamp(torch.floor((u - r) / TILE), 0, W // TILE)
    x1 = torch.clamp(torch.ceil((u + r) / TILE), 0, W // TILE)
    y0 = torch.clamp(torch.floor((v - r) / TILE), 0, H // TILE)
    y1 = torch.clamp(torch.ceil((v + r) / TILE), 0, H // TILE)
    tiles = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    tiles[~vis] = 0
    return float(tiles.sum())


# Calibrated peak-VRAM model (b12 SB, 2026-07-22, tools calib_peak.py). The OLD
# estimator predicted binning intersections (load) and picked K from that — but the
# real OOM driver is backward ACTIVATION, which scales with N (points), NOT with
# tile-load. Measured: render peak(N,K) = A_RENDER*N + B_RENDER*N/K bytes, where
# B_RENDER is the K-splittable backward activation (the big term). Plus a small
# load-dependent binning term for monster frames. Fitting {0.7,1.0,1.45}M x {1,2,4}
# gave A_RENDER~427, B_RENDER~1493 B/pt.
A_RENDER = 500.0   # per-point render cost NOT cut by K (rounded up from 427 for margin)
B_RENDER = 1550.0  # per-point backward activation, cut by K (rounded up from 1493)
# 2026-08-06: measured, and the "the split inflates these" hypothesis was WRONG.
# `tools/probe_backward_peak.py` at fixed N, same camera, split on vs off:
#     N=250k  0.541G vs 0.544G      N=500k  0.746G vs 0.766G
#     slope   883 B/pt (split) vs 953 B/pt (merged) -- MERGING IS 7.9% WORSE
# The forward graph has to live until the last backward either way; `retain_graph` only delays the
# free. What actually differs is the BACKWARD WORKSPACE: two sequential backwards hold one branch's
# intermediate gradients at a time, while `(a+b).backward()` holds both at the merge point.
# So these constants are NOT stale on account of the split, and the claim that it explains
# "b12@2M hits a ~1.5M wall" is retracted. (Skipping the split remains a SPEED win: one less pass.)
# ⚠ The probe used a simplified loss, not the real SSIM+depth metric, so 883/953 is not directly
# comparable to 1550 -- it bounds the split's CONTRIBUTION, not the constant itself.
#
# ⚠⚠ 2026-08-07 -- DO NOT SIZE A CAP FROM THESE CONSTANTS UNTIL THEY ARE RE-FITTED.
# Two full-run anchors say the per-point model is not just miscalibrated, it is the wrong shape:
#
#   cap4m_b12       N=4.0M, K=1, surface-fitted (depth-init grown)  ->  peak 5.51 G
#                   predicted by A+B: 4.0M * 2050 = 8.2 G render alone. Actual render, after
#                   subtracting 1.6 G of model state, is 3.91 G = 978 B/pt -- 2.1x LOWER.
#   uniform_60k_b12 N=0.5M, K=1, uniform VOLUME fill                ->  peak ~5.5 G
#                   i.e. the SAME peak with 8x fewer points, and it did not fall as N was pruned
#                   from 1.0M to 0.47M.
#
# So peak is dominated by how the primitives are ARRANGED (overdraw: layers per pixel), not by N.
# A volume fill costs ~7x per point what a surface cloud does. The `load_isect` term exists for
# exactly this but is calibrated as a small correction, and 2450 B/pt was fitted over 0.7-1.45M of
# ONE arrangement, then extrapolated 3x beyond its range.
#
# Consequence already paid: predict_num_strips said 4M would OOM with the wall at 2.04M; the run
# finished at 3.6M using 5.51/6.1 G. tools/calibrate_block_caps.py takes its caps from this model,
# so every per-block cap it has produced is likely far below what the card holds.
# Re-fit needs points spanning arrangements (surface vs volumetric), not just N. See 紀錄 2026-08-07.


def predict_num_strips(load_isect: float, n_points: int, floats_per_point: int,
                       vram_target_gb: float, v_os_gb: float, safety: float, max_strips: int,
                       state_mult: int = 4) -> int:
    """Pick K so the predicted PEAK VRAM stays under budget. Peak =
        model_state(N, not cut by K) + A_RENDER*N (not cut) + B_RENDER*N/K (cut by K)
                                      + binning(load)/K (monster spikes, cut by K)
    Solve peak <= safety*(V_target - V_os) for the smallest K in [1, max_strips].
    This predicts the quantity that actually OOMs (backward activation ~ N), unlike
    the old load-only formula that undercounted the N-scaling term and picked K=1
    into an OOM at N=1.45M."""
    if not math.isfinite(load_isect):
        return max_strips                                                # monster detected -> full defense
    GB = 2 ** 30
    model_bytes = n_points * floats_per_point * 4 * state_mult          # params+grad+2Adam, resident
    fixed = model_bytes + A_RENDER * n_points                            # not reducible by K
    splittable = B_RENDER * n_points + load_isect * BYTES_PER_ISECT      # reducible ~1/K
    budget = safety * (vram_target_gb * GB - v_os_gb * GB)
    room = budget - fixed
    if room <= 0.1 * GB:
        return max_strips                                                # fixed cost alone near budget -> max split
    k = int(math.ceil(splittable / room))
    return max(1, min(max_strips, k))


def crop_batch(batch, v0: int, v1: int, strip_camera):
    """Crop the (camera, image_info, gt_inverse_depth) train batch to rows [v0, v1)."""
    _, image_info, gt_inverse_depth = batch
    image_name, gt_image, masked_pixels = image_info

    gt_image_s = gt_image[:, v0:v1, :]
    masked_s = masked_pixels[..., v0:v1, :] if isinstance(masked_pixels, torch.Tensor) else masked_pixels

    if isinstance(gt_inverse_depth, tuple):
        depth_s = tuple(t[..., v0:v1, :] for t in gt_inverse_depth)
    elif isinstance(gt_inverse_depth, torch.Tensor):
        depth_s = gt_inverse_depth[..., v0:v1, :]
    else:
        depth_s = gt_inverse_depth

    return strip_camera, (image_name, gt_image_s, masked_s), depth_s
