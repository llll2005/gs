# -*- coding: utf-8 -*-
"""CPU prediction of monster-defense options (X) without burning GPU.

Loads a near-OOM b12 ckpt and, for a sample of training cameras, computes each
gaussian's screen-space tile footprint (projection math only, no rasterization).
The binning buffer that OOMs = sum of per-gaussian tiles (clipped to frame).

Answers the decisive question:
  Scenario M (few monsters): a handful of points cover the whole frame
     -> option 1 (kill points over a tile LIMIT) removes almost all the load.
  Scenario S (spread overdraw): the whole population has high tiles/point
     -> option 1 barely helps; need K (more strips) or fewer points.

Then simulates option 1 (per-gaussian tile cap) and option 3 (radius clamp) and
reports the resulting worst-view intersection count + affected-point fraction,
predicting whether each keeps the buffer under the ~200M-intersection (4.74 GiB)
death line.

Usage: python tools/predict_monster_defense.py <ckpt> --cameras <cameras.json> [--n_views 16]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

TILE = 16
BYTES_PER_ISECT = 24.0  # key(8) + value(4), doubled for CUB radix ping-pong
GB = 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--n_views", type=int, default=16)
    ap.add_argument("--near", type=float, default=0.2, help="rasterizer near plane (p_view.z <= this culled)")
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    xyz = sd["gaussian_model.gaussians.means"].float().numpy()
    scales = np.exp(sd["gaussian_model.gaussians.scales"].float().numpy())  # [N,2] tangent
    s_max = scales.max(1)                                                    # worst-case facing
    N = xyz.shape[0]

    cams = json.load(open(args.cameras))
    idxs = np.linspace(0, len(cams) - 1, min(args.n_views, len(cams))).astype(int)

    per_view = []
    worst = None
    for i in idxs:
        cam = cams[int(i)]
        fx, fy, W, H = cam["fx"], cam["fy"], int(cam["width"]), int(cam["height"])
        R = np.asarray(cam["rotation"], dtype=np.float64)   # cam-from-world? cameras.json stores c2w rotation
        pos = np.asarray(cam["position"], dtype=np.float64)
        # world -> camera: p_cam = R^T (p_world - pos)   (position is camera center, rotation is c2w)
        pc = (xyz - pos) @ R
        z = pc[:, 2]
        vis = z > args.near
        # screen radius (px), 3-sigma of the projected tangent extent, facing-worst-case
        r = np.zeros(N)
        r[vis] = 3.0 * fx * s_max[vis] / z[vis]
        # projected center
        u = np.full(N, -1e9); v = np.full(N, -1e9)
        u[vis] = fx * pc[vis, 0] / z[vis] + cam["cx"]
        v[vis] = fy * pc[vis, 1] / z[vis] + cam["cy"]
        tiles = tiles_touched(u, v, r, W, H)
        total = tiles.sum()
        per_view.append((int(i), total, tiles, r, vis, v.copy()))
        if worst is None or total > worst[1]:
            worst = per_view[-1]

    print("===== MONSTER-DEFENSE PREDICTION (%s, N=%d, %d views) =====" % (os.path.basename(args.ckpt), N, len(idxs)))
    print("per-view total intersections (M): " + " ".join("%.0f" % (pv[1] / 1e6) for pv in per_view))
    vi, vtot, vtiles, vr, vvis, vv = worst
    print("\nWORST view id=%d: %.0fM intersections -> %.2f GiB binning buffer" % (
        vi, vtot / 1e6, vtot * BYTES_PER_ISECT / GB))

    # K-strip sweep on the WORST view, accounting for the ACTUAL vertical
    # distribution (a point's whole footprint lands in the strip its center is
    # in — approximates the per-strip binning load; spread load -> ~total/K,
    # concentrated load -> one strip dominates). This is the rigorous answer to
    # "does bigger K just fix it?".
    print("\n-- K-strip sweep (max-strip binning buffer on worst view; no quality loss) --")
    H_full = None
    for cam in [cams[vi]]:
        H_full = int(cam["height"])
    for K in (1, 2, 4, 8, 16):
        edges = np.linspace(0, H_full, K + 1)
        strip_of = np.clip(np.digitize(vv, edges) - 1, 0, K - 1)
        strip_of[~vvis] = -1
        loads = np.array([vtiles[strip_of == k].sum() for k in range(K)])
        mx = loads.max()
        print("  K=%2d: max-strip %.0fM -> %.2f GiB  (mean-strip %.0fM, concentration %.2fx)" % (
            K, mx / 1e6, mx * BYTES_PER_ISECT / GB, loads.mean() / 1e6, mx / max(loads.mean(), 1)))
    nz = vtiles[vtiles > 0]
    print("tiles/point (visible, worst view): mean %.1f  P50 %.0f  P90 %.0f  P99 %.0f  MAX %.0f" % (
        nz.mean(), np.percentile(nz, 50), np.percentile(nz, 90), np.percentile(nz, 99), nz.max()))

    # Scenario diagnosis: what fraction of load do the top-K biggest points carry?
    order = np.sort(vtiles)[::-1]
    cum = np.cumsum(order) / vtot
    for frac_pts in (0.001, 0.01, 0.05):
        k = int(N * frac_pts)
        print("  top %.1f%% biggest points carry %.1f%% of the load" % (100 * frac_pts, 100 * cum[k]))

    # Option 1: per-gaussian tile cap (kill points over LIMIT)
    print("\n-- Option 1: kill points over a per-gaussian tile LIMIT --")
    frame_tiles = (W // TILE + 1) * (H // TILE + 1)
    for lim_frac in (0.5, 0.25, 0.1, 0.05):
        lim = frame_tiles * lim_frac
        killed = vtiles > lim
        remain = vtiles[~killed].sum()
        print("  LIMIT=%.0f%% frame (%.0f tiles): kills %d pts (%.3f%%) -> buffer %.2f GiB (%.0fM)" % (
            100 * lim_frac, lim, int(killed.sum()), 100 * killed.mean(),
            remain * BYTES_PER_ISECT / GB, remain / 1e6))

    # Option 3: clamp each point's radius to R_MAX px, recompute tiles
    print("\n-- Option 3: clamp screen radius to R_MAX px (point kept, footprint bounded) --")
    for rmax in (200, 100, 50):
        rc = np.minimum(vr, rmax)
        u = np.full(N, -1e9); v = np.full(N, -1e9)
        u[vvis] = 0; v[vvis] = 0  # centers already on-frame-ish; reuse original center via tiles calc
        tiles_c = tiles_touched_from_center_tiles(vtiles, vr, rc, W, H)
        rem = tiles_c.sum()
        print("  R_MAX=%dpx: affects %d pts (%.2f%%) -> buffer %.2f GiB (%.0fM)" % (
            rmax, int((vr > rmax).sum()), 100 * (vr > rmax).mean(), rem * BYTES_PER_ISECT / GB, rem / 1e6))

    print("\nDEATH LINE ~4.74 GiB = ~%.0fM intersections. Any option whose buffer stays" % (4.74 * GB / BYTES_PER_ISECT / 1e6))
    print("well under that is predicted to prevent the single-frame OOM.")


def tiles_touched(u, v, r, W, H):
    """AABB of each splat clipped to frame, in units of 16px tiles."""
    x0 = np.clip(np.floor((u - r) / TILE), 0, W // TILE)
    x1 = np.clip(np.ceil((u + r) / TILE), 0, W // TILE)
    y0 = np.clip(np.floor((v - r) / TILE), 0, H // TILE)
    y1 = np.clip(np.ceil((v + r) / TILE), 0, H // TILE)
    t = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    t[r <= 0] = 0
    return t


def tiles_touched_from_center_tiles(orig_tiles, r_orig, r_clamped, W, H):
    """Approximate clamped tiles by scaling: tiles ~ (min(r,rmax)/r)^2 * orig for
    frame-unclipped, but clip-aware via sqrt. Good enough for buffer prediction."""
    ratio = np.ones_like(r_orig)
    m = r_orig > 0
    # tiles scale ~ area ~ r^2 until AABB fills frame; use min with orig (clip already in orig)
    ratio[m] = np.minimum(1.0, (r_clamped[m] / r_orig[m]) ** 2)
    return orig_tiles * ratio


if __name__ == "__main__":
    main()
