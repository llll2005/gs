"""
Per-Gaussian off-surface (free-space floater) diagnostic — GT-free, real-scene applicable.

Idea (user, 2026-06-23): after densify, before fine-tune, identify object-boundary / free-space
Gaussians and remove the ones floating OFF the surface. Before pruning anything, MEASURE whether
the flagged set is genuinely free-space garbage (the prior post-hoc floater_cleanup failed because
its criterion selected important edge/occlusion points). This script only diagnoses; it prunes
nothing.

Criterion (the key fix vs the failed attempt): a Gaussian is a floater if, across the views that
see it, its own depth z_c sits consistently IN FRONT of the model's consensus surface depth
D_surf (rel = (z_c - D_surf)/D_surf < -thresh). "In front" (poking toward the camera), NOT merely
"inconsistent": occlusion pushes a point BEHIND the surface (rel>0), so the front-criterion
auto-avoids occlusion/silhouette edges — exactly the points the previous attempt wrongly killed.
Multi-view agreement (front in a large fraction of views) separates true free-space floaters
(front almost everywhere) from surface points (front almost nowhere).

Outputs: per-Gaussian front/behind agreement stats; opacity split of candidates vs rest (low-op =
already handled by importance-prune; high-op = the complementary niche worth pruning); and subset
RENDERS (full / candidates-removed / candidates-only) from a few views so you can SEE whether the
flagged set is floaters or surfaces. Saves the candidate mask for a later prune experiment.

Usage:
  python tools/diag_offsurface_gaussians.py mcmc_b7_capmax --block 7 \
      --rel-thresh 0.05 --front-frac 0.4 --min-views 5 --render-views 4 --out outputs/diag_offsurface
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools.diag_multiview_consistency import find_block_dir, find_ckpt, find_config, build_train_set
from internal.utils.gaussian_model_loader import GaussianModelLoader

EPS = 1e-8


def save_png(rgb_chw: torch.Tensor, path: str) -> None:
    arr = (rgb_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--block", type=int, default=7)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-views", type=int, default=60, help="views used to accumulate per-Gaussian stats")
    ap.add_argument("--rel-thresh", type=float, default=0.05, help="|rel depth| beyond this = off-surface")
    ap.add_argument("--front-frac", type=float, default=0.4, help="flag if in-front in >= this fraction of seen views")
    ap.add_argument("--min-views", type=int, default=5, help="need >= this many views seeing it to judge")
    ap.add_argument("--render-views", type=int, default=4, help="how many views to render full/removed/only for")
    ap.add_argument("--out", default="outputs/diag_offsurface")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    block_dir = find_block_dir(args.run, args.block)
    ckpt = find_ckpt(block_dir, args.ckpt)
    cfg_path = find_config(block_dir, args.run)
    print(f"[load] ckpt={ckpt}")

    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        ckpt, device, eval_mode=True)
    means = model.get_xyz.detach()           # [N,3]
    opac = model.get_opacities().detach().squeeze(-1)  # [N]
    n = means.shape[0]
    print(f"[load] gaussians={n:,}")

    train_set = build_train_set(cfg_path, args.block, args.run)
    n_total = len(train_set.cameras)
    idxs = list(range(n_total))
    if args.max_views and n_total > args.max_views:
        idxs = np.linspace(0, n_total - 1, args.max_views).astype(int).tolist()
    print(f"[data] {n_total} views, using {len(idxs)} to accumulate per-Gaussian stats")

    bg = torch.zeros(3, device=device)
    ones = torch.ones((n, 1), device=device)
    seen = torch.zeros(n, device=device)
    front = torch.zeros(n, device=device)
    behind = torch.zeros(n, device=device)
    rel_sum = torch.zeros(n, device=device)

    with torch.no_grad():
        for k, idx in enumerate(idxs):
            cam = train_set.cameras[idx].to_device(device)
            D = renderer(cam, model, bg_color=bg)["surf_depth"].squeeze().float()  # [H,W]
            H, W = D.shape
            W2C = cam.world_to_camera.float()  # [4,4], row: p_cam = p_world @ W2C
            pcam = torch.cat([means, ones], dim=-1) @ W2C  # [N,4]
            z = pcam[:, 2]
            u = cam.fx * pcam[:, 0] / (z + EPS) + cam.cx
            v = cam.fy * pcam[:, 1] / (z + EPS) + cam.cy
            gx = 2 * u / (W - 1) - 1
            gy = 2 * v / (H - 1) - 1
            grid = torch.stack([gx, gy], dim=-1).reshape(1, n, 1, 2)
            Dsamp = F.grid_sample(D[None, None], grid, mode="nearest", align_corners=True).reshape(n)
            inb = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1) & (z > 0) & (Dsamp > 0)
            rel = (z - Dsamp) / (Dsamp + EPS)
            seen += inb.float()
            front += (inb & (rel < -args.rel_thresh)).float()
            behind += (inb & (rel > args.rel_thresh)).float()
            rel_sum += torch.where(inb, rel, torch.zeros_like(rel))
            if (k + 1) % 10 == 0:
                print(f"  [{k+1}/{len(idxs)}] views processed")

    seen_c = seen.clamp(min=1)
    agree_front = front / seen_c
    agree_behind = behind / seen_c
    judged = seen >= args.min_views
    candidate = judged & (agree_front >= args.front_frac)

    n_judged = int(judged.sum())
    n_cand = int(candidate.sum())
    print("\n========= per-Gaussian off-surface diagnostic (GT-free) =========")
    print(f"gaussians={n:,}  judged(seen>={args.min_views})={n_judged:,} ({n_judged/n:.1%})")
    print(f"rel-thresh={args.rel_thresh:.0%}  front-frac>={args.front_frac:.0%}")
    print(f"★ FLOATER candidates (in-front, multi-view agreeing) = {n_cand:,} ({n_cand/n:.2%} of all)")
    # agreement distribution among judged
    af = agree_front[judged].cpu().numpy()
    print(f"  front-agreement among judged: mean={af.mean():.1%} p50={np.percentile(af,50):.1%} "
          f"p90={np.percentile(af,90):.1%} p99={np.percentile(af,99):.1%}")
    print(f"  behind-agreement among judged: mean={agree_behind[judged].mean().item():.1%}")

    # opacity split: is the candidate set already covered by importance-prune (low opacity) or a
    # complementary niche (opaque off-surface floaters)?
    if n_cand > 0:
        oc = opac[candidate]
        orest = opac[judged & ~candidate]
        print(f"\n  candidate opacity : mean={oc.mean():.3f} median={oc.median():.3f} "
              f">0.5: {(oc>0.5).float().mean():.1%}  >0.9: {(oc>0.9).float().mean():.1%}")
        print(f"  non-cand  opacity : mean={orest.mean():.3f} median={orest.median():.3f}")
        print(f"  -> high-opacity candidates (the niche importance-prune CAN'T remove): "
              f"{int((oc>0.5).sum()):,} ({(oc>0.5).float().mean():.1%} of candidates)")

    # save mask for a potential prune experiment (sanitize run name: may contain "/")
    safe_run = args.run.replace("/", "_").strip("_")
    mask_path = os.path.join(args.out, f"floater_mask_{safe_run}_b{args.block}.pt")
    torch.save({"candidate": candidate.cpu(), "agree_front": agree_front.cpu(),
                "seen": seen.cpu(), "opacity": opac.cpu(),
                "args": vars(args)}, mask_path)
    print(f"\n[save] candidate mask -> {mask_path}")

    # ---- visual confirmation: render full / candidates-removed / candidates-only ----
    print(f"[render] {args.render_views} views: full | removed | only  -> {args.out}")
    rview = np.linspace(0, n_total - 1, args.render_views).astype(int).tolist()
    raw_op = model.gaussians["opacities"]
    orig = raw_op.detach().clone()
    NEG = -1e4
    cand_dev = candidate.to(device)
    with torch.no_grad():
        for vi in rview:
            cam = train_set.cameras[vi].to_device(device)
            raw_op.copy_(orig)  # full
            save_png(renderer(cam, model, bg_color=bg)["render"], os.path.join(args.out, f"v{vi}_a_full.png"))
            raw_op.copy_(orig); raw_op[cand_dev] = NEG  # candidates removed
            save_png(renderer(cam, model, bg_color=bg)["render"], os.path.join(args.out, f"v{vi}_b_removed.png"))
            raw_op.copy_(orig); raw_op[~cand_dev] = NEG  # candidates only
            save_png(renderer(cam, model, bg_color=bg)["render"], os.path.join(args.out, f"v{vi}_c_only.png"))
        raw_op.copy_(orig)
    print("[done] inspect v*_c_only.png — if it shows floating blobs (not roofs/walls), candidates are real floaters.")
    print("=================================================================")


if __name__ == "__main__":
    main()
