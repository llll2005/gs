"""Is the geometry a real surface, or a shell of floaters hanging in front of the cameras?

PSNR cannot answer this here: our val split is a subset of train, so a shell that reproduces the
training views scores well while looking like fog from anywhere else. This audit compares the
model's height distribution against the SfM sparse points (the only geometry we know is real) and
counts how much of the OPAQUE mass sits above the surface.

Reference numbers on b12 (2026-07-31):
  depth-init PLY   z median 0.26   (matches SfM's 0.26 -- initialisation is correct)
  A' reg=0.007     z median 0.43   opaque-and-airborne  1.0% of all primitives
  reg000 reg=0     z median 1.01   opaque-and-airborne 78.7%   <- the fog seen in the viewer
"""
import argparse, glob, re, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--block", type=int, default=12)
ap.add_argument("--surface_z", type=float, default=0.26, help="SfM median z for this scene")
ap.add_argument("--topdown", action="store_true",
                help="also render a raised top-down view -- a NOVEL view, which is where a shell "
                     "of floaters shows up. The training views cannot reveal it and neither can "
                     "PSNR, because our val split is a subset of train.")
ap.add_argument("--height_mult", type=float, default=3.0, help="raise the camera this many times its height")
args = ap.parse_args()

d = f"outputs/{args.run}/blocks/block_{args.block}/checkpoints"
g = [(int(re.search(r'step=(\d+)', x).group(1)), x) for x in glob.glob(d + "/*.ckpt")]
if not g:
    print(f"[audit] 找不到 ckpt: {d}"); sys.exit(1)
sd = torch.load(sorted(g)[-1][1], map_location="cpu")["state_dict"]
m = sd["gaussian_model.gaussians.means"].float()
o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float().squeeze(-1))
n = m.shape[0]
air = m[:, 2] > args.surface_z + 0.25          # comfortably above the surface
opaque_air = air & (o > 0.5)
print(f"[audit] {args.run} block_{args.block}  N={n:,}")
print(f"  z: P10={m[:,2].quantile(.1):.2f} P50={m[:,2].median():.2f} P90={m[:,2].quantile(.9):.2f}  (SfM 地表 {args.surface_z})")
print(f"  懸空(z>{args.surface_z+0.25:.2f}): {100*air.float().mean():.1f}%")
print(f"  ★懸空且不透明(o>0.5): {100*opaque_air.float().mean():.1f}%   [A' 1.0% / reg000 78.7%]")
print(f"  opacity: P50={o.median():.3f}  o>0.9 佔 {100*(o>0.9).float().mean():.1f}%")


if args.topdown:
    # Reuse a real camera's orientation (aerial cameras already look down) and just lift it, so the
    # rotation convention cannot be got wrong. Raising it puts the viewpoint outside the training
    # distribution, which is the whole point: a shell that satisfies the training views turns into
    # visible fog from anywhere else.
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), "probes/p5_cost_aware"))
    from internal.utils.gaussian_model_loader import GaussianModelLoader
    from train_p5 import build_sets
    import torchvision

    dev = "cuda"
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        sorted(g)[-1][1], dev, eval_mode=False)
    ts, _ = build_sets("configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml",
                       "data/matrix_city/aerial/train/block_all", args.block, "/tmp/claude-1000/_ag")
    cam = ts.cameras[len(ts.cameras) // 2].to_device(dev)
    centre = -(cam.R.T @ cam.T)                       # world position of that camera
    lifted = centre.clone(); lifted[2] = centre[2] * args.height_mult
    cam.T = -(cam.R @ lifted)                          # same orientation, higher up
    out = f"outputs/{args.run}/topdown_x{args.height_mult:g}.png"
    with torch.no_grad():
        img = renderer(cam, model, bg_color=torch.zeros(3, device=dev))["render"].clamp(0, 1)
    torchvision.utils.save_image(img, out)
    print(f"  俯瞰圖(高度×{args.height_mult:g}，訓練分佈外): {out}")
