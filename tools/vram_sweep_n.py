"""Find the N at which a full training step stops fitting, by cloning an existing model.

The b12 ceiling question ("does the rasterizer change let us past ~1.5M?") is a memory question,
not a training question, so there is no reason to spend an hour growing a model to get there.
Clone the primitives of a real checkpoint up to a target N -- which preserves the opacity/scale
distribution that decides how much binning work a frame costs -- and run forward+backward+
optimizer at each N until it OOMs.

Both behaviours come out of ONE build. With EXACT_SUPPORT dropping o <= 1/255 from binning,
raising every opacity to just above 1/255 restores the old nothing-dropped behaviour, so
--no_drop reproduces the pre-change rasterizer without a rebuild.

Caveat: cloned geometry is not the geometry a real 2M run would converge to (a real run carries
more fog and smaller scales), so read this as a ceiling estimate, not a substitute for the run.
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "probes/p5_cost_aware"))
from internal.utils.gaussian_model_loader import GaussianModelLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--targets", default="1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--no_drop", action="store_true",
                    help="clamp o above 1/255 so EXACT_SUPPORT's cull cannot fire (= old behaviour)")
    args = ap.parse_args()

    dev = "cuda"
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        args.ckpt, dev, eval_mode=False)
    from train_p5 import build_sets
    ts, _ = build_sets("configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml",
                       "data/matrix_city/aerial/train/block_all", 12, "/tmp/claude-1000/_vs")

    base = {k: v.detach().clone() for k, v in model.gaussians.items()}
    n0 = base["means"].shape[0]
    # GaussianModelLoader hands back a PRE-ACTIVATED model: gaussians["opacities"] already
    # holds sigmoid(logit), so applying sigmoid again would map everything into [0.5, 0.73] and
    # report that nothing sits under 1/255. That mistake made the first run of this sweep
    # compare the new rasterizer against itself.
    o = model.get_opacities().detach().float().squeeze(-1)
    print(f"[base] N={n0:,}  o<=1/255 佔 {100 * (o <= 1 / 255).float().mean():.1f}%  no_drop={args.no_drop}")

    floor_o = 1 / 255 + 1e-5   # activated scale, matching the stored representation

    for tgt in [int(float(x) * 1e6) for x in args.targets.split(",")]:
        rep = math.ceil(tgt / n0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for k, v in base.items():
                big = v.repeat(*([rep] + [1] * (v.dim() - 1)))[:tgt].contiguous()
                if k == "means":  # jitter clones so they are not exactly co-located
                    big = big + torch.randn_like(big) * 1e-4
                if args.no_drop and k == "opacities":
                    big = big.clamp_min(floor_o)
                model.gaussians[k] = torch.nn.Parameter(big, requires_grad=True)
            opt = torch.optim.Adam(list(model.gaussians.values()), lr=1e-3)
            bg = torch.zeros(3, device=dev)
            peak = 0
            for i in range(args.views):
                cam = ts.cameras[i * (len(ts.cameras) // args.views)].to_device(dev)
                out = renderer(cam, model, bg_color=bg)
                out["render"].mean().backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                peak = max(peak, torch.cuda.max_memory_allocated())
            print(f"  N={tgt / 1e6:.1f}M  peak={peak / 2 ** 30:.2f} GiB   ✅ 存活")
        except torch.cuda.OutOfMemoryError as e:
            want = str(e).split("Tried to allocate ")[-1].split(";")[0] if "Tried to allocate" in str(e) else "?"
            print(f"  N={tgt / 1e6:.1f}M  ❌ OOM (要 {want})")
            break
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
