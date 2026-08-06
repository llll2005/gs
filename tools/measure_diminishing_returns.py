"""Does removing primitives cost *accelerating* quality? (the diminishing-returns / submodularity check)

§8.3 of 紀錄/現行方案與公式.md assumes the rendering objective has diminishing returns, and both
the (1-1/e)-style guarantees and the CELF lazy-evaluation speedup rest on that assumption. It is
an assumption, not a theorem: the coverage function f(S) = Σ_p [1 - Π_{j∈S}(1-α_j)] is provably
monotone submodular, but rendered *quality* is not the same thing (a wrong-coloured primitive
makes the image worse, so Q is not even monotone).

So measure it. Remove the bottom k% of primitives and record the quality loss as a function of k:

  submodular / diminishing returns  ->  loss curve is CONVEX (each additional slice costs more)
  additive                          ->  loss curve is LINEAR
  supermodular                      ->  loss curve is CONCAVE (early cuts hurt most)

Two rankings are swept, and the gap between them is the second result:
  contribution : v_i = Σ_views Σ_pixels T·α, from the trim rasterizer's record_transmittance.
                 §8.2 shows this is a computable UPPER BOUND on the submodular coverage marginal
                 (T here multiplies only the occluders in front, the coverage marginal multiplies
                 all of them), and an upper bound is the safe direction -- it cannot under-rate a
                 primitive and get it culled by mistake.
  random       : control. If the contribution ranking is not far flatter than random, then v_i is
                 not informative and no amount of combinatorial machinery on top of it will help.

Inference only -- no training, no optimizer. Peak is ~1.75 GiB at 1M primitives, so do not run it
alongside a training job on a 6GB card.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "probes/p5_cost_aware"))
from internal.utils.gaussian_model_loader import GaussianModelLoader


def psnr(a, b):
    mse = torch.mean((a - b) ** 2).clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml")
    ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--rank_views", type=int, default=24, help="views used to rank by contribution")
    ap.add_argument("--eval_views", type=int, default=8, help="views used to score each prune level")
    ap.add_argument("--fracs", default="0,10,20,30,40,50,60,70,80,90")
    ap.add_argument("--holdout_eval", action="store_true",
                    help="score on views the ranking never saw; without this the ranking and the "
                         "evaluation share views, which flatters any ranking")
    ap.add_argument("--out", default="/tmp/claude-1000/diminishing_returns.csv")
    args = ap.parse_args()

    dev = "cuda"
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        args.ckpt, dev, eval_mode=False)
    from train_p5 import build_sets
    ts, _ = build_sets(args.config, args.data, args.block, "/tmp/claude-1000/_dr")
    cams = ts.cameras
    n0 = model.get_xyz.shape[0]
    print(f"[load] N={n0:,}  views={len(cams)}")

    # ---- rank: multi-view Σ T·α ------------------------------------------------
    bg = torch.zeros(3, device=dev)
    contrib = torch.zeros(n0, device=dev)
    all_idx = list(range(len(cams)))
    if args.holdout_eval:
        # every 6th view is withheld from ranking and used only for scoring
        eval_pool = all_idx[3::6]
        rank_pool = [i for i in all_idx if i not in set(eval_pool)]
    else:
        eval_pool, rank_pool = all_idx, all_idx
    step = max(1, len(rank_pool) // args.rank_views)
    rank_idx = rank_pool[::step]
    used = 0
    with torch.no_grad():
        for i in rank_idx:
            t = renderer(cams[i].to_device(dev), model, bg_color=bg, record_transmittance=True)
            contrib += (t if torch.is_tensor(t) else t["transmittance"]).float()
            used += 1
    print(f"[rank] contribution over {used}/{len(rank_pool)} rankable views"
          f"{' (eval views held out)' if args.holdout_eval else ''}: "
          f"zero={100 * float((contrib <= 0).float().mean()):.1f}%  "
          f"top5%/mean={float(contrib.topk(max(1, n0 // 20)).values.mean() / contrib.mean().clamp_min(1e-12)):.1f}")

    base = {k: v.detach().clone() for k, v in model.gaussians.items()}
    eval_idx = [eval_pool[i] for i in np.linspace(0, len(eval_pool) - 1, min(args.eval_views, len(eval_pool))).astype(int)]
    # Reference = the UNPRUNED model's own render, not the ground truth. The question here is the
    # SHAPE of the degradation curve (convex / linear / concave), and measuring against the full
    # model isolates exactly the damage pruning does, without dragging in the error the model
    # already had. It also avoids depending on how the dataset happens to hand back GT tensors.
    def render_set():
        outs = []
        with torch.no_grad():
            for i in eval_idx:
                outs.append(renderer(cams[i].to_device(dev), model, bg_color=bg)["render"]
                            .clamp(0, 1).detach().clone())
        return outs

    ref = render_set()
    print(f"[ref] 未剪模型的 {len(ref)} 張渲染已存為參考")

    order_contrib = torch.argsort(contrib)            # ascending: worst first
    g = torch.Generator(device="cpu").manual_seed(0)
    order_random = torch.randperm(n0, generator=g).to(dev)

    rows = []
    for name, order in [("contribution", order_contrib), ("random", order_random)]:
        for frac in [float(x) for x in args.fracs.split(",")]:
            k = int(n0 * frac / 100)
            keep = torch.ones(n0, dtype=torch.bool, device=dev)
            if k > 0:
                keep[order[:k]] = False
            for key, v in base.items():
                model.gaussians[key] = torch.nn.Parameter(v[keep].contiguous(), requires_grad=False)
            vals = []
            with torch.no_grad():
                for j, i in enumerate(eval_idx):
                    out = renderer(cams[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
                    vals.append(psnr(out, ref[j]))
            p = float(np.mean(vals)) if vals else float("nan")
            rows.append((name, frac, int(keep.sum()), p))
            print(f"  {name:12s} cut {frac:4.0f}%  N={int(keep.sum()):>9,}  vs未剪 PSNR={p:6.2f} dB")
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        f.write("ranking,cut_pct,n_kept,psnr\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f}\n")

    # curvature: convex loss => second difference of (loss vs cut%) is positive
    print("\n[curvature] 失真(MSE vs 未剪)的二階差分（>0 = 凸 = 邊際遞減 = 支持次模）")
    for name in ("contribution", "random"):
        sub = [r for r in rows if r[0] == name]
        # loss = MSE relative to the unpruned render, which is what accumulates additively if the
        # objective is additive; PSNR is a log of it and would bend the curve on its own.
        loss = [10 ** (-r[3] / 10) for r in sub]
        d2 = [loss[i + 1] - 2 * loss[i] + loss[i - 1] for i in range(1, len(loss) - 1)]
        pos = sum(1 for x in d2 if x > 0)
        print(f"  {name:12s} 二階差分 {pos}/{len(d2)} 為正  "
              f"(MSE@50%={loss[len(loss)//2]:.2e}, MSE@90%={loss[-1]:.2e})")
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
