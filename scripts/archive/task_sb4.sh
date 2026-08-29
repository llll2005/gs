#!/bin/bash
# More lobes instead of more SH: SB with 4 lobes, byte-aligned cap, on the noprior recipe.
#
# The bottleneck is primitives per unit area (tools/diagnose_hard_views.py: PSNR correlates -0.643
# with GT gradient and -0.415 with camera altitude, while block-ownership correlates -0.063 and
# view support +0.125 -- the hard views demand more detail per pixel than our density supports).
# Per-block cap is VRAM-bound and splitting blocks finer costs linear time, so the free lever is
# BYTES PER PRIMITIVE.
#
#   F = 13 + 6n   (xyz3 + scale2 + rot4 + opacity1 + sh0 3)      bytes/pt = 4*F*4 + 978
#   SB n=2  F=25  1378 B/pt  3.63M      (measured: loses to SH3 by 0.19 dB)
#   SB n=4  F=37  1570 B/pt  3.57M      <- this run, +53% primitives over SH3
#   SB n=6  F=49  1762 B/pt  3.18M
#   SH3     F=59  1922 B/pt  2.34M      (current best: noprior 26.083)
#
# Why lobes rather than SH degree: SH2 -> SH3 costs 21 floats, SB 2 -> 4 lobes costs 12. And the
# view-dependent content here is water reflection -- a SHARP directional lobe, which is sparse in
# angle. SH needs many low-order bases to build a peak; an SB lobe IS a peak with a tunable width.
# So SB-2 losing to SH3 may mean "two lobes is too few", not "SB is the weaker representation" --
# and the lobe count has never been swept; 2 is the DBS default we inherited.
#
# ⚠ Confounded on purpose: colour model AND count both change, because equal-count is the wrong
#   comparison when the question is what a byte buys. Report both.
# ⚠ sb_lobe_init defaults to "uniform_angle", which over-samples the poles 2.5x. That mattered
#   little at 2 lobes; at 4 it may. If this arm underperforms, retry with uniform_sphere before
#   concluding anything about lobe count.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sb4_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.gaussian.init_args.sb_number 4 \
  --model.density.init_args.cap_max 3500000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sb4_b12
