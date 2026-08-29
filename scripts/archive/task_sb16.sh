#!/bin/bash
# The other end of the lobe axis: 16 lobes, which costs MORE per primitive than SH3 does.
#
#   n= 2  F= 25  1378 B/pt  4.36M
#   n= 4  F= 37  1570 B/pt  3.83M     <- task_sb4.sh, buys density
#   n=16  F=109  2722 B/pt  2.21M     <- this run, buys expressiveness and PAYS in count
#   SH3   F= 59  1922 B/pt  2.34M     <- current best (noprior 26.083)
#
# Run together with sb4 these separate two things that a single arm confounds:
#   sb4  wins, sb16 loses   -> count dominates; stop adding lobes, spend bytes on primitives
#   sb16 wins               -> angular expressiveness dominates COUNT, since sb16 has fewer
#                              primitives than SH3 and still won. That is the result that would
#                              justify building adaptive per-primitive lobe allocation.
#   both lose to SH3        -> SH is simply the better basis here and the lobe axis is closed
#
# FREE FOLLOW-UP, no GPU: whatever these score, the checkpoints answer "how many lobes does a
# primitive actually use" -- read the lobe amplitudes and look at the distribution. If most
# primitives carry one significant lobe and only water/glass carry several, that is the measurement
# that motivates variable-length storage (and tells us the exact saving) before writing any CUDA.
#
# ⚠ Colours are evaluated in Python and handed over as colors_precomp, so any lobe count runs
#   without touching the rasteriser -- but 2.21M x 16 lobe evaluations per camera per step is real
#   work. Expect it to be slower per step than SH3; report it/s alongside the metrics.
# ⚠ sb_lobe_init defaults to "uniform_angle", which over-samples the poles 2.5x. With 16 lobes that
#   distortion is at its worst. If this arm underperforms, retry with uniform_sphere before
#   concluding anything about lobe count.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sb16_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.gaussian.init_args.sb_number 16 \
  --model.density.init_args.cap_max 2200000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sb16_b12
