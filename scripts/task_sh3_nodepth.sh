#!/bin/bash
# opacity_reg 0.002 -> 0.008 on the SH3 recipe. Single variable against sh3_viewdep_b12 (24.672 /
# 0.7025 / 0.4500 / 0.3921), which is currently the best model on all four metrics.
#
# Two things it answers at once:
#  1. MECHANISM. The low-frequency contrast compression carries 45% of the residual energy and is
#     immune to count (4x), init, LR floor, blur split and colour model -- everything tried. The one
#     variable held constant across all of those runs is opacity_reg = 0.002. Lower opacity means
#     more semi-transparent layers per pixel, and a weighted average of more layers is exactly what
#     pulls large regions toward the mean. If that is the cause, 0.008 should make it WORSE, and
#     measurably so on tools/attribute_lowfreq_error.py.
#  2. RECIPE. 0.002 was tuned on the SB colour model at 900k (0.741 dB over reg=0). It has never
#     been retuned for SH3, whose own config ships 0.007, and the optimum may have moved with the
#     colour model.
# ⚠ Prior: on SB at ~900k, reg 0.007 scored 23.315 vs 0.002's 23.848 -- clearly worse. So a PSNR
#    loss here is expected and is not the readout; the contrast measurement is.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sh3_nodepth_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.depth_loss_weight.init 0.0
  --model.metric.init_args.opacity_reg 0.002 \
  -n sh3_nodepth_b12
