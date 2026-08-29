#!/bin/bash
# SH3 view-dependence + blur split, at the count SH3's byte budget actually allows.
#
# sh3_viewdep_b12 won all four metrics with 35% FEWER primitives (2.34M vs 3.60M):
#   PSNR 24.672  SSIM 0.7025  LPIPS 0.4500  texratio 0.3921
# Correcting for the count handicap (-0.62 doublings) the true gains are PSNR +0.32,
# LPIPS -0.034 (15x noise), texratio +0.040 (11x noise). It broke the Pareto front that
# blurlr had established -- pixel accuracy and perceptual quality improved together, which
# no placement or schedule lever managed.
#
# blur split acts on WHERE primitives go, view-dependence on what they can express -- different
# axes, and unlike the LR floor it does not fight detail (it creates it). blurbudget's own gains
# over its control were LPIPS -0.0129 (5.9x noise) and texture ratio +0.0191 (5.2x).
#
# CAP IS 2.6M, MATCHING sh3_viewdep EXACTLY. The first attempt used cap 3.4M reasoning from the
# byte ceiling (3.13M) and OOM'd at 3.02M -- the ceiling was computed from SB's 978 B/pt render
# term, and blur split raises it by cloning large primitives. Matching the control's cap is worth
# more than the extra count anyway: sh3_viewdep landed at 2,339,999, so this is a genuine
# single-variable comparison against 24.672 / 0.7025 / 0.4500 / 0.3921. Running at 1.5M or 2M
# instead would move count AND mechanism, and the count correction (~0.1-0.2 dB) is the same size
# as the effect being measured.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sh3blur_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.blur_split_budget 0.3 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n sh3blur_b12
