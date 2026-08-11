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
# Two things it leaves on the table:
#  1. It was CAP-BOUND: 2,339,999 = 0.9 x cap 2.6M. The byte ceiling for F=59 is 3.13M
#     (944 B/pt state + 978 render), so cap 3.4M reaches ~3.06M and stays inside it.
#  2. blur split acts on WHERE primitives go, view-dependence on what they can express --
#     different axes, and unlike the LR floor it does not fight detail (it creates it).
#     blurbudget's own gains were LPIPS -0.0129 and texratio +0.0191 over its control.
# ⚠ If this OOMs the fix is cap, not the mechanism: 3.06M x 1922 B/pt = 5.6 GB is right at
#    the wall measured for SB (3.78M survived, 4.00M did not, at 1378 B/pt).
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sh3blur_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 3400000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.blur_split_budget 0.3 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n sh3blur_b12
