#!/bin/bash
# The only lever today's data actually supports: spend steps, not primitives.
#
# cap4m_b12's own trajectory, split at densify_until_iter = 42,000:
#   growth 5,680 -> 39,760 : 313k -> 3.78M primitives (12x)  PSNR +1.26
#   refine 39,760 -> 60,000: count frozen at 3.6M            PSNR +2.19   <- 1.7x the gain
# So reach the cap sooner and hand the rest to refinement. interval 150 -> 100 puts the net growth
# at x1.148 per 500 steps, which reaches 4M around step 12,000; densify_until 25,000 leaves margin
# and gives 35,000 refinement steps instead of 18,000.
#
# Expected 24.8-25.2, from the measured refinement marginal (+0.100 -> +0.040 -> +0.016 dB/1000,
# already decelerating). This is NOT a route to 26 and is not presented as one.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=fastgrow_b12
rm -rf outputs/$NAME
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.densification_interval 100 \
  --model.density.init_args.densify_until_iter 25000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
