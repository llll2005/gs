#!/bin/bash
# Reach the cap sooner, then refine for far longer -- without giving up any primitives.
#
# Measured on cap4m_b12, split at densify_until_iter = 42,000:
#   growth 5,680->39,760 (34,080 steps, 313k -> 3.78M primitives)   PSNR +1.26
#   refine 39,760->60,000 (20,241 steps, count frozen)              PSNR +2.19
# Per step, refinement is 2.9x more productive. The model at 39,760 already had ~23.5 in it and
# the churn was masking it.
#
# interval 150 -> 100 roughly doubles the densification rate, so the cap is reached near step
# 12,000 instead of 39,000; densify_until 25,000 then hands 35,000 steps to refinement instead of
# 20,241 -- at the SAME final count. Both levers are kept, which is why this is preferred over
# densifystop_b12 (that arm stops densify early and pays ~0.2 dB in lost count, an amount
# comparable to the effect being measured, so neither outcome would be interpretable).
#
# Two parameters change, deliberately: they serve one hypothesis (shorten the churn window while
# keeping the count). The question is whether the schedule is better, not which knob did it.
# An earlier attempt was killed at step 6,000 before it could answer anything.
# ⚠ Watch VRAM: faster growth means more primitives earlier, and that attempt was at 5.67/6.1G by
# step 5,961. If it OOMs, the fix is a lower cap, not a slower interval.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/fastgrow_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.densification_interval 100 \
  --model.density.init_args.densify_until_iter 25000 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n fastgrow_b12
