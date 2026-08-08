#!/bin/bash
# Which half of the churn is suppressing quality: densification, or the contribution trim?
#
# All three 60k runs gain more in the ONE 5,680-step window straddling densify_until_iter=42,000
# than in the entire 34,000-step growth phase before it:
#   window 39k->45k : cap4m +1.34   blursplit +1.42   lrfloor +1.90
#   growth  5k->39k : cap4m +1.26   blursplit +1.29   lrfloor +0.88
# Per step, refinement is 2.9x more productive. The model at 39,760 already had ~23.5 in it and
# churn was masking it.
#
# But step 42,000 stops TWO things, because after_training_step gates the contribution trim on the
# density controller's densify_until_iter. Densification adds 5% every 150 steps; the trim removes
# 10% every 500 steps -- including primitives that are already well optimised. The trim is the
# better suspect: adding primitives cannot undo optimisation, deleting them can.
#
# contribution_prune_until_iter (added 2026-08-06, -1 = follow densify_until_iter) separates them.
# Here the trim stops at 25,000 while densification runs to 42,000 as usual.
#   big early gain -> the trim was the suppressor, and the fix is to stop culling early
#   nothing        -> densification churn is, and relocation/SGLD is what disturbs optimisation
# Control is cap4m_b12 (24.307); nothing else changes.
#
# Expected side effect: without the 10% cull the count reaches cap sooner. That is a confound, but
# a favourable one -- if quality does NOT improve despite reaching cap earlier, the trim is
# exonerated all the more strongly.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=trimstop_b12
rm -rf outputs/$NAME
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.renderer.init_args.contribution_prune_until_iter 25000 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
