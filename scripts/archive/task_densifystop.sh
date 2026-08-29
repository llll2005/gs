#!/bin/bash
# Which half of the churn cost 1.3-1.9 dB: densification, or the contribution trim?
#
# Every 60k run gains more in the ONE 5,680-step window straddling densify_until_iter=42,000 than
# in the whole 34,000-step growth phase before it (+1.34/+1.42/+1.90 vs +1.26/+1.29/+0.88). Step
# 42,000 stops BOTH densification and the trim, because the trim is gated on densify_until_iter.
#
# trimstop_b12 tested the trim half and came back null: with the trim off from 25,000 it tracked
# its control within the noise floor (20.80/20.94/20.98/21.11 vs 20.86/20.88/21.05/21.02) for the
# 2,600 steps it survived. It then OOM'd at N=4.00M -- without the 10%-per-500-steps cull the count
# ran straight into cap_max, where cap4m had been held at 3.78M. That was a design error of mine:
# removing a brake without lowering the ceiling.
#
# So this is the other half: densification stops at 25,000, the trim keeps running to 42,000. It
# cannot OOM for the same reason -- densify stops early, so the count never approaches the cap.
#   big gain from 25,000 -> densification churn (relocation moving optimised primitives, SGLD
#                           noise on positions) is what suppresses quality
#   null                 -> neither half alone explains the jump; something about the two together
# ⚠ Confound worth stating: stopping densify early also caps the final count (~2M vs 3.6M), so a
# loss of ~0.2 dB from count alone is expected and must be subtracted before reading the result.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/densifystop_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.densify_until_iter 25000 \
  --model.renderer.init_args.contribution_prune_until_iter 42000 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n densifystop_b12
