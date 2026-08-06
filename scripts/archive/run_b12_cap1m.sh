#!/bin/bash
# block_12 attempt #5: screen_size_prune_px=300 + cap_max 1.0M.
# Attempt #4 (screenprune, cap 1.7M) survived 10x longer (death 3151 -> 29402) but
# died in the late-densify growth regime: 1.24M pts x center-block perspective =
# bulk load alone fills 5.66G; prune was recycling 50-74k monsters per event
# (production ~500/step) and still lost the volume war.
# cap 1.0M rationale: b12's own trim/add equilibrium sat at 0.67-0.9M for most of
# the run, so the cap only clips the final runaway; at 0.9M the run measured 5.30G.
# Launch ONLY after b13 finishes.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mv logs/mcmc_60k_sh3_aggr17_b12.log logs/mcmc_60k_sh3_aggr17_b12.log.failed_screenprune_oom29402 2>/dev/null
rm -rf outputs/mcmc_60k_sh3_aggr17_b12

echo "[quad] block_12 CAP1M start $(date +%F_%T)" >> logs/quad_progress.log
conda run -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 1000000 \
  -n mcmc_60k_sh3_aggr17_b12 > logs/mcmc_60k_sh3_aggr17_b12.log 2>&1
rc=$?
psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_60k_sh3_aggr17_b12/blocks/block_12/results.txt 2>/dev/null | head -1)
echo "[quad] block_12 CAP1M done rc=${rc} ${psnr} $(date +%F_%T)" >> logs/quad_progress.log
