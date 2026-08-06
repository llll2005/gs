#!/bin/bash
# b12+b13 uniform rescue: FULL depth-init + screen_size_prune_px=300 (single documented
# deviation). See 紀錄/SF_MCMC_實驗計畫_2026-07-04.md b12/b13 診斷:
#   - subsampled init failed (death 3151 -> 3751 only): the step-1000 trim renormalizes
#     the population to a content-determined survivor set, so init count is a dead lever.
#   - measured killer = near-camera monsters: block cameras fly z 1.5-5.0 through content
#     airspace (nearest point-camera distance 0.017; 106k pairs < 1.0); radius ∝ scale/depth
#     -> single splat covers the frame -> rasterizer buffers (∝ intersections) exceed 6GB.
#     Mid-opacity monsters survive BOTH relocation (op > min_opacity) and contribution trim
#     (transmittance > 0) — MCMC lacks vanilla ADC's max_screen_size prune.
#   - fix = screen_size_prune_px 300: at each densify event, Gaussians whose max screen
#     radius over the interval exceeds 300px are recycled via MCMC relocation.
# b12 runs first (hardest block = the real test; a mechanism bug shows at the first densify
# event ~10 min in; flag failure shows at the old death zone ~1h in).
# Fallback if b12 still OOMs: threshold 150 + cap_max 1.4M (record deviation).
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mv logs/mcmc_60k_sh3_aggr17_b12.log logs/mcmc_60k_sh3_aggr17_b12.log.failed_sub05_oom3751 2>/dev/null
rm -rf outputs/mcmc_60k_sh3_aggr17_b12

for X in 12 13; do
  echo "[quad] block_${X} SCREENPRUNE start $(date +%F_%T)" >> logs/quad_progress.log
  conda run -n gspl python -u main.py fit \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
    --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_${X}.ply \
    --data.parser.block_id ${X} \
    --model.density.init_args.screen_size_prune_px 300 \
    -n mcmc_60k_sh3_aggr17_b${X} > logs/mcmc_60k_sh3_aggr17_b${X}.log 2>&1
  rc=$?
  psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_60k_sh3_aggr17_b${X}/blocks/block_${X}/results.txt 2>/dev/null | head -1)
  echo "[quad] block_${X} SCREENPRUNE done rc=${rc} ${psnr} $(date +%F_%T)" >> logs/quad_progress.log
done
echo "[quad] SCREENPRUNE ALL DONE $(date +%F_%T)" >> logs/quad_progress.log
