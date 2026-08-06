#!/bin/bash
# Quad merge-validation runs: blocks 8, 12, 13 with base aggr17 recipe (sequential).
# block_7 slot = existing outputs/mcmc_60k_sh3_aggr17_b7 (24.30/0.755/0.309 @ 1.53M).
# Decision per 紀錄/SF_MCMC_實驗計畫_2026-07-04.md: arm1 tied arm0 (< +0.3) -> quad uses base config.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
# fragmentation-OOM fix (torch 2.0.1, see 紀錄/開發日誌_2026-06-30 §4): block_8 died at
# step ~13.9k with reserved 4.75G >> allocated 1.72G without this.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
for X in 8 12 13; do
  echo "[quad] block_${X} start $(date +%F_%T)" >> logs/quad_progress.log
  conda run -n gspl python -u main.py fit \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
    --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_${X}.ply \
    --data.parser.block_id ${X} \
    -n mcmc_60k_sh3_aggr17_b${X} > logs/mcmc_60k_sh3_aggr17_b${X}.log 2>&1
  rc=$?
  psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_60k_sh3_aggr17_b${X}/blocks/block_${X}/results.txt 2>/dev/null | head -1)
  echo "[quad] block_${X} done rc=${rc} ${psnr} $(date +%F_%T)" >> logs/quad_progress.log
done
echo "[quad] ALL DONE $(date +%F_%T)" >> logs/quad_progress.log
