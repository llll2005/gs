#!/bin/bash
# b12+b13 rerun with 50%-subsampled depth-init (documented deviation, see
# 紀錄/SF_MCMC_實驗計畫_2026-07-04.md b12/b13 診斷).
# Both center blocks died in the init-lineage-dominated window on genuine
# VRAM blowups of the same family (deep alpha-blending / intersection spikes
# during the early opacity-mush phase):
#   b12: gradual full OOM @step3151 (allocated 5.29G ~= reserved), 2x reproducible
#   b13: single 4.64GiB binningBuffer alloc @step7499 (intersection spike)
# Fix: halve init points -> halves sustained depth AND spike magnitude in the
# window; after ~10k the count is cap/add-driven so the recipe is untouched.
# Fallback if the regrowth window (10k-42k) still OOMs: cap_max 1.4M for that block.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

mv logs/mcmc_60k_sh3_aggr17_b12.log logs/mcmc_60k_sh3_aggr17_b12.log.failed_oom3151 2>/dev/null
rm -rf outputs/mcmc_60k_sh3_aggr17_b12

for X in 12 13; do
  echo "[quad] block_${X} SUB05 start $(date +%F_%T)" >> logs/quad_progress.log
  conda run -n gspl python -u main.py fit \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
    --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_${X}_sub05.ply \
    --data.parser.block_id ${X} \
    -n mcmc_60k_sh3_aggr17_b${X} > logs/mcmc_60k_sh3_aggr17_b${X}.log 2>&1
  rc=$?
  psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_60k_sh3_aggr17_b${X}/blocks/block_${X}/results.txt 2>/dev/null | head -1)
  echo "[quad] block_${X} SUB05 done rc=${rc} ${psnr} $(date +%F_%T)" >> logs/quad_progress.log
done
echo "[quad] SUB05 ALL DONE $(date +%F_%T)" >> logs/quad_progress.log
