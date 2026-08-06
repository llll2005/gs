#!/bin/bash
# Evaluate merged.ckpt on each quad block's val set (same val split as per-block runs)
# -> per-block merge drop = merged val PSNR - per-block val PSNR.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
for X in 7 8 12 13; do
  echo "[merge-eval] block_${X} start $(date +%F_%T)" >> logs/quad_progress.log
  conda run -n gspl python -u main.py validate \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
    --ckpt_path outputs/quad_merge_aggr17/checkpoints/merged.ckpt \
    --data.parser.block_id ${X} \
    -n quad_merge_eval_b${X} > logs/quad_merge_eval_b${X}.log 2>&1
  rc=$?
  echo "[merge-eval] block_${X} done rc=${rc} $(date +%F_%T)" >> logs/quad_progress.log
done
echo "[merge-eval] ALL DONE $(date +%F_%T)" >> logs/quad_progress.log
