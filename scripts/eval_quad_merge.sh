#!/bin/bash
# Evaluate merged.ckpt on each quad block's val set (same val split as per-block runs)
# -> per-block merge drop = merged val PSNR - per-block val PSNR.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
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
