#!/bin/bash
# Quad recovery after Hyprland crash (2026-07-05 19:33) + reboot (21:27):
#   b8: resume from step 41999 ckpt (died at 92% via session SIGKILL, val 23.46 @54.5k)
#   b12/b13: full reruns (b12's giant-alloc OOM was dirty-GPU aftermath, PLY is small)
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "[quad] b8 RESUME from 41999 $(date +%F_%T)" >> logs/quad_progress.log
conda run -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_8.ply \
  --data.parser.block_id 8 \
  -n mcmc_60k_sh3_aggr17_b8 \
  --ckpt_path outputs/mcmc_60k_sh3_aggr17_b8/blocks/block_8/checkpoints/epoch=185-step=41999.ckpt \
  > logs/mcmc_60k_sh3_aggr17_b8_resume.log 2>&1
rc=$?
psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_60k_sh3_aggr17_b8/blocks/block_8/results.txt 2>/dev/null | head -1)
echo "[quad] block_8 resume done rc=${rc} ${psnr} $(date +%F_%T)" >> logs/quad_progress.log

for X in 12 13; do
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
