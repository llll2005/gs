#!/bin/bash
# T1: does tiling (K) let b12 escape its ~500k count limit in real TRAINING?
# Control : b12 K=1 cap 2M -> expect OOM in the densify-growth regime (b7 K=1 died at 1.7M).
# Main    : b12 K=4 cap 2M -> expect survive + hold ~2M + quality >> the count-limited 21.83@500k.
# cost_aware OFF (isolate K's payoff); no near_plane (K handles the per-iter spike now).
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

# --- control: K=1 (expect OOM) ---
echo "[t1] K1 cap2M control start $(date +%F_%T)" >> $PROG
rm -rf outputs/t1_b12_K1
timeout 3600 python probes/p5_cost_aware/train_p5.py --block_id 12 --init_ply $PLY \
  --max_steps 15000 --cap_max 2000000 --refine_start 500 --refine_every 150 \
  --val_every 5000 --K_strips 1 --out outputs/t1_b12_K1 > logs/t1_b12_K1.log 2>&1
echo "[t1] K1 done rc=$? $(date +%F_%T)" >> $PROG

# --- main: K=4 (expect survive at high count) ---
echo "[t1] K4 cap2M main start $(date +%F_%T)" >> $PROG
rm -rf outputs/t1_b12_K4
python probes/p5_cost_aware/train_p5.py --block_id 12 --init_ply $PLY \
  --max_steps 15000 --cap_max 2000000 --refine_start 500 --refine_every 150 \
  --val_every 5000 --K_strips 4 --out outputs/t1_b12_K4 > logs/t1_b12_K4.log 2>&1
rc=$?
psnr=$(grep -oE "VAL step 15000: psnr [0-9.]+" logs/t1_b12_K4.log | tail -1)
echo "[t1] K4 done rc=${rc} ${psnr} $(date +%F_%T)" >> $PROG
echo "[t1] ALL DONE $(date +%F_%T)" >> $PROG
