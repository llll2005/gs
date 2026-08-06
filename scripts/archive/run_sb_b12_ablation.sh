#!/bin/bash
# b12 SAME-KERNEL count ablation on trim+SB (answers the 2026-07-17 confound:
# is "more points don't help b12" a gsplat artifact, or real content ceiling?).
#
# Anchor: mcmc_60k_sh3_aggr17_b12 (trim SH3, cap 1.0M, screenprune300) = 22.45.
# Arm A: trim+SB, cap 1.0M   -> isolates the SB color effect at matched count.
# Arm B: trim+SB, cap CAP_B  -> count ablation within one kernel (SB's F=25
#         cuts model state 944->400 B/pt, buying count headroom the SH3 run
#         never had). CAP_B from tools/calibrate_block_caps.py caps table.
#
# Run serially; each ~10-14h on the 4050. Usage:
#   bash scripts/run_sb_b12_ablation.sh <CAP_B>   # e.g. 2000000
set -u
CAP_B=${1:?need CAP_B (formula cap for arm B, e.g. 2000000)}
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

run_arm () {
  local name=$1 cap=$2
  echo "[sb-ablation] $name cap=$cap start $(date +%F_%T)" >> $PROG
  rm -rf "outputs/$name"
  conda run -n gspl python -u main.py fit \
    --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
    --model.initialize_from $PLY \
    --data.parser.block_id 12 \
    --model.density.init_args.screen_size_prune_px 300 \
    --model.density.init_args.cap_max "$cap" \
    -n "$name" > "logs/$name.log" 2>&1
  local rc=$?
  local psnr
  psnr=$(grep -oE 'val/psnr: [0-9.]+' "outputs/$name/blocks/block_12/results.txt" 2>/dev/null | head -1)
  echo "[sb-ablation] $name done rc=$rc $psnr $(date +%F_%T)" >> $PROG
}

run_arm mcmc_sb_60k_aggr17_b12_cap1m 1000000
run_arm mcmc_sb_60k_aggr17_b12_capfx "$CAP_B"
