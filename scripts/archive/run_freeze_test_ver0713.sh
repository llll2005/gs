#!/bin/bash
# Closed-form budget FREEZE TEST: validate N_max = (V_target-V_os)/(M·F·4 + γτ/K).
# Formula (b12 calibration) predicts K=2 -> N_max = 2.50M. Test: grow to that cap with
# K=2, freeze, pure-optimize. Validates: (1) reaches 2.5M with NO OOM (formula's safety
# margin holds), (2) VRAM stays under target, (3) quality climbs past 21.83 (the count-
# limited 500k number) toward b7's regime.
# cost_aware ON but budget non-binding (1e9) -> λ stays 0 (no count pruning; cap controls
# count), keeps only the cost>2000-tile ceiling as monster insurance. K=2 bounds render.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

echo "[freeze] K8 cap2.5M start $(date +%F_%T)" >> $PROG
rm -rf outputs/freeze_b12_K8
python -u probes/p5_cost_aware/train_p5.py --block_id 12 --init_ply $PLY \
  --max_steps 10000 --cap_max 2500000 --refine_start 500 --refine_every 150 \
  --refine_stop_frac 0.6 --val_every 5000 --cost_aware on --intersection_budget 1e9 \
  --K_strips 8 --out outputs/freeze_b12_K2 > logs/freeze_b12_K2.log 2>&1
rc=$?
psnr=$(grep -oE "VAL step 10000: psnr [0-9.]+" logs/freeze_b12_K2.log | tail -1)
final=$(grep -oE "n [0-9,]+ peakVRAM [0-9.]+G" logs/freeze_b12_K2.log | tail -1)
echo "[freeze] K2 done rc=${rc} ${psnr} ${final} $(date +%F_%T)" >> $PROG
