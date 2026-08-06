#!/bin/bash
# Closed-form budget FREEZE TEST: validate N_max = (V_target-V_os)/(M·F·4 + γτ/K).
# Formula (b12 calibration) predicts K=2 -> N_max = 2.50M. Test: grow to that cap with
# K=2, freeze, pure-optimize. Validates: (1) reaches 2.5M with NO OOM (formula's safety
# margin holds), (2) VRAM stays under target, (3) quality climbs past 21.83 (the count-
# limited 500k number) toward b7's regime.
# cost_aware ON but budget non-binding (1e9) -> λ stays 0 (no count pruning; cap controls
# count), keeps only the cost>2000-tile ceiling as monster insurance. K=2 bounds render.
#!/bin/bash
# 終極預算解封測試：依賴動態天花板與 K=8 釋放 3M+ 容量，並輔以幾何約束推升 PSNR。

cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply
OUT_DIR=outputs/freeze_b12_K8_v5
LOG_FILE=logs/freeze_b12_K8_v5.log

echo "[freeze] K8-v5 Full Unbound + Geo start $(date +%F_%T)" >> $PROG
rm -rf $OUT_DIR

python -u probes/p5_cost_aware/train_p5.py \
  --block_id 12 \
  --init_ply $PLY \
  --max_steps 20000 \
  --cap_max 10000000 \
  --refine_start 500 \
  --refine_every 100 \
  --refine_stop_frac 0.95 \
  --val_every 1000 \
  --intersection_budget 1e9 \
  --cost_aware on \
  --lambda_mode vram \
  --vram_target_gb 5.7 \
  --depth_reg on \
  --lambda_normal 0.005 \
  --normal_from_iter 2000 \
  --K_strips 8 \
  --out $OUT_DIR > $LOG_FILE 2>&1

rc=${PIPESTATUS[0]}
psnr=$(grep -oE "VAL step 60000: psnr [0-9.]+" $LOG_FILE | tail -1)
final=$(grep -oE "n [0-9,]+ peakVRAM [0-9.]+G" $LOG_FILE | tail -1)
echo "[freeze] K8-v4-b12 done rc=${rc} ${psnr} ${final} $(date +%F_%T)" >> $PROG