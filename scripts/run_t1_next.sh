#!/bin/bash
# After K4 finishes: decide if K8 is needed, then run the λ+K dual-integration test.
# Decision rule:
#   K4 "sufficient" = it survived to step ~15k (log has "VAL step 15000" or "step 14500").
#     -> skip K8, use K=4 for the λ+K run.
#   K4 OOM'd (no VAL 15000) -> run K8 first (cap 2M, K=8); use K=8 for λ+K.
# λ+K run = the unified formula's first autonomous run:
#   cost_aware ON (λ dynamic budget) + K_strips (tiling) + NO binding hard cap (safety 4M).
#   λ self-sizes the count; K bounds the per-iteration spike. b12 = the pathological block.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

# 1. wait for K4 to finish
until grep -qE "\[t1\] K4 done|\[t1\] ALL DONE" $PROG 2>/dev/null; do sleep 60; done
while pgrep -f "K_strips 4" >/dev/null; do sleep 20; done
sleep 8

# 2. decide K
if grep -qE "VAL step 15000|step 14500 " logs/t1_b12_K4.log 2>/dev/null; then
  KUSE=4
  echo "[t1-next] K4 sufficient (survived to 15k) -> skip K8, λ+K with K=4 $(date +%F_%T)" >> $PROG
else
  echo "[t1-next] K4 insufficient (OOM before 15k) -> running K8 first $(date +%F_%T)" >> $PROG
  rm -rf outputs/t1_b12_K8
  python probes/p5_cost_aware/train_p5.py --block_id 12 --init_ply $PLY \
    --max_steps 15000 --cap_max 2000000 --refine_start 500 --refine_every 150 \
    --val_every 5000 --K_strips 8 --out outputs/t1_b12_K8 > logs/t1_b12_K8.log 2>&1
  rc=$?
  psnr=$(grep -oE "VAL step 15000: psnr [0-9.]+" logs/t1_b12_K8.log | tail -1)
  echo "[t1-next] K8 done rc=${rc} ${psnr} $(date +%F_%T)" >> $PROG
  KUSE=8
fi

# 3. λ+K dual-integration run (the formula's first autonomous run: no binding hard cap)
echo "[t1-next] λ+K start (cost_aware on, K=${KUSE}, safety cap 4M, budget 1.5e7) $(date +%F_%T)" >> $PROG
rm -rf outputs/t1_b12_lambdaK
python probes/p5_cost_aware/train_p5.py --block_id 12 --init_ply $PLY \
  --max_steps 15000 --cap_max 4000000 --refine_start 500 --refine_every 150 \
  --val_every 5000 --cost_aware on --intersection_budget 1.5e7 --K_strips ${KUSE} \
  --out outputs/t1_b12_lambdaK > logs/t1_b12_lambdaK.log 2>&1
rc=$?
psnr=$(grep -oE "VAL step 15000: psnr [0-9.]+" logs/t1_b12_lambdaK.log | tail -1)
final_n=$(grep -oE "n [0-9,]+ peakVRAM" logs/t1_b12_lambdaK.log | tail -1)
echo "[t1-next] λ+K done rc=${rc} ${psnr} ${final_n} $(date +%F_%T)" >> $PROG
echo "[t1-next] ALL DONE $(date +%F_%T)" >> $PROG
