#!/bin/bash
# Wait for the b12 cap-ladder to finish (GPU frees), then run the tiling feasibility
# verification on b12's pathological geometry. Logs to quad_progress.log for the Monitor.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log

# 1. wait for cap-ladder terminal state
until grep -qE "ca-b12-cap.*(DONE cap|ALL CAPS FAILED)" $PROG 2>/dev/null; do
  sleep 60
done
# ensure the training process is really gone (GPU free) before we grab the GPU
while pgrep -f "train.py -s.*probe_b12" >/dev/null; do sleep 20; done
sleep 10

echo "[tiling] verify start $(date +%F_%T)" >> $PROG
python probes/p5_cost_aware/verify_tiling.py \
  --block_id 12 \
  --init_ply data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --Ks 1 2 4 8 \
  --out outputs/verify_tiling_b12.txt > logs/verify_tiling_b12.log 2>&1
rc=$?
echo "[tiling] verify done rc=${rc} $(date +%F_%T)" >> $PROG
