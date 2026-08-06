#!/bin/bash
# Beta deep-dive: (1) b12 acid test (pathological block; VRAM behavior decides
# 25-block base), (2) ablation A: --sb_number 0 (SB color contribution),
# (3) ablation B: --beta_lr 0 (kernel frozen at init => learnable-kernel contribution).
# All: cap 1M, --data_device cpu, eval at iteration_30000 (step-matched yardstick).
cd /home/LnoArch/Projects/專題/beta-splatting || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=/home/LnoArch/Projects/專題/beta-splatting/_pylibs
LOGD=/home/LnoArch/Projects/專題/CityGaussian/logs
PROG=$LOGD/quad_progress.log
DATA=/home/LnoArch/Projects/專題/CityGaussian/data

run_arm() {
  local name=$1 scene=$2 extra=$3
  echo "[beta-dd] $name start $(date +%F_%T)" >> $PROG
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done \
      > $LOGD/betadd_${name}_vram.log 2>/dev/null & echo $! > /tmp/vram_pid )
  python train.py -s $DATA/$scene --eval -r 1 --data_device cpu $extra \
      -m output/dd_${name} > $LOGD/betadd_${name}_train.log 2>&1
  rc=$?
  kill $(cat /tmp/vram_pid) 2>/dev/null
  peak=$(sort -n $LOGD/betadd_${name}_vram.log 2>/dev/null | tail -1)
  if [ $rc -eq 0 ]; then
    python eval.py -s $DATA/$scene -m output/dd_${name} --data_device cpu --iteration 30000 \
        > $LOGD/betadd_${name}_eval.log 2>&1
    rc=$?
  fi
  res=$(grep -oE "\{.*\}" $LOGD/betadd_${name}_eval.log 2>/dev/null | tail -1)
  echo "[beta-dd] $name done rc=${rc} peakVRAM=${peak}MiB ${res} $(date +%F_%T)" >> $PROG
}

run_arm b12acid   probe_b12 ""
run_arm b7_noSB   probe_b7  "--sb_number 0"
run_arm b7_gaussK probe_b7  "--beta_lr 0"
echo "[beta-dd] ALL DONE $(date +%F_%T)" >> $PROG
