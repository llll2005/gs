#!/bin/bash
# Kernel probes P1/P2/P2.5: beta / triangle / convex splatting official repos on the
# exported block_7 scene (data/probe_b7, 1600x900, val_names.txt protocol patch).
# Each: train 30k (repo defaults) -> render -> metrics. Numbers land in each repo's
# output dir; comparison table assembled afterwards.
# NOTE: no `set -u` — conda's (de)activate scripts reference unbound vars and die under it
cd /home/LnoArch/Projects/專題 || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SCENE=/home/LnoArch/Projects/專題/CityGaussian/data/probe_b7
LOGD=/home/LnoArch/Projects/專題/CityGaussian/logs
PROG=$LOGD/quad_progress.log

run_one() {
  local name=$1 dir=$2 extra_env=$3 extra_args=$4
  echo "[probe] $name start $(date +%F_%T)" >> $PROG
  (cd $dir && env $extra_env python train.py -s $SCENE --eval -r 1 $extra_args \
      -m output/probe_b7 > $LOGD/probe_${name}_train.log 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    (cd $dir && env $extra_env python render.py -m output/probe_b7 --skip_train \
        > $LOGD/probe_${name}_render.log 2>&1 \
      && env $extra_env python metrics.py -m output/probe_b7 \
        > $LOGD/probe_${name}_metrics.log 2>&1)
    rc=$?
  fi
  echo "[probe] $name done rc=${rc} $(date +%F_%T)" >> $PROG
}

run_one beta     beta-splatting     "PYTHONPATH=/home/LnoArch/Projects/專題/beta-splatting/_pylibs" ""
run_one triangle triangle-splatting "IGNORE=1" "--outdoor --max_shapes 1000000"  # default 4M OOMs 6GB
run_one convex   convex-splatting   "IGNORE=1" "--outdoor --light"  # full version OOMs 6GB at ~600 steps
echo "[probe] ALL DONE $(date +%F_%T)" >> $PROG
