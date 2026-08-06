#!/bin/bash
# Self-healing cost-aware b12, tier 2: CAP-reduction ladder.
# Diagnosis (2026-07-12): near_plane 0.2 died @Iter3800, 0.5 @6302 — BOTH at the moment
# count filled cap=1M ("Now 1000000"). So the killer is the aggregate isect_tiles load at
# 1M gaussians on b12's overdraw geometry, NOT a single monster. gsplat's rasterizer memory
# model is heavier than the CityGS trim (which survived b12 at cap 1M). Fix = lower cap.
# Keep near_plane=0.5 (demonstrably delays death = cheap insurance) + screen_recycle.
# eval uses BETA_NEAR_PLANE env so a monster-bearing ckpt doesn't OOM at eval.
cd /home/LnoArch/Projects/專題/beta-splatting || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=/home/LnoArch/Projects/專題/beta-splatting/_pylibs
DATA=/home/LnoArch/Projects/專題/CityGaussian/data/probe_b12
LOGD=/home/LnoArch/Projects/專題/CityGaussian/logs
PROG=$LOGD/quad_progress.log

for CAP in 500000 300000 200000; do
  out=output/ca_b12_cap${CAP}
  rm -rf $out
  echo "[ca-b12-cap] cap=$CAP start $(date +%F_%T)" >> $PROG
  python train.py -s $DATA --eval -r 1 --data_device cpu --iterations 30000 \
    --save_iterations 30000 --cap_max $CAP --screen_recycle_px 300 \
    --render_near_plane 0.5 -m $out > $LOGD/ca_b12_cap${CAP}.log 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    if [ -d $out/point_cloud/iteration_30000 ]; then
      sleep 15; kill $TP 2>/dev/null
      echo "[ca-b12-cap] cap=$CAP SURVIVED to 30k $(date +%F_%T)" >> $PROG
      break
    fi
    sleep 30
  done
  wait $TP 2>/dev/null
  if [ -d $out/point_cloud/iteration_30000 ]; then
    BETA_NEAR_PLANE=0.5 python eval.py -s $DATA -m $out --data_device cpu --iteration 30000 \
      > $LOGD/ca_b12_cap${CAP}_eval.log 2>&1
    res=$(grep -oE "\{'SSIM.*\}" $LOGD/ca_b12_cap${CAP}_eval.log | tail -1)
    echo "[ca-b12-cap] DONE cap=$CAP ${res:-EVAL_FAILED} $(date +%F_%T)" >> $PROG
    exit 0
  fi
  lastiter=$(grep -oE "Iter=[0-9]+" $LOGD/ca_b12_cap${CAP}.log | tail -1)
  echo "[ca-b12-cap] cap=$CAP DIED ${lastiter}, reducing cap $(date +%F_%T)" >> $PROG
done
echo "[ca-b12-cap] ALL CAPS FAILED — b12 may need <200k or a lighter rasterizer $(date +%F_%T)" >> $PROG
