#!/bin/bash
# Self-healing cost-aware b12: near_plane fallback ladder. Tries increasingly
# aggressive near-camera culling until b12 survives past the death zone (~3400).
# Each rung: launch beta+cost-aware, watch-and-kill at iteration_30000. If it OOMs
# before 30k, escalate near_plane. This is the autonomous pattern — no per-iteration
# human diagnosis needed for THIS failure family (near-camera monster OOM).
cd /home/LnoArch/Projects/專題/beta-splatting || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=/home/LnoArch/Projects/專題/beta-splatting/_pylibs
DATA=/home/LnoArch/Projects/專題/CityGaussian/data/probe_b12
LOGD=/home/LnoArch/Projects/專題/CityGaussian/logs
PROG=$LOGD/quad_progress.log

for NP in 0.2 0.5 1.0; do
  out=output/ca_b12_np${NP}
  rm -rf $out
  echo "[ca-b12-heal] near_plane=$NP start $(date +%F_%T)" >> $PROG
  python train.py -s $DATA --eval -r 1 --data_device cpu --iterations 30000 \
    --save_iterations 30000 --cap_max 1000000 --screen_recycle_px 300 \
    --render_near_plane $NP -m $out > $LOGD/ca_b12_np${NP}.log 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    if [ -d $out/point_cloud/iteration_30000 ]; then
      sleep 15; kill $TP 2>/dev/null
      echo "[ca-b12-heal] near_plane=$NP SURVIVED to 30k $(date +%F_%T)" >> $PROG
      break
    fi
    sleep 30
  done
  wait $TP 2>/dev/null
  if [ -d $out/point_cloud/iteration_30000 ]; then
    # survived — eval and stop the ladder
    python eval.py -s $DATA -m $out --data_device cpu --iteration 30000 \
      > $LOGD/ca_b12_np${NP}_eval.log 2>&1
    res=$(grep -oE "\{'SSIM.*\}" $LOGD/ca_b12_np${NP}_eval.log | tail -1)
    echo "[ca-b12-heal] DONE near_plane=$NP ${res} $(date +%F_%T)" >> $PROG
    exit 0
  fi
  lastiter=$(grep -oE "Iter=[0-9]+" $LOGD/ca_b12_np${NP}.log | tail -1)
  echo "[ca-b12-heal] near_plane=$NP DIED ${lastiter}, escalating $(date +%F_%T)" >> $PROG
done
echo "[ca-b12-heal] ALL RUNGS FAILED $(date +%F_%T)" >> $PROG
