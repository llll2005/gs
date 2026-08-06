#!/bin/bash
# Beta 60k A/B: does long training climb, and does SB-color's memory saving let beta
# break our project-wide ~1.5M-primitive 6GB ceiling?
#   A: cap 1M, GPU data, 60k (fast control — resolves "24.72 -> ?" at 60k)
#   B: cap 4M, CPU data, densify_until 42k, 60k (CEILING PROBE — our Gaussian gsplat
#      dies at 1.7M on 6GB; beta @30 floats/primitive vs our 59 should reach higher.
#      4M = CityGS regime. VRAM sampler + checkpoint counts capture how far it grows.)
# --eval => train.py loops forever (L79-81); watch-and-kill at iteration_60000.
cd /home/LnoArch/Projects/專題/beta-splatting || exit 1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gspl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=/home/LnoArch/Projects/專題/beta-splatting/_pylibs
DATA=/home/LnoArch/Projects/專題/CityGaussian/data/probe_b7
LOGD=/home/LnoArch/Projects/專題/CityGaussian/logs
PROG=$LOGD/quad_progress.log

run_kill_at_60k() {
  local name=$1 outdir=$2 args=$3 sample_vram=$4
  echo "[beta60k] $name start $(date +%F_%T)" >> $PROG
  rm -rf $outdir
  local vrampid=""
  if [ "$sample_vram" = "yes" ]; then
    ( while :; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"; sleep 10; done > $LOGD/${name}_vram.log 2>/dev/null ) &
    vrampid=$!
  fi
  python train.py -s $DATA --eval -r 1 --iterations 60000 --save_iterations 30000 60000 $args \
      -m $outdir > $LOGD/${name}_train.log 2>&1 &
  local tp=$!
  while kill -0 $tp 2>/dev/null; do
    if [ -d $outdir/point_cloud/iteration_60000 ]; then
      sleep 20; kill $tp 2>/dev/null
      echo "[beta60k] $name reached 60k, killed $(date +%F_%T)" >> $PROG; break
    fi
    sleep 30
  done
  wait $tp 2>/dev/null
  [ -n "$vrampid" ] && kill $vrampid 2>/dev/null
  local peak=$(awk '{print $2}' $LOGD/${name}_vram.log 2>/dev/null | sort -n | tail -1)
  local lastiter=$(grep -oE "Iter=[0-9]+" $LOGD/${name}_train.log | tail -1)
  echo "[beta60k] $name done peakVRAM=${peak:-NA}MiB last=${lastiter:-NA} $(date +%F_%T)" >> $PROG
}

run_kill_at_60k b7_60k_cap1m  output/b7_60k_cap1m  ""                                          no
run_kill_at_60k b7_60k_cap4m  output/b7_60k_cap4m  "--data_device cpu --cap_max 4000000 --densify_until_iter 42000"  yes
echo "[beta60k] ALL DONE $(date +%F_%T)" >> $PROG
