#!/bin/bash
# Clean same-kernel count ablation, BOTH arms on the new (pinned+pp_shifty) scale:
#   B' : trim+SB @ cap 2.0M, K=2 strips  (K bounds rasterizer buffers ~1/2; the
#        old arm B OOM'd at ~30k/5.15G on K=1)
#   A' : trim+SB @ cap 1.0M, K=1         (re-train of arm A on the new binary,
#        because old A was trained on the lost March binary = mismatched model)
# Verdict = B' vs A', same binary, same recipe, only cap+K differ.
# New-scale context anchors (Lightning val): SH3=21.609, old-A evals 20.756.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

run_arm () {
  local name=$1 cap=$2 strips=$3
  while pgrep -f "main.py fit" > /dev/null; do sleep 300; done
  sleep 30
  echo "[count-ns] $name cap=$cap K=$strips start $(date +%F_%T)" >> $PROG
  rm -rf "outputs/$name"
  conda run -n gspl python -u main.py fit \
    --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
    --model.initialize_from $PLY \
    --data.parser.block_id 12 \
    --model.density.init_args.screen_size_prune_px 300 \
    --model.density.init_args.cap_max "$cap" \
    --model.train_strips "$strips" \
    -n "$name" > "logs/$name.log" 2>&1
  local rc=$?
  local psnr
  psnr=$(grep -oE 'val/psnr: [0-9.]+' "outputs/$name/blocks/block_12/results.txt" 2>/dev/null | head -1)
  echo "[count-ns] $name done rc=$rc $psnr $(date +%F_%T)" >> $PROG
}

run_arm mcmc_sb_ns_b12_cap2m_K2 2000000 2
run_arm mcmc_sb_ns_b12_cap1m_K1 1000000 1
