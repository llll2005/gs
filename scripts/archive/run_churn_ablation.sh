#!/bin/bash
# CHURN ABLATION — attacks the root cause found 2026-07-28, instead of stacking more
# mechanisms on top of it.
#
# Finding: at step 30k on b12, 63.4% of points sit below the death line (0.005) and are
# relocated EVERY densify event (616k points moved onto 355k live hosts, every 150 steps).
# Mechanism: opacity_reg (0.007) pushes every point with o<0.9 down; the photometric loss
# only defends the ones that matter; the rest cross the line, get relocated (opacity
# renormalised back up), and are pushed down again — an L1 death treadmill. The population
# never settles during densify. This explains b12's slowness (0.40 it/s), its proximity to
# OOM, and why every mechanism that ADDS condemnation (condensation annealing, v/c) OOMs:
# they add pressure to an already 63%-overloaded system.
#
# Churn is a real cost that NO existing metric prices (not point count, not accounted VRAM,
# not PSNR) — so this ablation is also a direct cost-aware finding.
#
# Arms (b12, SB, cap 1M, everything else identical to A' = 22.12 / 0.40 it/s / churn 63%):
#   reg002 : opacity_reg 0.002  — weaker downward push
#   reg000 : opacity_reg 0.0    — no push at all (extreme control)
# Measure per arm: churn%/event (the [dead-mask] log), it/s, peak VRAM, final PSNR, final N.
#
# Read: lower churn + PSNR held/better + faster => the treadmill is waste, root cause found.
#       lower churn + PSNR worse => 63% churn is MCMC's necessary exploration; then attack
#       its efficiency (relocation cost), not its existence.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
SB=configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

# Progress ledger: START/DONE/DIED are emitted by the training process itself
# (internal/callbacks.py TrainConsole._progress) with structured fields straight from the live
# footer. Scripts must NOT write those lines: the old approach grepped them back out of the
# training log, which dragged tqdm bars and ANSI escapes into the ledger and let every script
# drift to its own column widths. Use log() only for things Python cannot know -- queue-level
# events like waiting on the GPU or a batch boundary.
log () { printf '%s | %-22s | %s\n' "$(date '+%m-%d %H:%M')" "$1" "$2" >> $PROG; }

wait_gpu () {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && used=0
    [ "$used" -lt 1500 ] && break
    sleep 300
  done
  sleep 30
}

run_arm () {
  local name=$1 reg=$2
  wait_gpu
  rm -rf "outputs/$name"
  conda run -n gspl python -u main.py fit --config $SB \
    --model.initialize_from $PLY --data.parser.block_id 12 \
    --model.density.init_args.screen_size_prune_px 300 \
    --model.density.init_args.cap_max 1000000 \
    --model.metric.init_args.opacity_reg "$reg" \
    -n "$name" > "logs/$name.log" 2>&1
  local rc=$?
}

log "churn-ablation" "QUEUE BATCH START — root-cause attack on the 63% relocation treadmill"
# reg002 跳過 (2026-07-28): Adam 尺度不變性已用 reg002 自己的 1499 ckpt 免費證實 —
# opacity_reg 0.007→0.002 (差 3.5x) 只讓死亡線下比例 23.3%→20.5%, median o 0.0212→0.0296.
# 對光度梯度≈0 的塵埃點, Adam 的 m/sqrt(v) 把 L1 係數大小吃掉, 每步位移≈lr 與係數無關.
# → 此臂無資訊價值; 真正的旋鈕是「繞過 Adam 的顯式衰減」或「relocation 閥門」.
# run_arm churn_reg002_b12 0.002
run_arm churn_reg000_b12 0.0
log "churn-ablation" "QUEUE BATCH DONE"
