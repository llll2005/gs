#!/bin/bash
# Why did the DAR arm collapse? Two arms that the hypothesis separates.
#
# What happened: DAR at lambda=0.05 / C_t=0.25 / opacity_reg=0 scored 10.40 PSNR against
# reg000's 23.16. The mechanism itself ran clean (applied 59501/59501, clipped 0%, churn 0%,
# o<0.005 only 0.3%), and opacity came out normal (P50 0.80). What died was SCALE: median
# log-scale -20.94 (~8e-10) with 100% of primitives under 1e-6, where reg000 sits at -6.74 with
# 38.8% under. So an opacity-only intervention killed the scales -- an indirect coupling.
#
# Hypothesis: the low-pass FilterSize means shrinking a primitive does NOT reduce its peak alpha
# but does reduce how many pixels it touches. Under a standing external push on opacity the
# primitive is permanently "not yet fitted", so shrinking is the cheap way to cut its loss
# contribution -- shrink, cover less, receive less gradient, shrink further.
#
# The hypothesis predicts the two arms diverge:
#   noscale : scale_reg=0. Photometric gradient is the driver, not the scale L1 -> should STILL
#             collapse. If it recovers instead, the hypothesis is wrong and the scale L1 is the
#             co-conspirator.
#   lam005  : lambda 0.05 -> 0.005. Less standing push -> should recover.
#
# Read at 8k steps: reg000 sat at loss ~0.094 by 6.5k, the collapsed DAR was stuck at 0.3-0.5
# from ~5k on. Loss alone separates them; the scale histogram at the end confirms.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SB=configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply
# START/DONE/DIED are written by the training process (internal/callbacks.py); see 完整指令手冊.
log () { printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "$1" "$2" >> logs/quad_progress.log; }

run_arm () {
  local name=$1; shift
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 1500 ]; do sleep 120; done
  sleep 20
  rm -rf "outputs/$name"
  conda run --no-capture-output -n gspl python -u main.py fit --config $SB \
    --model.initialize_from $PLY --data.parser.block_id 12 \
    --model.density.init_args.screen_size_prune_px 300 \
    --model.density.init_args.cap_max 1000000 \
    --model.metric.init_args.opacity_reg 0.0 \
    --trainer.max_steps 8000 "$@" -n "$name" > "logs/$name.log" 2>&1
}

log "dar-diagnose" "QUEUE BATCH START — 兩臂分辨 DAR 崩潰真因 (假說預測 noscale 仍崩 / lam005 復原)"
run_arm dar_diag_noscale --model.density.init_args.dar_lambda 0.05 --model.metric.init_args.scale_reg 0.0
run_arm dar_diag_lam005  --model.density.init_args.dar_lambda 0.005
log "dar-diagnose" "QUEUE BATCH DONE"
