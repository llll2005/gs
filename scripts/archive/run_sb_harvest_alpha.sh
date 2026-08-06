#!/bin/bash
# Harvest-phase opacity test (RTG-freeze idea, narrow no-conflict version).
# Resumes arm A (SB@1.0M b12) from its step=41999 ckpt — densify already over,
# relocation off — and replays ONLY the 18k settle phase under 3 opacity policies:
#   cprime : unchanged           -> validates resume mechanics; expect ~21.99
#   freeze : opacity grads None  -> RTG-style hold (tests if L1 attrition is pure loss)
#   noreg  : opacity_reg = 0     -> photometric may move alpha, no shrink pressure
# ~75-90 min/arm at ~4 it/s. Waits for the GPU (arm B) before starting.
# Judgment: freeze/noreg > cprime => harvest alpha pressure is attrition, RTG idea
# partially revives; ~= cprime => MCMC self-consistent end to end, close the line.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
CKPT=$(ls outputs/mcmc_sb_60k_aggr17_b12_cap1m/blocks/block_12/checkpoints/*step=41999.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "no 41999 ckpt found" >> $PROG; exit 1; }

# wait until the GPU is free (arm B still training)
while pgrep -f "main.py fit" > /dev/null; do sleep 600; done
sleep 60

run_arm () {
  local name=$1; shift
  echo "[harvest-alpha] $name start $(date +%F_%T)" >> $PROG
  rm -rf "outputs/$name"
  conda run -n gspl python -u main.py fit \
    --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
    --model.initialize_from "$CKPT" \
    --ckpt_path "$CKPT" \
    --data.parser.block_id 12 \
    --model.density.init_args.screen_size_prune_px 300 \
    --model.density.init_args.cap_max 1000000 \
    "$@" \
    -n "$name" > "logs/$name.log" 2>&1
  local rc=$?
  local psnr
  psnr=$(grep -oE 'val/psnr: [0-9.]+' "outputs/$name/blocks/block_12/results.txt" 2>/dev/null | head -1)
  echo "[harvest-alpha] $name done rc=$rc $psnr $(date +%F_%T)" >> $PROG
}

run_arm sb_harvest_cprime
run_arm sb_harvest_freeze --model.density.init_args.freeze_opacity_after_densify true
run_arm sb_harvest_noreg  --model.metric.init_args.opacity_reg 0.0
