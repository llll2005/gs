#!/bin/bash
# DAR FULL ARM (60k) — the unified framework's first quality verdict.
# Smoke passed 2026-07-28 15:16: applied 1501/1501, c_hat med 0.79 stable (no sawtooth),
# u med 1.000 (preconditioner inert as CPU-predicted), churn 8.2% vs A' 63% = treadmill
# shrank 7.7x. WATCH: clipped% rose 0.1 -> 14.6% by step 1501 as geometry spread; if it
# climbs past ~30% the cost ratio compresses and dar_clip should go up.
#
# TO BEAT: reg000 (opacity_reg=0, no pressure at all) = 23.158 / 0% churn / 4.34 it/s.
# NOT A' 22.117. DAR's case is not "pressure helps" — reg000 already showed it does not on
# b12@1M where cap binds. DAR's case is "pressure priced by cost keeps quality while
# controlling the cost profile", i.e. it should hold ~23 AND kill fog/monsters so a higher
# cap becomes feasible. Losing to reg000 by a lot = cost-aware pressure is also net-negative
# here, and the finding moves to blocks/caps where pressure is actually needed.
#
# original smoke header follows ---
# DAR SMOKE — first training-time test of the unified cost-aware regularizer (紀錄/現行方案與公式.md §1).
#
# Why a smoke and not the full arm: the mechanism is CPU-verified only (stride distribution
# computed from b12@30k's own Adam state). A 60k arm is 11h; this is ~8min and answers
# "does it fire, and do the measured strides match the prediction" before spending that.
#
# What to check in logs/dar_smoke.log — the [DAR-cost] line every 1500 steps:
#   c_hat[min/med/max]  expect ~0.10 / ~0.6-0.9 / ~4.0     (screen-diag² cap working)
#   stride[med/max]     expect ~0.015-0.03 / <0.25          (max < dar_clip ⇒ 0% saturated)
# If stride[max] == 0.25 exactly for most steps, the clip is binding again → cost signal
# flattened (the ε-placement failure mode); stop and re-derive λ before running the arm.
#
# Also compare fraction below the death line vs A' at the same step: prediction is that dust
# pressure DROPS (3621 steps to die vs 64-106 under stock L1), so the sub-0.005 population
# should be markedly smaller than A''s 23.3% at 1.5k / 63% at 30k.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
SB=configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply
NAME=dar_arm_b12

# Progress ledger: START/DONE/DIED are emitted by the training process itself
# (internal/callbacks.py TrainConsole._progress) with structured fields straight from the live
# footer. Scripts must NOT write those lines: the old approach grepped them back out of the
# training log, which dragged tqdm bars and ANSI escapes into the ledger and let every script
# drift to its own column widths. Use log() only for things Python cannot know -- queue-level
# events like waiting on the GPU or a batch boundary.
log () { printf '%s | %-22s | %s\n' "$(date '+%m-%d %H:%M')" "$1" "$2" >> $PROG; }

# nvidia-smi, not pgrep: `conda run` rewrites cmdline so pgrep -f "python -u main.py fit"
# never matches and the whole queue used to die at CUDA init (2026-07-2x).
wait_gpu () {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && used=0
    [ "$used" -lt 1500 ] && break
    sleep 300
  done
  sleep 30
}

wait_gpu
rm -rf "outputs/$NAME"
conda run --no-capture-output -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 1000000 \
  --model.density.init_args.dar_lambda 0.05 \
  --model.density.init_args.dar_clip 0.25 \
  --model.metric.init_args.opacity_reg 0.0 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
rc=$?
dar=$(grep '\[DAR-cost\]' "logs/$NAME.log" | tail -1)
if [ -z "$dar" ]; then
  err=$(grep -oE '(OutOfMemoryError|RecursionError|TypeError|AttributeError|Error)[^"]{0,70}' "logs/$NAME.log" | tail -1)
else
fi
