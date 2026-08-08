#!/bin/bash
# Queued single-block experiments, run serially. Kill this driver to stop the queue.
#
# Restored from scripts/archive/ on 2026-08-09 after three days of ad-hoc /tmp watchers, one of
# which sat blocked for 2.5 hours behind a hung `main.py test` that was using 151 MiB and doing
# nothing. This file already carried the fix for that exact failure, dated 2026-07-25:
#
#   pgrep on the command line is not a GPU-availability test. conda run rewrites the cmdline, a
#   dead-but-unreaped process still matches, and the checking shell's own command line can match
#   itself. Ask the driver how much VRAM is actually resident.
#
# Add new experiments at the BOTTOM; each block waits for the GPU, runs, and reports to the ledger.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
SB=configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml
PLY_DIR=data/matrix_city/aerial/train/block_all/depth_init

wait_gpu () {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && used=0
    [ "$used" -lt 1500 ] && break     # <1.5GB = no training resident
    sleep 300
  done
  sleep 30
}

# A run that dies at init leaves no results.txt and a short log. Surface that rather than silently
# reporting an empty PSNR (2026-07-25: a whole queue was lost to silent init failures).
report () {  # name, block, note
  local psnr; psnr=$(grep -oE 'val/psnr: [0-9.]+' "outputs/$1/blocks/block_$2/results.txt" 2>/dev/null | head -1)
  if [ -z "$psnr" ]; then
    local errline; errline=$(grep -oE "(Error|error|Exception|OutOfMemory)[^\"]{0,80}" "logs/$1.log" 2>/dev/null | tail -1)
    echo "[queue] !! $1 FAILED (no results.txt). last error: ${errline:-<none, check logs/$1.log>} $(date +%F_%T)" >> $PROG
    return
  fi
  local tex; tex=$(grep -oE 'val/texratio: [0-9.]+' "outputs/$1/blocks/block_$2/results.txt" 2>/dev/null | head -1)
  echo "[queue] $1 done $psnr $tex ($3) $(date +%F_%T)" >> $PROG
}

# --- 1. blur split, budget-share version --------------------------------------------------------
# The first attempt (blursplit_b12, a multiplier) was a design error: candidates are 0.01% of the
# population, so 3x moved them from 0.0084% to 0.0253% of the sampling mass = 15 of 60,000
# additions, and the run duly tracked its control. A budget share means the same thing however few
# candidates there are. Control cap4m_b12 24.307; noise floor measured at 0.053 dB.
wait_gpu; echo "[queue] blurbudget start $(date +%F_%T)" >> $PROG
rm -rf outputs/blurbudget_b12
conda run -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY_DIR/block_12.ply --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.blur_split_budget 0.3 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n blurbudget_b12 > logs/blurbudget_b12.log 2>&1
report blurbudget_b12 12 "vs cap4m 24.307, 噪音 0.053"

# --- 2. blursplit test (renders) ----------------------------------------------------------------
# Failed on 2026-08-08 with "Option 'blur_split_weight' is not accepted": renaming the config field
# broke configs already on disk. A deprecated alias restores loading. results.txt holds the
# training-time val numbers already quoted in 紀錄, so back it up around the test.
wait_gpu
D=outputs/blursplit_b12/blocks/block_12
cp -f $D/results.txt $D/results.txt.bak 2>/dev/null
conda run -n gspl python -u main.py test --config $D/lightning_logs/version_0/config.yaml \
  --save_val --test_speed > logs/test_blursplit_b12.log 2>&1
rc=$?
mv -f $D/results.txt.bak $D/results.txt 2>/dev/null
echo "[queue] test blursplit_b12 rc=$rc 圖 $(find $D/test -name '*.png' 2>/dev/null | wc -l) 張 $(date +%F_%T)" >> $PROG

# --- 3. ★ trimstop: which half of the churn suppresses quality? ---------------------------------
# Every 60k run gains more in the ONE 5,680-step window straddling densify_until_iter=42,000 than
# in the whole 34,000-step growth phase before it (+1.34/+1.42/+1.90 vs +1.26/+1.29/+0.88).
# But 42,000 stops TWO things, because the contribution trim is gated on densify_until_iter.
# Densify adds 5% every 150 steps; the trim removes 10% every 500 -- including primitives that are
# already well optimised. Adding cannot undo optimisation, deleting can, so the trim is the better
# suspect. contribution_prune_until_iter separates them.
wait_gpu; echo "[queue] trimstop start $(date +%F_%T)" >> $PROG
rm -rf outputs/trimstop_b12
conda run -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY_DIR/block_12.ply --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.renderer.init_args.contribution_prune_until_iter 25000 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n trimstop_b12 > logs/trimstop_b12.log 2>&1
report trimstop_b12 12 "trim 停 25k, densify 照跑到 42k; vs cap4m 24.307"

# --- 4. blur budget + long-axis spread ----------------------------------------------------------
# Only after blurbudget has reported alone: two mechanisms at once cannot be attributed. Expected to
# be super-additive if either works -- blur split selects primitives that alone cover a large patch,
# and MCMC stacks all their children at the host's centre, which cannot cover that patch.
wait_gpu; echo "[queue] blurlas start $(date +%F_%T)" >> $PROG
rm -rf outputs/blurlas_b12
conda run -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY_DIR/block_12.ply --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.blur_split_budget 0.3 \
  --model.density.init_args.long_axis_spread 0.5 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n blurlas_b12 > logs/blurlas_b12.log 2>&1
report blurlas_b12 12 "blur 0.3 + 長軸 0.5"

echo "[queue] all done $(date +%F_%T)" >> $PROG
