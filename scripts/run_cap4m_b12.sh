#!/bin/bash
# Where is the primitive ceiling on 6GB, and does quality still climb to it?
#
# Every large b12 run so far ended at exactly 0.9 x cap_max -- 2M->1,799,999, 1M->900,000,
# 1M->900,000. That is not convergence, it is the cap: once N reaches cap_max, `add_new_gs`
# computes min(cap, 1.05N) - N = 0 and stops adding, while the contribution prune keeps firing
# every 500 steps, so the last prune before densify_until_iter knocks 10% off and nothing
# regrows it. Every one of those runs would have used more primitives if allowed.
#
# So raise the cap until the hardware says no. Matched control is `oreg_0p002_b12`: identical
# in every other parameter, cap 1M -> 900,000 points -> val 23.848. This is cap x4.
#
# WHY opacity_reg 0.002 AND NOT 0
# -------------------------------
# reg=0 wins on val PSNR (+1.04 dB) but LOSES across the board in building regions
# (memory feedback_metric_resolution). b12 is half flat water, which inflates val; the same trap
# that made "23x fewer points for only 1.37 dB" look real tonight before the texture ratio came
# back 0.145 vs 0.393. Measuring a ceiling for a recipe we would not ship is not worth the GPU
# hours. Note reg=0 also costs LESS VRAM, so this is the conservative choice for a ceiling test.
#
# PREDICTION (strip_cameras.py constants, SB F=25, K=1)
#     model state N*25*4B*4(Adam) = 400 B/pt | A_RENDER 500 (fixed) | B_RENDER 1550 (K-splittable)
#     K=1 total 2450 B/pt, usable budget ~5.0G  ->  wall at N ~ 2.0M
#     EXACT_SUPPORT measured the training ceiling at ~2.5M, so expect 2.0-2.5M.
# Growth from the ~1.2M depth-init: per 1500 steps, 10 densify (x1.63) and 3 prunes (x0.729)
# = net x1.19  ->  reaches 4M or the wall by step ~11,000. THE ANSWER ARRIVES IN ~2 HOURS;
# there is no need to let 60k finish to learn the ceiling.
#
# K-strip is deliberately OFF. It is the obvious way to push past the wall, but it is 0 for 2 in
# practice (2026-07-23 and 07-24, both rc=1 at cap 2M, the second dying at 1.26M) while the plain
# K=1 run at the same cap reached 1.8M and finished. Those failures predate EXACT_SUPPORT so they
# do not condemn the mechanism, but they do disqualify it as this run's safety net.
#
# BOTH OUTCOMES ARE THE ANSWER
#   OOM   -> internal/callbacks.py writes DIED with the step, N and VRAM = the ceiling, measured
#   4M    -> the predictor is too pessimistic, and we get a real cap-4M quality point
# Watch outputs/cap4m_b12/blocks/block_12/train_status.txt for the N-vs-VRAM curve; the wall can
# be extrapolated from it before the crash.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"

NAME=cap4m_b12
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

rm -rf outputs/$NAME
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from $PLY \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
