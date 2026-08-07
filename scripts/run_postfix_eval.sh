#!/bin/bash
# Full evaluation of everything trained since the 2026-08-06 fixes, on the metrics that have
# resolution on this scene.
#
# WHY THE RUN LIST IS DRAWN WHERE IT IS
# ------------------------------------
# 2026-08-06 changed two things that are NOT interchangeable:
#   P7 depth-coverage fix   -- a pure bugfix. Runs before it are comparable if their d_reg never
#                              spiked (60 of them were clean); only 7 were contaminated.
#   contribution top-K -> max -- a MECHANISM CHANGE. The old loop approximated `max` but was
#                              camera-order dependent, so it altered which primitives the prune
#                              kept. Runs across this line are NOT strictly comparable.
# So the four PRIMARY runs below were all trained after both, and only they may be compared to each
# other without caveat. The three REFERENCE runs are pre-change: their numbers are indicative, and
# the "4x count = +0.459 dB" figure quoted on 2026-08-07 crosses this line.
#
# WHAT IS MEASURED AND WHY NOT PSNR
# ---------------------------------
# b12 is roughly half flat water, where a near-uniform colour blob still scores 32-40 dB. That
# average hid a 2.7x building texture gap behind 1.37 dB of PSNR (77k vs 1.8M, measured 08-06), and
# on 08-07 `uniform_60k_b12` had PSNR FALL 22.21 -> 21.93 at a step where both LPIPS and texture
# ratio improved -- blur is the MSE-optimal answer, so sharpening costs PSNR.
#   1. rescore_by_content   -- splits views by GT gradient, reports building/water separately.
#                              Texture ratio is the only metric verified to separate known-good
#                              from known-bad here (7x, no overlap) where building LPIPS overlapped.
#   2. measure_overdraw_waste -- the wasted fraction w of blend work. If w is large, the cost-aware
#                              thesis has room; if w < 15%, its ceiling is low and we need to know
#                              now rather than after writing.
# Both need the GPU to itself: a 3.6M-point model will not load beside a training run.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1

PRIMARY="v1clean_24k_b12 uniform25m_24k_b12 cap4m_b12 uniform_60k_b12"
REFERENCE="oreg_0p002_b12 oreg_0p0_b12 b12_cap2m_reg000_exact"

echo "[eval] start $(date +%F_%T)" >> logs/quad_progress.log

PYTHONPATH=. conda run -n gspl --no-capture-output python tools/rescore_by_content.py \
  --runs $PRIMARY $REFERENCE --views 8 \
  --out 紀錄/rescore_postfix.csv > logs/eval_texratio.log 2>&1
rc1=$?

PYTHONPATH=. conda run -n gspl --no-capture-output python tools/measure_overdraw_waste.py \
  --runs $PRIMARY $REFERENCE --views 16 > logs/eval_overdraw.log 2>&1
rc2=$?

echo "[eval] done texratio=$rc1 overdraw=$rc2 $(date +%F_%T)" >> logs/quad_progress.log
