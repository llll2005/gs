#!/bin/bash
# Is 24.307 real, or another water-inflated average?
# b12 is half flat water; on 2026-08-06 that hid a 2.7x building texture gap behind 1.37 dB of PSNR.
# cap4m's val/texratio 0.3555 is over the whole val set and the other arms predate the metric, so
# the only apples-to-apples comparison is the offline tool, which splits views by GT gradient.
# Needs the GPU to itself: a 3.6M model will not load alongside a training run.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/rescore_by_content.py \
  --runs cap4m_b12 oreg_0p002_b12 b12_cap2m_reg000_exact uniform_60k_b12 v1clean_24k_b12 \
  --views 8 --out 紀錄/rescore_capseries.csv > logs/texratio_capseries.log 2>&1
echo "[texratio] done rc=$? $(date +%F_%T)" >> logs/quad_progress.log
