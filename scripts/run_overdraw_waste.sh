#!/bin/bash
# The one number the cost-aware thesis rests on: what fraction of blend work buys nothing.
# Needs the GPU to itself (3.6M-point models). Runs after the texture-ratio comparison.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/measure_overdraw_waste.py \
  --runs cap4m_b12 oreg_0p002_b12 oreg_0p0_b12 b12_cap2m_reg000_exact v1clean_24k_b12 \
  --views 16 > logs/overdraw_waste.log 2>&1
echo "[overdraw] done rc=$? $(date +%F_%T)" >> logs/quad_progress.log
