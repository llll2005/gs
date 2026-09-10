#!/usr/bin/env bash
# speed3 的 test + 糊掉統計：釐清「PSNR +4.9sd 但紋理比 -3.1sd」是不是真的變糊。
#   糊掉% 上升 / corr 中位下降 => 確實更糊 => 拒絕 noise_gate_eps
#   糊掉% 持平                 => 紋理比下降可能是別的東西 => 再議
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/task_test.sh speed3_b12
conda run -n gspl --no-capture-output python -u tools/blur_persistence.py \
  agd2_b12 speed3_b12 --blk 12
conda run -n gspl --no-capture-output python -u tools/veil_detect.py \
  agd2_b12 speed3_b12 --blk 12 2>/dev/null || true
