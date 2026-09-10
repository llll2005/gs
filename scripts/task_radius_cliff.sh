#!/usr/bin/env bash
# 半徑懸崖診斷：failure 是不是 footprint 跨過閾值造成的？決定 ac_shrink 這條線的生死。
# 兩個輸出：① corr vs 半徑有沒有懸崖（有 => 閾值就是 target_px）
#           ② |g|/AC 是不是隨半徑下降（否 => 抵消不由 footprint 驅動，整個家族否證）
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/radius_cliff.py agd2_b12 sched30_b12 --blk 12
