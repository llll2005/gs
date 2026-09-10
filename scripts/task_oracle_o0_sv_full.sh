#!/usr/bin/env bash
# O0 控制組：**單一視角 + 整張 loss**（--multi-view 1）。
# 沒有它，第三臂無法解讀 —— 第三臂同時改了「多視角」與「整張 loss」兩件事。
# 三點比較才能拆開：
#   1 視角 + tile 局部 loss  = +0.65（arm 1/4 已測）
#   1 視角 + **整張 loss**   = 本臂        <- 只含**空間**競爭（同視角其他區域在搶同一批粒子）
#   6 視角 + 整張 loss       = arm 3       <- 再加上**視角間**衝突
# 本臂掉很多 => 主因是空間競爭，與視角數無關
# 本臂不掉、arm 3 掉 => **多視角衝突為真**
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --multi-view 1 --n-tile 6
