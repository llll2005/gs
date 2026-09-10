#!/usr/bin/env bash
# O0 第三臂（決定性）：6 台相機一起優化同一批影響集、整張 loss（= 訓練的形狀）。
#   仍達 ~0.85 => 多視角一致解存在，訓練只是沒找到 => 往全域優化找
#   卡在低點   => 多視角衝突為真 => blur 是正確的妥協，根因在幾何
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --multi-view 6 --n-tile 6
