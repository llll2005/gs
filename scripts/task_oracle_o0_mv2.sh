#!/usr/bin/env bash
# O0 第三臂重跑：覆核相機**真正留出**（前一次 6 台覆核有 5 台正在被優化 => 量到的是
# 訓練表現不是泛化）。改用 12 台留出相機（vset 36 台 - 6 台優化 = 30 台可選）。
# 這一欄決定「6 視角的解到底有沒有泛化」：
#   ΔPSNR >= 0 => 泛化，訓練確實漏掉了免費的改善
#   ΔPSNR < 0  => 仍是過擬合（只是從 1 視角變成 6 視角），284 視角下可能真的做不到
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --multi-view 6 --n-tile 6 --verify-cams 12
