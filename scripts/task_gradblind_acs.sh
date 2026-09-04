#!/bin/bash
# ★★★★ acs 跑完後：**縮小 footprint 有沒有真的把盲區變淺？**（~3 分鐘）
#
# 這是與分數**獨立**的機制檢查。四種可能，結論完全不同：
#   盲區變淺 + 分數升  => 機制成立且有效，進配方
#   盲區變淺 + 分數平  => 機制成立但**盲區不是品質的瓶頸** => 整條假說線要重估
#   盲區沒變 + 分數平  => 介入沒作用到成因（如同 acd）=> 加大強度或換作用點
#   盲區變深           => 縮小 footprint 反而更糟 => 機制方向錯
# 對照：agd2_b12 的比值的比 = 0.218x
set -u
cd "$(dirname "$0")/.." || exit 1
conda run -n gspl --no-capture-output python tools/grad_blindspot.py \
  agd2_b12 sched30_b12 --blk 12 --model-run acs_b12 --step 60000
