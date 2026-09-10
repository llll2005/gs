#!/usr/bin/env bash
# O0 第四臂：凍結顏色（shs_dc）。第二臂顯示 means 用真實 LR 也能恢復、
# scales 只動 2.9%、opacity 只動 0.2% => 做事的應該是顏色。凍住它就知道。
#   Δ 掉到接近 0 => 恢復純粹是**每顆粒子重新上色**，那是單視角幾乎必然成功的事
#                  => O0「basis 足夠」的結論要大幅降級
#   Δ 仍大      => 幾何（位置/旋轉）本身就能恢復 => 結論站得住
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --freeze shs_dc --n-tile 8
