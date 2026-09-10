#!/usr/bin/env bash
# O0 第六臂：視角數 6 -> 24（其餘完全相同）。測「視角數就是關鍵」這個預測。
# 機制：Adam 的 beta1=0.9 等效窗約 10 步，而 O0 一步一張圖輪替。
#   6 視角  => 窗涵蓋全部視角，共識估得準
#   284 視角(訓練) => 10 步只採樣 3.5%
# 預測：Δcorr 應隨視角數單調下降。
#   24 視角掉到接近 0.09（對照組水準）=> **視角數/共識估計就是機制**
#     => 單參數介入：加長 beta1（EMA 窗），不動任何排程
#   Δ 幾乎不變                       => 視角數不是機制，回頭找別的
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --multi-view 24 --n-tile 6 --verify-cams 8
