#!/usr/bin/env bash
# O0 第五臂：多視角 + 整張 loss + **訓練的兩個 L1 正則項**（權重從 ckpt 讀：
# opacity_reg 0.002 / immune>0.9、scale_reg 0.007）。
# 第三臂證明「訓練的終點不是局部最優」——但 O0 只優化光度，訓練優化「光度+正則」。
#   Δ 塌掉（~+0.09，與對照組同級）=> **正則就是元兇**，而且是「全域統一定價傷到
#     異質區域」的直接證據 => 直通論文命題（成本感知/異質定價）
#   Δ 仍是 +0.40 => 正則不是元兇 => 剩下視角數(6 vs 284) 與優化路徑
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --multi-view 6 --n-tile 6 --with-reg
