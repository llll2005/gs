#!/bin/bash
# E5：修正清除半徑後的 graded init（P3 的修正）
#
# 【改了什麼】fill_radius 0.02456 → 0.08，anchor_voxel 解耦保持 0.02456。
#   原本用 depth-init 的「間距」當清除半徑，但該用的是它的「誤差」：
#   depth-init 離表面 0.0795，填充點 opacity=0.99 散在錨點前後，約一半擋在前面。
#   E1 已證實：被砍點有 69.1% 在存活點後方、徑向間距 0.0600（對應誤差尺度非體素尺度）。
#
#   錨點 203,840（幾乎不變）/ 填充 788,001（−29%）/ 總計 991,841（depth-init 的 0.82×）
#   ⚠ 顆數不再對齊 v1，位置與顆數同時變，這是已知 confound。
#
# 【關鍵讀數在 step 499，不是 24k】錨點留存率：
#     原版 fill_radius 0.02456 → step 499 錨點留存 10.7%（audit_start_trim 的嚴格配對口徑）
#     若遮擋假說成立，修正後應顯著上升；沒上升 ⇒ P3 診斷錯，立刻砍掉這一臂
#
# 【對照】aggr24k_b12：5,679 → 20.644/.582/.764，24,000 → 22.070/.604/.695
set -euo pipefail
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=graded_r08_24k_b12
rm -rf "outputs/$NAME"
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_24k_aggr_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/graded_k4r08_block_12.ply \
  --data.parser.block_id 12 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
