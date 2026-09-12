#!/bin/bash
# 現行最佳配方在**修正後資料**上的基準線。用法：task_speed3.sh <block_id>
# ⚠ 與舊的 26.6957 等數字**完全不可比**：資料修正了（§16.14）、相機數也變了
#   （b12 284 -> 653）。這是新年代的第一個基準。
source "$(dirname "$0")/_common.sh"
BLK=${1:?用法: task_speed3.sh <block_id>}
run_fit lab/speed3 "$BLK" \
  --model.initialize_from null \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0
