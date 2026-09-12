#!/bin/bash
# 命題唯一沒測過的一側：`cost_budget` 取代 `cap_max`
#   max Q s.t. (1/K)Σc_i <= B 的字面形式。取樣端三變體全輸（cost_aware_sampling_refuted），
#   只剩約束端沒測。⚠ 在修正後的資料上重做才有意義（舊基準全作廢）。
# 用法：task_costbudget.sh <block_id>
source "$(dirname "$0")/_common.sh"
BLK=${1:?}
run_fit "${RUN_PREFIX}costbudget" "$BLK" \
  --model.initialize_from null \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.cost_budget_report 2000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0
