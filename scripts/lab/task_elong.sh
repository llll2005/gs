#!/bin/bash
# Elongation filter（CityGSV2 有、我方換成 MCMC 後沒有的那個）。
# 用法：task_elong.sh <block_id> <prune|relocate>
# ⚠ 2026-09-12 在**錯開的資料**上量到「長寬比與失敗率負相關」=> 那個否證作廢，要重量。
#   binning 外接盒浪費 76.5% 的量測不依賴 GT 對位，仍然成立（記憶 elongation_binning_waste）。
source "$(dirname "$0")/_common.sh"
BLK=${1:?}; MODE=${2:?prune 或 relocate}
case "$MODE" in
  prune)    FLAG=(--model.density.init_args.elongation_prune 20.0) ;;
  relocate) FLAG=(--model.density.init_args.elongation_relocate 20.0) ;;
  *) echo "❌ 不認得 $MODE"; exit 2 ;;
esac
run_fit "lab/elong_$MODE" "$BLK" \
  --model.initialize_from null \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  "${FLAG[@]}"
