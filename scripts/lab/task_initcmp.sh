#!/bin/bash
# init 來源的**單變數**比較，20,000 步。用法：task_initcmp.sh <block_id> <arm>
#   arm = sfm      SfM 稀疏點（--model.initialize_from null）
#         depth    depth-init PLY（需要先跑 task_depthprep.sh）
#         random   100k 隨機點（對照「起點到底重不重要」的下界）
# 設計：除了 init 之外**所有參數相同**，而且刻意用**最少的機制**
#   （cap 固定、不 absgrad、不 noise、無正則）以免機制與 init 交互作用
#   —— 這是本機 `configs/normal.yaml` 那條線的 lab 版。
source "$(dirname "$0")/_common.sh"
BLK=${1:?用法: task_initcmp.sh <block_id> <arm>}; ARM=${2:?}
COMMON=(--trainer.max_steps 20000
        --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 20000
        --model.density.init_args.cap_max 2600000
        --model.density.init_args.densify_until_iter 10000
        --model.density.init_args.absgrad_densify 0.0
        --model.density.init_args.fast_noise false
        --model.metric.init_args.opacity_reg 0.002
        --model.metric.init_args.lambda_normal 0.0
        --model.metric.init_args.depth_loss_weight.init 0.0)
case "$ARM" in
  sfm)    run_fit "lab/init_sfm"    "$BLK" --model.initialize_from null "${COMMON[@]}" ;;
  depth)  P="data/matrix_city/aerial/train/block_all/depth_init/block_$BLK.ply"
          [ -f "$P" ] || { echo "❌ 缺 $P —— 先跑 scripts/lab/task_depthprep.sh"; exit 2; }
          run_fit "lab/init_depth"  "$BLK" --model.initialize_from "$P" "${COMMON[@]}" ;;
  random) run_fit "lab/init_random" "$BLK" --model.initialize_from null \
            --data.parser.points_from random --data.parser.n_random_points 100000 "${COMMON[@]}" ;;
  *) echo "❌ 不認得的 arm：$ARM（sfm/depth/random）"; exit 2 ;;
esac
