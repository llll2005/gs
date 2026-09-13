#!/bin/bash
# init 來源的**單變數**比較，20,000 步。用法：task_initcmp.sh <block_id> <arm>
#   arm = sfm      SfM 稀疏點（--model.initialize_from null）
#         depth    depth-init PLY（需要先跑 task_depthprep.sh）
#         random   100k 隨機點（對照「起點到底重不重要」的下界）
#         sfmfill  SfM 點 + 空處低密度補點（需先跑 tools/make_sfm_fill_init.py）
# 設計：除了 init 之外**所有參數相同**，而且刻意用**最少的機制**
#   （cap 固定、不 absgrad、不 noise、無正則）以免機制與 init 交互作用
#   —— 這是本機 `configs/normal.yaml` 那條線的 lab 版。
source "$(dirname "$0")/_common.sh"
BLK=${1:?用法: task_initcmp.sh <block_id> <arm>}; ARM=${2:?}
# ⚠ STEPS 可覆寫，用來當**閘門**（例：STEPS=1500 先驗 PLY 載入正確再上 lab 跑全長）。
#   LR 排程跟著一起縮放，否則閘門跑的是另一個 regime（v2 §7 的教訓）。
STEPS=${STEPS:-20000}
COMMON=(--trainer.max_steps "$STEPS"
        --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps "$STEPS"
        --model.density.init_args.cap_max 2600000
        --model.density.init_args.densify_until_iter 10000
        --model.density.init_args.absgrad_densify 0.0
        --model.density.init_args.fast_noise false
        --model.metric.init_args.opacity_reg 0.002
        --model.metric.init_args.lambda_normal 0.0
        --model.metric.init_args.depth_loss_weight.init 0.0)
case "$ARM" in
  sfm)    run_fit "${RUN_PREFIX}init_sfm"    "$BLK" --model.initialize_from null "${COMMON[@]}" ;;
  depth)  P="data/matrix_city/aerial/train/block_all/depth_init/block_$BLK.ply"
          [ -f "$P" ] || { echo "❌ 缺 $P —— 先跑 scripts/lab/task_depthprep.sh"; exit 2; }
          run_fit "${RUN_PREFIX}init_depth"  "$BLK" --model.initialize_from "$P" "${COMMON[@]}" ;;
  random) run_fit "${RUN_PREFIX}init_random" "$BLK" --model.initialize_from null \
            --data.parser.points_from random --data.parser.n_random_points 100000 "${COMMON[@]}" ;;
  # ★ SfM 點 + 空處低密度補點（使用者 2026-09-13 提案；tools/make_sfm_fill_init.py 產生）
  # ⚠ 走 `points_from ply` 而**不是** `--model.initialize_from` —— 後者載入已存好的
  #   Gaussian 模型（scale 由檔案決定），而 sfm 臂走 setup_from_pcd（scale 從點雲算）
  #   => 用 initialize_from 會多出「scale 初始化」這個變數，就不是單變數了。
  # ⚠ ply_file 的路徑是**相對於 dataset 目錄**（internal/dataparsers 用 os.path.join(self.path, ...)）
  sfmfill) P="sfmfill_init/block_$BLK.ply"
           F="data/matrix_city/aerial/train/block_all/$P"
           [ -f "$F" ] || { echo "❌ 缺 $F —— 先跑 tools/make_sfm_fill_init.py"; exit 2; }
           run_fit "${RUN_PREFIX}init_sfmfill" "$BLK" --model.initialize_from null \
             --data.parser.points_from ply --data.parser.ply_file "$P" "${COMMON[@]}" ;;
  *) echo "❌ 不認得的 arm：$ARM（sfm/depth/random/sfmfill）"; exit 2 ;;
esac
