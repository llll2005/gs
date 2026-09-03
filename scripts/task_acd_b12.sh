#!/bin/bash
# ★★★★★ 顯性高頻觸發增生：現行最佳 + ac_densify 1.0（§11.80 梯度盲區的直接對策）
#
# 實測（tools/grad_blindspot.py，36 台驗證相機一次 backward）：
#   失敗區高頻殘差多 86%、位置梯度只有 41% => **每單位殘差的訊號少 4.6 倍**
#   且盲區在 **step 1,499 就存在**（當時失敗區殘差還比成功區低 0.504x）
#   => 盲區是**原因**不是結果。軌跡：1499/14999/29999/60000 = 0.331/0.453/0.364/0.218
# 機制：probs = opacity x (1 + w x AC/mean(AC))，AC = tile 內殘差的**標準差**
#   ⚠ 與 err_guided_densify 的訊號**不同**：那是 avg_pool(|render-gt|)（DC 誤差，
#     天花板只有 1.22x）；本旗標用的是零均值高頻能量 —— 正是梯度看不見的量。
# ⚠ 與已否證的十次密度控制介入**不同類**：那些調「往哪裡分配」而訊號本身是死的；
#   本旗標換的是**觸發訊號**。
# 對照 = agd2_b12（26.5602 / 0.7755 / 0.3467 / 糊掉 46.08%）。唯一變數 = ac_densify。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/acd_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.ac_densify 1.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n acd_b12
