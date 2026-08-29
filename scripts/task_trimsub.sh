#!/bin/bash
# ★★★★★ trim 用 24 台相機 vs 全部 284 台，剪掉的會是同一批嗎？（約 8 分鐘）
#
# 依據（§11.48 逐 op profile）：`before_training_step`（trim pass）佔 profile 視窗 **80%**
# （53.85s / 67.50s），284 台 => 約 190 ms/次全幅 transmittance 渲染。
# 正常訓練每 500 步一次、到 30k 共 60 次 => 約 3,240 秒 ≈ **9 小時跑次的 10%**。
# 而 `_measure_multiview_contribution`（vpc 用）**早就在用 24 台取樣**，trim 卻用全部 284 台。
#
# ⚠ 先前我把 profile 裡 to_device 的 48.2s 讀成「搬運是渲染的 5 倍」—— **誤讀已收回**：
#   CUDA 非同步，渲染的 GPU 時間在下一次 .to() 才被迫同步，cProfile 記到搬運頭上。
#
# 做法：探針在**同一次 trim 事件內**額外用 stride=12（284/12≈24 台）再算一次 contribution，
# 只讀、不改變剪枝。用同一條程式碼路徑，不重造 main.py 的組裝。
# ⚠ ckpt init 讓模型一開始就在 2.34M 顆（真實工作點）；trim 改成每 100 步一次好在 400 步內取樣 4 次。
#
# 判準：Spearman > 0.99 **且** 底部 10% 遮罩重疊 > 95% => 可以取樣，直接省約 9%
#       否則 => trim 需要全視角，此路不通
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/trimsub
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from "outputs/agd_b12/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt" \
  --data.parser.block_id 12 \
  --trainer.max_steps 400 \
  --trainer.enable_checkpointing false \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 400 \
  --model.renderer.init_args.trim_subsample_probe 12 \
  --model.renderer.init_args.contribution_prune_from_iter 100 \
  --model.renderer.init_args.contribution_prune_interval 100 \
  --model.density.init_args.cap_max 2600000 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n trimsub 2>&1 | tee logs/trimsub.txt | grep -E "trim-subsample|DIED|Error"
echo; echo "===== 開獎 ====="; grep -a "trim-subsample" logs/trimsub.txt || echo "(無輸出)"
