#!/bin/bash
# ★★ coarse-init + 當前最佳機制，但 coarse 用**修正後的 scales**（2026-08-24）
#
# 與 `coarseft_b12`（26.518，現行最佳）**唯一的差別是 coarse 那一端的 scales**。
# 見 `task_coarse2.sh` 的檔頭：修正後 scale 中位變 11.0%，而 coarse 的
# `depth_loss_weight.init` 是 0.5 => 它比一般跑次更吃 scales 的正確性。
# 對照 coarseft_b12：平均 26.510 / 中位 26.56 / 最差 17.41 / 建築低頻 0.02504。
#
# 這一臂回答：**我方調出來的機制，疊在官方的 coarse-init 流程上，能不能贏過 depth-init？**
# 先前的 coarse 對照（`official_ft_blk5` 23.342、實驗 A 的 22.27）都是 **GT bug 期**且
# 用**舊 coarse**，全部不可引用（研究總覽 §7.0）。
#
# 本臂 = `sched30` 的全部機制（cap 2.6M / densify_until 30k / opacity_reg 0.002 /
#        lambda_normal 0 / depth_loss 0），只把 init 從 depth-init 換成 `coarse_fix`。
#
# ⚠ `initialize_from` 給**目錄**即可（`citygs_ft60k_coarse3x.yaml` 檔頭註明 fit 會用
#   `search_load_file` 自動找最新 ckpt）。若失敗，改指到明確的 .ckpt 檔。
# ⚠ `_initialize_from_trained_model` **不做 block 過濾**（記憶 _ctx 陷阱）—— 官方不裁切
#   coarse，我們也不裁，讓起始 trim 自己處理。
# ⚠ 判讀對照 `sched30_b12`（depth-init，26.377）與 `fixdepth_b12`（修好的 depth-init）。
#   多指標 + 看最差 10% + 看圖。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
[ -d outputs/coarse_fix2 ] || { echo "✘ outputs/coarse_fix2 不存在 —— 先跑 task_coarse.sh"; exit 1; }
rm -rf outputs/coarseft2_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from outputs/coarse_fix2 \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n coarseft2_b12
