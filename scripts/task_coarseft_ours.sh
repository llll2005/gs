#!/bin/bash
# ★★★ 用**我方配方訓練的 coarse** 當 init + 我方機制（2026-08-24）
#
# 三方比較（都是 b12、都用 sched30 的機制）：
#   `sched30_b12`      26.377   depth-init（PLY 路徑）        trimming **啟用**
#   `coarseft_b12`     26.518   官方 coarse（ckpt 路徑）      trimming **意外被關掉**（§11.11）
#   `coarseft_ours_b12`   ?     我方 coarse（ckpt 路徑）      trimming **啟用**（我方 renderer）
#
# ⇒ 本臂與 `sched30_b12` 的差別**只剩 init**（trimming 兩邊都開）
#   => 這才是「coarse-init 到底值多少」的乾淨答案，`coarseft_b12` 的 +0.141 混了兩個變數。
# ⚠ 判準多指標 + 最差 10% + 看圖。噪音底 PSNR 0.0325。
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
[ -d outputs/coarse_ours ] || { echo "✘ outputs/coarse_ours 不存在 —— 先跑 task_coarse.sh"; exit 1; }
rm -rf outputs/coarseft_ours_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from outputs/coarse_ours \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n coarseft_ours_b12
