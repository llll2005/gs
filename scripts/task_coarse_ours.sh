#!/bin/bash
# ★★★ 用**我方配方**訓練 coarse（使用者提議，2026-08-24）
#
# 現有的 `coarse_fix` 用的是官方原版設定，和我方配方差很多：
#   sh_degree 2 vs 3 ／ CityGSV2 grad-densify vs MCMC ／ **diable_trimming: true** vs 啟用
#   depth_loss 0.5 vs 0 ／ lambda_normal 0.0125 vs 0 ／ opacity_reg 預設 vs 0.002 ／ 無 cap vs 有
#
# ★ 附帶價值：coarse 的 renderer 會被微調端整個繼承（ckpt 路徑，§11.11）。
#   用我方 config 訓練 coarse => 它的 renderer 是**我方的（trimming 啟用）**
#   => 後續 `coarseft_ours` 不會再有「意外關掉 trim」那個混淆。
#
# 設計選擇：
#   全域（不帶 block_id；`block_id` 預設 None = 吃全部 5,621 張）
#   down_sample_factor 2（coarse 就該便宜；我方主線是 1.2）
#   max_steps 30,000 + **means_lr_scheduler.max_steps 一起改**（只改前者 LR 會停在 10%）
#   densify_until 15,000（50%，sched30 的比例）
#   cap_max 550,000 => 交付約 495,000（trim 造成 0.9x，§12.8），**刻意對齊 `coarse_fix` 的
#     494,317 顆** => 「同樣大小的 init，只差訓練演算法」
#   ⚠ **`train_max_num_images_to_cache 512` 必須設**：我方 config 沒設它，預設 -1 = 全部快取，
#     5,621 張 x ~23 MB ≈ 129 GB，會撐爆 46 GB RAM。（官方 coarse config 有設，我方沒有。）
#
# ⚠ 但書：我方配方的 `depth_loss 0` 與 `lambda_normal 0` 是在**單塊 b12** 上測出來的；
#   全域 coarse 的工作是幾何，這兩個關掉未必對。這是本臂的主要風險，但使用者要的是
#   「用我們的組態」，所以照原樣搬，不另外調。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/coarse_ours
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --data.parser.down_sample_factor 2 \
  --data.train_max_num_images_to_cache 512 \
  --trainer.max_steps 30000 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 30000 \
  --model.density.init_args.cap_max 550000 \
  --model.density.init_args.densify_until_iter 15000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n coarse_ours
