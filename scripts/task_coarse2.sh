#!/bin/bash
# ★★ 用**修正後的 scales** 重跑 coarse（2026-08-24）
#
# `coarse_fix`（08-23）用的是舊的、汙染的 `estimated_depth_scales.json`
# —— `get_depth_scales.py:73` 的 off-by-one 到 08-23 才修（§11.6），修正後 scale 中位變 11.0%、
# p90 36.5%，而且補回一張被跳過的影像（5620 -> 5621 鍵）。
# ⚠ **coarse config 的 `depth_loss_weight.init` 是 0.5**（主線是 0）=> coarse 受 scales 影響
#    比一般跑次大得多 => `coarse_fix` 幾乎確定不是它的上限。
# 對照 `coarse_fix`：全域 val 20.966 / 0.5719 / 0.6807，1.8 小時。
#
# 用**官方原版設定** `configs/_official_citygsv2_mc_aerial_coarse_sh2.yaml`
# （記憶 citygs_official_configs：sh_degree **2** + density **全預設** grad-densify
#  + down_sample_factor 2 + 30k 步 + `diable_trimming: true`；本機 configs/citygsv2_*.yaml
#  已被改過，不可當官方依據）。
#
# ⚠ **coarse 是全域的**（不帶 block_id，吃全部 5,621 張影像）—— 沒有「某一塊的 coarse」。
# ⚠ 它用 `EstimatedDepthBlockColmap` 讀偽深度，而 dataparser 的 zfill off-by-one 已於
#   2026-08-12 修好 => **這個 coarse 會用到正確的深度**（與舊 coarse 不同世代，不可混用）。
# ⚠ 舊 coarse（aerial_train_block_all_3x 等）是**舊 SfM 座標系**，永久不可當 init（鐵律）。
# ⚠ 預估很久：5,621 張 @ down_sample 2、30k 步。跑之前確認 GPU 沒有更急的事。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/coarse_fix2
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/_official_citygsv2_mc_aerial_coarse_sh2.yaml \
  -n coarse_fix2
