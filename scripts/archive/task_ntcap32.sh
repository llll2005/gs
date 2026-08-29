#!/bin/bash
# ★★★★★ 顆數槓桿重開：notrim2 + cap 3.2M
#
# 為什麼現在可以重開（2026-08-25，研究總覽 §11.23）：
# 「顆數槓桿在 b12/b7 都關閉、cap 2.6M 是對的」這個結論**是在有 trim 的條件下得到的**：
#   有 trim   2.34M -> 26.377    2.70M -> 26.306    3.13M -> **23.706（崩潰）**
#   無 trim   2.60M -> 26.535
# 退化的機制就是 churn：族群越大，每 500 步剪掉的 10% 就越多，churn 越兇。
# **trim 關掉之後那個機制不存在了** => 顆數槓桿的否證前提消失，必須重測。
#
# 三個支持的量測：
#   1. 顆數對照（免費，不用新實驗）：cap30 2.70M 有 trim 26.306 vs notrim2 2.60M 無 trim 26.535
#      => 顆數少 4% 卻高 0.229 dB(7x 底) => 之前的顆數否證其實在量 trim 的傷害
#   2. VRAM 反而降了：notrim2 2.60M 用 5.51G，sched30 2.34M 用 5.57G => 有餘裕
#   3. 修正後的顆數斜率 +0.541 dB/加倍（[[cheap_lever_scaling]]）
#      => 2.60M -> 3.20M 預期 +0.162 dB = 5x 噪音底
#
# ⚠ 無 trim 時交付 = 1.0 x cap（不是 0.9）=> cap 3.2M 就是 3.2M 顆。
# ⚠ OOM 風險真實：ledger 裡 N=3.24M 跑過（5.67G），但那是不同組態。若 OOM 就是答案
#   （代表無 trim 的 VRAM 餘裕沒有想像中大），不要重試更高的 cap。
# 判準：多指標 + 最差10% + 建築低頻，對照 notrim2_b12（26.535 / 0.7840 / 0.3357 / 建築低頻 0.02447）。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/ntcap32_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.renderer.init_args.diable_trimming true \
  --model.density.init_args.cap_max 3200000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n ntcap32_b12
