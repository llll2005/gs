#!/bin/bash
# ★★★★★ AC 導向的**強制縮小**：Gemini 提案裡我先前漏做的那一半
#
# acd_b12（ac_densify 1.0）輸了，但量測顯示它**根本沒搬動族群**：
#   失敗區高斯/像素 0.62 -> 0.64（+3%），半徑與 opacity 完全不變
#   ⇒ 那個實驗檢驗的是「太弱的介入」，不是機制。原因可算：AC 天花板 2.83x，
#     w=1 時最大加權 3.83x，而每次事件搬動的粒子本就少。
# 本臂**繞過取樣瓶頸**，直接作用在成因上：梯度抵消的程度取決於 footprint
#   相對於殘差空間頻率的大小 ⇒ 把 footprint 縮小就能脫離盲區。
# 每次 densify 事件把 AC 分數最高的 5% 粒子 scale x0.5（log 空間加 log(0.5)）。
# ⚠ 與 err_unlock 同構（那個壓 opacity、這個壓 scale），都是直接改參數。
# 對照 = agd2_b12（26.5602 / 0.7755 / 0.3467 / 糊掉 46.08%）。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/acs_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.ac_shrink 0.5 \
  --model.density.init_args.ac_shrink_frac 0.05 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n acs_b12
