#!/bin/bash
# ★★★★ 跨塊驗證「梯度盲區」機制本身（~5 分鐘）：b7 的盲區應該比 b12 小
#
# b12 糊掉 46.08% ／ b7 糊掉 22.16%（差 2.1 倍）。
# 若盲區是真機制而非 b12 的特例，**b7 的「梯度per殘差 比值的比」應該明顯接近 1**
# （b12 是 0.218x）。這是對機制的**獨立跨塊檢驗**，不是對某個配方的檢驗。
#
# 判準：
#   b7 明顯 > 0.218（更接近 1）=> 盲區深度與糊掉率**同向** => 機制跨塊成立
#   b7 與 b12 一樣深            => 盲區與糊掉率無關 => 機制解釋力大幅下降，要重想
set -u
cd "$(dirname "$0")/.." || exit 1
conda run -n gspl --no-capture-output python tools/grad_blindspot.py \
  agd2_b7 agd_b7 --blk 7 --model-run agd2_b7 --step 60000
