#!/usr/bin/env bash
# O0 第二臂：只把 means 的 LR 換成訓練末期的**真實值 1.767e-6**，其餘維持 oracle 的 1e-3。
# ⚠ 從 config 推的 6.4e-7 是錯的：ckpt 的 optimizer_states 顯示末期實際是 1.767e-6（2.8 倍）。
# ⚠ 本臂同時做**多視角覆核**（--verify-cams 6），那才是現在最關鍵的數字：
#   §11.89 已推翻「動不了」的解釋（Adam 步長兩組一樣），剩下的主嫌是**多視角衝突**。
# 問的是：oracle 之所以救得回來，是不是只因為它用了 1,563 倍的位置學習率？
#   仍救得回來 => 位置移動不是瓶頸，瓶頸在**訊號**（梯度盲區）=> 繼續走觸發訊號那條線
#   救不回來   => **末期 LR 太低**就是主因之一，而那是一個 config 數字，幾乎零成本
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --means-lr 1.767e-6
