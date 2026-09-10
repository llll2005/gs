#!/bin/bash
# ★★★★★ O0 oracle：凍結拓撲，只讓失敗 tile 的影響集重新優化（~20 分鐘）
#
# 回答最後一個分岔：
#   救得回來 => 現有 basis 夠，是**優化到不了** => 該改優化（LR/排程/觸發訊號）
#   救不回來 => 才輪到 O1（加 basis / 改 basis）
# **必須先 O0 再 O1** —— 順序反了因果鏈就不乾淨。
#
# 已採納外部指出的四個要點：影響集按光線覆蓋取（非中心落點）／對照組同流程跑成功 tile／
# 量鄰域避免只是把誤差搬走／措辭限制為「此局部預算下無法恢復」。
set -u
cd "$(dirname "$0")/.." || exit 1
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py \
  agd2_b12 sched30_b12 --blk 12 --steps 300 --n-tile 8
