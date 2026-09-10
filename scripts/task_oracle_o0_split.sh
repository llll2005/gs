#!/usr/bin/env bash
# O0 第七臂（因果測試）：與第三臂重跑完全相同，唯一差別是**影響集先分裂成 4 顆**。
# §11.98 的假說「大粒子跨 tile/跨視角共享 => 無法特化 => 動它就傷別處」預測：
#   分裂後每顆的約束變少 => **留出視角的代價應該下降**
# 對照（K=1，已測）：Δcorr(失敗) +0.401 ／ 留出視角 **-0.675 dB**（對照組 -0.556）
#   Δ 上升且留出代價明顯縮小 => 分裂確實解耦 => 按足跡增生（fpd）有因果依據
#   留出代價不變              => 「耦合」只是相關，該區本來就難 => fpd 的理由要降級
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --multi-view 6 --n-tile 6 --verify-cams 12 --split 4
