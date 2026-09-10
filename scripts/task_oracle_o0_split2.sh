#!/usr/bin/env bash
# O0 第七臂**乾淨版**：子代共位（--split-jitter 0）+ 用 MCMC 真正的 Eq.9
# （`gsplat.relocation.compute_relocation`，非我原本的 scale/sqrt(K) 近似）。
# 前一次用了抖動 1.0 => `corr 前` 從 0.270 掉到 0.140（分裂在優化前就先弄壞畫面），
# 因為 Eq.9 的推導**假設子代共位**（本專案 long_axis_spread 的 docstring 早就記過）。
# 判準（與 K=1 對照，corr 前應維持在 ~0.270 才算乾淨）：
#   終值 > 0.672 且留出代價 < 0.675 dB => 分裂確實解耦 => 按足跡增生有因果依據
#   corr 前 仍明顯下降                => Eq.9 在 2DGS 上也不保外觀，此探針無法用
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/oracle_o0.py agd2_b12 sched30_b12 \
  --blk 12 --steps 300 --multi-view 6 --n-tile 6 --verify-cams 12 --split 4 --split-jitter 0
