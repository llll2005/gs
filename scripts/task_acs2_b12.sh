#!/bin/bash
# ★★★★★ AC 導向強制縮小（**修正版**：加了尺度下限）
#
# ⛔ 前一版 acs_b12 在 step 14999 尺度塌陷：p1 = 1.2e-20（對照 agd2 的 4.8e-04，
#   差 16 個數量級）、**15.19% 的粒子 scale < 1e-4**（對照 0.78%），而 N 幾乎不變
#   => 那 15% 佔著 cap 卻渲染不出東西 => 量到的是「刪掉 15% 粒子」不是「縮小 footprint」。
# 成因：shrink 在**每個** densify 事件（150 步）都做，30k 步內約 193 次。
#   我曾在 step 1499（3 次事件）驗過「成員每次都在換」而判安全 —— **外推錯了**。
# 修正：加下限 = 族群 scale 中位數 / 50（仍遠小於典型 footprint，足以脫離盲區，但不歸零）。
# 對照 = agd2_b12（26.5602 / 0.7755 / 0.3467 / 糊掉 46.08%）。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/acs2_b12
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
  -n acs2_b12
