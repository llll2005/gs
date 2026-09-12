#!/bin/bash
# ★★★★★ 等 VRAM 的表徵-容量權衡：sh3/2.34M  vs  **sh2/3.06M**（使用者提議）
#
# §11.68 實測：儲存 2.022 GB > 渲染 1.214 GB，**儲存才是 VRAM 的大頭**。
#   sh3 = 58 floats/顆（實測），sh2 = 37 ⇒ 訓練期 864 vs 552 B/顆。
#   等 VRAM 解：0.552N + 0.519N = 3.236 GB ⇒ N ~ 3.0M ⇒ cap 3.4M（交付 90%）。
# 誠實預測：**打平或小輸**。+31% 顆數 x 0.541 dB/加倍 ~ +0.21 dB；
#   而 config 註解記著 sh3 值 +0.295 dB ⇒ 淨約 -0.09 dB。兩個係數都是別的工作點量的。
# ⚠ OOM 風險：cap30（sh3/cap 3.0M）已知可跑，本臂儲存更低、渲染略高 ⇒ 應安全但要盯。
# 對照 = agd2_b12（26.5602 / 0.7755 / 0.3467 / 糊掉 46.08%）。變數 = sh_degree + cap。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/sh2cap34_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.gaussian.init_args.sh_degree 2 \
  --model.density.init_args.cap_max 3400000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sh2cap34_b12
