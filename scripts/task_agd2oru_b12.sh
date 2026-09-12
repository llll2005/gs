#!/bin/bash
# ★★★★★ 收割期移除 opacity L1（§11.62 三臂小實驗的 B 臂，兩個判準都過且超額）
#
# 小實驗實測（2,500 步收割期）：判死 +3.57pp -> **-0.06pp**（損耗歸零）、
#   確信 o>0.5 +2.91pp -> **+8.24pp**（兩極化不但沒被破壞，還快 2.8 倍）
# ⚠ 不等於 opacity_reg=0：L1 在**生長期照常**（回收器在，它有下游），
#   只在收割期關掉（回收器已 early-return，它只剩損耗）。
# 對照 = agd2_b12（26.5602 / 0.7755 / 0.3467 / 糊掉 46.08%）。唯一變數 = 這個旗標。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/agd2oru_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.opacity_reg_until_iter 30000 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n agd2oru_b12
