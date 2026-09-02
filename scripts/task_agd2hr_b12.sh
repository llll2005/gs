#!/bin/bash
# ★★★★★ 收割期回收死粒子（使用者提案；§11.63 分析 + 三臂小實驗的 C 臂）
#
# 小實驗實測（2,500 步收割期）：判死 +3.57pp -> **-3.48pp**（死粒子幾乎消滅）、
#   確信 o>0.5 +2.91pp -> +2.61pp（兩極化只慢 0.30pp）
# ⚠ 我原本擔心「relocate 會 reset 宿主 Adam 動量、傷收割期收斂」—— 實測影響極小。
# ⚠ 與已知淨負的 churn（notrim2 的 27,700 步）是同一操作，差別在目的地（absgrad 導向）與強度。
# 對照 = agd2_b12（26.5602 / 0.7755 / 0.3467 / 糊掉 46.08%）。唯一變數 = harvest_relocate。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/agd2hr_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.harvest_relocate true \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n agd2hr_b12
