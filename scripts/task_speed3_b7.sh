#!/bin/bash
# ★★★★★ 三項省時一次驗證：現行最佳 + fast_noise + noise_gate_eps 1e-3
#   （`densify_blind_report` 預設 False，err_score 累積已自動關閉）
#
# CUDA event 精確歸因（tools/knob_cost.py，2026-09-05，N=2.34M）：
#   _add_xyz_noise 全量               61.24 ms  10.17%
#   + noise_gate_eps 1e-3             20.10 ms   3.34%   => 省 41.14 ms = **6.83%**
#   + fast_noise（代數改寫，單獨）     35.76 ms   5.94%   => 省 25.48 ms = **4.23%**
#   _accumulate_error_score            3.83 ms   0.64%   => 已閘門，**訓練行為不變**
#   兩者互補（一個減 N、一個減每顆成本）=> 預估合計約 8%
#
# ⚠ 為什麼需要端到端驗、不能只看 ms：
#   `noise_gate_eps` 改變 **RNG 串流**（randn 只對子集抽）=> 非位元等價。
#   `fast_noise` 代數等價但浮點運算次序不同 => 也非位元等價。
#   前車之鑑：`fused_ssim` 看起來等價、省 4.9%，但 SSIM 是 **-4.06sd 的離群值**，已撤回（§11.44）。
# ⚠ 安全性（已算）：eps=1e-3 被跳過的粒子每步位移 <= scale^2*noise_lr*lr*eps = 4.15e-8
#   => 60,000 步隨機遊走累積 = 自身尺寸的 **0.15%**。
#
# 判準：對照 agd2_b7（25.1361 / SSIM / LPIPS / 紋理比），四指標都在 ±3sd 內 => 採用。
#       ⚠ 優先看 SSIM（噪音底 0.0005，四個指標裡唯一沒被低估的）。
#       ⚠ 時間本身也是判準：9.6h 應降到約 8.8h。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/speed3_b7
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 1e-3 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n speed3_b7
