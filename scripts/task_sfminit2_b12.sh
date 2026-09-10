#!/bin/bash
# ★★★★★ SfM-init 重做：**唯一變數 initialize_from null**，對照 speed3_b12
#
# 為什麼要重做（2026-09-11）：`speed3_sfminit_b12` 是手動啟動的，resolved config 與
# `speed3_b12` 差了四項，不是單變數：
#   initialize_from  depth_init.ply -> null      <- 想測的
#   dynamic_strips   false -> **true**           <- 污染源
#   strip_vram_target_gb 5.4 -> 5.2 / strip_safety 0.6 -> 0.85
#   skip_surf_normal false -> true
# `dynk_K.log` 600 個採樣點：**K>1 佔 77.7%、K>=6 佔 69.3%**，而
# `_strip_forward_backward` 的 docstring 自承逐條帶 SSIM 是「boundary-window approximation」
# ⇒ 77.7% 的訓練步跑在被改過的 loss 上 ⇒ 那次的 26.21@56.8k 不可用來判定 SfM-init。
#
# K 是白付的（tools/strip_k_audit.py，純 CPU）：
#   兩個 60k 模型在同參數下**都**被判 K=6（load 中位 5.4M vs 6.9M）
#   predict 的 budget = 0.85*(5.2-0.8) = 3.740 GiB
#   而 depth-init 同 N=2.34M 用 K=1 **實測**峰值 4.87 GB 沒 OOM（卡 6.1 GB）
#   => 預算訂低約 1.1 GB。本次沿用 speed3 的 config（dynamic_strips 預設 false）。
#
# 判準：對照 speed3_b12（26.6957 / SSIM 0.781 / LPIPS 0.332 / 紋理比 0.473），
#       用 tools/cmp_runs.py 的「最後四個 val 點平均」。噪音底 PSNR 3sd = 0.24。
#       ⚠ 依鐵律，新最佳要 b7 也驗過才算數。
# 已知（未受污染，K=1 期間）：5,680 +0.72 / 11,360 +0.78 dB —— 早期領先是真的。

set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sfminit2_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from null \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 1e-3 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sfminit2_b12
