#!/bin/bash
# ★★★★★ 驗證 14% 加速為真且不掉分（同時是第 4 個同組態樣本）
#
# 累計省時（全部不換品質，§11.35/§11.37/§11.42）：
#   零權重 loss 閘門  -39 ms  -9%     位元級無損
#   fused_ssim       -24 ms  -4.9%   數值差 1e-12（相對 1.8e-6）
#   depth_to_normal   -4 ms  -1%     位元級無損（本臂用 skip_surf_normal）
#   => 約 -14%，9.6h 應降到 ~8.2h
#
# 判準（`tools/cmp_runs.py`，最後 4 點平均）：
#   對照 sched30_b12 / sched30rep_b12 / egd_b12（三個同組態樣本）
#   四指標都在 ±3sd 內 => 加速無損，寫進配方
#   任一項超出      => fused_ssim 的 1.8e-6 相對誤差被放大了，退回 fused_ssim: false
# ★ 附帶：這是第 4 個同組態樣本，把 sigma 再收緊（目前 n=3, sd=0.0277）。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sched30fast_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.renderer.init_args.skip_surf_normal true \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sched30fast_b12
