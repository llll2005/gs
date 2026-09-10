#!/bin/bash
# ★★★★★ SfM-init 跨塊確認：block_7（新最佳的必要條件，鐵律）
# b12 實測（同配方，唯一變數 `initialize_from null`，SfM 320,513 顆起步）：
#   5,680 +0.72 ／ 11,360 +0.78 ／ 17,040 **+0.61（7.5sd）** ／ 22,720 +0.73 ／ 28,400 +0.37 dB
#   PSNR/SSIM/紋理比 三指標**每個檢查點同向**
# ⚠ notrim2 與 speed3 都是「b12 贏、b7 不複製」=> **沒有 b7 不可宣稱**。
# ⚠ b7 的 SfM 點數與 b12 不同（內容重的塊），起始顆數會不一樣，這是機制的一部分不是瑕疵。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/speed3_sfminit_b7
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from null \
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
  -n speed3_sfminit_b7
