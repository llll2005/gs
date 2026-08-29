#!/bin/bash
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# blur budget + raised LR floor. The two mechanisms buy different things, measured against the
# noise floor (blursplit_b12 is a near-replicate of cap4m: 0.053 dB / 0.0022 LPIPS / 0.0037 tex):
#   lrfloor     PSNR +0.158 (3.0x)  LPIPS unchanged (0.0x)
#   blurbudget  PSNR  0.0x          LPIPS -0.0129 (5.9x), texratio +0.019 (5.2x)
# Neither touches the other's metric, so if they are independent this lands near 24.5 with LPIPS
# ~0.467. If they interact that is worth knowing too -- both act on primitive positions.
rm -rf outputs/blurlr_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.blur_split_budget 0.3 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.lr_final 0.0000064 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n blurlr_b12
