#!/bin/bash
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# blur budget + long-axis spread. Runs only after blurbudget reported alone (24.353 / LPIPS 0.4674
# / texratio 0.3783 -- PSNR flat, LPIPS 5.9x the noise floor, texratio 5.2x): two mechanisms at
# once cannot be attributed. Expected to be super-additive -- blur split selects primitives that
# alone cover a large patch, and MCMC stacks every child at the host's centre, which cannot cover
# that patch. See 紀錄/研究總覽.md §11.9.3.
rm -rf outputs/blurlas_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.blur_split_budget 0.3 \
  --model.density.init_args.long_axis_spread 0.5 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n blurlas_b12
