#!/bin/bash
# 顆數消融：同 60k 排程下把顆數釘在 ~80k，判「少而準」成不成立。
# 設計理由與對照數據全部寫在 configs/mcmc_2dgs_sb_60k_cap80k_aerial.yaml 檔頭。
# 判準＝建築區紋理比（tools/rescore_by_content.py），不是整體 PSNR（被水面稀釋，已知無鑑別力）。
set -euo pipefail
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=cap80k_60k_b12
rm -rf "outputs/$NAME"
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_cap80k_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
