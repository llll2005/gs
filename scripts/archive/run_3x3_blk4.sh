#!/bin/bash
# 3x3 block 4 (centre cell of the 3x3 grid), full 60k, current best recipe.
#
# Why 3x3: a 3x3 block covers 16/9 = 1.78x the area of a 4x4 one, so at equal primitive density it
# needs ~3.4M -- right at the measured ceiling. 2x2 would need ~7.6M and is out of reach.
# The point is not VRAM but SEAMS: merged quality measured 7.0 dB below single-block, and going
# 25 -> 16 -> 9 blocks cuts seam count 40 -> 24 -> 12. That shows up in the global number the
# north-star comparison actually uses.
#
# cap 3M is deliberately above the 2.5M ceiling measured on cloned geometry: real 2M geometry cost
# 1.4x what cloning predicted, so this may OOM. If it does, the ledger records the step and N,
# which is the number worth having.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
D3=data/matrix_city/aerial/train/block_all/depth_init_3x3
BLK=4
[ -f "$D3/block_${BLK}.ply" ] || { echo "缺 $D3/block_${BLK}.ply（prep_3x3 未完成）"; exit 1; }
# GPU serialisation is the runner's job now (scripts/runner.sh waits before每個 non-[cpu] task);
# this script used to carry its own copy of that loop. Twelve scripts had duplicated it.
rm -rf outputs/b3x3_blk${BLK}_reg000
conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial_3x3.yaml \
  --model.initialize_from "$D3/block_${BLK}.ply" --data.parser.block_id $BLK \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 3000000 \
  --model.metric.init_args.opacity_reg 0.0 \
  -n b3x3_blk${BLK}_reg000 > logs/b3x3_blk${BLK}_reg000.log 2>&1
