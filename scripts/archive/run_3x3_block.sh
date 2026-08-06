#!/bin/bash
# One block of the 3x3 full-scene run. Usage: run_3x3_block.sh <block_id>
#
# Purpose: get a MERGEABLE full scene, so the official 741-view held-out test set can finally be
# scored. Per-block scoring on that test set is meaningless -- a block model owns ~1/9 of the city
# while an aerial test frustum covers far more, so most of the frame renders empty and PSNR sits at
# ~15 dB no matter how good the block is (measured 2026-08-01 on b7, whose own training views
# reconstruct cleanly). Only the merged model can be compared against anything published.
#
# All blocks train into ONE output dir, because merge_citygs_ckpts.py consumes
# outputs/<name>/blocks/block_*/ -- separate per-block dirs cannot be merged without shuffling
# files around afterwards.
#
# Settings are locked to what block 4 already ran with (cap 3M, opacity_reg 0,
# screen_size_prune_px 300). Do not "improve" them for later blocks: a merged scene assembled from
# blocks trained under different regimes is not interpretable, and changing one would mean redoing
# block 4's 10.5 hours. opacity_reg=0 is also the current best single-block recipe (best val LPIPS
# and the least needle-like geometry of the sweep).
#
# ~10.5 h per block on this card (block 4: 2026-07-29 22:28 -> 07-30 09:01, reached 2.70M
# primitives at 60k steps). Eight blocks is roughly 3.5 days of continuous GPU.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
BLK=${1:?用法: run_3x3_block.sh <block_id>}
NAME=b3x3_full
D3=data/matrix_city/aerial/train/block_all/depth_init_3x3
[ -f "$D3/block_${BLK}.ply" ] || { echo "缺 $D3/block_${BLK}.ply"; exit 1; }

# Only this block's dir -- NOT outputs/$NAME, which holds every block trained so far.
rm -rf "outputs/$NAME/blocks/block_${BLK}"

conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial_3x3.yaml \
  --model.initialize_from "$D3/block_${BLK}.ply" --data.parser.block_id "$BLK" \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 3000000 \
  --model.metric.init_args.opacity_reg 0.0 \
  -n "$NAME" > "logs/${NAME}_blk${BLK}.log" 2>&1
rc=$?
n=$(ls -d outputs/$NAME/blocks/block_*/ 2>/dev/null | wc -l)
printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "3x3-full" \
  "BLOCK $BLK rc=$rc — outputs/$NAME 現有 $n 塊" >> logs/quad_progress.log
exit $rc
