#!/bin/bash
# One 5x5 block with the 24k aggressive recipe. Usage: run_24k_aggr_block.sh <block_id>
#
# Purpose: find out whether our recipe survives 24k steps at cap 3M, and what geometry and quality
# it lands at. If it does, a 25-block full scene costs ~2 days instead of ~5, and the full-scene
# official held-out number -- the one thing we still do not have -- becomes affordable.
#
# Reports depth-bias and per-content-group quality at the end so the run is self-scoring. The
# reference to beat is the 60k cap2M run on the same block: slope 1.082, corr 0.903.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
BLK=${1:?用法: run_24k_aggr_block.sh <block_id>}
NAME=aggr24k_v3_b${BLK}
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_${BLK}.ply
[ -f "$PLY" ] || { echo "缺 $PLY"; exit 1; }
rm -rf outputs/$NAME
conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_24k_v3_aerial.yaml \
  --model.initialize_from "$PLY" --data.parser.block_id "$BLK" \
  -n $NAME > logs/$NAME.log 2>&1
rc=$?
printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "aggr24k" "BLOCK $BLK rc=$rc" >> logs/quad_progress.log
if [ $rc -eq 0 ]; then
  ck=$(find outputs/$NAME -name '*step=24000.ckpt' | head -1)
  echo "===== 幾何（對照：同塊 60k cap2M = slope 1.082 / corr 0.903）====="
  conda run --no-capture-output -n gspl python tools/measure_depth_bias.py \
    --ckpt "$ck" --block "$BLK" --block_dim 5 5 --content_bounds --views 10 2>&1 \
    | grep -vE "pkg_resources|__import__|Warning"
fi
exit $rc
