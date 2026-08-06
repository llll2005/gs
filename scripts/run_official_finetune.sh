#!/bin/bash
# One 4x4 block fine-tuned from the official coarse, with the upstream config.
# Usage: run_official_finetune.sh <block_id>
#
# This is the experiment the coarse run could not answer. Measuring the coarse's own depth was the
# wrong test: a coarse is a deliberately rough proxy (499,682 primitives for the whole city, val
# PSNR 20.9), its geometry is not meant to be accurate, and its readings swing wildly with the view
# sample (corr +0.585 over 8 views vs -0.193 over 6). What matters is the geometry that comes OUT of
# fine-tuning from it -- that is the artefact our own per-block models are comparable to.
#
# 4x4 = 16 blocks is the upstream layout for MatrixCity aerial (block_dim [4,4] in their config),
# not our 5x5 or 3x3.
#
# OOM outlook, revised downward by what the coarse run showed: gradient densification is uncapped,
# which is the standing risk, but the starting point is a per-block slice of a 499k model rather
# than the 3.83M SfM cloud that killed the coarse at 1.2x, and the coarse itself peaked at 1.72 GiB.
# Run ONE block and watch before committing the other fifteen.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
BLK=${1:?用法: run_official_finetune.sh <block_id>}
NAME=official_ft_blk${BLK}
COARSE=outputs/official_coarse_sh2/checkpoints/epoch=6-step=30000.ckpt
[ -f "$COARSE" ] || { echo "缺 coarse: $COARSE"; exit 1; }
rm -rf outputs/$NAME
conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/_official_citygsv2_mc_aerial_sh2_trim.yaml \
  --data.parser.block_id "$BLK" \
  -n $NAME > logs/$NAME.log 2>&1
rc=$?
printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "official-ft" "BLOCK $BLK rc=$rc" >> logs/quad_progress.log
[ $rc -eq 0 ] && conda run --no-capture-output -n gspl python tools/measure_depth_bias.py \
  --ckpt "$(find outputs/$NAME -name '*step=60000.ckpt' | head -1)" \
  --block "$BLK" --block_dim 4 4 --views 10 2>&1 | grep -vE "pkg_resources|__import__|Warning"
exit $rc
