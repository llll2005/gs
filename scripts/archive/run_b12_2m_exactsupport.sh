#!/bin/bash
# Does b12 reach 2M with the lossless half of EXACT_SUPPORT? Previously it OOM'd around 1.5M.
#
# The rasterizer now drops primitives with o <= 1/255 from binning entirely. That is provably
# free: peak alpha equals o, and the blend loop already discards alpha < 1/255 in both the
# forward and backward pass, so those primitives contribute nothing to the image or to any
# gradient. Verified on a fixed ckpt: renders are byte-identical (maxdiff exactly 0), 49% of
# primitives drop out of binning, peak render VRAM falls 20.3% (1747 -> 1393 MiB).
# (The second half -- the exact support radius, another -32% -- is NOT enabled here: it is not
# bit-identical and needs its own quality A/B first.)
#
# Step budget (corrected after the first attempt): 8k was chosen on the theory that +5% per
# event compounds 0.34M -> 2M in ~5.4k steps. It does not -- early events give back most of the
# gain, and reg000 on the same recipe needed ~10.5k steps just to reach 0.90M. The 8k run
# stopped at 0.77M, still under the ~1.5M where b12 used to die, so it answered nothing.
#
# Also note the -20.3% was measured on the INFERENCE path (forward only). The training peak also
# carries backward activation and Adam state, so the saving does not transfer 1:1 -- at 0.77M
# this run sat at 1.84G, not obviously below reg000 at comparable N. Watch VRAM at matched N.
#
# Read: survives to 2M => the lossless drop alone moved the ceiling, and the cap table should be
# regenerated against the new binary. Dies below 2M => report the step and N it died at; the
# 20.3% was not enough and the exact-radius half (or K) is needed.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
NAME=b12_2m_exactsupport_16k
# Progress ledger: START/DONE/DIED are emitted by the training process itself
# (internal/callbacks.py TrainConsole._progress) with structured fields straight from the live
# footer. Scripts must NOT write those lines: the old approach grepped them back out of the
# training log, which dragged tqdm bars and ANSI escapes into the ledger and let every script
# drift to its own column widths. Use log() only for things Python cannot know -- queue-level
# events like waiting on the GPU or a batch boundary.
log () { printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "$1" "$2" >> $PROG; }
rm -rf "outputs/$NAME"
conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 2000000 \
  --model.metric.init_args.opacity_reg 0.0 \
  --trainer.max_steps 16000 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
rc=$?
st=$(sed -n '5p' "outputs/$NAME/blocks/block_12/train_status.txt" 2>/dev/null)
if [ $rc -eq 0 ]; then
else
  err=$(grep -oE '(OutOfMemoryError|Error)[^"]{0,70}' "logs/$NAME.log" | tail -1)
fi
