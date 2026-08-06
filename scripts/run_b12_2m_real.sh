#!/bin/bash
# b12 at cap 2M on the best recipe (reg000: opacity_reg=0), full 60k.
#
# Why this can now be expected to survive: the rasterizer drops primitives with o <= 1/255 from
# binning. That is free -- peak alpha equals o, and the blend loop already discards alpha < 1/255
# in both passes -- and it was verified byte-identical on a fixed ckpt (maxdiff exactly 0).
# Measured on the training path (tools/vram_sweep_n.py, cloning a real ckpt to each N and running
# fwd+bwd+optimizer):
#     N       old (nothing dropped)   new
#     1.0M    2.62 GiB                2.28 GiB
#     2.0M    4.68 GiB                3.98 GiB
#     2.5M    OOM                     4.90 GiB
#     3.0M    -                       OOM
# So the ceiling moved 2.0M -> 2.5M and b12@2M went from hugging the line to ~0.7 GiB of slack.
#
# The caveat this run exists to settle: that sweep used CLONED points, i.e. the opacity/scale
# distribution of a converged 1M model. A real 2M run carries more fog and smaller scales, which
# is exactly why b12 historically died around 1.5M. If it dies anyway, the log will say at which
# step and N -- that difference is the thing worth knowing.
#
# Baseline to beat: reg000 = 23.158 PSNR at 0.90M (cap 1M). If 2M is reachable but scores no
# better, "b12 更多點有沒有用" gets answered the clean way -- same kernel, same recipe, only cap.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
NAME=b12_cap2m_reg000_exact
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
  -n "$NAME" > "logs/$NAME.log" 2>&1
rc=$?
st=$(sed -n '5p' "outputs/$NAME/blocks/block_12/train_status.txt" 2>/dev/null)
psnr=$(grep -oE 'val/psnr: [0-9.]+' "outputs/$NAME/blocks/block_12/results.txt" 2>/dev/null | grep -oE '[0-9.]+$')
if [ -n "$psnr" ]; then
else
  err=$(grep -oE '(OutOfMemoryError|Error)[^"]{0,70}' "logs/$NAME.log" | tail -1)
fi
