#!/bin/bash
# 4x4 full scene, 16 blocks, SH3 recipe on the FIXED data mapping. ~7 days.
#
# This is the only run that can produce a number comparable to the paper's 27.23: that is a merged
# whole-scene model scored on the official 741-frame held-out set, and every number we have is
# per-block val on the reconstruction split. The one previous attempt (b3x3_full merged -> 16.42)
# was trained against neighbour-frame GT and is void.
#
# Why 4x4: each block gets the whole 6 GB, so scene capacity = blocks x per-block cap.
#   3x3   9 x 2.52M = 22.7M    4 days
#   4x4  16 x 2.34M = 37.4M    7 days     <- 65% more capacity than 3x3
#   5x5  25 x 2.34M = 58.5M   11.5 days
# Partition granularity is itself the capacity lever, and needs no new mechanism.
#
# Each block writes to outputs/sh3_4x4/blocks/block_N and appends START/DONE/DIED to
# logs/quad_progress.log, so progress is visible per block and a crash costs one block, not the run.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# -n is BOTH the config basename (configs/<n>.yaml) and the output name; --config is ambiguous
# with --config_name/--config_dir and argparse rejects it. Caught by --dry-run before committing
# seven days of GPU to a run that would have died in the first second.
conda run -n gspl --no-capture-output python utils/train_citygs_partitions.py \
  -n sh3_4x4_aerial \
  --init_mode depth \
  --depth_init_dir data/matrix_city/aerial/train/block_all/depth_init_4x4
