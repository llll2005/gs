#!/bin/bash
# Official 741-frame eval with the offset VERIFIED BY POSE, not guessed.
#
# transforms.json carries the authoritative file<->pose mapping. Matching each sparse camera centre
# to it gives camera N -> image N+1 for all 741, residual 0.000000 -- so in this tool's indexing
# (files = the 1-based symlinks 0001..0741, j = int(camera_name) + offset) the correct offset is 0.
#
# scripts/merge_3x3_and_eval.sh hardcodes --offset -1, and 紀錄 recorded -1 as established "on
# training views with a 7 dB margin". That was the TRAIN set: train and test use different naming
# conventions, and the value was carried across without rechecking. The calibrator actually picked
# +0 on its own (16.63 vs 16.60/16.42); we overrode it. The side-by-side that looked like a
# different place was rendered one frame off because of that override.
set -u
cd "$(dirname "$0")/.." || exit 1
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/eval_official_test.py \
  --ckpt outputs/b3x3_full/checkpoints/merged.ckpt --offset 0 \
  --save_dir outputs/b3x3_full/official_test_vis_off0
