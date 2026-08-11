#!/bin/bash
# The comparison the whole project is FOR, and it has never been run.
#
# Every number we have -- 24.67, 24.50, 24.35 -- is per-block val on the reconstruction split
# (val subset of train), on one block of a 5x5 grid. The paper's 27.23 is a MERGED, whole-scene
# model scored on the official 741-frame test set. Those are different yardsticks, and we have
# never put ourselves on theirs.
#
# We can: outputs/b3x3_full has all nine blocks trained (per-block val 20.82-25.16, mean 22.59).
# Merging them and scoring the 741 frames costs about an hour and produces the first number that
# is directly comparable to 27.23. Until that exists, every claim about the gap -- including my own
# "26 dB needs 100x primitives" -- is extrapolation from the wrong measurement.
#
# ⚠ These nine were trained with the OLD recipe (SB colour, reg000, cap 3M), not the current best
#   (SH3, reg 0.002, which is +2.3 dB over reg000's own control on b12). So this measures the
#   PIPELINE, not our best quality. Read it as a floor and as validation that merge+eval works.
# ⚠ Merge fidelity was questioned before (Gaussian drift made an earlier merged eval untrustworthy,
#   S8/S9 scored 19.51). If the merged number is far below the per-block mean, suspect the merge
#   before concluding anything about quality.
set -u
cd "$(dirname "$0")/.." || exit 1
python utils/merge_citygs_ckpts.py outputs/b3x3_full
CK=$(ls -t outputs/b3x3_full/*.ckpt 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "merge 沒有產出 ckpt"; exit 1; }
echo "[merge] $CK"
PYTHONPATH=. python tools/eval_official_test.py --ckpt "$CK" --block_dim 3 3
