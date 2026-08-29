#!/bin/bash
# Merge the 16 blocks and score the official 741-frame test set. ~1 h.
#
# --offset 0 is NOT the value 紀錄 recorded for the train set (-1); the test set uses a different
# naming convention and pose matching against transforms.json (position AND orientation) puts the
# test cameras at image N+1, which is offset 0 in this tool's indexing. Passing it explicitly also
# skips the calibration gate, which cannot discriminate on this capture: neighbouring frames along
# the flight path are similar enough that all three candidates land within 0.03 dB.
# The renders go to official_test_vis so the pairing can be confirmed by eye -- on 2026-08-12 a
# wrong offset produced a plausible-looking 16.4 dB that was pure garbage.
set -u
cd "$(dirname "$0")/.." || exit 1
conda run -n gspl --no-capture-output python utils/merge_citygs_ckpts.py outputs/sh3_4x4_aerial || exit 1
CK=outputs/sh3_4x4_aerial/checkpoints/merged.ckpt
[ -f "$CK" ] || { echo "merge 未產出 $CK"; exit 1; }
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/eval_official_test.py \
  --ckpt "$CK" --offset 0 --block_dim 4 4 \
  --save_dir outputs/sh3_4x4_aerial/official_test_vis | tee -a 紀錄/official_test_merged.txt
