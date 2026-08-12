#!/bin/bash
# Does "few primitives, high quality" work now that the GT mapping is fixed? ~1.5 h, not 11.
#
# The 24k recipe runs densification_interval 350, above the 231.5 break-even against the
# contribution prune, so the count DECAYS instead of growing: v1clean_24k_b12 ended at 77,089
# primitives and 22.022 against a 3.6M model's 24.307 -- 23x fewer for 1.37 dB. That looked like
# the strongest evidence for the cost-aware thesis until the building texture ratio came back
# 0.145 vs 0.393 and showed it was "few and blurry", the 1.37 dB being flattered by b12's water.
#
# But every one of those numbers was measured against neighbour-frame GT (memory
# zfill_gt_offset_bug). Training on a shifted target is equivalent to training on a blurred one,
# and a small model has the least capacity to absorb that -- so the low-count arm is exactly where
# the bug should have hurt most, and where the fix should show up largest. Worth 1.5 h before
# committing to the large-count direction.
#
# Same config as v1clean_24k_b12, nothing changed but the (now fixed) dataparser.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/v1_refix_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_24k_aggr_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n v1_refix_b12
