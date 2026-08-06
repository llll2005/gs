#!/bin/bash
# B: does pruning cost accelerating quality, and is v_i informative at all?
#
# Wrapped in a script rather than inlined in the queue because the checkpoint lookup needs a regex
# with a backslash, and passing that through the queue file was what made the runner loop forever
# on 2026-07-29 (awk -v expands escapes). Queue lines stay simple; complexity lives here.
#
# Smoke result at low sample count already showed v_i is strongly informative: cutting 60% by
# contribution distorted the render less (47.0 dB vs the unpruned render) than cutting 30% at
# random (26.5 dB). This run is the full sweep.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
CK=$(python -c "
import glob,re
g=[(int(re.search(r'step=(\d+)',x).group(1)),x) for x in glob.glob('outputs/churn_reg000_b12/blocks/block_12/checkpoints/*.ckpt')]
print(sorted(g)[-1][1])")
[ -n "$CK" ] || { echo "找不到 reg000 ckpt"; exit 1; }
conda run --no-capture-output -n gspl python tools/measure_diminishing_returns.py \
  --ckpt "$CK" --block 12 --rank_views 24 --eval_views 8 \
  --fracs 0,10,20,30,40,50,60,70,80,90 \
  --out 紀錄/dim_returns_b12_reg000.csv
