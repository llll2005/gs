#!/bin/bash
# B2: the honest version of B -- rank on ALL views, evaluate on views the ranking never saw.
#
# B gave "22% of primitives contribute nothing", and the knee in the damage curve sat exactly
# there. Two things inflate that number and both are fixed here:
#   (a) the zero-contribution fraction falls as views are added (48.9% at 9 views, 22.0% at 26,
#       still falling) -- so rank over all 284;
#   (b) B ranked and evaluated on overlapping view sets, which flatters any ranking -- so hold
#       out an evaluation set the ranking never touched.
# Whatever survives both is the number a pruning mechanism can actually be built on.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
CK=$(python -c "
import glob,re
g=[(int(re.search(r'step=(\d+)',x).group(1)),x) for x in glob.glob('outputs/churn_reg000_b12/blocks/block_12/checkpoints/*.ckpt')]
print(sorted(g)[-1][1])")
[ -n "$CK" ] || { echo "找不到 reg000 ckpt"; exit 1; }
conda run --no-capture-output -n gspl python tools/measure_diminishing_returns.py \
  --ckpt "$CK" --block 12 --rank_views 284 --eval_views 12 --holdout_eval \
  --fracs 0,5,10,15,20,25,30,40,50,70,90 \
  --out 紀錄/dim_returns_b12_reg000_full.csv
