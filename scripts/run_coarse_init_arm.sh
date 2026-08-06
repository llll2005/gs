#!/bin/bash
# Full arm: same recipe, coarse-init instead of depth-init.
#
# Three things this separates that are currently tangled:
#  1. Is the floater shell a property of depth-init's starting geometry, or of opacity_reg=0?
#     If coarse-init at reg=0 also stays on the ground, the cause was the starting point.
#  2. Our claim "depth-init already fixes initialisation, so MCMC's relocation is redundant" is
#     what justified reg000. Under coarse-init that claim should stop holding.
#  3. The official pipeline is coarse-init, so this is the comparable route for the self-run
#     baseline the eval protocol requires.
#
# Bails out if the compatibility check never confirmed the SB colour model survived, since a
# silent fall back to sh0 would make this a comparison of two different models.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
if ! grep -q "floats/point=25" logs/coarse_init_check.log 2>/dev/null; then
  echo "中止：coarse_init_check 沒有確認 SB 存活（floats/point=25）——先看 logs/coarse_init_check.log"
  exit 1
fi
CK="outputs/coarse_mc_aerial_1.2x/checkpoints/epoch=6-step=30000.ckpt"
rm -rf outputs/coarse_init_reg000_b12
conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from "$CK" --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 1000000 \
  --model.metric.init_args.opacity_reg 0.0 \
  -n coarse_init_reg000_b12 > logs/coarse_init_reg000_b12.log 2>&1
rc=$?
C=$(find outputs/coarse_init_reg000_b12 -name config.yaml | head -1)
cp outputs/coarse_init_reg000_b12/blocks/block_12/results.txt outputs/coarse_init_reg000_b12/blocks/block_12/results_train.txt 2>/dev/null
[ -n "$C" ] && conda run --no-capture-output -n gspl python main.py test --config "$C" --save_val >> logs/coarse_init_reg000_b12.log 2>&1
conda run --no-capture-output -n gspl python tools/audit_geometry.py coarse_init_reg000_b12 --block 12 --topdown 2>&1 | tail -6
exit $rc
