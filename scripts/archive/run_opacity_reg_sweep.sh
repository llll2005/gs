#!/bin/bash
# Which opacity_reg keeps the geometry on the ground? Same block, same recipe, one variable.
#
# The viewer showed reg000's scene is fog: 78.7% of primitives are opaque AND airborne, against
# A'(reg=0.007)'s 1.0%, while depth-init starts correct (z median 0.26, matching SfM). PSNR never
# saw it because val is a subset of train -- a shell that reproduces the training views scores
# fine. So reg000's +1.04 dB bought a better cheat, not a better model.
#
# Mechanism: immune_opacity_threshold=0.9 exempts o>0.9 from opacity_reg, so the L1 is really a
# GATE on who gets to become opaque -- with it, only load-bearing primitives climb past 0.9;
# without it, everything does.
#
# (The depth-scale hypothesis is dead: the dataparser's bounds are RELATIVE to the median scale,
# so the real rejection rate is 0.52%, not the 29.9% an absolute reading suggested.)
#
# Each arm trains, then tests, then audits geometry -- the audit is the one that matters.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply
CFG=configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml
REG=$1
NAME=oreg_${REG//./p}_b12
# GPU serialisation is the runner's job now (scripts/runner.sh waits before每個 non-[cpu] task);
# this script used to carry its own copy of that loop. Twelve scripts had duplicated it.
rm -rf outputs/$NAME
conda run --no-capture-output -n gspl python -u main.py fit --config $CFG \
  --model.initialize_from $PLY --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 1000000 \
  --model.metric.init_args.opacity_reg $REG \
  -n $NAME > logs/$NAME.log 2>&1
rc=$?
cp outputs/$NAME/blocks/block_12/results.txt outputs/$NAME/blocks/block_12/results_train.txt 2>/dev/null
C=$(find outputs/$NAME -name config.yaml | head -1)
[ -n "$C" ] && conda run --no-capture-output -n gspl python main.py test --config "$C" --save_val >> logs/$NAME.log 2>&1
conda run --no-capture-output -n gspl python tools/audit_geometry.py $NAME --block 12 --topdown 2>&1 | tail -6
exit $rc
