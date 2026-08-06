#!/bin/bash
# Does a shapeless init still hold up where our GOOD models live?
#
# On the 24k decaying schedule a uniform volume fill matched depth-init and cost nothing:
#   uniform25m  70,488 pts  val 22.182  building 18.37  texratio 0.151
#   v1clean     77,089 pts  val 22.022  building 18.13  texratio 0.145
# But both arms were ground down to ~70k with texture ratio 0.15 against the 1.8M model's 0.393
# -- both are bad, and a schedule that destroys both arms can hide a real difference. Our actual
# best single-block model is the 60k GROWTH recipe at 900k points and 23.848.
#
# So repeat the comparison there. Control is `oreg_0p002_b12`, identical in every parameter;
# the only change is which PLY seeds it, and the two PLYs carry the SAME point count
# (1,211,537 = 1,211,537), so this isolates spatial arrangement, not budget.
#
# ⚠ CONFOUND, stated up front: uniform seeds opacity 0.05, depth-init 0.99. That is not cosmetic --
# it decides how hard the start trim bites, because the trim deletes points whose contribution is
# EXACTLY zero (prune_mask = c <= quantile(c, 0.0)), and at 0.99 transmittance collapses after
# ~2 layers while at 0.05 it takes ~108. Measured: depth-init loses 73%, uniform 20%. So this
# compares the two initialisations AS CONFIGURED, which is the decision-relevant question
# ("can the free thing replace the expensive pipeline"), not a clean shape-only ablation.
#
# WHAT IT DECIDES
#   matches 23.848  -> utils/estimate_dataset_depths.py (5,621 images, the most expensive step in
#                      the whole pipeline), the per-image scale/offset calibration, the normals and
#                      the voxel dedup can all be deleted, and the dependency on a monocular depth
#                      network goes with them -- which also cleans up the GT-free story.
#   falls short     -> depth-init earns its cost in the growth regime, and 2026-08-06's tie was an
#                      artifact of a schedule that wrecks both arms.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

NAME=uniform_60k_b12
PLY=data/matrix_city/aerial/train/block_all/depth_init/uniform_matched_block_12.ply

rm -rf outputs/$NAME
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from $PLY \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 1000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
