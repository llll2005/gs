#!/bin/bash
# Can the whole depth-init pipeline be deleted? Tested at the count where our best model lives.
#
# At a matched 900k, a shapeless uniform fill beat depth-init on 5 of 7 content-split metrics
# (water PSNR 34.41 vs 33.62, building PSNR 20.27 vs 20.04, building texture ratio 0.412 vs 0.372),
# tying one and losing one by 0.002. But its overall val PSNR was 0.42 dB LOWER, which the content
# split does not explain -- the split samples the 8 most and 8 least textured views, so uniform may
# win at both ends and lose in the middle. That discrepancy is unresolved and this run does not
# resolve it either; it answers the other question: does the advantage survive at cap 4M, where
# uniform has never been tested.
#
# If it holds, utils/estimate_dataset_depths.py (5,621 images, the most expensive step in the
# pipeline), the per-image scale/offset calibration, the normals and the voxel dedup all go, and
# the dependency on a monocular depth network goes with them.
#
# The LR floor is chosen from lrfloor_b12's own result rather than fixed in advance: that run is
# the test of whether the 60k plateau was convergence or just the schedule shutting off, and there
# is no reason to guess when the answer will be on disk.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CTRL=24.307        # cap4m_b12, the depth-init control at the same cap
LRF=$(grep -oE 'best_val_psnr: [0-9.]+' outputs/lrfloor_b12/blocks/block_12/best_val.txt 2>/dev/null | grep -oE '[0-9.]+')
if [ -n "$LRF" ] && [ "$(echo "$LRF > $CTRL" | bc -l)" = "1" ]; then
  LR_FINAL=0.0000064          # B_opt route confirmed: keep the raised floor
  WHY="lrfloor $LRF > $CTRL"
else
  LR_FINAL=0.00000064         # plateau was real: fall back to the shipped schedule
  WHY="lrfloor ${LRF:-無結果} <= $CTRL"
fi
echo "[uniform_cap4m] lr_final=$LR_FINAL ($WHY) $(date +%F_%T)" >> logs/quad_progress.log

NAME=uniform_cap4m_b12
rm -rf outputs/$NAME
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/uniform_matched_block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.lr_final $LR_FINAL \
  -n "$NAME" > "logs/$NAME.log" 2>&1
