#!/bin/bash
# SB colour, re-measured on the FIXED data mapping. The control is sh3_viewdep_refix_b12.
#
# Every mechanism conclusion from 2026-08-07..12 was drawn from models trained against the wrong
# GT frame (dataparser zfill, see 紀錄 / memory zfill_gt_offset_bug), so none of them carries over.
# "SH3 beats SB" is the one that most deserves re-testing first, because the explanation I gave for
# it was CROSS-VIEW AVERAGING -- one primitive seen from many views whose true appearance differs,
# so a diffuse model can only store the mean. Training every image against its NEIGHBOUR'S frame
# manufactures exactly that inconsistency. So SH3's advantage may have been largely compensation
# for the bug, and could shrink or vanish now.
#
# Byte-aligned cap, not equal cap: SB is 25 floats/point vs SH3's 59, so at the same VRAM SB fits
# more. State is 4*F*4 B/pt and the render term (~978 B/pt) is independent of F:
#   SH3 2.6M x (944+978) = 5.00 GB   ->   SB at the same budget = 5.00e9/(400+978) = 3.63M
# Rounded to 3.6M, which also matches what cap4m actually reached, so the VRAM ceiling is known.
# ⚠ Therefore this is NOT single-variable: colour model AND count both change. That is deliberate
#   -- the decision it informs is "which representation buys more per byte", not "which is better
#   at equal count". Report both numbers.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sb_refix_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 3600000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n sb_refix_b12
