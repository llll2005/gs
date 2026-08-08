#!/bin/bash
# Blur split + long-axis spread: pick the right hosts, AND put the children somewhere useful.
#
# Runs only after blurbudget_b12, never instead of it: two mechanisms at once cannot be attributed.
# The pair is expected to be super-additive if either works at all -- blur split selects primitives
# that alone cover a large patch, and stacking their children at the host's centre (what MCMC does
# today) cannot cover that patch. Selecting the right host is useless if the copies land on top of
# each other; spreading is useless if the hosts are the wrong ones.
#
# Control is cap4m_b12 (24.307). ONLY blur_split_weight changes, so the comparison is clean.
#
# Premise measured before writing any of it (紀錄 §11.9.1, tools/measure_blur_split_candidates.py):
#   - 303 of 3.6M primitives exceed Mini-Splatting's threshold, carrying 12.6% of blending weight
#   - their median local image gradient is 0.0118 vs 0.0064 for all visible primitives
#   - 74.6% of that weight sits on textured content, only 25.4% on flat water
# so the criterion already points at genuine under-reconstruction here, and the texture gate
# proposed in §11.9.1 would recover the remaining quarter -- worth adding later, not a blocker.
#
# budget 0.3: 30% of each densification step is cloned from over-threshold hosts.
#
# The FIRST attempt (blursplit_b12, blur_split_weight=2.0) was a design error, not a null result: a
# multiplier cannot move a population that is 0.01% of the total. 3x took the candidates from
# 0.0084% to 0.0253% of the sampling mass -- 15 of 60,000 additions -- and the run duly tracked its
# control within noise (20.83/20.79/20.94 vs cap4m's 20.86/20.88/21.05 at the same steps).
# Mini-Splatting splits ALL over-threshold Gaussians; a budget share is the closest equivalent that
# still fits MCMC's sample-with-replacement scheme, and it means the same thing however few
# candidates there are.
#
# What would falsify it: 紀錄 §11.4 shows selection within a fixed set is degenerate (v_i ~ c_i).
# Blur split escapes that only because it CHANGES the set -- if it comes back at 24.3 +- 0.1, the
# escape did not work either and the degeneracy is stronger than Mini-Splatting's result suggests.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=blurlas_b12
rm -rf outputs/$NAME
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.blur_split_budget 0.3 \
  --model.density.init_args.long_axis_spread 0.5 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
