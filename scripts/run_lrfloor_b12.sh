#!/bin/bash
# Is the refinement plateau convergence, or just the LR schedule shutting off?
#
# cap4m_b12's refinement marginal fell 0.100 -> 0.040 -> 0.016 dB/1000 steps over three windows.
# Across the same span the position LR fell 1.0e-5 -> 1.78e-6. The ratios are 0.16 and 0.18 --
# the PSNR marginal rate is proportional to the position learning rate, which is what a
# RATE-LIMITED optimisation looks like, not a converged one.
#
# For an exponential schedule LR(t) = L0 * f^(t/T), the total "learning" available is
#   int_0^T LR dt = L0*T*(f-1)/ln f
# f = 0.01 gives 0.215*L0*T; f = 0.1 gives 0.391*L0*T. So raising the floor 10x buys 1.8x the
# integral for FREE, where doubling T would cost 2x the wall clock for 2x.
#
# Only lr_final changes from cap4m_b12 (6.4e-7 -> 6.4e-6). Everything else identical, so the
# comparison is clean: 24.307 is the control.
#   beats it  -> not converged, the schedule was the ceiling, and a stretched 120k run is next
#   loses     -> converged and the extra LR is just noise; the plateau is real and count/steps are
#                both exhausted, which is itself the answer to "what is the ceiling"
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=lrfloor_b12
rm -rf outputs/$NAME
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 4000000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.lr_final 0.0000064 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
