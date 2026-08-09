#!/bin/bash
# Is the residual a missing VIEW-DEPENDENT term? Swap SB colour back to SH3, change nothing else.
#
# The low-frequency band (>=32px) carries 45% of the residual energy, and attributing it showed the
# same thing in every model: contrast compression toward the mean.
#   GT dynamic range 0.271;  render 0.179-0.188  =>  60-69% of it, on 8 building views
#   darkest quartile rendered +0.043..+0.052 too BRIGHT, brightest -0.039..-0.057 too DARK
#   4x the primitives barely moves it (uniform 900k 60% -> cap4m 3.6M 68%), so it is not capacity
#
# What compresses large regions toward the mean while being insensitive to count: one primitive
# seen from many views, whose true appearance differs per view, can only store the average. Our
# colour model is sh_degree 0 (diffuse DC) plus 2 SB lobes; SH3 was measured at +0.295 dB on this
# scene, so view-dependent content demonstrably exists here.
#
# It also predicts the water ghosts. Water is the most view-dependent surface in the scene -- it
# mirrors sky from one angle and buildings from another. A diffuse model cannot express that, so
# the only way left to lower the loss is to place REAL geometry above the water to explain the
# reflected buildings some views see. CityGS's original recipe uses sh_degree 3 and does not show
# the artefact; we dropped that capacity to save memory (59 floats -> 25).
#
# READ THE CONTRAST, NOT THE PSNR. Equal-VRAM alignment gives SH3 only 2.6M primitives against
# blurbudget's 3.6M (state 944 vs 400 B/pt; the 978 B/pt render term is independent of F), so
# ~0.35 dB of the comparison is a count handicap. Decisive readouts:
#   tools/attribute_lowfreq_error.py  -- does the 0.179 dynamic range widen towards GT's 0.271?
#   the test renders                  -- do the ghost buildings over water go away?
# opacity_reg is forced to 0.002 (the config ships 0.007) so the colour model is the only change.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sh3_viewdep_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  -n sh3_viewdep_b12
