#!/bin/bash
# Explicit price on the band both photometric terms ignore. On top of dssim05_b12
# (25.519 / 0.7709 / 0.3614 / 0.4959, floater 0.958%).
#
# tools/loss_footprint_pricing.py: inject a FIXED total amount of error into a real render and vary
# only how it is spread. The SSIM penalty relative to L1 falls monotonically with blob radius --
# 15.1x at r=2 down to 1.14x at r=64, and 7.47x -> 1.34x on a second sweep with different amplitude
# and area. A large-footprint error is therefore in the blind spot of BOTH terms: L1 sees a faint
# tint spread thin, SSIM sees nothing structural. That is how large floaters survive.
# Meanwhile the fixed model's band decomposition puts 37% of the residual ENERGY at >=32 px
# (2.57e-3 of 6.87e-3) despite only 2.5% relative error there -- large signal, no prioritisation.
#
# L1 at 1/8 resolution prices exactly that band, at the cost of one avg_pool.
#   floaters fall and LPIPS holds  -> the blind spot was real and is now covered
#   nothing moves                  -> the residual at that scale is not what the floaters cost us
#   PSNR/LPIPS worsen              -> another added regulariser that does not pay, like AtomGS
#     and EdgeAware before it (both lost on PSNR *and* LPIPS -- checked 2026-08-15, no mechanism in
#     the archive was killed by PSNR alone while helping LPIPS)
# ⚠ Weight 0.3 is a guess. The term's scale is comparable to L1's, so 0.3 makes it roughly a third
#   of the photometric objective; if it dominates, the model will go flat and texture ratio drops.
#
# The loss is currently 0.8*L1 + 0.2*(1-SSIM) -- the 3DGS default we never questioned. L1 is
# precisely the term that prefers blur when uncertain, so 80% of the objective is pulling against
# detail. That is why perceptual convergence is slow: we are not optimising perception.
#
# The trajectory says the run is not converged in that sense. Both fixed runs peak in PSNR at
# 56,800 and then LOSE 0.09-0.10 by 60,000 while texture ratio keeps climbing (+0.009/+0.010) --
# i.e. training is still trading pixel accuracy for detail when the clock runs out. Raising the
# structure weight changes what that trade optimises rather than how fast it runs.
#
# READ LPIPS AND TEXTURE, NOT PSNR. This arm deliberately de-weights L1, so a PSNR drop is the
# expected cost, not the result.
#   success : LPIPS clearly better, texture ratio up and still below ~0.6 (real detail)
#   failure : texture ratio shoots past 0.6 without LPIPS following (manufacturing noise)
# ⚠ Expect a modest effect. A CPU upper-bound test found LPIPS is best around texture ratio 0.53
#   and worsens beyond it -- and the fixed model already sits at 0.45. (That test ran on a
#   pre-fix model, so the ceiling itself may have moved; this run also re-measures it.)
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/coarsel1_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.lambda_dssim 0.5 \
  --model.metric.init_args.coarse_l1_weight 0.3 \
  -n coarsel1_b12
