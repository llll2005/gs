#!/bin/bash
# Third point on the SSIM-weight axis: 0.2 -> 0.5 -> 0.8. Everything else as dssim05_b12.
#
# 0.5 won on three of four photometric metrics AND cut floaters 20% (1.193% -> 0.958%), which is
# the interesting part: nothing in that run targets floaters. The mechanism would be that SSIM is a
# WINDOW comparison, so a near-camera primitive with a huge screen footprint wrecks the structure of
# everything it covers, while L1 only sees a faint tint. That makes SSIM an implicit penalty
# proportional to screen footprint -- which is exactly the c_i this project spent days trying to
# price explicitly in the density controller before the degeneracy theorem killed that route.
#
# Two points cannot distinguish a mechanism from a coincidence. A third does:
#   floaters keep falling and LPIPS keeps improving  -> the implicit c_i pricing is real, and the
#       cost-aware idea belongs in the LOSS, not in primitive selection
#   floaters bottom out or LPIPS turns around at 0.5 -> 0.5 was just a good operating point
# ⚠ Expect PSNR to fall further -- L1 is down to 20% of the objective. Judge on LPIPS, texture
#   ratio and the floater fraction that task_test.sh now reports.
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
rm -rf outputs/dssim08_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.lambda_dssim 0.8 \
  -n dssim08_b12
