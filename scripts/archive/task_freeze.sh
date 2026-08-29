#!/bin/bash
# ⛔ 已退役（2026-08-17）：研究總覽 §12.12 證明那 10% 是 trim 每 500 步剪 10% 造成的，
#    不是 opacity 流失 => 本臂在測一個被推翻的假說。已從 queue 移除，保留檔案供追溯。
# Can BOTH geometric priors go? lambda_normal 0 AND depth loss 0, on top of SH3 / cap 2.6M /
# opacity_reg 0.002.
#
# Each was measured alone against sh3_viewdep_refix_b12 (25.435 / 0.7498 / 0.3990 / 0.4504):
#   lambda_normal 0   -> 25.828  (+0.393)
#   depth loss 0      -> 25.675  (+0.240)
# Both positive on all four metrics, but they cannot be added: both constrain geometry, so the
# gains very likely overlap. This run measures the pair.
#
# What it decides beyond the number: the depth loss is the LAST dependency on DA2 (depth-init was
# already shown replaceable by a uniform fill). If it can go, utils/estimate_dataset_depths.py over
# 5,621 images, the per-image scale/offset calibration, the normals and the voxel dedup all go with
# it, and the GT-free story stops involving a monocular depth network at all.
#
# ⚠ Photometric metrics cannot see geometry degradation, and 2DGS's surface quality is part of what
#   CityGSV2 sells. Removing BOTH geometric priors is exactly the case where that blind spot bites.
#   Check surf_normal / surf_depth before this becomes the recipe -- a win here is not sufficient.
#   (Supporting datum, not proof: nodepth's val/d_reg rose only 0.1498 -> 0.170, i.e. geometry drifts
#   13.5% further from the pseudo-depth without supervision, consistent with "already exhausted".)
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
rm -rf outputs/freeze_b12
# ★ 收割期的 opacity L1：是流失還是正則？（2026-08-16 排入）
#
# 發現（研究總覽 §12.8）：四個跑次全部停在 cap 的 90.00%，逐 ckpt 追蹤顯示 10% 全掉在
# 42k->60k 的收割期 —— densify 停掉後 relocation 關閉，opacity_reg 繼續壓 opacity、
# min_opacity 剪掉，沒有東西再補。峰值 VRAM 是按 N=cap 付的，交付卻只有 0.9*cap。
#
# freeze_opacity_after_densify 在 mcmc_2dgs_density_controller.py:233 已實作但從未跑過。
# 它在 global_step >= densify_until_iter 之後把 opacity 梯度歸零（Adam 直接跳過 None-grad）。
#
# 判讀（對照 noprior 26.083 / 2.34M / floater 1.726%）：
#   顆數回到 ~2.6M 且 PSNR 持平或微升  -> 那 10% 是純流失，收割期的 opacity L1 白付
#   顆數回到 ~2.6M 但 PSNR 掉          -> 它是需要的正則，10% 是該付的代價
#   顆數沒回來                          -> 流失來源不是 opacity 梯度，我的歸因錯，要重查
#
# ⚠ 預期 PSNR 效果本來就小（+10% 顆數 ~ +0.03 dB，遠低於 1.147 dB 缺口）。它的價值是
#   判定 count-blind 的 opacity L1 在收割期的代價，直接餵成本感知論述，不是衝分數的臂。
# ⚠ 峰值 VRAM 不會升：峰值早就發生在 30k~42k 的 N=cap 平台，這只是不讓它掉下來。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.density.init_args.freeze_opacity_after_densify true \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n freeze_b12
