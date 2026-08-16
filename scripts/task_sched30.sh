#!/bin/bash
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
rm -rf outputs/noprior_b12
# ★★ 排程壓縮：densify 42k -> 30k，收割期從 18k 拉到 30k（2026-08-17）
#
# 動機來自逐步軌跡分析（研究總覽 §12.12）。三個修正後跑次型態完全一致：
#
#            39,759    45,439    51,119    56,799    60,000
#   noprior  24.233 -> 25.580 -> 25.934 -> 26.149 -> 26.083
#   delta     (densify 停)  +1.35     +0.35     +0.22     -0.07
#
#   * **收割期（42k->60k）貢獻 +1.85 dB，比我們測過的所有機制加起來還多。**
#   * 顆數在 step 28,399 就撞到 cap => **28.4k~42k 這 13,600 步完全沒有成長，只有churn**：
#     每 500 步 trim 剪 10%、每 150 步 densify 加 5%，在 0.9cap~cap 之間震盪，
#     實測總流動量 12.8M = 族群的 **5.2 倍**（每顆平均被換掉 5 次）。
#   * 那段期間 PSNR 是**下降**的（noprior 34,079 的 24.551 -> 39,759 的 24.233）。
#
# 唯一變數：densify_until_iter 42,000 -> 30,000（trim 的 contribution_prune_until_iter=-1
# 會自動跟著改，兩者本來就綁在一起）。max_steps 維持 60,000。
#
# 判讀（noprior = 26.083，best_val 26.149 @ 56,800）：
#   >26.3   -> 那 13,600 步的 churn 是淨傷害，排程該壓縮；順帶訓練可以更短 => 直接進最終配方
#   26.0~26.2 -> 打平：churn 無害但也無用 => 仍值得採用（省 13,600 步的 trim 開銷）
#   <25.8   -> 晚期 densify 真的有在改善配置，churn 是必要的探索 => 維持 42k
#
# ⚠ 不要同時改 max_steps。收割已知在 ~57k 飽和（三個跑次最後 3,200 步 PSNR 全部倒退），
#   但那是另一個變數，混在一起就分不出是排程還是步數。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sched30_b12
