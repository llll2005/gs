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
rm -rf outputs/cap30_b12
# ★ 顆數天花板的安全利用：cap 3.0M（2026-08-20）
#
# `cap35`（cap 3.5M）**OOM 於 step 22,952、N=3.28M**：
#   `Tried to allocate 52.00 MiB (5.66 GiB total; 5.06 GiB allocated)`
#   => **不是硬性容量不足，是碎片化**（要不到 52 MiB 而死，帳面還有 0.6 G）。
#   碎片化的正解 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **需要 torch >= 2.1，
#   本環境是 2.0.1，不可用**；升級會動到整個鎖定的 CUDA/rasterizer 堆疊，風險遠大於報酬。
#
# ⇒ **SH3 在這台機器的實測天花板 ≈ 3.28M 顆**，而我們一直跑 2.34M ⇒ 留了 40% 沒用。
#   cap 3.0M 的族群在 2.70M~3.00M 之間震盪，峰值離 3.28M 有 **8.5% 邊際**。
#
# 預期（顆數斜率 +0.541 dB/加倍，§12.24）：2.70/2.34 = 1.154x = **+0.112 dB = 3.4x 噪音底**。
# 疊在 sched30 上（對照 sched30_b12 26.377）=> 目標約 26.49。
#
# 判讀：
#   >= 26.45  -> 顆數增益如預期且與 sched30 可加 => 進配方，並考慮 cap 3.15M 再擠
#   26.38~26.45 -> 增益被吃掉一部分（疊加不可加）=> 記錄非可加性，維持 cap 2.6M
#   OOM       -> 天花板比 3.28M 更低（碎片化隨時間累積）=> 顆數線就此封頂
#
# ⚠ 成長期會落後對照組（cap 大則族群長得慢），勝負要看收割期。cap35 死前 step 22,720
#   是 23.706 vs sched30 同 step 23.944 —— 那個落後是正常的，不是失敗訊號。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 3000000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n cap30_b12
