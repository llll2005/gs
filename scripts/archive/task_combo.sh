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
rm -rf outputs/combo_b12
# ★★ 綜合最佳：cap30 + blur split（2026-08-21，多指標排名後的合成）
#
# 多指標重審（研究總覽 §12.30，13 個修正後跑次 x 5 指標）的結果：
#   **五個指標五個不同的第一名**，Pareto 前緣有四個（sched30 / cap30 / blurcum / npd）。
#   平均名次：**cap30 2.0（第一）** > sched30 2.8 > sched36 4.0 > blurcum 4.6 > ... > npd 7.2
#
#   run       PSNR SSIM LPIPS 最差10% 建築低頻
#   cap30       2    2     3     2      1     <- 只輸 PSNR(-0.071=2.2x底)，其餘勝或平
#   sched30     1    4     2     5      2
#   blurcum     4    7     6     1      5     <- **最差10% 全體第一**
#   npd        10    1     1    12     12     <- 感知第一但結構墊底
#
# 本臂 = **cap30（最佳全能）+ blurcum 的 blur_split_budget 0.3（最佳尾巴）**。
# 兩者針對不同東西（顆數/排程 vs densify 判準），所以有機會疊加。
# ⚠ **疊加不可假設可加** —— 本專案已量到兩對互相抵消（blur split x LR 下限、blur split x SH3）。
# ⚠ npd 的旋鈕（dssim 0.5 + depth off）**刻意不納入**：它感知第一但結構墊底（尾巴與低頻都第 12），
#   而結構正是使用者看到的問題。
#
# 判讀（**必須多指標，禁止用單一指標宣布勝負**，跑 `python tools/audit_all.py`）：
#   平均名次進前一 -> 可加，進最終配方
#   PSNR 升但最差10%/低頻退 -> 又一次「移動平均而非尾巴」，記為不可加
#   全面持平 -> 兩者作用在同一個瓶頸上
# ★ 重點看**最差 10%**：§12.29 量到修好最差 25% 值 +1.55 dB，而所有機制至今只把尾巴動了 0.50 dB。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 3000000 \
  --model.density.init_args.blur_split_budget 0.3 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n combo_b12
