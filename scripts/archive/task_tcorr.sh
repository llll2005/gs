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
rm -rf outputs/tcorr_b12
# ★★★★ RTG-SLAM 式的透明修正層（2026-08-24）
#
# 【背景】RTG-SLAM 原文（`參考論文/3641519.3657455.pdf` §3.1-3.2，已逐句核對）對
# 「已承諾的不透明高斯做錯了」的處理**不是解鎖它**（RTG 的 opacity 是二元 0.99/0.1 且
# `lr_α = 0`，根本沒有梯度），而是「加一顆透明的（α=0.1）在前面修顏色」。
# 而 §11.7 量到：**好視角靠大量半透明粒子做出來**（opacity<0.1 佔 47~49%），
# 壞視角反而高不透明多（>0.9 佔 18~23%）=> RTG 的設計正好解釋了為什麼半透明多是好事。
#
# 判讀（對照 `sched30_b12` 平均 26.369 / **最差 17.43** / 建築低頻 0.02570；噪音底 0.0325）：
#   **最差 10% 明顯上升** -> 尾巴的第一個真入口
#   平均升尾巴不動        -> 有用但不打在尾巴上
# ⚠ 判準看**最差 10%** 與**看圖**，不是只看平均。⚠ 多指標 `tools/audit_all.py`。
#
# 本臂 = `sched30` + `transparent_corrector 0.1`：relocation 的目的地若其父代是
# 「opacity>0.5 且 _err_score 前 30%」，新粒子的 opacity 直接設 0.1，**不走 MCMC Eq.9 的分裂**。
# ⚠ 這刻意破壞 Eq.9 的 alpha 守恆 —— 那正是重點：RTG 是「加修正」不是「分裂」。
# ⚠ 與 `errunlock` 是**互斥的兩個假說**（那個壓低已存在的粒子＝我自己發明的；
#   這個新增低不透明伴隨粒子＝RTG 的做法）。**刻意分開跑**，否則分不出是誰的功勞。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.transparent_corrector 0.1 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n tcorr_b12
