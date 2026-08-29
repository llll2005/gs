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
rm -rf outputs/halfcap_b12
# ★ 修正後的顆數標度律 —— 全專案唯一一個同基底顆數對照（2026-08-16 排入）
#
# 使用者問「4.96M 只拿到 24.16 是在 GT 正確還是錯誤時跑的」，查證結果：2026-06-30，
# 而 zfill 修正是 2026-08-12(faeb4d4) => 那是 bug 期數字，不可引用。連帶作廢的還有
# 「+0.229 dB/加倍」「SH3 見頂 1.5M/24.30」「顆數不是王道，別再跑 count ablation」。
# 修正後**沒有任何同基底的顆數對照**：所有 SH3 跑次都卡在 cap 2.6M，只有 SB 上到 3.15~3.24M
# （換了基底，不可比）。所以 dQ/dN 目前是零數據，而整個成本感知論述的前提就是它。
#
# 往上走是 VRAM 死路（SH3 實測 2247 B/pt，6GB 頂多 cap ~2.9M，只有 +11%），
# 所以往下量：cap 1.3M（交付 1.17M）對 noprior 的 2.34M ＝ 乾淨的 2 倍對照，
# 而且顆數少所以比 12.5h 更快。唯一變數是 cap_max。
#
# 判讀（noprior = 26.083 @ 2.34M）：
#   26.0 附近      -> 斜率近乎零，顆數在此區間已飽和 => 別再為省位元組做任何事，
#                     成本感知要改成「同顆數下的品質」而不是「同 VRAM 下的顆數」
#   24.5~25.5      -> 斜率 0.5~1.5 dB/加倍，遠陡於 bug 期的 0.229 => 省位元組非常值得，
#                     而 SB 失敗是因為它每位元組的品質太差，不是路線錯
#   <24            -> 斜率極陡，顆數是主導變數，該全力找更便宜的表徵
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 1300000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n halfcap_b12
