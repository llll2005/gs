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
# ★★ 配方轉移測試：現行最佳（noprior）換到 block_7（2026-08-16 排入）
#
# 動機（使用者提議）：整套配方都是在 b12 上調的，而**b12 在 25 塊裡內容密度排第 13、
# 低於中位數**（GT 梯度 0.01547），半面是水。b7 是**最重的一塊**（0.02189 = b12 的 1.42×），
# 而且是官方 CityGSV2 baseline 重跑用的那塊。同時是最硬的內容測試與有歷史對照的塊。
#
# 唯一變數是 block_id 與 init PLY，cap 刻意維持 2.6M 與 b12 完全一致 —— 換 cap 就失去可比性。
#
# ★事前押注（pre-registration，跑之前先寫死，免得事後湊）：**b7 = 24.22**
#   由既有 8 塊 run（5x5、同 SH3 aggr17 族）擬出的內容模型：扣掉壞塊 block3(11.704) 後
#   r=-0.652、斜率 -2.9 dB / 0.01 GT梯度。b7 比 b12 重 0.00642 => 26.083 - 1.86 = 24.22。
#   ⚠ 該關係與「視角數」高度混淆（視角數 vs PSNR r=-0.647，幾乎一樣強，n=7 分不開）。
#   命中 => 內容模型可用，能外推 25 塊；差很多 => 逐塊差異由別的東西主導，單塊推不了全場景。
#
# 判讀（b12 noprior = 26.083）：
#   b7 落在 25~26      -> 配方會轉移，內容更重只是稍難，可以放心推全場景
#   b7 明顯更低(<24)   -> 配方過擬合到 b12 的低內容/水面，逐塊 cap 或參數要按內容調
#   b7 反而更高        -> b12 的水面在拖累它，真實的建築區能力被低估了
#   OOM                -> **這本身就是重要發現**：VRAM 餘裕是 b12 特有的，內容重的塊
#                         在同 cap 下 overdraw 更高，全場景 25 塊會一路踩雷。ledger 會記
#                         DIED 與死在第幾步，不算白跑。
#
# ⚠ b7 只有 192 視角（b12 是 284）=> epoch 數會不同，但 max_steps 都是 60,000，可比。
# ⚠ 峰值 VRAM 的儲存部分與內容無關（2247 B/pt x 2.6M = 5.44G），變動的是 overdraw 那部分。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n best_b7
