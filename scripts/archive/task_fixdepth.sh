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
rm -rf outputs/fixdepth_b12
# ★★★★★ 修好 depth-init 的 off-by-one 之後重跑（2026-08-22）—— 目前最重要的一臂
#
# ⛔ 發現：`utils/depth_init_blocks.py:126` 有**和 dataparser 一模一樣的 zfill off-by-one**，
#    而 2026-08-12 的修正（faeb4d4）**只修了 dataparser，漏了這裡**，PLY 從 2026-05-29
#    起就沒重生過 => **每一顆點都是用鄰幀的深度圖擺位置的**。
#      COLMAP 相機名 0 起算（0000.png ~ 5620.png），磁碟深度圖 1 起算（000001 ~ 005621）
#      舊版 `base.zfill(6)`：相機 1729.png -> 001729.png.npy（錯，正確是 001730）
#      而相機 0000.png -> 000000.png.npy 不存在 => **那台相機被整個跳過**
#
# 汙染量級（實測相鄰兩張深度圖的逐像素相對差）：
#      中位 **35.6%**（範圍 3.2%~87.7%）   vs 偽深度自身的誤差 8.6%（§7.1）
#    => 這個 bug 注入的誤差是偽深度本身噪音的 **4.1 倍**。
#    => 「20 倍厚的殼」「起始 trim 砍掉 72% 看得見的點」「錯的前層被鎖死」
#       全都是這個 bug 的直接產物（§11.2 要據此改寫）。
#
# 本臂 = `sched30`（現行最佳配方，26.377）唯一換掉 init PLY：`depth_init_fix/block_12.ply`
# （`tools/` 無關，是 `utils/depth_init_blocks.py` 修好後重生的）。**唯一變數是 init 的正確性。**
#
# 判讀（對照 `sched30_b12` 平均 26.369 / 最差 17.43 / 建築低頻 0.02570；噪音底 PSNR 0.0325）：
#   全面上升 + **最差 10% 明顯上升** -> 這個 bug 就是尾巴的病灶，
#                                       且所有 depth-init 跑次的絕對值都要重測
#   只有平均動、尾巴不動              -> 尾巴另有成因，但絕對值仍要重測
#   幾乎不動                          -> 起始 trim 把汙染吸收掉了（它砍的正是最錯的那些點）
# ⚠ 必須看圖。⚠ 多指標（tools/audit_all.py），不可只看 PSNR。
PLY=data/matrix_city/aerial/train/block_all/depth_init_fix/block_12.ply
# 守衛：等 PLY 重生完（上限 4 小時）。**刻意用等待而不是失敗** —— runner 不管退出碼
# 都會把任務行移除，失敗一次這一臂就從佇列消失了（2026-08-22 踩過）。
for _ in $(seq 1 240); do [ -f "$PLY" ] && break; sleep 60; done
[ -f "$PLY" ] || { echo "✘ 等了 4 小時 $PLY 仍不存在，中止"; exit 1; }
echo "✔ 使用修正後的 init: $PLY"

conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init_fix/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n fixdepth_b12
