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
rm -rf outputs/notrim2_b12
# ★★★★★ 週期性 contribution trim 的乾淨消融（2026-08-24）
#
# 【怎麼發現的】`coarseft_b12`（26.518）被我當成「新高、coarse-init 有效」，**歸因是錯的**：
#   `coarse_fix` 的官方 config 有 `diable_trimming: true`，而 `initialize_from` 走 **ckpt 路徑
#   會把 renderer 整個換掉**（`gaussian_splatting.py:183`）=> coarseft 與 half30k 都繼承了
#   「完全不 trim」。實證：顆數下降事件 sched30 有 **59 次**、coarseft/half30k **0 次**；
#   最終顆數 sched30 = 0.900xcap（trim 造成的 §12.8）、coarseft = **1.000xcap**。
#   ⇒ coarseft vs sched30 有**兩個**變數（coarse-init + 不 trim），+0.141 不可歸給 coarse-init。
#   ⚠ 這是我自己記憶 `_ctx` 裡就寫著的陷阱（「ckpt 路徑會連 renderer 一起換掉；PLY 路徑不會」）。
#
# 【為什麼值得單獨測】週期性 contribution trim（每 500 步剪貢獻度最低的 10%）**從來沒被消融過**。
#   §11.4 的 `notrim` 只關掉了**起始** trim（step 1 那一刀），不是這個。
#   而它每 500 步剪掉 10% 的族群，60k 步內共 59 次 —— 是動作量最大的機制之一。
#   旁證：`errunlock`（把高不透明粒子壓透明）**全面變差 −0.635 dB** => 那些粒子是承重的；
#   若「貢獻度低」的粒子也同樣承重，trim 就是在持續破壞。
#
# 本臂 = `sched30`（depth-init，**PLY 路徑，renderer 旗標不會被換掉**）+ `diable_trimming true`。
#   唯一變數是週期性 trim 的開關。⚠ 用 depth-init 而不是 coarse-init，正是為了讓旗標生效。
#
# 判讀（對照 `sched30_b12` 平均 26.369 / 最差 17.43 / 建築低頻 0.02570；噪音底 0.0325）：
#   明顯上升 -> **trim 是淨負的**，而 coarseft 的增益其實來自這裡 => 主線配方要改
#   打平     -> trim 中性，coarseft 的增益真的來自 coarse-init
#   下降     -> trim 是必要的，coarseft 是「coarse-init 的增益大到蓋過失去 trim 的損失」
# ⚠ 顆數會停在 1.000xcap 而非 0.900（§12.8 的 0.9 就是 trim 造成的）=> 多 11% 顆粒是**共生變數**，
#   但 §12.27 已測 cap +15% 是 −0.071 => 不會製造假的正結果。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.renderer.init_args.diable_trimming true \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n notrim2_b12
