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
rm -rf outputs/sfminit_b12
# ★★★ 換初始化：SfM-init（2026-08-21，由尾巴診斷指定）
#
# 診斷結果（`task_earlyckpt.sh`，sched30_b12 的 15k/30k/42k/60k 四組 test）：
#              14999     29999     41999     60000     改善
#   1729(最差) 0.01891   0.01806   0.01825   0.01806    −4.5%
#   平均       0.00776   0.00636   0.00509   0.00472   **−39%**
#   最差10%    0.01831   0.01514   0.01455   0.01439    −21%
#
# **45,000 步的最佳化把平均改善 39%，卻只把最差視角改善 4.5%；1729 在 30k 之後完全不動。**
# ⇒ 尾巴**不是被 densify/trim 弄壞的，是從頭就沒成功過** ⇒ 事前判準指向**換初始化**。
# 對上 3DGS 論文（`參考論文/3dgs.pdf` 行 1180-1182）的 init 消融：init 不良的 floater
# "cannot be removed by optimization" —— 與我們觀察到的「損失一直給最大梯度卻修不好」一致。
#
# 本臂 = `sched30` 但 `--model.initialize_from null` => 走 SfM-init（`points_from: sfm` 本來就是
# 預設，`colmap_block_dataparser` 讀 points3D 時已帶 selected_image_ids 逐塊過濾）。
# **零成本、不用寫程式**。b12 實得約 320,513 顆（vs depth-init 的 1,211,537），cap 不用改
# （5% 成長，約 5,700 步即達 2.6M）。
#
# ⚠ 非純單變數：位置與起始顆數同時變（少 3.8 倍），報告要照實寫。
# ⚠ 弱反證：bug 期的 `uniform25m_24k_b12`（完全無形狀的初始化）與 depth-init **打平**
#   ——但那是 24k 衰減排程、兩臂都只剩 7 萬顆、且在 GT bug 下量的，不足以否定本臂。
# ⚠ `surface_distance` 定義是「到最近 SfM 點的距離」⇒ **不可用它論證 SfM-init 較好**（循環論證）。
#
# 判讀（對照 sched30_b12：PSNR 26.377 / 最差10% 18.49 / 建築低頻 0.02570；**多指標 + 看圖**）：
#   **最差10% 明顯上升** -> 初始化就是尾巴的病灶 => 這是 +1.55 dB 那條路的入口
#   平均升但尾巴不動      -> init 影響的是整體不是尾巴，尾巴另有原因
#   全面下降             -> depth-init 確實較好，尾巴的病灶不在初始化
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from null \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sfminit_b12
