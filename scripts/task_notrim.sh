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
rm -rf outputs/notrim_b12
# ★★ 歸因對照：關掉起始 trim（2026-08-22）
#
# 【共同背景】起始 trim 在 step 1 鎖死了錯的前層（研究總覽 §11.2）
#
# depth_init 給**每一顆**都設 opacity 0.99，而那層殼比表面厚 20 倍。起始 trim 用
# `sum T*alpha` 算貢獻度 => 前層吸收掉幾乎所有 transmittance => 後層歸零被砍。
#   b12 實測：1,211,537 顆裡**只有 157 顆真的在視錐外**，卻砍掉 **873,504 顆（72.1%）**
#   => 砍的是「看得見但被前層遮住」的點，其中包含正確的表面。
#   => 錯的前層在**任何最佳化之前**就被鎖死。
#
# 這解釋三件先前對不起來的事：尾巴為什麼「從頭就爛」（§12.36）、為什麼五種旋鈕全距只有
# 0.08 dB（§2.3）、為什麼換 SfM-init 只動了 +0.15（它沒有殼，所以沒有前層問題，但太稀疏）。
#
# 判讀（對照 `sched30_b12`：平均 26.369 / 最差 17.43 / 建築低頻 0.02570；噪音底 PSNR 0.0325）：
#   **最差 10% 明顯上升** -> 機制證實，這是 +1.55 dB 那條路的入口
#   只有平均動、尾巴不動   -> 鎖死的不是尾巴的成因
#   全面變差             -> 那層殼是必要的，trim 砍對了
# ⚠ 判讀看**最差 10%**（`tools/tail_analysis.py`）與**看圖**，不是只看平均。
#
# 本臂 = `sched30` 但 `--model.renderer.init_args.diable_start_trimming true`
# => 保留全部 1,211,537 顆（opacity 仍是 0.99）。唯一變數是「step 1 那一刀砍不砍」。
#
# 這是 `op05` 的**歸因對照**：
#   op05 有效、本臂無效 -> 問題是**遮擋**（後層拿不到梯度），不是那一刀本身
#   兩臂都有效          -> 問題是那一刀刪掉了可用的點
#   本臂有效、op05 無效 -> 出乎預料，要重查
# ⚠ 預期本臂偏中性：即使不砍，被遮住的點仍然拿不到梯度，只是改由 opacity_reg 慢慢殺。
#   **那個「中性」本身就是資訊**（它會把矛頭指向遮擋而非刪除）。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.renderer.init_args.diable_start_trimming true \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n notrim_b12
