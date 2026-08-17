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
rm -rf outputs/sched30_b12
# ★★ 排程壓縮：densify_until 42,000 -> 30,000（2026-08-17，理由已被 CPU 驗證修正）
#
# ⛔ 撤回原本的理由。我原本寫「收割期 LR 積分變 3.02 倍」。CPU 實測（10 個修正後跑次的
#    收割段擬合，scratchpad/lrfit.py）：把自變數換成 LR 積分 vs 換成步數，**R² 幾乎一樣，
#    10 個裡 8 個是步數擬得更好**（例：sb4 0.9858 vs 0.9878）。收割期內 LR 是步數的平滑單調
#    函數 => 兩種參數化分不開，「LR 積分是資源」只是換座標軸重講同一條曲線，不可當機制。
#
# ✅ 站得住的是外推出來的收斂餘量（同一次擬合，A = 漸近值）：
#      b7                   A=26.051   餘量 **1.234 dB**   <- 差一整個 dB 沒收完
#      b12 各跑次(n=9)                 餘量 0.10 ~ 0.20 dB <- 已收斂
#    b12 在 56.8k 見頂、b7 到 60k 仍在爬(+0.107)，與此一致。
#    ⚠ 1.234 是**外推**不是量測，且 b7 的時間常數 c=0.0014 比 b12 群(0.0057~0.0095)小 5 倍、
#      是離群值 => 當方向與量級參考，不要當預測值引用。
#
# ⚠ 本臂(b12)預期只是打平：b12 餘量只有 0.10~0.20 dB，多的收斂預算沒地方去。
#   它的作用是**對照**，讓「內容重的塊才需要更多收斂」成為 2x2 而不是 n=1。判準：
#     sched30_b7 > 26.05  => **停 churn 提高了可達品質**（不只是收斂更快）=> 機制成立，進配方
#     24.8 < x < 26.05    => 只買到收斂速度（同樣有用，但宣稱不同：是省時間不是提品質）
#     x <= 24.82(基線)    => 晚期 densify 真的在改善配置，churn 是必要探索 => 維持 42k
#
# ★ 提早中止判準（省 GPU）：baseline b7 在 step 42,239 是 23.815。本臂在 42,000 時已經
#   收割了 12,000 步 => **若 42k 附近的 val 沒有明顯高於 23.815，機制就沒發生，可以直接砍掉**，
#   不必等到 60k（省約 4 小時）。用 `tail -f logs/runner.log` 看 val 即可。
#
# 附帶（與機制無關但成立）：30k/60k=50% 正好是上游 vanilla 的比例（15k/30k）。我方 42k/60k=70%
# 是 2026-06-25 設的（config 未追蹤、註解自承「主力積極槓桿」），**早於 GT 修正 => 依據已作廢**。
# 顆數 26,879(b7)/28,399(b12) 撞 cap => 30,000 是族群建完後最早的切點。
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
