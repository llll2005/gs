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
rm -rf outputs/errunlock_b12
# ★★★★ 錯誤觸發的 opacity 解鎖（RTG-SLAM B4 移植到 MCMC）（2026-08-24）
#
# 使用者問：「這是不是就跟 RTG-SLAM 的凍結機制一樣？RTG 有相應的算法解決嗎？」
# 查證：**有，而且我們早就移植了，只是從沒開過、而且在一個現在不用的 controller 裡。**
#   `rtg_stable_density_controller.py` 的 **B4 stable->unstable reversion**：
#   把 stable 粒子投影到當前相機，若其像素誤差連續 N 個 densify 週期超過門檻，就退回 unstable。
#   `stable_revert_enabled` 預設 False，configs/ 與 scripts/ 全庫沒有任何地方開過它。
#   ⚠ 該檔的「RTG-SLAM Sec 3.2」章節標註是先前的人寫的，**論文不在 參考論文/，未經核對**。
#
# 病灶（研究總覽 §11.7）：b12 最差視角（1729/2999/3007）與最好的（2652/2643/1292）
#   內容難度相同、粒子更多（489k vs 322k）、沒有錯位（平移掃描峰值在 (0,0)），
#   唯一分得開的是 **opacity 分布**：壞視角 opacity>0.9 佔 18~23%，好視角只有 8~9%。
#   ⇒ 早期把某層推到高 opacity ⇒ 後面的正確幾何永遠收不到梯度 ⇒ 自我強化的局部極小。
#
# 為什麼不用 vanilla 的 `opacity_reset_interval`（3,000 步全體壓低）：
#   那會連 90% 正常的區域一起打斷。B4 是**錯誤觸發**的，只動持續做錯的那些。
#   而 MCMC 把 opacity reset 整個拿掉了（我們的 controller 裡 `opacity_reset` 出現 0 次）。
#
# 本臂 = `sched30` + `err_unlock_frac 0.10`（每個 densify 事件，在 opacity>0.5 的候選裡
#        取誤差最高的 10%，把 opacity 壓回 0.05 並**重置其 Adam 狀態**）。
#   ⚠ 不重置 Adam 動量會在幾步內把 opacity 推回去 => 機制靜默失效。MCMC 自己的 relocation
#     也是這樣處理的（`mcmc_density_controller.py:298`）。
#   ⚠ 壓到 0.05 而非 0：低於 `min_opacity`(0.005) 會被判 dead 並被 relocate 搬走，
#     那會連位置一起換掉（多一個變數）。留在原地變透明，讓最佳化自己決定。
#
# 判讀（對照 `sched30_b12` 平均 26.369 / **最差 17.43** / 建築低頻 0.02570；噪音底 0.0325）：
#   **最差 10% 明顯上升** -> 病灶確認，這是尾巴那 +1.55 dB 的第一個真入口
#   平均升但尾巴不動      -> 解鎖有用但不是打在尾巴上
#   全面下降             -> 那些高 opacity 是必要的，假說錯
# ⚠ 判準看**最差 10%**（tools/tail_analysis.py）與**看圖**，不是只看平均。
# ⚠ 這是我今天第五個假說（前四個都被自己的資料推翻），先驗機率不高，但零件現成、代價一臂。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.err_unlock_frac 0.10 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n errunlock_b12
