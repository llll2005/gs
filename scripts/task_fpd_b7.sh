#!/bin/bash
# ★★★★★ 足跡加權增生：現行最佳（sched30 + absgrad 2.0）+ cost_aware_densify **-0.5**
#
# 負權重 = 往**大足跡**的地方增生（`probs /= (c/median)^w`，w<0 ⇒ 乘上 (c/median)^0.5）。
# 這是 **Taming 3DGS 的符號方向**（他們 `+0.1 * c^i_g`，理由 "large projections ...
# lead to a blurry appearance"），而他們的權重只佔 score 約 **0.2%** ⇒ 等於沒測過。
# 我方原本的定價方向是**除以 c**；符號以實測為準，不為敘事挑（研究總覽 §11.94）。
#
# 為什麼相信有空間（全部實測，2026-09-05）：
#   足跡面積天花板 ceiling(top5%)/mean = **18.07x**（|g| 本次實測 12.40x）
#   r² = 14.90x；opacity（現況取樣）只有 3.25x
#   |g| 的 top10% 與足跡 top10% 只重疊 **39.96%** ⇒ **還有六成不重合**，值得疊加
#   §11.88 半徑懸崖：投影半徑 >12.25px 的 tile，corr 從 0.980 掉到 0.732
#     （最大跌幅 = 全距的 49.5%，均勻下降只會是 14%）⇒ 大足跡確實對應糊掉
#
# ⚠ 強度 -0.5 而非 -1.0：`(c/median)^w` 是**冪次**形式，比 absgrad 的加法式
#   `1 + w*g/mean` 更激進（小 c 會被壓到極低）。vpc 的教訓＝從保守強度開始（§11.34）。
#   -0.5 讓 top5% 得到約 sqrt(14.9) = 3.9x 的加權。
# ⚠ 疊在現行最佳之上（不是取代 absgrad）—— 問的是「足跡有沒有 |g| 之外的東西」。
#   若贏，再跑「只開足跡、關掉 absgrad」分辨是相加還是取代。
#
# 判準：對照 agd2_b12（26.5602 / SSIM / LPIPS / 建築低頻 / 最差10%），多指標同向才算。
# ⚠ 噪音底（n=3 標定）：PSNR <0.24 / SSIM <0.0015 不可宣稱。優先看 SSIM。
# ⚠⚠ 2026-09-05 第一次跑（08:00~）**完全白跑**：`screen_size_prune` 在 `add_new_gs` 之前
#   把 `_max_radii2D` 清成 None ⇒ cost_aware 讀到 None、整段跳過 ⇒ 三個量測全部「沒差」
#   （足跡分布比值 0.99~1.01、val 三點 PSNR Δ<0.07、SSIM 完全相同）。
#   順序已修（清除移到 add_new_gs 之後），並加了「首次觸發 / 拿不到訊號」的明確 log。
# ★ 本次開跑後**先確認 log 出現 `[cost-aware] ✅ 首次觸發`**，沒有就立刻中止。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/fpd_b7
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.cost_aware_densify -0.5 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n fpd_b7
