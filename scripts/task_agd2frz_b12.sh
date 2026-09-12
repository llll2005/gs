#!/bin/bash
# ★★★★★ 收割期凍結 opacity：補上 §11.61 量到的「生產者還在跑、回收器已關」缺口
#
# 實測（agd2_b12 三個 ckpt）：判死比例 30k 時 3.77% -> 60k 時 **15.03%**。
# 成因：`mcmc_2dgs_density_controller.py:487` 讓回收路徑在 densify_until 後 early-return，
# 而 `opacity_reg 0.002` 繼續壓 30,000 步 => 死掉的沒人回收，只能躺著佔儲存。
# 兩項損失都在收割期：trim 移除 260k（2.6M->2.34M）+ 存活者中 15.03% 死去未移除。
#
# `freeze_opacity_after_densify` 早就實作（:381，預設 False），docstring 寫的正是這個理由，
# 但**從沒跑過**（研究總覽 0 次提及）。當初只觀察到 1.0M->0.90M，沒量到今天這麼利的數字。
#
# ⚠ 這不等於 `opacity_reg=0`：正則在**生長期**照常作用（那時回收器在，它有下游），
#   只在**收割期**關掉（那時回收器已關，它只剩損耗）。
# ⚠ 風險：收割期正是 +1 dB 的來源。凍結的只有 opacity（SH/means/scales 照常學），
#   但若外觀收斂其實依賴 opacity 微調，可能傷到那 +1 dB => 看 val 曲線在 30k 後的形狀。
# 對照 = agd2_b12（26.5602 / 0.7755 / 0.3467 / 糊掉 46.08%）。唯一變數 = 這個布林旗標。
# ★ 除了四指標，另看：終點 N（預期 > 2.34M）、判死比例（預期 << 15.03%）。
# 判準：`tools/cmp_runs.py agd_b12 agd2frz_b12`（最後 4 個 val 點平均），
#       外加 `tools/veil_detect.py agd_b12:12 agd2frz_b12:12` 看糊掉%（agd_b12 = 46.17%）。
#       ⚠ 對照是 **agd_b12（26.432）**，不是 sched30_b12 —— 問的是「更強有沒有更好」。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/agd2frz_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.freeze_opacity_after_densify true \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n agd2frz_b12
