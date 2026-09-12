#!/bin/bash
# ★★★★★ cost_budget **短探針**（~30 分）：預算綁不綁得住？綁住能省多少 VRAM？
#
# ⚠ 為什麼先做探針而不是直接跑 9.6h —— 實測的天花板很低（§11.66）：
#   逐顆儲存 928 B/顆（**成本預算動不到**）／binning 191 B/顆（能動的）=> **儲存 83%**
#   => 就算把 binning 砍半也只省 8.5% VRAM，換成顆數 +8.5%
#      依 +0.541 dB/加倍 => **上限約 +0.06 dB**，遠低於 3sd 的 0.24 dB 宣稱門檻。
# ⚠ 且 09-07 首次嘗試 **OOM 於 step 22,352（N 衝到 3.10M）**：執行碼是對的
#   （`_load >= _cb => num_gs = 0`），所以只可能是**預算從未綁住** => B 設得太**高**。
#   （我當時診斷成「B 太緊」是錯的。）
#
# 本探針從 depth-init 從頭跑 8,000 步（涵蓋 densify_from 1000 之後約 47 個事件），
# cap 用**基準的 2.6M**（不放寬 => 不可能 OOM），B 掃一個明顯偏低的值強迫它綁住：
#   看 `[cost-budget]` 印出的 Load 是否達到 B、N 是否被壓在 cap 之下。
# 判準：
#   Load 達到 B 且 N < cap        => 預算**有綁住** => 量 VRAM 差額，再決定要不要跑 9.6h
#   Load 從未接近 B               => 線上 Load 的量級與預期差很多 => 重新標定，別跑 9.6h
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/cbprobe_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --trainer.max_steps 8000 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 60000 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.cost_budget 3000000 \
  --model.density.init_args.cost_budget_report 500 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n cbprobe_b12
echo "===== [cost-budget] 全部輸出 ====="
grep -o "\[cost-budget\][^|]*" logs/runner.log | tail -20
