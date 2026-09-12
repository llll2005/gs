#!/bin/bash
# ★★★★★ AbsGS 增生取樣「加強度」：sched30 + absgrad_densify **2.0**（§11.47 的延伸）
#
# w=1.0 已經是現行最佳且**跨塊驗證過**（b12 26.43 / b7 25.28，四指標同向全勝）。
# 本臂唯一變數 = 強度 1.0 -> 2.0，其餘與 task_absgrad_densify.sh 逐字相同。
#
# 為什麼相信還有空間（§11.30 實測）：
#   `|g|` 的 ceiling(top5%)/mean = **16.68**，而到 step 600 opacity 取樣只吃到
#   2.76/16.68 = **17%** => 還有約 6 倍未被利用。
#   w=1.0 讓最高 |g| 的粒子權重加倍；w=2.0 讓它加三倍。
#
# ⚠ 這**不是**「調大一定更好」——vpc 的教訓正好相反（強度 5% 直接崩 -2.34 dB）。
#   本臂就是要找出 w 的轉折點在 1 和 2 之間還是更後面。若 w=2 就開始掉，
#   代表 w=1 已在峰值附近，這條線收斂，不必再往上試。
# ⚠ 需要光柵器以 ABSGRAD=1 編譯（子模組分支 citygs-6gb，c58fb57）。
# ⚠ 只開這一個，不要同時開 cost_aware_densify（§11.49 已證它在顆數約束下無效）。
#
# 判準：`tools/cmp_runs.py agd_b12 agd2_b12`（最後 4 個 val 點平均），
#       外加 `tools/veil_detect.py agd_b12:12 agd2_b12:12` 看糊掉%（agd_b12 = 46.17%）。
#       ⚠ 對照是 **agd_b12（26.432）**，不是 sched30_b12 —— 問的是「更強有沒有更好」。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/agd2_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n agd2_b12
