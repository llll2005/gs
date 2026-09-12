#!/bin/bash
# ★★★★ 速度：週期 trim 相機取樣 stride=4（71/284 台），端到端驗分數
#
# 探針（§11.52）：71 台 vs 284 台的**底部 10% 遮罩重疊 99.8%**，而 trim 只用到這個集合
# （`contribution <= quantile`），全排序的 Spearman(0.951) 沒有被任何程式碼讀到。
# ⚠ 但只量過 step=1 一個時點，而一次跑次約 120 次 trim => 誤差可能累積 => 必須端到端驗。
# 對照 = agd2_b12（26.5602 / 0.7755 / 0.3467，2.15it/s）。唯一變數 = TRIM_SUBSAMPLE。
# 判準：四指標都在 +-3sd 內 => 無損則採用。
# ⚠ 預期省約 **5%**（非先前宣稱的 7.5~9%）：§11.56 修正——「trim 佔 profile 80%」是
#   40 步視窗被**一次性**起始 trim 佔滿的假影；免費對照（notrim2 多 11% 顆粒卻快 2.7%）
#   顯示 trim 總成本約 wall time 的 7~8%，而 stride=4 只砍週期 trim 的 75%。
#   2.76/16.68 = **17%** => 還有約 6 倍未被利用。
#   w=1.0 讓最高 |g| 的粒子權重加倍；w=2.0 讓它加三倍。
#
# ⚠ 這**不是**「調大一定更好」——vpc 的教訓正好相反（強度 5% 直接崩 -2.34 dB）。
#   本臂就是要找出 w 的轉折點在 1 和 2 之間還是更後面。若 w=2 就開始掉，
#   代表 w=1 已在峰值附近，這條線收斂，不必再往上試。
# ⚠ 需要光柵器以 ABSGRAD=1 編譯（子模組分支 citygs-6gb，c58fb57）。
# ⚠ 只開這一個，不要同時開 cost_aware_densify（§11.49 已證它在顆數約束下無效）。
#
# 判準：`tools/cmp_runs.py agd_b12 tsub4_b12`（最後 4 個 val 點平均），
#       外加 `tools/veil_detect.py agd_b12:12 tsub4_b12:12` 看糊掉%（agd_b12 = 46.17%）。
#       ⚠ 對照是 **agd_b12（26.432）**，不是 sched30_b12 —— 問的是「更強有沒有更好」。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
export TRIM_SUBSAMPLE=4   # 週期 trim 只用 71/284 台
rm -rf outputs/tsub4_b12
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
  -n tsub4_b12
