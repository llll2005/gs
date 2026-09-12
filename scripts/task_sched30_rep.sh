#!/bin/bash
# ★★★★★ 把 sigma 釘死：sched30_b12 同組態重跑（使用者指示）
#
# 今天所有比較都建立在 **n=1** 的噪音估計上，而它已經讓我誤判至少三次：
#   notrim2 vs sched30 +0.158 = 1.5 sigma（我宣稱成新高）
#   notrim2_b7        -0.137 = 1.3 sigma（我宣稱成「反轉」）
#   egd vs sched30    -0.106            （根本是同組態重跑，旗標是死碼）
# `_ctx.md` 記載的噪音底 0.0325 是 n=2 單對、自述「不是 sigma」；
# 唯一的同組態重跑（egd vs sched30）實測 PSNR 全距 **0.107**，是它的 3.3 倍（§11.29）。
#
# 本跑次是**第三個**同組態樣本 => 把全距從 n=1 變成 n=2，並給 sigma 一個真正的下界。
# ⚠ 它不會提高分數。它讓之後每一個結論站得住。
# ⚠ 與 sched30_b12 / egd_b12 的差異僅在：期間加了零權重 loss 閘門（§11.35，位元級等價）
#   與 ABSGRAD（寫入未被讀的 .z，位元級等價）=> 仍是同組態。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/sched30rep_b12
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
  -n sched30rep_b12
