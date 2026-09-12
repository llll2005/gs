#!/bin/bash
# ★★★★★ 成本感知**重做**：加法式 + **精確** c_i（cost_add_densify 2.278）
#
# 為什麼重做（§11.108 / §11.109，兩次 9.6h 的失敗換來的）：
#   冪次式 `(c/med)^-w` 兩個符號**都輸**：w=-0.5 PSNR -4.8sd ／ w=+0.5 PSNR -11.6sd、SSIM -59.2sd
#   而它的動態範圍是 **300x**（實測權重 0.083~25.0）
#   同期 `absgrad_densify` 用**加法式** `1 + w*ĝ`、範圍僅 ~26x => **贏**，是現行最佳的一部分
#   ⇒ 疑點在**形式與強度**，不在訊號本身。
# 兩處都換掉：
#   形式  冪次 -> **加法**（`probs *= 1 + w*ŝ`，下界 1.0，不會把任何粒子壓到接近 0）
#   訊號  代理 `_max_radii2D²` -> **精確** `num_covered_pixels`（trim 每 500 步已免費算好，
#         現行配方下算完就丟）。代理把離散度**誇大 66%**（ceiling 15.26x vs 精確 9.19x）
# w 的校準（不是猜的）：令 `1 + w*ceiling(1/ĉ)` = 25.8 = absgrad 已證有效的動態範圍
#   實測 ceiling(1/ĉ) = 10.89x  =>  **w = 2.278**
# 方向：w > 0 = 往**便宜**（小足跡）的地方增生 = **我方命題的方向**
#
# ⚠ 判準以 **PSNR/LPIPS** 為主；紋理比在本工作點跟著塵埃走（§11.106），不可當正面證據。
#   對照 agd2_b12 26.5602 / 0.7755 / 0.3467 ／ floater 1.660%。
# ⚠⚠ 2026-09-07 第一次啟動（09:07）**已中止並修正**：實現的權重範圍是 **1.001~148.7x**，
#   而非設計的 25.8x。成因：w 是用 `ceiling`（top5% **平均**）解出來的，但 `1/c` 在 c->1
#   （幾乎看不見的塵埃）有**無界尾巴**，max 遠大於 ceiling。
#   這是真缺陷：加法式**乘在 opacity 上** => 塵埃 0.005x148=0.74 > 正常 0.15x1=0.15
#   => 塵埃會壓過正常粒子成為主要分裂父代 —— 正是冪次式失敗的同一種病。
#   已修：把 ŝ 夾在它自己的 ceiling => max 權重 = 1+w*ceiling（依構造），校準點上 = 25.8x。
# ★ 開跑後必須看到 `[cost-add] ✅ 首次觸發`，且**權重範圍上限應在 ~26x 附近**；
#   若又看到三位數，代表夾子沒生效，立刻中止。
#   （trim 必須開著 —— 精確 c 從那裡來；densify_from_iter 1000 > 第一次 trim 500 ✅）
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/cadd_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.cost_add_densify 2.278 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n cadd_b12
