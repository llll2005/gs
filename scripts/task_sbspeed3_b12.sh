#!/bin/bash
# ★★★★ DBS/SB 色彩 + **現行最佳配方**（~8.8h）—— 使用者 2026-09-12 指定
#
# 為什麼值得重測：先前的 SB 負結果是**對著舊基準**量的。
#   `sb_refix_b12` 建於 2026-08-12 22:51，而 sched30(08-18) / absgrad(08-29) / speed3(09-06)
#   都在那之後 => 那次比的是「SB + 舊配方」vs「SH3 + 舊配方」。
#   本次：SB + speed3 配方（sched30 + absgrad 2.0 + fast_noise + noise_gate）。
#
# ⚠ 但要先講清楚落差有多大（`tools/cmp_runs.py speed3_b12 sb_refix_b12`）：
#     PSNR -1.5771 (-56.9sd)   SSIM -0.0391 (**-390.9sd**)
#     LPIPS -0.0861 (-99.9sd)  紋理比 -0.0573 (-11.1sd)
#   而配方本身只值約 +0.3 dB => 不預期翻盤，但「對著現行基準量過」本身有價值。
#
# ⚠ **本次刻意維持 cap 2.6M（單一變數＝色彩基底）**，不順便放大顆數：
#   ① SB 25 floats vs SH3 58 floats 省下的空間換顆數，最多值 +0.5 dB/加倍（cheap_lever_scaling），
#      補不上 1.58 dB 的洞 => 先確認基底本身在新配方下還輸多少
#   ② 同 cap 不可能 OOM。`bf16mom_b12` 就是因為我用「省下的儲存」直接放大 cap 而 OOM
#      （漏算渲染側也隨 N 長：邊際成本 696+2050=2746 B/顆，不是 696）=> 燒掉 3.9 小時。
#   若同 cap 的差距縮到 3sd 內，再跑等 VRAM 的高 cap 版（估 cap ~3.0M）。
#
# ⚠ 時間/VRAM 的判準只能在**本機 6GB** 量 —— lab 主機是 24GB RTX 3090，不可比。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/sbspeed3_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 1e-3 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sbspeed3_b12
