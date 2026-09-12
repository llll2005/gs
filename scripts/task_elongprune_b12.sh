#!/bin/bash
# ★★★★ Elongation Filter 路徑 (a)：**硬剪**（CityGaussian V2 的做法）—— 使用者 2026-09-12 指定兩路都測
#
# 目標（tools/elongation_ceiling.py，speed3_b12 @60k，純 CPU 代理）：
#   forward.cu:281 的 tile 數 = `max(extent.x, extent.y)` 撐出的**正方形外接盒**
#   => 長寬比 k 的粒子有 1-1/k 的 binning 是空白。
#   長寬比 p50=3.17 / p90=13.28 / p99=40.93   <- 拉長是**常態**不是罕見病理
#   Σs_max² vs Σs_max·s_min => **浪費 76.5%**
#   剔除 >20（4.85% 顆）=> binning -14.0%，真覆蓋只 -1.8%   （7.9:1 的交換）
#   而逐段計時量到 backward 佔 37~47%、逐顆項佔 77% => 打的是最大宗。
#
# ⚠ 門檻 20 是照代理量測選的（4.85% 顆數、7.9:1）。`task_binning_trim_probe.sh` 會用
#   光柵器回傳的**精確** radii/num_covered_pixels 重算 —— 代理忽略了 truncated_R 與
#   FilterSize 下限，會**高估**可回收比例 => 探針回來後可能要改門檻，改了再跑。
# ⚠ 被剔除的那批帶著約 5% 的不透明度質量（>20 門檻）=> 刪掉會改變畫面，不是免費的。
# 先驗：(a) 有 CityGSV2 的外部先例；(b) 有我方的內部反證（vpc_prune_frac -2.34 dB）
#       => 預期 (a) 比 (b) 有機會，但兩個都測。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/elongprune_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
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
  --model.density.init_args.elongation_prune 20.0 \
  -n elongprune_b12
