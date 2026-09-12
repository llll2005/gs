#!/bin/bash
# ★★★★★ 收割期關掉 opacity L1（`opacity_reg_until_iter 30000` = densify 停的那一步）
#
# 前提**早就量過**但從沒端到端跑（§11.69，受控 2,500 步三臂，唯一變數就是這個旗標）：
#   A 對照    判死 3.77 -> 7.34（+3.57pp）  確信 o>0.5  9.12 -> 12.03（+2.91pp）
#   B 移除L1  判死 3.77 -> 3.71（**-0.06pp**）        -> 17.36（**+8.24pp**）
#   => `opacity_reg` 在收割期是**純損耗**，拿掉它連兩極化都快 2.8 倍。
#
# 它同時打中兩個獨立觀察到的損失：
#   ① `cap` 只交付 **90%**，那 10% **全掉在收割期**（densify 停但 opacity_reg 續壓）
#      => 回收它 ≈ 2.34M -> 2.6M（+11% 顆數）≈ +0.08 dB
#   ② **106 個完賽跑次裡有 60 個在 step 56,800 見頂、60,000 更低**（-0.02 ~ -0.20 dB）
#      => 末段的緩慢劣化，機制與 ① 同源
#
# ⚠ 風險：opacity_reg 是塵埃的主要抑制者（`scale_reg=0` 曾大輸 -18sd 是同一類）
#   => **必須連同 floater 一起判**（`tools/geometry_health.py`）；floater 若上升，
#      那就是「用幾何品質換光度分數」，不可採用。
# ⚠ 判準：對照 speed3_b12（26.6957 / 0.7757 / 0.3446 / 0.4744，floater 1.540%）。
#   ★ 特別看**終點 N** 是不是回到 ~2.6M（機制是否生效的直接證據）。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/oregstop_b12
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
  --model.metric.init_args.opacity_reg_until_iter 30000 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n oregstop_b12
