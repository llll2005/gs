#!/bin/bash
# ★★★★ scale_reg = 0：與 opacity_reg=0 同形狀的單變數實驗（§11.32）
#
# `scale_reg` 是現行配方裡**唯一從沒被動過**的正則（還是 config 預設 0.007），而它同時：
#   1. **幾乎不做它該做的事** —— 實測 2.9 倍正則只把尺度壓下 4%（重建損失頂回來）
#   2. **卻是塵埃的主犯** —— 紀錄兩處：「96.6% 是次像素塵埃，被 scale L1 碾的」
#      「scale L1 把 97% 的點碾成次像素塵埃」
#   實測現在仍有 15.5%(b12) / 7.2%(b7) 的塵埃（7/24 當時是 86.5%，opacity_reg 降低已修掉大半）。
#
# 這與 `opacity_reg` 0.01->0.002 那個發現**同一個形狀**：一個成本盲的 L1，做不到宣稱的事，
# 卻在製造 dead weight。那次拿掉換到 +1.04 dB、快 10.9x、省 3G。
#
# ⚠ 但**不要期待同樣的量級**：塵埃只佔 VRAM 約 6%，換成顆數只值 +0.045 dB。
#   真正的問題是**拿掉之後動力學會不會整個變好**（那才是 +1.04 的來源），那是開放的。
# ⚠ 風險：scale_reg 是唯一在壓「尺度」的機制，拿掉可能讓大足跡粒子失控 => **看 VRAM 與 it/s**，
#   以及 `tools/veil_detect.py`（大而透明的粒子正是疊影的來源）。
#
# 判準：多指標 + 建築低頻 + 最差10% + 疊影%，對照 sched30_b12
# （26.377 / 0.7711 / 0.3542 / 建築低頻 0.02570 / 疊影 17.81%）。
# ⚠⚠ 噪音底已重新標定（§11.29）：同組態重跑的 PSNR 全距實測 **0.107**，
#    所以 **<0.3 dB 的差異一律不可宣稱**，要看多指標是否同向。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/noscalereg_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.scale_reg 0.0 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n noscalereg_b12
