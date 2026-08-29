#!/bin/bash
# ★★★★★ AbsGS 訊號當增生取樣權重：sched30 + absgrad_densify（§11.30 / §11.34）
#
# 這是 `vpc_prune_frac` 慘敗（-2.34 dB，§11.34）之後**接線改對**的版本：
#   ⛔ 舊：把低 v/c 的粒子併進 dead_mask => relocation => 丟到隨機宿主、毀掉它在做的事
#   ✅ 新：用訊號決定**往哪裡增生**（改變集合），既有幾何完全不動
#   兩者共用 `mcmc_density_controller.py:320` 的 `probs` 插槽（blur_split_budget 也在那）。
#
# 為什麼相信有空間（§11.30，實測）：
#   我方誤差分數 ceiling(top5%)/mean = **1.22** => 完美取樣器也只能看到 1.22 倍平均誤差
#     => 沒東西可賺（`egd` 也確實無效，而且那個旗標根本是死碼，§11.29）
#   `|g|` 的同一個量 = **16.68**，13 倍；top10% 與 opacity 只重疊 4.4~20.6%（隨機 10%）
#     => 選的是幾乎不相干的一批 => **換訊號會換掉增生位置**
#   到 step 600 opacity 取樣只吃到 2.76/16.68 = 17%，還有 6 倍空間
#
# ⚠ 需要光柵器以 ABSGRAD=1 編譯（已編，2026-08-25，.so mtime 已驗）。
# ⚠ **只開這一個**，不要同時開 cost_aware_densify —— 先分清楚哪個有效。
# ⚠ 強度 w=1.0：最高 |g| 的粒子權重加倍。vpc 的教訓是**先從保守強度開始**（那次選 5% 直接崩）。
#
# 判準：多指標 + 建築低頻 + 最差10% + `tools/veil_detect.py`，對照 sched30_b12
# （26.377 / 0.7711 / 0.3542 / 建築低頻 0.02570 / 疊影 17.81%）。
# ⚠⚠ 噪音底已重新標定（§11.29）：同組態重跑 PSNR 全距實測 **0.107**
#    => **<0.3 dB 一律不可宣稱**，要看多指標是否同向。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/agd_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 1.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n agd_b12
