#!/bin/bash
# ★★★★★ 成本感知回收：sched30 + vpc_prune_frac 0.05（使用者提問帶出來的整條線）
#
# 這是本 session **第一個有正面前提量測支持**的介入（notrim2／egd 都是先做再發現前提不成立）。
#
# 前提量測（研究總覽 §11.28，`vpc_report` 在 notrim2_b12 的 60k 成品模型上，只讀不剪枝）：
#   rho(v,c)=+0.127  pearson(log v,log c)=+0.085   <- `v = kappa*c` 決定性推翻
#   等成本節省下的價值損失：砍 50% 渲染成本，按 v/c 只需剪 10% 顆粒、損 0.64% 價值；
#   按 v 剪 30% 才砍到 21.3%，要追上得剪到約 70%，價值損失高一個量級。
#   => 成本感知在 cost-value frontier 上壓倒性勝出。
#
# ⚠ 兩個必須講清楚的缺口：
#   1. **`Σv` 不是品質**。損失 0.64% 的 Σv 不等於損失 0.64% 的 PSNR —— 只有訓練跑次能回答。
#   2. 診斷量的是「**移除**」，但 `vpc_mask` 併進 `dead_mask`（controller :455-458）
#      => 實際走 **relocation（搬移）**，N 不變。搬移會繼承宿主尺度，霧狀巨獸換成正常大小，
#      成本應該會降，但**這一步沒有被診斷直接支持**。
#
# 強度：每個 densify 事件（150 步）回收 v/c 最低的 5%。`vpc_interval` 預設 1500 步刷新一次
# 多視角 value（24 視角 transmittance pass，攤下來 0.016 renders/step，可忽略）。
# ⚠ 這是**額外的 churn**，而 churn 已知淨負（§12.21/§12.25）=> 若輸，先懷疑強度太高再懷疑機制。
#
# 判準（多指標 + 建築低頻 + 最差10% + `tools/veil_detect.py`），對照 sched30_b12
# （26.377 / 0.7711 / 0.3542 / 建築低頻 0.02570）。**特別看 VRAM 與 it/s** —— 若命題成立，
# 成本感知回收應該同時**省 VRAM**，那是 count-based 剪枝做不到的（§11.23 量到 trim 省不到）。
# ⚠ 有效的話必須在 b7 重跑才算數（2026-08-26 教訓）。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/vpc_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.vpc_prune_frac 0.05 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n vpc_b12
