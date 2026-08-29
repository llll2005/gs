#!/bin/bash
# ★★★★★ 拆開 §11.24 的混淆：notrim2 + densify_until 12,000（b7）
#
# 使用者注意到曲線在 30k 附近停滯 => 查下去發現 §11.26 的混淆：
#   trim 一關，族群 3,000 步內就撞 cap（每 150 步 +5%，34 萬->260 萬只要約 6,200 步），
#   之後做了 **27,700 步純 churn**，是 trim-on 各臂（9,600）的 **2.9 倍**，而那段實測淨負。
#   根源：`densify_until=30,000` 是為 **trim 開啟時**調的（那時 cap 在 ~22k 才撞到）。
#   => **notrim2 從來沒有被公平測試過**，§11.24 的「b7 反轉」不可歸給「缺少 trim」。
#
# 本臂：densify_until 30,000 -> 12,000 => churn 12,000-2,300 = 9,700 步 ≈ sched30_b7 的 9,600。
# ⚠ 無法只動一個變數：收割期同時 30,000 -> 48,000 步（三時間尺度問題，§11.17）。
#
# 判準（多指標 + 建築低頻 + 最差10% + `tools/veil_detect.py`），對照兩個：
#   sched30_b7  24.993 / 0.8009 / 0.2554 / 建築低頻 0.02195   （trim 開，churn 9,600）
#   notrim2_b7  24.856 / 0.8140 / 0.2374 / 建築低頻 0.02322   （trim 關，churn 27,700）
#   贏過 notrim2_b7 且建築低頻回到 sched30 水準 => **churn 是主因**，trim 本身不是
#   仍輸給 sched30_b7                          => churn 不是主因，§11.24 的歸因成立
# ⚠ 有效的話必須在 b12 重跑才算數（2026-08-26 教訓：單塊結論不可推廣）。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/ntdu12_b7
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.renderer.init_args.diable_trimming true \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 12000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n ntdu12_b7
