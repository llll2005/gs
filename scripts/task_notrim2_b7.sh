#!/bin/bash
# ★★★★★ 新最佳配方的跨塊確認：notrim2 換到 block_7
#
# 為什麼這是最優先（2026-08-25）：`notrim2_b12` 26.535 是新高且**全判準有利**
# （建築低頻 +4.4x 底、floater 1.087%->0.474%、VRAM 更低、還快 5%，見 研究總覽 §11.23），
# 但它是 **n=1**。本專案方法論一向要求 2x2；b7 是內容重的塊，最有可能推翻它。
# **如果 b7 推翻，後面所有建立在「trim 是淨負」上的規劃都要重來** => 先做這個。
#
# 唯一變數＝ block_id 與 init PLY。cap 刻意維持 2.6M 與 b12 完全一致（換 cap 就失去可比性），
# 其餘旗標與 `task_notrim2.sh` 逐字相同。
#
# 對照基準：`sched30_b7` 24.993 / 0.8009 / 0.2554（同組態但有 trim，2.34M 顆）
# 判準（一律多指標 + 最差10% + 建築低頻，`tools/audit_all.py`）：
#   四指標同向勝 且 建築低頻改善 => 「trim 淨負」跨塊成立 => 進配方，改寫 _ctx 與記憶
#   b7 打平或變差                => 效應是 b12 特有（半面水？顆數天花板？）=> 不可推廣，
#                                   §11.23 要降級為「b12 上成立」並查兩塊差在哪
#
# ⚠ b7 的收斂餘量外推是 1.234 dB（b12 只有 0.10~0.20）=> b7 到 60k 仍在爬，
#   兩塊的絕對值不可互比，只能各自對自己的 sched30 對照。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/notrim2_b7
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.renderer.init_args.diable_trimming true \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n notrim2_b7
