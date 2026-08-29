#!/bin/bash
# ★ 每步 179 ms 的固定成本到底花在哪？（2026-08-25，約 5 分鐘 GPU）
#
# 由 ledger 擬合得到（68 個跑完的跑次）：
#   每步成本(ms) = 178.9 + 100.5 x N(百萬顆)     R^2 = 0.673
#   => 在 N=2.34M，固定成本佔 43%，且顆數砍半只省 28%
#   => 「少一點高斯」這個古典槓桿在我方很弱；真正該問的是那 43% 是什麼
#
# 已知線索：N=0.08M 的跑次每步仍要 163~188 ms（所以不是光柵化），
# 而 down_sample 2 的跑次固定成本掉到約 47 ms（所以是逐像素的東西）。
#
# 已用讀碼確認的浪費（現行最佳配方 sched30 全中）：
#   gs2d_metrics.py:28-33      normal_error 與 rend_dist 無條件算完再乘 lambda(=0)，且進 loss => backward 也走
#   citygsv2_metrics.py:202-3  深度 loss（含一個 SSIM 窗積分）無條件算完再乘 weight(=0)
# ⚠ 但我沒量過它值多少 —— 本任務就是要量，不要先改碼再說「有效」。
#
# 用 Lightning 內建 profiler，不改任何程式碼。跑 300 步就停。
# ⚠ config 用 probe_ctrl.yaml = **正式跑次 noprior_b12 的 resolved config**（截到 2000 步），
#   不是已作廢的 proxy 組態。300 步時 N 只有約 34 萬，正好是固定成本主導的區間。
# 讀法：看 `run_training_batch` / `backward` / `optimizer_step` / `get_train_batch` 的佔比。
# 若 dataloader 等待佔比高 => 加 num_workers / 快取；若 backward 佔比高 => 零權重 loss 值得砍。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
OUT=logs/profile_$(date +%m%d_%H%M).txt
rm -rf outputs/profile_probe
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/proxy/probe_ctrl.yaml \
  --trainer.profiler simple \
  --trainer.max_steps 300 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 300 \
  -n profile_probe 2>&1 | tee "$OUT"
echo
echo "===== profiler 摘要 ====="
sed -n '/Action.*Mean duration/,/^$/p' "$OUT" | head -30
echo
echo "完整輸出: $OUT"
echo "⚠ 這是 300 步、族群還很小的狀態 => 固定成本的相對佔比會被高估，"
echo "  但它要回答的是「那 179 ms 是什麼」，而 179 ms 本來就是 N->0 的截距，正好對應。"
