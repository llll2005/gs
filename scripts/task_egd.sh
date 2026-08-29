#!/bin/bash
# ★★★★ 誤差導向增生：sched30 + err_guided_densify（零實作成本，接口早就寫好但從沒開過）
#
# MCMC 用 `probs = get_opacities()` 抽父代 —— **完全沒有誤差項**，所以它往「質量已經在的地方」
# 增生，而不是往「渲染錯的地方」。我方讀過的每一個 densifier 都用某種誤差訊號
# （FastGS 數足跡內的高誤差像素、Taming 加權 8 個訊號、Mini-Splatting 用最大貢獻面積），
# **我方是其中資訊最少的**。
#
# 開啟後：`probs = opacity * (1 + k * 正規化誤差)`（乘法不是取代 —— MCMC 的分裂公式
# o_new = 1-(1-o)^(1/N) 會把父代的 opacity 分給子代，所以抽到低 opacity 的父代只會產生更淡的
# 子代；保留 opacity 當基底才不破壞那個性質，只改變質量往哪去。Taming 的分數是加權乘積，同理。）
#
# 為什麼是現在（2026-08-25）：
#   * ⛔ 本臂原本掛在 notrim2 上，但 notrim2 已於 2026-08-26 撤回（b7 反轉，§11.24）
#     => 改掛回**唯一兩塊都驗證過的** sched30。機制與 trim 正交，前提不受影響。
#   * §11.15 量到**低頻/結構是唯一跨配方有差異的維度**（高頻全距只有 1.066x），
#     而 AbsGS 的診斷正是「大足跡的東西在既有訊號下是隱形的」
#   * `_report_densify_blindness` 會在 log 印出「opacity 取樣看到的誤差 / 全體平均」，
#     1.00 = 完全盲目。那個數字本身就是這條線值不值得繼續的判準
#
# ⚠ 已知缺陷（先跑再改）：`_accumulate_error_score` 在**投影中心**取 16px tile 的 |render-gt|，
#   **不乘足跡** => 一顆鋪滿一整面牆的大高斯與一顆灰塵拿到一樣的分數。
#   這正是 AbsGS 指出的盲點。若本臂有效但不多，下一步就是把它換成
#   `|g|`（已在 `viewspace_points.grad[:, 2]`，ABSGRAD=1 已編進 .so）或誤差 x 足跡。
#
# 判準：多指標 + 建築低頻 + 最差10%，對照 sched30_b12（26.377 / 0.7711 / 0.3542 / 建築低頻 0.02570）。
# ⚠ 有效的話**必須在 b7 重跑才算數**（2026-08-26 教訓：notrim2 在 b12 看起來全勝、b7 反轉）。
# k=1.0：最高誤差的粒子權重加倍。先用保守值，有效再往上調。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/egd_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.err_guided_densify 1.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n egd_b12
