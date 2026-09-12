#!/bin/bash
# ★★★★★ 訓練時間到底花在哪？（真實迴圈逐段，CUDA event + wall clock，~15 分）
#
# 要回答的是使用者/教授的問題：「是因為搬運才慢還是什麼原因（找到前幾耗時的動作）」。
#
# 為什麼不能用既有的東西回答：
#   `logs/opprofile_*.txt`（cProfile）顯示 `Tensor.to` 佔 48.2s/67.5s = 71%，看起來就是「搬運」
#   —— 但那是**假的**，兩個已知陷阱同時中：
#     ① cProfile 把非同步工作記到下一個同步點，`.to()` 正是那個點 => 它吸收了光柵化的等待
#        算術也對不上：6,804 次小張量 H2D x 典型 10~20us = 約 0.14 秒，不可能是 48 秒
#     ② 40 步的視窗只含一次 trim pass（每 500 步）=> 它的 79.8% 是視窗效應
#   `tools/step_breakdown.py` 避開了那兩個坑，但**只量 5 段**，沒有 trim/optimizer/
#   density controller/相機搬移 —— 而 trim 正是先前調查裡「沒想到的真兇」。
#   `tools/vram_pressure.py` 臂 A 量到 render+backward+opt ≈ 155ms/百萬顆 + **21ms** 固定，
#   而全訓練擬合是 `178.9 + 100.5*N`（§11.18）=> **158ms 的固定成本在微基準之外**。
#
# 做法：在**真實迴圈**裡插 CUDA event（不強制同步、不破壞重疊）**同時**記 wall clock。
#   兩者的差 = CPU 側阻塞（dataloader / H2D 搬運 / Python 開銷）= 「是不是搬運」的直接答案。
#   計時器預設關閉，靠 `CITYGS_STEP_PROFILE=1` 開啟。
#
# 兩段，因為低顆數與高顆數的成本結構完全不同：
#   A  depth-init 從頭跑 2,000 步（N 約 0.5M）—— 含起始 trim、4 次 trim pass、約 7 次 densify
#   B  從 speed3_b12 的 60k ckpt 起跑 1,200 步（N = 2.34M）—— 高顆數穩態，且能抓到 step 1000 的 trim
#      ⚠ 用 ckpt init 時 renderer 會被 ckpt 的取代（_ctx 陷阱 1）——**這裡正是我們要的**，
#        因為那就是 speed3 自己的 renderer（trim 開、EXACT_SUPPORT 開）。
#      ⚠ 本次刻意**不設** max_split_size_mb（使用者 2026-09-12 決定）=> 順便是它的第一次實測。
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
export CITYGS_STEP_PROFILE=1
R=logs/stepcost_$(date +%m%d_%H%M).log

COMMON="--config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml
  --data.parser.block_id 12
  --model.density.init_args.cap_max 2600000
  --model.density.init_args.absgrad_densify 2.0
  --model.density.init_args.densify_until_iter 30000
  --model.density.init_args.screen_size_prune_px 300
  --model.density.init_args.fast_noise true
  --model.density.init_args.noise_gate_eps 1e-3
  --model.metric.init_args.opacity_reg 0.002
  --model.metric.init_args.lambda_normal 0.0
  --model.metric.init_args.depth_loss_weight.init 0.0"

{
  echo "##### A：depth-init 從頭 2,000 步（低顆數，含起始 trim 與週期性事件）#####"
  rm -rf outputs/stepcost_lo
  conda run -n gspl --no-capture-output python -u main.py fit $COMMON \
    --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
    --trainer.max_steps 2000 -n stepcost_lo 2>&1 | grep -vE "^Epoch|it/s\]|^ *$"

  echo
  echo "##### B：從 speed3_b12 的 60k ckpt 起跑 600 步（N = 2.34M 高顆數穩態）#####"
  rm -rf outputs/stepcost_hi
  conda run -n gspl --no-capture-output python -u main.py fit $COMMON \
    --model.initialize_from outputs/speed3_b12/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt \
    --trainer.max_steps 1200 -n stepcost_hi 2>&1 | grep -vE "^Epoch|it/s\]|^ *$"
} 2>&1 | grep -vE "pkg_resources|declare_namespace" | tee "$R"

echo
echo "===== 低顆數（stepcost_lo）====="; cat outputs/stepcost_lo/blocks/block_12/step_cost.txt 2>/dev/null
echo
echo "===== 高顆數（stepcost_hi）====="; cat outputs/stepcost_hi/blocks/block_12/step_cost.txt 2>/dev/null
echo "報告：$R"
