#!/bin/bash
# ★★★★★ 命題的**約束端**（唯一未測的一塊）：cost_budget 取代 cap_max 的顆數計價
#
# 取樣端已定案否證（3 變體 / 29h / 全輸，§11.111）。約束端是**不同的機制**：
#   取樣 = 「往哪裡增生」  ／  約束 = 「用什麼當預算單位」
# 命題的字面形式：max Q(θ_S)  s.t.  (1/K)Σ_{i∈S} c_i <= B
#
# ⚠ 兩個 Gemini 建議中我不採納的地方（有實測依據）：
#   1 **不拔除 cap_max**。本檔 config 的 docstring 自己就寫「cap_max 設高當安全閥」。
#     這個 session 才因為拔掉 screen_size_prune 這個安全機制而 OOM（§11.103）。
#     => cap 3.5M 當上限，讓 cost_budget 成為實際綁住的約束。
#   2 現行 cost_budget 用 `(2r/16)²` 代理而非精確 c_i。換成精確要改碼，
#     且代理**誇大離散度 66%**（§11.109）=> 本次先驗「約束端有沒有用」，訊號精度是下一步。
#
# ⚠ B 不可用探針的 23.1M（docstring 明載：同定義**不同實作路徑**）=> 本腳本自行標定：
#   階段 1  用現行最佳的 60k ckpt 起，跑數十步只為印出**成熟族群**的線上 Load
#           （在 2,500 步標定是錯的 —— 那時 N 只有 0.3M，Load 不具代表性）
#   階段 2  B = 0.90 x 觀測值，從頭跑滿 60k
# ★ 兩階段都會印 `[cost-budget]`；階段 2 若 Load 從未逼近 B => 約束沒綁住，等於 no-op。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
CKPT=$(ls outputs/speed3_b12/blocks/block_12/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls outputs/agd2_b12/blocks/block_12/checkpoints/*step=60000.ckpt | head -1)
echo "=== 階段 1：用 $CKPT 標定成熟族群的線上 Load ==="
rm -rf outputs/cbcal
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from "$CKPT" \
  --data.parser.block_id 12 \
  --trainer.max_steps 60 --trainer.enable_checkpointing false \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 60 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 0 \
  --model.density.init_args.cost_budget_report 20 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --model.dynamic_strips true \
  --model.strip_max 8 \
  --model.strip_vram_target_gb 5.4 \
  --model.strip_v_os_gb 0.8 \
  --model.strip_safety 0.6 \
  -n cbcal 2>&1 | tee logs/cbcal.txt
LOAD=$(grep -o "Load(區間最壞視角)=[0-9,]*" logs/cbcal.txt | tail -1 | tr -d ',' | cut -d= -f2)
N_VAL=$(grep -o "N=[0-9,]*" logs/cbcal.txt | tail -1 | tr -d ',' | cut -d= -f2)

if [ -z "$LOAD" ] || [ -z "$N_VAL" ]; then
  echo "⛔ 標定失敗：沒抓到 Load 或 N，中止"
  exit 1
fi

# B = (Load / N_VAL) * 2340000 * 0.90
B=$(awk -v l="$LOAD" -v n="$N_VAL" 'BEGIN{printf "%d", (l/n) * 2340000 * 0.90}')
echo "=== 階段 2：觀測 Load=$LOAD  =>  B=$B（0.90x）；cap 3.5M 只當安全閥 ==="
rm -rf outputs/cb_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 3500000 \
  --model.density.init_args.cost_budget "$B" \
  --model.density.init_args.cost_budget_report 2000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n cb_b12
