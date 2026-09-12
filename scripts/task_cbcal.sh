#!/bin/bash
# ★★★ 成本預算的標定（~4 分鐘）—— 寫 controller 之前的最後一個前提步驟
#
# §11.60 用 tools/cost_budget_probe.py（解析投影）量到 agd2_b12 終點 B = 23.1M。
# 但線上的 Load 用的是**光柵器回傳的 radii**，兩者是同定義、不同實作路徑
# ⇒ **不可直接把 23.1M 當預算**，用錯的單位標定就是白跑 9.6 小時。
#
# 本任務：從 agd2_b12 的 60k ckpt 起跑 200 步，只開 `cost_budget_report`
# （`cost_budget=0` ⇒ **完全不改變行為**），讀出 N=2.34M 時線上量到的 Load。
# ⇒ 之後把 `cost_budget` 設成這個值，問題就變成「同樣的渲染成本能塞多少顆」。
#
# ✅ ckpt-init 只替換 gaussian_model 與 renderer（gaussian_splatting.py:149-185 已查），
#    density controller 不受影響 ⇒ 這個旗標會生效（不同於 trim_subsample_probe 那次的坑）。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/cbcal
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from "outputs/agd2_b12/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt" \
  --data.parser.block_id 12 \
  --trainer.max_steps 200 \
  --trainer.enable_checkpointing false \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 200 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.cost_budget_report 25 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n cbcal 2>&1 | tee logs/cbcal.txt | grep -E "cost-budget|DIED|Error"
echo; echo "===== 開獎：把 cost_budget 設成下面的 Load 值 ====="
grep -a "cost-budget" logs/cbcal.txt | tail -4 || echo "(無輸出 => 旗標沒生效，先查再排)"
