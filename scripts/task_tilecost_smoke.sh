#!/bin/bash
# ★★★★ 精確 tile 成本（exact_tile_cost）功能測試：本機 b6 短跑 500 步
#   要看到的：① [exact-tile-cost] ✅ 首次生效那行，含精確 vs 代理的比值（= 預算換算係數）
#             ② [cost-budget] 的 Load 打印改用精確單位 ③ 訓練正常、trim 正常觸發
#   ⚠ 這不是品質實驗；cost_budget 沒設（純標定模式）。
set -u
cd "$(dirname "$0")/.." || exit 1
unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
R=logs/tilecost_$(date +%m%d_%H%M).log
{
name=tilecost_on
[ -d "outputs/$name" ] && mv "outputs/$name" "outputs/$name.old_$(date +%m%d_%H%M%S)"
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6 \
  --model.initialize_from null --trainer.max_steps 500 \
  --data.image_uint8 true --data.skip_unused_depth true \
  --model.density.init_args.cap_max 2600000 --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.absgrad_densify 2.0 --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.density.init_args.exact_tile_cost true \
  --model.density.init_args.cost_budget_report 200 \
  --model.metric.init_args.opacity_reg 0.002 --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 -n "$name" > "logs/$name.log" 2>&1
echo "rc=$?"
echo "════ 首次生效與預算打印 ════"
grep -E "exact-tile-cost|cost-budget|Trimming done" "logs/$name.log" | head -8
echo "════ 錯誤檢查 ════"
grep -E "Traceback|Error|nan" "logs/$name.log" | head -3 || echo "（無）"
} 2>&1 | tee "$R"
