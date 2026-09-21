#!/bin/bash
# ★★★★★★ 驗證 add_new_gs 真的有被呼叫（2026-09-21 的縮排 bug 回歸測試）
#   ⚠ 必跑 >= 1,600 步：densify_from_iter=1000、interval=150 => 事件在 1050/1200/1350/1500…
#     上一次我只跑 1,200 步就說「功能正常」，等於只驗到 trim 觸發，**完全沒驗到增生**。
#   判準：N 必須**成長**。壞掉時 N 會單調衰減（只有 trim 在剪，每 500 步 -10%）。
#   兩臂都要驗：exact_tile_cost 開/關 —— 壞掉的正是**關**的那條（_max_tiles2D 恆為 None）。
set -u
cd "$(dirname "$0")/.." || exit 1
unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
R=logs/densify_verify_$(date +%m%d_%H%M).log
run1 () {   # run1 <名稱> <extra...>
  local name=$1; shift
  [ -d "outputs/$name" ] && mv "outputs/$name" "outputs/$name.old_$(date +%m%d_%H%M%S)"
  conda run -n gspl --no-capture-output python -u main.py fit \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6 \
    --model.initialize_from null --trainer.max_steps 1600 \
    --data.image_uint8 true --data.skip_unused_depth true \
    --model.density.init_args.cap_max 2600000 --model.density.init_args.densify_until_iter 30000 \
    --model.density.init_args.absgrad_densify 2.0 --model.density.init_args.fast_noise true \
    --model.density.init_args.noise_gate_eps 0.001 \
    --model.density.init_args.churn_report true \
    --model.metric.init_args.opacity_reg 0.002 --model.metric.init_args.lambda_normal 0.0 \
    --model.metric.init_args.depth_loss_weight.init 0.0 "$@" -n "$name" > "logs/$name.log" 2>&1
  echo "##### $name rc=$? #####"
  grep -E "^\[churn\]" "logs/$name.log" | head -5
}
{ run1 dv_off
  run1 dv_on --model.density.init_args.exact_tile_cost true
  echo "════ 判準：新增 必須 > 0，且 N 要成長 ════"
  for n in dv_off dv_on; do
    a=$(grep -oP '(?<=\[churn\] step 1050: N=)[0-9,]+' "logs/$n.log" | head -1 | tr -d ,)
    b=$(grep -oP '新增=\K-?[0-9,]+' "logs/$n.log" | head -1 | tr -d ,)
    echo "  $n: step1050 起始 N=${a:-?}  首次新增=${b:-?}  $([ "${b:-0}" -gt 0 ] 2>/dev/null && echo '✅' || echo '⛔ 仍然是 0')"
  done
} 2>&1 | tee "$R"
