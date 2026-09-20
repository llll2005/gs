#!/bin/bash
# ★★★★★ 驗證「uint8 影像快取（修正版）＋ skip_unused_depth」是否逐位元不改變訓練（2026-09-17，本機短跑）
#   A   原樣
#   A2  原樣再跑一次 => 先確認訓練本身可重現；不可重現就無法判定，報告會明說
#   D   兩個開關都開（--data.image_uint8 true --data.skip_unused_depth true）
# 1,200 步：涵蓋 step 1 的起始 trim、step 1,000 的週期 trim 與增生事件；b6、speed3 旗標、SfM init、原生 6GB
# 比對：最終 ckpt 的全部參數與 Adam 狀態逐位元；另記每次訓練行程的 RAM 峰值（RSS 取樣）
set -u
cd "$(dirname "$0")/.." || exit 1
unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
STEPS=${STEPS:-1200}
R=logs/bitcheck_data_$(date +%m%d_%H%M).log
COMMON=(--config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6
  --model.initialize_from null --trainer.max_steps "$STEPS"
  --model.density.init_args.cap_max 2600000 --model.density.init_args.densify_until_iter 30000
  --model.density.init_args.absgrad_densify 2.0 --model.density.init_args.fast_noise true
  --model.density.init_args.noise_gate_eps 0.001 --model.metric.init_args.opacity_reg 0.002
  --model.metric.init_args.lambda_normal 0.0 --model.metric.init_args.depth_loss_weight.init 0.0)
run1 () {   # run1 <跑次名> [額外參數...]
  local name=$1; shift
  [ -d "outputs/$name" ] && mv "outputs/$name" "outputs/$name.aborted_$(date +%m%d_%H%M%S)"
  echo "##### $name  額外參數：${*:-（無）}  $(date) #####"
  conda run -n gspl --no-capture-output python -u main.py fit "${COMMON[@]}" -n "$name" "$@" \
    > "logs/$name.log" 2>&1 &
  local bg=$! peak=0 s
  while kill -0 "$bg" 2>/dev/null; do
    s=$(ps -eo rss,args | awk -v n="-n $name" 'index($0, "main.py fit") && index($0, n) && !/awk/ {t+=$1} END {print t+0}')
    [ "$s" -gt "$peak" ] && peak=$s
    sleep 5
  done
  wait "$bg"; local rc=$?
  grep -E "\[depth\]|Traceback|Error" "logs/$name.log" | head -3
  echo "[RAM] $name 行程 RSS 峰值 $((peak/1024)) MiB（rc=$rc）"
  return "$rc"
}
last_ckpt () { find "outputs/$1/blocks/block_6/checkpoints" -maxdepth 1 -name '*.ckpt' | sed -E 's/.*step=([0-9]+)\.ckpt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-; }
{ bad=0
  run1 bitcheck_A || bad=1
  run1 bitcheck_A2 || bad=1
  run1 bitcheck_D --data.image_uint8 true --data.skip_unused_depth true || bad=1
  echo "════ 逐位元比對 ════"
  if conda run -n gspl --no-capture-output python tools/compare_ckpt_bits.py "$(last_ckpt bitcheck_A)" "$(last_ckpt bitcheck_A2)" --label "A vs A2（可重現性）"; then
    conda run -n gspl --no-capture-output python tools/compare_ckpt_bits.py "$(last_ckpt bitcheck_A)" "$(last_ckpt bitcheck_D)" --label "A vs D（uint8＋不載深度）" || bad=1
  else
    echo "⚠⚠ 原樣重跑就不相同 => 訓練本身不可重現，A vs D 的比較無法判定逐位元（仍印出供參考）"
    conda run -n gspl --no-capture-output python tools/compare_ckpt_bits.py "$(last_ckpt bitcheck_A)" "$(last_ckpt bitcheck_D)" --label "A vs D（僅供參考）"
    bad=1
  fi
  exit "$bad"; } 2>&1 | grep -vE "pkg_resources|declare_namespace" | tee "$R"
st=${PIPESTATUS[0]}; echo "報告：$R"; exit "$st"
