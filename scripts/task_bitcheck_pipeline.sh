#!/bin/bash
# ★★★★★ 管線層級逐位元驗證（2026-09-18）：送進訓練的 GT 影像是否逐位元相同
# 為什麼不比最終 ckpt：訓練本身不可逐位元重現（原樣跑兩次，25 個張量有 18 個不同，最大差 8e-5）
# => 改比「每一步送進 training_step 的 gt_image」的 sha1；這一段在任何 GPU 運算之前，不受非確定性影響
#   P0  原樣（float 快取）
#   P1  --data.image_uint8 true --data.skip_unused_depth true
# 兩者 shuffle 種子相同 => 影像順序相同；比對 60 步的（名稱, 形狀, dtype, 雜湊）整串
set -u
cd "$(dirname "$0")/.." || exit 1
unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
export CITYGS_BATCH_HASH_STEPS=60
R=logs/bitcheck_pipeline_$(date +%m%d_%H%M).log
COMMON=(--config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6
  --model.initialize_from null --trainer.max_steps 60
  --model.density.init_args.cap_max 2600000 --model.metric.init_args.depth_loss_weight.init 0.0)
run1 () {
  local name=$1; shift
  [ -d "outputs/$name" ] && mv "outputs/$name" "outputs/$name.aborted_$(date +%m%d_%H%M%S)"
  conda run -n gspl --no-capture-output python -u main.py fit "${COMMON[@]}" -n "$name" "$@" > "logs/$name.log" 2>&1
  local rc=$?
  grep -a "\[batch-hash\]" "logs/$name.log" | sed -E 's/^.*\[batch-hash\] //' > "logs/$name.hash"
  echo "$name rc=$rc 雜湊行數 $(wc -l < logs/$name.hash)；dtype：$(awk '{print $4}' logs/$name.hash | sort | uniq -c | tr '\n' ' ')"
  grep -aE "\[depth\]" "logs/$name.log" | head -1
  return "$rc"
}
{ bad=0
  run1 bitpipe_P0 || bad=1
  run1 bitpipe_P1 --data.image_uint8 true --data.skip_unused_depth true || bad=1
  # 比 (步數, 名稱, 形狀, 雜湊)，不比 dtype（兩者本來就該都是 float32；uint8 會在 on_after_batch_transfer 轉掉）
  # ⚠ 2026-09-18 修：形狀 "(3, 900, 1600)" 含空格會多切欄位，舊版取第 2~7 欄剛好漏掉第 8 欄的 sha1 => 改比整行
  a=$(cat logs/bitpipe_P0.hash); b=$(cat logs/bitpipe_P1.hash)
  n=$(wc -l < logs/bitpipe_P0.hash)
  if [ "$n" -ge 50 ] && [ "$a" = "$b" ]; then
    echo "✅ 管線層級逐位元相同：$n 步的 GT 影像（名稱、形狀、sha1）完全一致"
  else
    echo "⛔ 不一致或行數不足（P0 $n 行）；差異前 5 行："
    diff <(echo "$a") <(echo "$b") | head -10
    bad=1
  fi
  exit "$bad"; } 2>&1 | grep -vE "pkg_resources|declare_namespace" | tee "$R"
st=${PIPESTATUS[0]}; echo "報告：$R"; exit "$st"
