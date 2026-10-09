#!/bin/bash
# 2026-10-09 使用者的 trim 改進構想（只剪幾乎無貢獻、剪了不補、只剪幾次）的離線閘門：
#   對 cs60_sfmdup4 b6／b13 的 15k／30k／42k／60k ckpt，量「v <= tau 的刪掉」在各門檻下的顆數、Load、val 與塊內 held-out 變化。
#   不重訓、只渲染評分 => 不比時間，可平行。輸出 logs/zeroprune/<塊>_<步>.txt。
# 用法：bash scripts/lab/task_zeroprune.sh [跑次=lab/cs60_sfmdup4]
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
RUN=${1:-lab/cs60_sfmdup4}
L=logs/zeroprune; mkdir -p "$L"
for B in 6 13; do
  for S in 14999 29999 41999 60000; do
    o="$L/$(basename "$RUN")_b${B}_${S}.txt"
    [ -s "$o" ] && { echo "已有 $o"; continue; }
    ck=$(ls outputs/$RUN/blocks/block_$B/checkpoints/*step=$S.ckpt 2>/dev/null | head -1)
    [ -n "$ck" ] || { echo "⛔ 缺 $RUN b$B step $S"; continue; }
    echo "════ $RUN b$B @$S"
    conda run -n gspl --no-capture-output python tools/zero_contrib_prune.py --ckpt "$ck" 2>&1 \
      | grep -vE "pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample" | tee "$o.tmp" \
      && grep -q "不剪" "$o.tmp" && mv "$o.tmp" "$o"
  done
done
