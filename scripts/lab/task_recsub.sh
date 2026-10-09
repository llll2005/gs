#!/bin/bash
# 2026-10-09 trim 的 record 只用部分相機會剪錯多少（新年代重量）：tools/record_subsample_overlap.py
#   cs60_sfmdup4 b6／b13 @14,999 與 @29,999（增生期中段與結尾，trim 只在增生期跑）。只渲染 => 不比時間，可平行。
# 用法：bash scripts/lab/task_recsub.sh
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
L=logs/recsub; mkdir -p "$L"
for B in 6 13; do
  for S in 14999 29999; do
    o="$L/b${B}_${S}.txt"; [ -s "$o" ] && { echo "已有 $o"; continue; }
    ck=$(ls outputs/lab/cs60_sfmdup4/blocks/block_$B/checkpoints/*step=$S.ckpt 2>/dev/null | head -1)
    [ -n "$ck" ] || { echo "⛔ 缺 b$B @$S"; continue; }
    conda run -n gspl --no-capture-output python tools/record_subsample_overlap.py --ckpt "$ck" 2>&1 \
      | grep -vE "pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample" | tee "$o.tmp" \
      && grep -q "half_even" "$o.tmp" && mv "$o.tmp" "$o"
  done
done
