#!/bin/bash
# 2026-10-11 使用者：同 VRAM 的 Ap 池是否太多 => 去重版取代 Ap，先離線檢查（tools/pool_keep_check.py）。
#   對象：g44_best0poolA b6／b12 的 60k ckpt（池在載入時凍結）；K＝該塊 dup 份數（logs/full44_init.tsv）。
#   池只留 1／0.5／1/K／1/(2K)（隨機）與 opacity 最高的 1/K，量訓練視角「塊內內容／塊外背景」的 PSNR。只渲染、[solo]。
# 用法：[solo] bash scripts/lab/task_pool_keep.sh
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
L=logs/pool_keep_$(date +%m%d_%H%M).log
bad=0
for B in 6 12; do
  CK=$(ls outputs/lab/g44_best0poolA/blocks/block_$B/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
  K=$(awk -v b="$B" '$1==b {print $2}' logs/full44_init.tsv)
  [ -n "$CK" ] && [ -n "$K" ] || { echo "⛔ block $B 缺 ckpt 或 K（ck=[$CK] K=[$K]）"; bad=1; continue; }
  conda run -n gspl --no-capture-output python tools/pool_keep_check.py --ckpt "$CK" --K "$K" --ncam 24 2>&1 \
    | grep -vE "pkg_resources|declare_namespace|WARNING|找不到深度|caching|colmap|down sample|loading|appearance" || bad=1
done 2>&1 | tee "$L"
exit $bad
