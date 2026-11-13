#!/bin/bash
# ★ 2026-11-13 只合併其中幾塊（使用者「排 20k 的 1 2 6 7 幫我 merge，我要處理大樓破掉的問題」）。
# 用法：bash scripts/lab/merge_subset.sh <跑次家族（不帶 lab/）> <輸出名稱> <塊> [<塊> ...]
#   例：bash scripts/lab/merge_subset.sh g44tn_s20g70np16tricap0p5cmflattrimasklowresgeo8log0_best1maskdistrfed10b4 base3_b1267 1 2 6 7
# 做法：outputs/lab/<輸出名稱>/blocks/block_<B> 用符號連結指到家族的那幾塊，再跑 utils/merge_citygs_ckpts.py
#   => outputs/lab/<輸出名稱>/checkpoints/merged.ckpt（各塊依分區盒裁切後合併；同 task_fullscene.sh merge）。各塊步數要一樣。
case "${1:-}" in -h|--help|"") sed -n 2,6p "$0"; exit 0 ;; esac
cd "$(dirname "$0")/../.." || exit 1
FAM=$1; NAME=$2; shift 2
SRC=outputs/lab/$FAM; OUT=outputs/lab/$NAME
mkdir -p "$OUT/blocks"
st=
for b in "$@"; do
  d=$SRC/blocks/block_$b
  s=$(ls "$d"/checkpoints/*.ckpt 2>/dev/null | grep -oE 'step=[0-9]+' | cut -d= -f2 | sort -n | tail -1)
  [ -n "$s" ] || { echo "⛔ 缺 $d 的 ckpt"; exit 2; }
  [ -z "$st" ] && st=$s
  [ "$s" = "$st" ] || { echo "⛔ 步數不一致：block_$b step $s ≠ $st"; exit 2; }
  ln -sfn "$(realpath "$d")" "$OUT/blocks/block_$b"
done
echo "合併 ${*}（step $st）=> $OUT/checkpoints/merged.ckpt"
conda run -n gspl --no-capture-output python utils/merge_citygs_ckpts.py "$OUT" 2>&1 | grep -vE 'pkg_resources|declare_namespace' | tail -5
rc=${PIPESTATUS[0]}
ls -la "$OUT/checkpoints/" 2>/dev/null
exit "$rc"
