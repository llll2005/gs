#!/bin/bash
# 2026-10-01：dup 系列更激進的變體（使用者：「dup4 獨有的參數更激進」）。只產 PLY，不動佇列。
#   dup 系列專屬參數只有兩個：--dup（複製幾份）、--jitter（複製份的抖動，單位＝最近鄰間距的倍數，預設 0.5）。
#   jitter 在先前的掃描裡從沒測過。其餘參數同 sfmfill 基準（fill-ratio 0.35、fill-voxel 0.35）。
#     dup5      複製 5 份（起始約 2.96M > cap 2.6M：MCMC 不再加點，靠週期 trim 降回 cap 附近；終點 N 仍由 cap 決定）
#     dup4j10   dup4、抖動 1.0（散得更開）
#     dup4j025  dup4、抖動 0.25（更接近原點）
#   ⚠ 一律用 --blocks 6 12 13（亂數在塊迴圈外，清單不同會改變產出；與 task_sfmfill_sweep2.sh 一致）
set -u
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial/train/block_all
OUT=$D/sfmfill_sweep
for v in dup5 dup4j10 dup4j025; do
  case "$v" in
    dup5)     a="--dup 5" ;;
    dup4j10)  a="--dup 4 --jitter 1.0" ;;
    dup4j025) a="--dup 4 --jitter 0.25" ;;
  esac
  if [ -f "$OUT/$v/block_6.ply" ] && [ -f "$OUT/$v/block_13.ply" ]; then echo "略過 $v（已存在）"; continue; fi
  echo "=== 產生 $v：$a $(date) ==="
  conda run -n gspl --no-capture-output python tools/make_sfm_fill_init.py "$D" --blocks 6 12 13 \
    --out-dir "$OUT/$v" --fill-ratio 0.35 --fill-voxel 0.35 $a || exit $?
  for b in 6 13; do echo "  $v/block_$b：$(head -c 400 "$OUT/$v/block_$b.ply" | grep -a -m1 'element vertex' | awk '{print $3}') 點"; done
done
echo "✅ dup 變體產生完成"
