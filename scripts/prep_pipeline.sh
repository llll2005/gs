#!/bin/bash
# ★ 11-17 資料準備一鍵流程（使用者「資料準備有沒有新底用不到的可以去掉？init 資料準備能不能簡化／合併讓它更快？」「可以作」）
# 只做新底用得到的步驟，能平行的平行，每步計時；輸出全部寫到 <目標資料夾>（不動現行資料），最後和現行檔案逐檔比對。
#   1 單目深度（GPU）‖ 預縮影像（CPU，同時跑）         2 單目深度的逐張尺度（get_depth_scales）
#   3 SfM 三角形校正深度（多程序）                      4 分區（partition_from_colmap，不用 coarse）
#   5 SfM＋補點底（CITYGS_SFM_FAST=1：SfM 只陣列化讀一次並快取；塊順序與現行檔案相同 => 產出應逐位元相同）＋裁盒
#   6 校正深度版塊外遮罩（逐塊平行）                    7 稠密起點（逐塊平行；--seed_per_block，與現行檔案差一個亂數種子的程度）
# 驗證用的接法：第 1、2 步的新產出寫到 *_new，下游照用現行的單目深度與尺度（符號連結）=> 第 3 步之後能逐位元比對；時間照樣是完整的。
# 拿掉的（新底用不到）：舊版塊外遮罩 block_masks44（65 分）、補點的其他 dup、舊 5x5 depth_init、平面外推、SfM 可見性快取（v4 已關）。
# 用法（lab，[solo]：第 1 步要 GPU、其餘吃滿 CPU，不能和計時中的訓練同時跑）：
#   bash scripts/prep_pipeline.sh <現行資料夾> <目標資料夾> [平行數]
case "${1:-}" in -h|--help|"") sed -n 2,13p "$0"; exit 0 ;; esac
set -u
SRC=$(realpath "$1"); DST=$2; J=${3:-8}
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$DST"; DST=$(realpath "$DST")
for x in input sparse estimated_depths estimated_depth_scales.json; do [ -e "$DST/$x" ] || ln -s "$SRC/$x" "$DST/$x"; done
PY="conda run -n gspl --no-capture-output python -u"
TS="$DST/prep_timing.tsv"; : > "$TS"
LOG="$DST/prep.log"; : > "$LOG"
stage () {   # stage <名稱> <指令...>
  local n=$1; shift; local t0; t0=$(date +%s)
  echo "▶ $(date '+%H:%M:%S') $n" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%d\t%d\n' "$n" "$dt" "$rc" >> "$TS"
  echo "✔ $n：$((dt / 60)) 分 $((dt % 60)) 秒（rc=$rc）" | tee -a "$LOG"
  [ "$rc" = 0 ] || { echo "⛔ $n 失敗，停（看 $LOG）"; exit 3; }
}
s1 () {
  $PY utils/run_depth_anything_v2.py "$DST/input" --output "$DST/estimated_depths_new" &
  local p1=$!
  $PY tools/make_resized_images.py "$DST" --nproc 12 &
  local p2=$!
  wait $p1 || return 1; wait $p2 || return 2
}
s5 () {
  CITYGS_SFM_FAST=1 $PY tools/make_sfm_fill_init.py "$DST" --blocks 0 1 2 3 4 5 7 8 9 10 11 13 14 15 --block_dim 4 4 --dup 1 --out-dir "$DST/sfmfill_sweep44" || return 1
  CITYGS_SFM_FAST=1 $PY tools/make_sfm_fill_init.py "$DST" --blocks 6 12 --block_dim 4 4 --dup 1 --out-dir "$DST/sfmfill_sweep44" || return 2
  $PY tools/block_mask44.py --data "$DST" --blocks $(seq 0 15) --crop_ply sfmfill_sweep44 --skip_masks || return 3
}
s6 () { seq 0 15 | xargs -P "$J" -I{} $PY tools/block_mask44.py --data "$DST" --blocks {} --depth_dir estimated_depths_tri --depth_raw --out "$DST/block_masks44_tri"; }
s7 () { seq 0 15 | xargs -P "$J" -I{} $PY tools/make_tri_dense_init.py "$DST" --blocks {} --seed_per_block; }
T0=$(date +%s)
stage "1 單目深度 ‖ 預縮影像" s1
stage "2 單目深度尺度" $PY utils/get_depth_scales.py "$DST" --depth_dir "$DST/estimated_depths_new" --output "$DST/estimated_depth_scales_new.json"
stage "3 校正深度" $PY tools/make_tri_depths.py "$DST" --nproc 16
stage "4 分區" $PY utils/partition_from_colmap.py "$DST" --block_dim 4 4 --content_threshold 0.08
stage "5 SfM＋補點底＋裁盒" s5
stage "6 塊外遮罩（校正深度版）" s6
stage "7 稠密起點" s7
echo "合計 $(( ($(date +%s) - T0) / 60 )) 分" | tee -a "$LOG"
$PY tools/prep_compare.py "$SRC" "$DST" 2>&1 | grep -v -i "warn\|pkg_res\|declare_ns" | tee -a "$LOG"
echo "DONE prep_pipeline"
