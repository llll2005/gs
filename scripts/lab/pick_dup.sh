#!/bin/bash
# ★★★★★★ 使用者 10-19：init 要跟 cap 走——全部組合都維持「起始接近滿額」（同 task_full44.sh prep 的規則：
#   起始顆數 <= 0.9 x cap 的最大 dup 份數），cap 才是單一變數。缺的 dup PLY 現場產（tools/make_sfm_fill_init.py），
#   crop=1（塊外遮罩臂）時改看裁盒後的顆數（tools/block_mask44.py --crop_ply；同 bestmask 的初始化）。
# 用法：bash scripts/lab/pick_dup.sh <塊> <cap> [crop 0|1] [最大份數 24]   => stdout 最後一行印出份數 K
set -u
cd "$(dirname "$0")/../.." || exit 1
B=$1; CAP=$2; CROP=${3:-0}; KMAX=${4:-24}
D=data/matrix_city/aerial/train/block_all; S=$D/sfmfill_sweep44
LIM=$((CAP * 9 / 10))
npts () { head -c 600 "$1" 2>/dev/null | grep -a 'element vertex' | awk '{print $3}'; }
best=
for k in $(seq 1 "$KMAX"); do
  F=$S/dup$k/block_$B.ply
  if [ ! -f "$F" ]; then
    mkdir -p "$S/dup$k"
    conda run -n gspl --no-capture-output python tools/make_sfm_fill_init.py "$D" --blocks "$B" --block_dim 4 4 \
      --out-dir "$S/dup$k" --fill-ratio 0.35 --fill-voxel 0.35 --dup "$k" >/dev/null 2>&1 || { echo "⛔ 產 $F 失敗" >&2; exit 2; }
  fi
  if [ "$CROP" = 1 ]; then
    C=${S}_crop/dup$k/block_$B.ply
    if [ ! -f "$C" ]; then
      T=$(mktemp -d); mkdir -p "$T/dup$k"; ln -s "$(realpath "$F")" "$T/dup$k/block_$B.ply"
      conda run -n gspl --no-capture-output python tools/block_mask44.py --blocks "$B" --skip_masks --crop_ply "$T" >/dev/null 2>&1
      mkdir -p "${S}_crop/dup$k"; mv "${T}_crop/dup$k/block_$B.ply" "$C" 2>/dev/null; rm -rf "$T" "${T}_crop"
      [ -f "$C" ] || { echo "⛔ 裁 $C 失敗" >&2; exit 2; }
    fi
    n=$(npts "$C")
  else
    n=$(npts "$F")
  fi
  echo "  dup$k：起始 $n（上限 $LIM）" >&2
  [ "${n:-0}" -le "$LIM" ] && best=$k || break
done
[ -n "$best" ] || { echo "⛔ dup1 起始已超過 0.9 cap" >&2; exit 3; }
echo "$best"
