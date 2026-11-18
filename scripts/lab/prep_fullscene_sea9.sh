#!/bin/bash
# ★ 11-18 全場景 v9 前置（使用者「全場景排在 60k 前面」；海面 v9＝不外推＋交界補齊（--gap2／--gap3）＋水體自動高度）：
#   obj <塊...>：這些塊產 v9 海面物件（同 b3 的參數；沒有水體的塊 make_sea_object 會跳過、不寫 npz）——可以分幾組同時跑（各寫各的塊）
#   init       ：sfmfill_tri44sea9_crop 缺的塊全部產（陸地＝v2、海面＝鋪磚；沒有海的塊沒有鋪磚），再檢查 16 塊起點與標記都在
# 用法（佇列）：[cpu] bash scripts/lab/prep_fullscene_sea9.sh obj 0 1 2 4 ；…；[solo] bash scripts/lab/prep_fullscene_sea9.sh init
case "${1:-}" in -h|--help|"") sed -n 2,6p "$0"; exit 0 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial/train/block_all
PY="conda run -n gspl --no-capture-output python -u"
mode=$1; shift
if [ "$mode" = obj ]; then
  [ $# -gt 0 ] || { echo "⛔ obj 要給塊號"; exit 2; }
  $PY tools/make_sea_object.py "$D" --blocks "$@" --masks --grow --gap --gap2 --gap3 --water_bodies "$D/water_bodies.npz" --far_levels 6 --out sea_object_v9 2>&1 \
    | grep -v -i "warn\|pkg_res\|declare_ns" | grep "水體\|gap3\|鋪磚邊長\|跳過\|DONE\|Error\|Trace"
  for b in "$@"; do [ -f "$D/sea_object_v9/block_$b.npz" ] && echo "  b$b 有 v9 海面物件" || echo "  b$b 沒有海面物件（沒有水體或跳過）"; done
  echo "DONE prep_sea9 obj $*"
elif [ "$mode" = init ]; then
  MISS=""; for b in $(seq 0 15); do [ -f "$D/sfmfill_tri44sea9_crop/dup1/block_$b.ply" ] || MISS="$MISS $b"; done
  echo "▶ $(date '+%H:%M:%S') 產 v9 起點：${MISS:-（都有）}"
  if [ -n "$MISS" ]; then
    $PY tools/make_tri_dense_init.py "$D" --blocks $MISS --drop_fill --plane_dir estimated_depths_plane --plane_sep --drop_outliers \
        --sea_object sea_object_v9 --out sfmfill_tri44sea9_crop 2>&1 | grep -v -i "warn\|pkg_res\|declare_ns" | tail -20
  fi
  bad=""; for b in $(seq 0 15); do [ -f "$D/sfmfill_tri44sea9_crop/dup1/block_$b.ply" ] && [ -f "$D/sfmfill_tri44sea9_crop/dup1/block_${b}_sea.npy" ] || bad="$bad $b"; done
  [ -z "$bad" ] || { echo "⛔ 缺起點或海面標記：$bad"; exit 7; }
  echo "✔ $(date '+%H:%M:%S') 16 塊 v9 起點齊全；v9 有海面物件的塊：$(ls $D/sea_object_v9/*.npz | wc -l)"
  echo "DONE prep_sea9 init"
else
  echo "⛔ 模式只有 obj／init"; exit 2
fi
