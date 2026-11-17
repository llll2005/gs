#!/bin/bash
# ★ 11-18 全場景 v5 前置（海面 v5＝水平面 −2.6 m＋遠海分層＋外推海面；全場景 b3 遠海變黑 −0.81 dB 的修法）：
#   1 sea_object_v5 缺的有海塊（b0～b3 在單塊驗證時已產）用同一組參數補產：make_sea_object --masks --grow --sea_level -0.026 --far_levels 6 --far_fill 2.5
#     （沒有海的塊 make_sea_object 會跳過、不寫 npz）
#   2 sfmfill_tri44sea5_crop 缺的塊（含沒有海的塊）用同一組參數產（陸地＝v2、海面＝鋪磚＋遠海＋外推）
#   3 沒有海的塊（make_sea_object 沒寫 npz）：海面元件全部不生效 => 配方和新底 7 候選全場景的同一塊完全一樣
#     => 把那幾塊的訓練結果硬連結到 v5 家族（不重訓；run_summary 照用），印出清單
#   4 檢查 16 塊的起點與標記都在
# 用法（佇列，[solo]：吃 CPU）：bash scripts/lab/prep_fullscene_sea5.sh
case "${1:-}" in -h|--help) sed -n 2,9p "$0"; exit 0 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial/train/block_all
PY="conda run -n gspl --no-capture-output python -u"
V2RUN=outputs/lab/g44tn_s17g70np16tricap0p5cmflattrimasklres3geo8log0sea3dgcapfcsboxpbwseabackgseaflgeo50_best1maskdistrfed10b4
V5RUN=outputs/lab/g44tn_s17g70np16tricap0p5cmflattrimasklres3geo8log0sea3dxcapfcsboxpbwseabackxseaflgeo50_best1maskdistrfed10b4
SEA=""; for b in $(seq 0 15); do [ -f "$D/sea_object_v2/block_$b.npz" ] && SEA="$SEA $b"; done
MISS=""; for b in $SEA; do [ -f "$D/sea_object_v5/block_$b.npz" ] || MISS="$MISS $b"; done
echo "▶ $(date '+%H:%M:%S') 有海的塊（v2 有海面物件）：$SEA｜v5 要補：${MISS:-（都有）}"
if [ -n "$MISS" ]; then
  $PY tools/make_sea_object.py "$D" --blocks $MISS --masks --grow --sea_level -0.026 --far_levels 6 --far_fill 2.5 --out sea_object_v5 2>&1 \
    | grep -v -i "warn\|pkg_res\|declare_ns" | grep "外推海面：\|鋪磚邊長\|跳過\|DONE\|Error\|Trace" || exit 5
fi
IMISS=""; for b in $(seq 0 15); do [ -f "$D/sfmfill_tri44sea5_crop/dup1/block_$b.ply" ] || IMISS="$IMISS $b"; done
echo "▶ $(date '+%H:%M:%S') 產 v5 起點：${IMISS:-（都有）}"
if [ -n "$IMISS" ]; then
  $PY tools/make_tri_dense_init.py "$D" --blocks $IMISS --drop_fill --plane_dir estimated_depths_plane --plane_sep --drop_outliers \
      --sea_object sea_object_v5 --out sfmfill_tri44sea5_crop 2>&1 | grep -v -i "warn\|pkg_res\|declare_ns" | tail -20 || exit 6
fi
bad=""; for b in $(seq 0 15); do [ -f "$D/sfmfill_tri44sea5_crop/dup1/block_$b.ply" ] && [ -f "$D/sfmfill_tri44sea5_crop/dup1/block_${b}_sea.npy" ] || bad="$bad $b"; done
[ -z "$bad" ] || { echo "⛔ 缺起點或海面標記：$bad"; exit 7; }
mkdir -p "$V5RUN/blocks"
for b in $(seq 0 15); do
  [ -f "$D/sea_object_v5/block_$b.npz" ] && continue
  [ -d "$V5RUN/blocks/block_$b" ] && { echo "  b$b 沒有海、v5 家族已有這塊 => 不動"; continue; }
  [ -d "$V2RUN/blocks/block_$b" ] || { echo "⛔ b$b 沒有海、但新底 7 候選全場景也沒有這塊"; exit 8; }
  cp -al "$V2RUN/blocks/block_$b" "$V5RUN/blocks/block_$b" && echo "  b$b 沒有海 => 沿用新底 7 候選全場景的這塊（硬連結，配方相同）"
done
echo "✔ $(date '+%H:%M:%S') 16 塊起點齊全；v5 有海面物件的塊：$(ls $D/sea_object_v5/*.npz | wc -l)"
echo "DONE prep_fullscene_sea5"
