#!/bin/bash
# ★ 11-18 全場景前置（使用者 11-17「這一批定案採用後安排 4x4 訓練並排 test 生成圖片」；S1＋A＋B＋加速組合）：
#   1 等佇列裡的 16 塊平面外推＋海面物件（[cpu]，`plane_depth_all_1117`／`sea_object_v2_all_1117` 兩個 log）跑完
#   2 sea_object_v2 缺的塊（b1／b3 以外）從 sea_object_v2_all 補過來（npz＋cap_b<b>；沒有海的塊 make_sea_object 會跳過、不寫 npz）
#   3 sfmfill_tri44sea2_crop 缺的塊用同一組參數產（陸地＝v2、海面＝長過的鋪磚；b1／b3 已有的不動）
#   4 檢查 16 塊的起點與標記都在
# ⚠ 11-18：runner 用 bash -c 執行腳本內容 => pgrep 的字樣要寫成 [m]ake_… 才不會比對到自己（第一版因此空等）
# 用法（佇列，[solo]：吃 CPU、不能和計時中的訓練同時跑）：bash scripts/lab/prep_fullscene_sea.sh
case "${1:-}" in -h|--help) sed -n 2,8p "$0"; exit 0 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial/train/block_all
PY="conda run -n gspl --no-capture-output python -u"
echo "▶ $(date '+%H:%M:%S') 等 16 塊平面外推＋海面物件跑完"
for i in $(seq 1 1440); do
  pgrep -f "[m]ake_plane_depth[.]py|[m]ake_sea_object[.]py" > /dev/null || break
  [ $((i % 20)) = 0 ] && echo "  …還在跑（$(date '+%H:%M:%S')）"
  sleep 30
done
pgrep -f "[m]ake_plane_depth[.]py|[m]ake_sea_object[.]py" > /dev/null && { echo "⛔ 等了 12 小時還在跑"; exit 3; }
grep -q "DONE\|跳過\|鋪磚邊長" logs/sea_object_v2_all_1117.log 2>/dev/null || { echo "⛔ logs/sea_object_v2_all_1117.log 看不到 16 塊海面物件 v2 的輸出"; tail -5 logs/sea_object_v2_all_1117.log; exit 4; }
echo "▶ $(date '+%H:%M:%S') 補 sea_object_v2"
mkdir -p "$D/sea_object_v2"
for b in $(seq 0 15); do
  [ -f "$D/sea_object_v2/block_$b.npz" ] && continue
  if [ -f "$D/sea_object_v2_all/block_$b.npz" ]; then
    cp "$D/sea_object_v2_all/block_$b.npz" "$D/sea_object_v2/" && cp -r "$D/sea_object_v2_all/cap_b$b" "$D/sea_object_v2/" && echo "  b$b 補上"
  else echo "  b$b 沒有海面物件（沒有海）"; fi
done
MISS=""; for b in $(seq 0 15); do [ -f "$D/sfmfill_tri44sea2_crop/dup1/block_$b.ply" ] || MISS="$MISS $b"; done
echo "▶ $(date '+%H:%M:%S') 產海面起點：${MISS:-（都有）}"
if [ -n "$MISS" ]; then
  $PY tools/make_tri_dense_init.py "$D" --blocks $MISS --drop_fill --plane_dir estimated_depths_plane --plane_sep --drop_outliers \
      --sea_object sea_object_v2 --out sfmfill_tri44sea2_crop 2>&1 | grep -v -i "warn\|pkg_res\|declare_ns" || exit 5
fi
bad=""; for b in $(seq 0 15); do [ -f "$D/sfmfill_tri44sea2_crop/dup1/block_$b.ply" ] && [ -f "$D/sfmfill_tri44sea2_crop/dup1/block_${b}_sea.npy" ] || bad="$bad $b"; done
[ -z "$bad" ] || { echo "⛔ 缺起點或海面標記：$bad"; exit 6; }
echo "✔ $(date '+%H:%M:%S') 16 塊起點齊全；有海面物件的塊：$(ls $D/sea_object_v2/*.npz | wc -l)"
echo "DONE prep_fullscene_sea"
