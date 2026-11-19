#!/bin/bash
# ★ 2026-11-19 論文③ 真實資料 Mill-19（Mega-NeRF；使用者同意下載）：把一個場景整理成我們流程要的樣子，再跑新底的資料準備。
#   協定照 CityGS 第一版：載入時降採樣 4 倍、train 訓練、val 評測（val 全圖，不做外觀嵌入）；COLMAP 用 CityGS 提供的 colmap_results。
#   新底 9 去掉海面元件（Mill-19 沒有水）；起點用 dense（sfmfill_tri44_crop）；4x4 分塊（沿用 44 的檔名慣例）。
# 用法（佇列）：[solo] bash scripts/lab/prep_mill19.sh <building|rubble>
#   之後訓練：CITYGS_DATA=data/mill19/<場景>_ours CITYGS_DSF=4 CITYGS_CAP_ALLOC=configs/cap_alloc_sfm_g44_mill19_<場景>.json CITYGS_DEMO=1 \
#             CITYGS_RUN_NAME=lab/mill19_<場景>_base9 bash scripts/lab/task_tune25.sh <塊> s17g70+np16+tricap0p5+cmflat+trimask+lres3+geo8+log0+dense+capfc+sbox+pbw+geo50+regvis
case "${1:-}" in -h|--help|"") sed -n 2,7p "$0"; exit 0 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
SC=$1; SRC=data/mill19/$SC-pixsfm; DS=data/mill19/${SC}_ours; TS=data/mill19/${SC}_ours_test; J=${PREP_J:-12}
PY="conda run -n gspl --no-capture-output python -u"
[ -d "$SRC/train/rgbs" ] && [ -d "$SRC/val/rgbs" ] || { echo "⛔ 缺 $SRC/{train,val}/rgbs（先下載解壓 $SC-pixsfm.tgz）"; exit 2; }
[ -d "data/colmap_results/$SC/train/sparse/0" ] && [ -d "data/colmap_results/$SC/val/sparse/0" ] || { echo "⛔ 缺 data/colmap_results/$SC/{train,val}/sparse/0"; exit 2; }
TL="$DS/prep_timing.tsv"; mkdir -p "$DS" "$TS"; : > "$TL"
stage () { local n=$1; shift; local t0; t0=$(date +%s); echo "▶ $(date '+%H:%M:%S') $n"; "$@"; local rc=$?
  printf '%s\t%d\t%d\n' "$n" "$(( $(date +%s) - t0 ))" "$rc" >> "$TL"; echo "✔ $n：$(( $(date +%s) - t0 )) s（rc=$rc）"; [ "$rc" = 0 ] || { echo "⛔ $n 失敗"; exit 3; }; }
# 0 目錄：input -> rgbs、sparse 複製（我們的工具會在 sparse 旁寫快取，不動 colmap_results）
[ -e "$DS/input" ] || ln -s "$(realpath $SRC/train/rgbs)" "$DS/input"
[ -e "$TS/input" ] || ln -s "$(realpath $SRC/val/rgbs)" "$TS/input"
[ -d "$DS/sparse/0" ] || { mkdir -p "$DS/sparse" && cp -r "data/colmap_results/$SC/train/sparse/0" "$DS/sparse/"; }
[ -d "$TS/sparse/0" ] || { mkdir -p "$TS/sparse" && cp -r "data/colmap_results/$SC/val/sparse/0" "$TS/sparse/"; }
$PY - "$DS" "$TS" <<'EOF'
import sys, os
sys.path.insert(0, ".")
from internal.utils.colmap import read_images_binary, read_cameras_binary
for d in sys.argv[1:]:
    ims = read_images_binary(f"{d}/sparse/0/images.bin"); cams = read_cameras_binary(f"{d}/sparse/0/cameras.bin")
    names = [im.name for im in ims.values()]; miss = [n for n in names if not os.path.exists(f"{d}/input/{n}")]
    c = next(iter(cams.values()))
    print(f"[檢查] {d}：COLMAP 影像 {len(names)}、在 input/ 找不到 {len(miss)}（例 {miss[:3]}）｜相機 {c.model} {c.width}x{c.height}")
    if miss: sys.exit(2)
EOF
[ $? = 0 ] || { echo "⛔ COLMAP 影像名與 rgbs 對不上"; exit 4; }
ALLOC=configs/cap_alloc_sfm_g44_mill19_$SC.json
s1 () { $PY utils/run_depth_anything_v2.py "$DS/input" --output "$DS/estimated_depths" -d 4 & local p1=$!
        $PY tools/make_resized_images.py "$DS" --factor 4 --nproc "$J" & local p2=$!; wait $p1 || return 1; wait $p2 || return 2; }
s5 () { $PY tools/make_sfm_fill_init.py "$DS" --blocks $(seq 0 15) --block_dim 4 4 --dup 1 --out-dir "$DS/sfmfill_sweep44" || return 1
        $PY tools/block_mask44.py --data "$DS" --blocks $(seq 0 15) --crop_ply sfmfill_sweep44 --skip_masks || return 2; }
s6 () { seq 0 15 | xargs -P "$J" -I{} $PY tools/block_mask44.py --data "$DS" --blocks {} --depth_dir estimated_depths_tri --depth_raw --out "$DS/block_masks44_tri"; }
s8 () { seq 0 15 | xargs -P "$J" -I{} $PY tools/make_tri_dense_init.py "$DS" --blocks {} --seed_per_block --alloc "$ALLOC"; }
stage "1 單目深度（存 1/4 解析度）‖ 預縮影像 4 倍" s1
stage "2 單目深度尺度" $PY utils/get_depth_scales.py "$DS" --depth_dir "$DS/estimated_depths" --output "$DS/estimated_depth_scales.json"
stage "3 校正深度（相機 / 8）" $PY tools/make_tri_depths.py "$DS" --nproc 16 --down 8
stage "4 分區 4x4" $PY utils/partition_from_colmap.py "$DS" --block_dim 4 4 --content_threshold 0.08
stage "5 SfM＋補點底＋裁盒" s5
stage "6 塊外遮罩（校正深度版）" s6
stage "7 顆數分配（SfM 點數）" $PY tools/sfm_cap_alloc.py "$DS" --out "$ALLOC"
stage "8 稠密起點" s8
stage "9 上限遮罩（躺平）" $PY tools/make_cap_masks.py --data "$DS" --kind flat
echo "合計 $(awk -F'\t' '{s+=$2} END{print int(s/60)}' "$TL") 分（$TL）"
echo "DONE prep_mill19 $SC"
