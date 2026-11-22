#!/bin/bash
# ★ 2026-11-22 Mill-19 一個場景的整條：有分區檔的塊依序訓練（現行配方去掉海面）=> 合併 => val 評測＋時間表。
#   11-22 起資料是轉成 +z 朝上的版本（prep_mill19.sh 第 0 步；11-21 的 mill19_*_cur1121 沒轉、作廢）。
#   重新分區後有相機的塊可能和之前不同 => 塊清單從分區檔讀，不寫死。各塊 [solo] 由佇列行本身保證（整條一行）。
# 用法（佇列）：[solo] bash scripts/lab/task_mill19.sh <building|rubble> <跑次名，例 mill19_building_z1122> [臂]
#   MILL19_ONLY=train|merge|eval 只跑其中一段；已有 run_summary.json 的塊跳過（中斷後可接續）。
#   ★ 11-23 MILL19_INIT_CAPAVG=<n>＝稠密起點照平均 cap n 重新產生到 sfmfill_tri44ia<n>_crop（沒有才產生），訓練改用它（起點跟著上限縮；預設起點是照 1M 產生的）。
case "${1:-}" in -h|--help|"") sed -n 2,6p "$0"; exit 0 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
SC=$1; RUN=$2
ARM=${3:-s17g70+np16+tricap0p5+cmflat+trimask+lres3+geo8+log0+dense+capfc+sbox+pbw+geo50+regvis+c15+nogate+nohf+m39}
DS=data/mill19/${SC}_ours; TS=data/mill19/${SC}_ours_test; ALLOC=configs/cap_alloc_sfm_g44_mill19_$SC.json
PART=$DS/partition/partitions-dim_4_4_visibility_0.08
[ -f "$DS/sparse/0/ALIGNED_UP.json" ] && [ -f "$TS/sparse/0/ALIGNED_UP.json" ] || { echo "⛔ $DS 或 $TS 還沒轉成 +z 朝上（先跑 prep_mill19.sh）"; exit 2; }
[ -f "$ALLOC" ] || { echo "⛔ 缺 $ALLOC"; exit 2; }
B=""; for b in $(seq 0 15); do [ -f "$PART/$(printf '%03d_%03d' $((b % 4)) $((b / 4))).txt" ] && B="$B $b"; done
echo "[mill19] $SC｜跑次 $RUN｜有分區檔的塊（$(echo $B | wc -w)）：$B｜臂 $ARM"
ONLY=${MILL19_ONLY:-all}
if [ -n "${MILL19_INIT_CAPAVG:-}" ]; then
  DD=sfmfill_tri44ia${MILL19_INIT_CAPAVG}_crop; export CITYGS_DENSE_DIR=$DD
  if [ ! -f "$DS/$DD/.complete" ]; then
    echo "[mill19] 產生稠密起點 $DD（cap_avg $MILL19_INIT_CAPAVG）"
    for b in $B; do echo $b; done | xargs -P 8 -I{} conda run -n gspl --no-capture-output python -u tools/make_tri_dense_init.py "$DS" --blocks {} --seed_per_block --alloc "$ALLOC" --cap_avg "$MILL19_INIT_CAPAVG" --out "$DD" \
      || { echo "⛔ 稠密起點產生失敗"; exit 6; }
    touch "$DS/$DD/.complete"
  fi
fi
if [ "$ONLY" = all ] || [ "$ONLY" = train ]; then
  for b in $B; do
    if [ -f "outputs/lab/$RUN/blocks/block_$b/run_summary.json" ]; then echo "[mill19] b$b 已完成，跳過"; continue; fi
    CITYGS_DATA=$DS CITYGS_DSF=4 CITYGS_CAP_ALLOC=$ALLOC CITYGS_DEMO=1 CITYGS_RUN_NAME=lab/$RUN \
      bash scripts/lab/task_tune25.sh "$b" "$ARM" || { echo "⛔ b$b 訓練失敗"; exit 3; }
  done
fi
if [ "$ONLY" = all ] || [ "$ONLY" = merge ]; then
  CITYGS_DATA=$DS FS_BLOCKS="$B" bash scripts/lab/task_fullscene.sh merge "$RUN" || { echo "⛔ 合併失敗"; exit 4; }
fi
if [ "$ONLY" = all ] || [ "$ONLY" = eval ]; then
  L=logs/${RUN}_test.log
  conda run -n gspl --no-capture-output python -u tools/eval_official_test.py --ckpt "outputs/lab/$RUN/checkpoints/merged.ckpt" \
    --test_dir "$TS" --image_subdir input --down_sample 4 --skip_marker --save_dir "outputs/lab/$RUN/test_val" > "$L" 2>&1
  rc=$?
  python3 tools/run_summary_report.py "outputs/lab/$RUN" >> "$L" 2>&1
  grep -E '^(PSNR|SSIM|LPIPS|渲染)|單卡依序' "$L"
  [ "$rc" = 0 ] || { echo "⛔ 評測失敗（$L）"; exit 5; }
fi
echo "DONE task_mill19 $SC $RUN"
