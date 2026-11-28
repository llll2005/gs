#!/usr/bin/env bash
# ★ 11-28 使用者「都先排兩塊驗證」：同卡同時訓練多塊（`紀錄/new_archived/已關閉的線.md` K，重開條件＝單塊 GPU 使用率明顯降下來；
#   現在前 25% 步數是 0.35 倍解析度，那段可能吃不滿卡）。
# 用法：bash scripts/lab/task_concurrent.sh <跑次名> <臂> <塊...>
#   一塊＝單獨跑＋每 0.5 s 抽樣 GPU 使用率，按解析度階段分開報；多塊＝同時啟動（不錯開）、等全部跑完，報整個任務的牆鐘。
#   佇列行要加 [solo]（量到的才乾淨）。訓練輸出逐行加 epoch 時間戳 => logs/conc_<跑次>_b<塊…>_blk<塊>.log；抽樣 => logs/gpuutil_<…>.csv
set -u
cd "$(dirname "$0")/../.." || exit 1
NAME=${1:?缺跑次名}; ARM=${2:?缺臂}; shift 2
[ $# -ge 1 ] || { echo "⛔ 至少一塊"; exit 2; }
TAG="${NAME}_b$(IFS=_; echo "$*")"
U=logs/gpuutil_${TAG}.csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$U" 2>/dev/null &
SPID=$!
t0=$(date +%s.%N)
pids=()
for b in "$@"; do
  L=logs/conc_${TAG}_blk${b}.log
  ( set -o pipefail
    CITYGS_DEMO=1 CITYGS_RUN_NAME=lab/$NAME bash scripts/lab/task_tune25.sh "$b" "$ARM" 2>&1 \
      | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > "$L" ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
t1=$(date +%s.%N)
kill "$SPID" 2>/dev/null
echo "[concurrent] $TAG：$# 塊、牆鐘 $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b - a}') s、rc=$rc"
python3 tools/probe/gpu_util_phase_1128.py --util "$U" --logs "logs/conc_${TAG}_blk*.log" --t0 "$t0" --t1 "$t1"
exit $rc
