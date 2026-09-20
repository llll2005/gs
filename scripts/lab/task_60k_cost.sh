#!/bin/bash
# 60k 完整配方五臂的**成本欄**（2026-09-17 使用者問「trimvpc 60k 成本為啥沒數據」）。
#
# 為什麼之前沒有：60k 跑次走 task_speed3.sh，訓練中不印 Load；短配方的成本是把 ckpt 拉回本機
# 用 task_load_compare.sh 離線量的，而 60k 的沒人量，cost_budget_calibrate.py 也直到 09-17 才上 lab。
#
# 量法與短配方 cs_* 一致（同工具、該塊**全部**相機），全部在 lab 上直接讀 ckpt、不拉檔：
#   load  離線 Load（tools/cost_budget_calibrate.py --max-cam 100000）；幾何量，三槽平行也不受干擾
#   time  穩態每步時間＋VRAM 分項（tools/step_breakdown.py --block）；⚠ 計時必須排 [solo]
# 用法：task_60k_cost.sh <塊> load|time
set -u
cd "$(dirname "$0")/../.." || exit 1
BLK=${1:?用法: task_60k_cost.sh <塊> load|time}; MODE=${2:?同上}
case "$MODE" in load|time) ;; *) echo "⛔ 模式只能是 load 或 time"; exit 2 ;; esac
RUNS="speed3 speed3_trimvpc elong_prune elong_relocate speed3_trimvpc_elong"
for t in tools/cost_budget_calibrate.py tools/step_breakdown.py; do [ -f "$t" ] || { echo "⛔ 缺 $t"; exit 3; }; done
R=logs/cost60k_b${BLK}_${MODE}_$(date +%m%d_%H%M).log
{ bad=0
  for r in $RUNS; do
    ck=$(find "outputs/lab/$r/blocks/block_$BLK/checkpoints" -maxdepth 1 -name '*step=60000.ckpt' 2>/dev/null | head -1)
    [ -n "$ck" ] || { echo "⛔ $r／b$BLK 沒有 60k ckpt"; bad=1; continue; }
    echo "════════ $r／b$BLK（$(basename "$ck")，$(du -h "$ck" | cut -f1)）════════"
    if [ "$MODE" = load ]; then
      conda run -n gspl --no-capture-output python tools/cost_budget_calibrate.py --ckpt "$ck" --max-cam 100000 \
        || { echo "⛔ 離線 Load 失敗"; bad=1; }
    else
      conda run -n gspl --no-capture-output python tools/step_breakdown.py --run "lab/$r" --block "$BLK" --step 60000 --repeat 20 \
        || { echo "⛔ 計時失敗"; bad=1; }
    fi
  done
  exit "$bad"; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|找不到深度圖|caching images|depth scale' | tee "$R"
st=${PIPESTATUS[0]}
echo "報告：$R"
[ "$st" -eq 0 ] || echo "⛔ 至少一臂失敗（rc=$st）—— 台帳的 DONE 不代表成功，看報告"
exit "$st"
