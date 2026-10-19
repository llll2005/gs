#!/bin/bash
# ★★★★★★ 使用者 10-18：「把這輪具代表性的實驗去跑 test、診斷、塊外塊內顆粒數比例分析、各項數據分析」
#   每個跑次（該塊）：缺 test 圖就補存（task_savepics.sh），再跑 tools/inbox_diag.py --hidden_all
#   （[0] 塊內外顆數／opacity 質量／實際混合貢獻、[1] 黑色透明、[2] 地底、[3] 邊緣、[4] 深度誤差、[5] 馬路、[6] 中空）。
#   official＝官方 release 同塊（不補 test 圖）。
# 用法（佇列）：[solo] bash scripts/lab/task_round_report.sh <塊> <跑次...>
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
B=$1; shift
bad=0
for r in "$@"; do
  [ "$r" = official ] && continue
  if ! ls -d outputs/lab/$r/blocks/block_$B/test/epoch=* >/dev/null 2>&1; then
    echo "── 補 test 圖：$r b$B"; bash scripts/lab/task_savepics.sh "lab/$r" "$B" | tail -2 || bad=1
  fi
done
bash scripts/lab/task_inbox_diag.sh "$B" "$@" || bad=1
exit "$bad"
