#!/bin/bash
# 存 val 渲染圖（給人眼看的），不是評測。
# 用法：task_savepics.sh <run_name> <block_id>
#
# ⚠⚠ 2026-09-13 補回：原本的佇列行是壞的，rc=2 掉了一個任務 ——
#     `python main.py test --config python main.py test --config /outputs/lab/...`
#     （`--config` 後面接了整串重複的指令，且路徑是絕對的 `/outputs/...` 少了 repo 根）
#     => 長指令一律進腳本，佇列行只呼叫它（這正是本專案的排程鐵律）。
# ⚠ 用 `test` 不是 `validate`：`validate` 會覆蓋 results.txt。
set -u
# ⚠ 2026-09-14 修：本檔在 scripts/lab/ 底下，`/..` 只回到 scripts/，要 `/../..` 才是 repo 根目錄
#   （原本照抄頂層 scripts/task_*.sh 的寫法 => 相對路徑的 config 找不到 => exit 1，lab 上 rc=1 / 90 秒）
cd "$(dirname "$0")/../.." || exit 1
RUN=${1:?用法: task_savepics.sh <run> <blk>}; BLK=${2:?}
CFG="outputs/$RUN/blocks/block_$BLK/lightning_logs/version_0/config.yaml"
[ -f "$CFG" ] || { echo "❌ 找不到 resolved config：$CFG"; exit 1; }
echo "=== 存圖：$RUN block $BLK ==="
conda run -n gspl python -u main.py test --config "$CFG" --save_val 2>&1 | tail -25
echo "圖在 outputs/$RUN/blocks/block_$BLK/val/ 之類的位置，用 ls 找"
