#!/bin/bash
# 用法：bash scripts/_noedit.sh <腳本> [參數...]
#
# 為什麼需要：**bash 是逐段讀取腳本檔的**，不是一次讀完。任務跑到一半若腳本被
# 原地覆寫（`git pull`、編輯器、甚至把同樣內容重寫一遍都算），bash 會從錯的
# 位元組偏移繼續讀，執行到半行殘段。
#   實例（2026-09-13）：16 行的 scripts/lab/task_speed3.sh 報
#   「列 17: .init: command not found」rc=127 —— 而那支訓練已經跑了 1h42m，
#   檔案內容與 HEAD 逐位元組相同（824 B），只是 mtime 落在跑次期間。
#   lab 上 3 個槽跑長跑次時 `git pull` 是常態 ⇒ 這是會重演的一類。
#
# 做法：把腳本內容讀進**記憶體**再交給 `bash -c`，之後檔案怎麼變都不影響。
# ★ 關鍵：`bash -c <code> <name> <args...>` 會把 <name> 設成 `$0`。
#   本專案每一支 task 腳本都用 `cd "$(dirname "$0")/.."` 自我定位，
#   所以 `$0` 必須設回**原本的腳本路徑**，否則全部會 cd 到錯的地方。
set -u
s="${1:?用法: _noedit.sh <腳本> [參數...]}"; shift
[ -r "$s" ] || { echo "⛔ 讀不到 $s"; exit 127; }
code=$(cat "$s") || { echo "⛔ 讀取失敗 $s"; exit 127; }
exec bash -c "$code" "$s" "$@"
