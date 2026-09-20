#!/bin/bash
# lab 端健康探針：印一行 `@HW ...` 給本機的 scripts/health_watch.sh 解析。**只讀，不改任何東西。**
# ⚠ 這支只在 lab 上有意義（讀 outputs/lab/ 與 lab 的 runner.log）。
#
# 為什麼獨立成檔（2026-09-13）：原本塞在 health_watch.sh 的 `lab.py run "..."` 字串裡，
#   多層跳脫無法單獨驗證，而那串裡有兩個誤判：
#   ① 把 `.aborted_*` 目錄裡早就死掉的 train_status 算進去（speed3 b13 停在 14,457、
#      1,007 分鐘前沒更新）=> 全體最大步數被永遠釘住 => 誤報「25 分鐘沒動」
#   ② 用「全體最大步數有沒有前進」判停滯：只要**別的**跑次在走，單一跑次卡住永遠抓不到；
#      而若改成「只看最近 30 分鐘更新過的檔」，卡住的跑次正是停止更新的那個 => 會被濾掉
#   正解：**逐跑次看 train_status 多久沒更新**（正常時幾秒到幾分鐘就寫一次），
#   排除 `.aborted_*` 與已完賽（100%）；超過 180 分鐘沒更新的視為遠古殘留也排除
#   （卡住的跑次在 25 分鐘就會被報，遠早於 180）。
#   「有沒有任務在跑」看行程而不是 train_status：task_test / savepics / allocfrag 不寫 status。
#
# 輸出：@HW fails=<累計FAIL> pending=<待執行> procs=<任務行程數> active=<活躍訓練數> stale=<最久沒更新的分鐘> name=<跑次/塊>
set -u
cd "$(dirname "$0")/../.." || exit 1
f=$(grep -c '✘ FAIL' logs/runner.log 2>/dev/null); f=${f:-0}
p=$(grep -vcE '^\s*(#|$)' scripts/queue.txt 2>/dev/null); p=${p:-0}
n=$(ps -eo cmd | awk '$1=="bash" && $2 ~ /^scripts\/lab\/task_/ {c++} END{print c+0}')
now=$(date +%s); a=0; worst=0; wn=-
for x in outputs/lab/*/blocks/*/train_status.txt; do
  [ -f "$x" ] || continue
  case "$x" in *.aborted_*) continue;; esac
  grep -qE '^step .*\(100%\)' "$x" && continue
  age=$(( (now - $(stat -c %Y "$x")) / 60 ))
  [ "$age" -gt 180 ] && continue
  a=$((a+1))
  if [ "$age" -ge "$worst" ]; then
    worst=$age
    wn=$(echo "$x" | sed -E 's#outputs/lab/([^/]+)/blocks/(block_[0-9]+)/.*#\1/\2#')
  fi
done
echo "@HW fails=$f pending=$p procs=$n active=$a stale=$worst name=$wn"
