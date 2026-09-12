#!/bin/bash
# Queue runner: takes the top task from scripts/queue.txt, runs it, removes it, repeats.
#
# The file is re-read before every task and the finished task is removed by matching its exact
# text against a FRESH read -- so you can add, reorder, edit or delete anything in the queue while
# the runner is working, and none of your edits get clobbered. The only line you cannot change is
# the one currently executing.
#
#   start:   setsid nohup bash scripts/runner.sh > /dev/null 2>&1 < /dev/null &
#   stop:    touch scripts/queue.stop      (finishes the current task, then exits)
#   status:  tail logs/runner.log
#
# GPU serialisation lives HERE, not in the task scripts. This is a single-card project, so every
# GPU task has to wait for the card, and twelve task scripts had each grown their own copy of the
# same wait loop -- which is also how prep_3x3 came to abort on a false positive (its private
# guard read a training run's normal growth as competition). Prefix a queue line with [cpu] to
# skip the wait for work that never touches the GPU (partitioning, depth-init, analysis).
#
# Idle cost is one `head` on a small file every 60s.
set -u
cd "$(dirname "$0")/.." || exit 1
Q=${RUNNER_QUEUE:-scripts/queue.txt}   # 可用 RUNNER_QUEUE 指向別的檔（測試用）
LOG=logs/runner.log
LEDGER=logs/quad_progress.log
LOCK=${RUNNER_LOCK:-scripts/.runner.lock}
IDLE=60
GPU_FREE_MIB=${RUNNER_GPU_FREE_MIB:-1500}   # below this the card counts as free
# ── 平行槽（2026-09-13，使用者要求；lab 的 3090 24GB 可同時放 3 個 6GB 信封的跑次）──
# SLOTS=1 走原本那條**完全不變**的路徑（本機單卡）；SLOTS>1 才走下面的平行路徑。
#   ⚠ 平行化必須「**啟動前就認領並移除該行**」，否則兩個槽會搶到同一行。
#     代價：runner 被殺掉時，正在跑的那行已經不在佇列裡（單槽模式是跑完才移除，會重跑）。
#   ⚠ GPU 准入改成「**還放得下一個**」而不是「卡是空的」：
#     free = total - used >= RUNNER_GPU_RESERVE_MIB
SLOTS=${RUNNER_SLOTS:-1}
GPU_RESERVE_MIB=${RUNNER_GPU_RESERVE_MIB:-6800}   # 一個 6GB 信封的跑次要留多少
STAGGER=${RUNNER_STAGGER:-90}                     # 兩次啟動之間隔多久（避開同時配置尖峰）

mkdir -p logs
exec 9>"$LOCK"   # 見下方 9>&-：子進程不得繼承這個 fd
flock -n 9 || { echo "另一個 runner 已在執行（$LOCK）"; exit 1; }

log () { printf '%s | %s\n' "$(date '+%m-%d %H:%M:%S')" "$1" | tee -a "$LOG"; }
ledger () { printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "runner" "$1" >> "$LEDGER"; }

wait_for_gpu () {
  local used waited=0
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && used=0
    [ "$used" -lt "$GPU_FREE_MIB" ] && break
    [ $((waited % 1800)) -eq 0 ] && log "   ⏳ 等 GPU（目前 ${used}MiB）"
    sleep 60; waited=$((waited + 60))
  done
  sleep 20      # let the previous process actually release
}

gpu_room () {   # 還放得下一個 6GB 信封的跑次嗎
  local used total
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "$used" ] && used=0
  [ -z "$total" ] && total=0
  [ $((total - used)) -ge "$GPU_RESERVE_MIB" ]
}

# 取「第一個非空非註解的行」與它上方註解區塊的第一行（標籤）
head_task () { grep -vE '^\s*(#|$)' "$Q" 2>/dev/null | head -1; }
label_of () {
  TASK_TXT="$1" awk 'BEGIN{t=ENVIRON["TASK_TXT"]}
    $0==t {print first; exit}
    /^[[:space:]]*#/ {if(!inblk){first=$0; inblk=1} next}
    /^[[:space:]]*$/ {inblk=0; first=""; next}
    {inblk=0; first=""}' "$Q" | sed 's/^[[:space:]]*#*[[:space:]]*//'
}
drop_task () {   # 從**重新讀取**的檔案裡移除恰好這一行，**連同它上方的註解區塊**
  # ⚠ 只刪任務行會讓它的註解留下來、與下一個任務的註解黏在一起 =>
  #   台帳把前一個任務的標題掛在下一個跑次上（queue_status.py 的說明記過這種錯位；
  #   2026-09-13 平行乾跑時四個任務全被標成「任務 A」就是這個）。
  local t="$1" tmp
  [ -f "$Q" ] || return 0
  tmp=$(mktemp)
  TASK_TXT="$t" awk '
    BEGIN{ t=ENVIRON["TASK_TXT"]; hit=0 }
    { n++; L[n]=$0; if(!hit && $0==t) hit=n }
    END{
      s=hit
      if(hit>0){ j=hit-1; while(j>=1 && L[j] ~ /^[[:space:]]*#/){ s=j; j-- } }
      for(i=1;i<=n;i++){ if(hit>0 && i>=s && i<=hit) continue; print L[i] }
    }' "$Q" > "$tmp" && mv "$tmp" "$Q"
}
short_of () { local x; x=$(printf '%s' "$1" | grep -oE '[a-zA-Z0-9_./-]+\.sh( [a-zA-Z0-9_]+)?' | head -1); echo "${x##*/}"; }

# ══════════════ 平行模式（SLOTS > 1）══════════════
# ⚠⚠ 這裡**不可以**用 `${#ARR[@]}` 判斷關聯陣列的大小：`set -u` 下對**空的**關聯陣列
#    取長度會丟「unbound variable」，條件式因此行為錯亂（2026-09-13 乾跑實測：
#    變成一次只啟動一個，看起來像平行沒生效）。改用明確的整數計數器 nrun。
if [ "$SLOTS" -gt 1 ]; then
  log "runner 啟動（**平行 $SLOTS 槽**；佇列 $Q；每槽保留 ${GPU_RESERVE_MIB}MiB；錯開 ${STAGGER}s）"
  declare -A P_LABEL P_CMD P_START P_SHORT P_SLOT
  declare -a FREE_SLOTS=()
  for i in $(seq 1 "$SLOTS"); do FREE_SLOTS+=("$i"); done
  nrun=0
  stopping=0

  reap () {   # 收割已結束的子行程；回傳收了幾個
    local got=0 pid rc dur
    [ "$nrun" -eq 0 ] && return 0
    for pid in "${!P_CMD[@]}"; do
      kill -0 "$pid" 2>/dev/null && continue
      wait "$pid"; rc=$?
      dur=$(( $(date +%s) - ${P_START[$pid]} ))
      if [ "$rc" -eq 0 ]; then
        log "✔ DONE   [槽 ${P_SLOT[$pid]}] ${P_LABEL[$pid]}  (rc=0, ${dur}s)"
        ledger "TASK DONE   [${P_SHORT[$pid]:-?}] ${P_LABEL[$pid]} (${dur}s)"
      else
        log "✘ FAIL   [槽 ${P_SLOT[$pid]}] ${P_LABEL[$pid]}  (rc=$rc, ${dur}s)"
        ledger "TASK FAIL   [${P_SHORT[$pid]:-?}] ${P_LABEL[$pid]} rc=$rc (${dur}s)"
      fi
      FREE_SLOTS+=("${P_SLOT[$pid]}")
      unset "P_LABEL[$pid]" "P_CMD[$pid]" "P_START[$pid]" "P_SHORT[$pid]" "P_SLOT[$pid]"
      nrun=$((nrun - 1)); got=$((got + 1))
    done
    return "$got"
  }

  while true; do
    if [ "$stopping" -eq 0 ] && [ -f scripts/queue.stop ]; then
      log "偵測到 queue.stop：不再啟動新任務，等現有 $nrun 個跑完"
      rm -f scripts/queue.stop; stopping=1
    fi
    # ── 有空槽就啟動 ──
    while [ "$stopping" -eq 0 ] && [ "$nrun" -lt "$SLOTS" ] && [ "${#FREE_SLOTS[@]}" -gt 0 ]; do
      task=$(head_task); [ -z "$task" ] && break
      cmd="$task"; needs_gpu=1
      case "$task" in \[cpu\]*) cmd="${task#\[cpu\]}"; needs_gpu=0 ;; esac
      if [ "$needs_gpu" -eq 1 ] && ! gpu_room; then
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        log "   ⏳ VRAM 不足以再開一個（目前 ${used}MiB，需留 ${GPU_RESERVE_MIB}MiB）；執行中 $nrun"
        break
      fi
      lbl=$(label_of "$task"); [ -z "$lbl" ] && lbl="(未命名)"
      slot="${FREE_SLOTS[0]}"; FREE_SLOTS=("${FREE_SLOTS[@]:1}")
      drop_task "$task"                      # ★ 啟動前就認領，避免兩槽搶同一行
      slog="logs/runner.slot${slot}.log"
      log "▶ START  [槽 $slot] $lbl"
      log "         $cmd   （輸出 -> $slog）"
      ledger "TASK START  [$(short_of "$cmd")] $lbl"
      { printf '\n===== %s | 槽 %s | %s =====\n' "$(date '+%m-%d %H:%M:%S')" "$slot" "$lbl" >> "$slog"
        bash -c "$cmd" >> "$slog" 2>&1 9>&- ; } &
      pid=$!
      P_LABEL[$pid]="$lbl"; P_CMD[$pid]="$cmd"; P_START[$pid]=$(date +%s)
      P_SHORT[$pid]=$(short_of "$cmd"); P_SLOT[$pid]="$slot"
      nrun=$((nrun + 1))
      sleep "$STAGGER"
    done
    reap
    if [ "$stopping" -eq 1 ] && [ "$nrun" -eq 0 ]; then break; fi
    if [ "$nrun" -eq 0 ]; then sleep "$IDLE"; else sleep 10; fi
  done
  log "runner 結束（平行模式）"
  exit 0
fi

fails=0
log "runner 啟動（佇列 $Q；GPU 門檻 ${GPU_FREE_MIB}MiB）"
while true; do
  [ -f scripts/queue.stop ] && { log "偵測到 queue.stop，結束"; rm -f scripts/queue.stop; break; }
  [ -f "$Q" ] || { sleep "$IDLE"; continue; }

  # first line that is neither blank nor a comment
  task=$(grep -vE '^\s*(#|$)' "$Q" | head -1)
  if [ -z "$task" ]; then
    sleep "$IDLE"; continue
  fi
  # label = the comment block immediately above it, for readability in the ledger
  # first line of the contiguous comment block directly above the task (blank line resets it)
  label=$(TASK_TXT="$task" awk 'BEGIN{t=ENVIRON["TASK_TXT"]}
    $0==t {print first; exit}
    /^[[:space:]]*#/ {if(!inblk){first=$0; inblk=1} next}
    /^[[:space:]]*$/ {inblk=0; first=""; next}
    {inblk=0; first=""}' "$Q" | sed 's/^[[:space:]]*#*[[:space:]]*//')
  [ -z "$label" ] && label="(未命名)"

  # [cpu] prefix = does not touch the GPU, so do not queue behind it
  cmd="$task"; needs_gpu=1
  case "$task" in
    \[cpu\]*) cmd="${task#\[cpu\]}"; needs_gpu=0 ;;
  esac

  log "▶ START  $label"
  log "         $cmd"
  [ "$needs_gpu" -eq 1 ] && wait_for_gpu
  # 台帳帶上腳本名：label 是用 awk 掃「任務上方的註解區塊」推的，多個區塊相鄰時會抓到別人的，
  # 而監控只看台帳那一行 => 已經多次讓人以為跑錯了實驗（實際上指令一直是對的）。
  # 腳本名直接從指令取，不會錯。
  short=$(printf '%s' "$cmd" | grep -oE '[a-zA-Z0-9_./-]+\.sh( [a-zA-Z0-9_]+)?' | head -1)
  short=${short##*/}
  ledger "TASK START  [${short:-?}] $label"
  start=$(date +%s)
  # 9>&- : the lock fd must NOT be inherited by the task. flock holds as long as ANY process has
  # the fd open, so an orphaned child (e.g. a training run that outlived a killed runner) would
  # keep the lock forever and no new runner could ever start (observed 2026-07-31).
  bash -c "$cmd" >> "$LOG" 2>&1 9>&-
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    log "✔ DONE   $label  (rc=0, ${dur}s)"
    ledger "TASK DONE   [${short:-?}] $label (${dur}s)"
  else
    log "✘ FAIL   $label  (rc=$rc, ${dur}s) — 佇列繼續往下跑"
    ledger "TASK FAIL   [${short:-?}] $label rc=$rc (${dur}s)"
  fi

  # Remove exactly this task from a FRESH read, so concurrent edits survive.
  if [ -f "$Q" ]; then
    tmp=$(mktemp)
    # via ENVIRON, never `-v`: awk's -v expands backslash escapes, so a task containing e.g.
    # `step=(\d+)` would arrive as `step=(d+)`, never match, never be removed, and the runner
    # would re-run it forever (observed 2026-07-29, ~60 repeats in an hour).
    TASK_TXT="$task" awk 'BEGIN{t=ENVIRON["TASK_TXT"]; done=0} {if(!done && $0==t){done=1; next} print}' "$Q" > "$tmp" && mv "$tmp" "$Q"
    # Belt and braces: if the task is somehow still at the head, it did not get removed.
    if [ "$(grep -vE '^\s*(#|$)' "$Q" | head -1)" = "$task" ]; then
      fails=$((fails + 1))
      log "⚠ 任務未被移除（第 ${fails} 次）——可能是比對失敗"
      if [ "$fails" -ge 3 ]; then
        log "✘ 連續 3 次無法移除，改用行號強制刪除以免無限迴圈"
        ln=$(grep -nvE '^\s*(#|$)' "$Q" | head -1 | cut -d: -f1)
        [ -n "$ln" ] && sed -i "${ln}d" "$Q"
        fails=0
      fi
    else
      fails=0
    fi
  fi
done
log "runner 結束"
