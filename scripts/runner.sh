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
  ledger "TASK START  $label"
  start=$(date +%s)
  # 9>&- : the lock fd must NOT be inherited by the task. flock holds as long as ANY process has
  # the fd open, so an orphaned child (e.g. a training run that outlived a killed runner) would
  # keep the lock forever and no new runner could ever start (observed 2026-07-31).
  bash -c "$cmd" >> "$LOG" 2>&1 9>&-
  rc=$?
  dur=$(( $(date +%s) - start ))
  if [ $rc -eq 0 ]; then
    log "✔ DONE   $label  (rc=0, ${dur}s)"
    ledger "TASK DONE   $label (${dur}s)"
  else
    log "✘ FAIL   $label  (rc=$rc, ${dur}s) — 佇列繼續往下跑"
    ledger "TASK FAIL   $label rc=$rc (${dur}s)"
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
