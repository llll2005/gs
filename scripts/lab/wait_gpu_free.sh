#!/bin/bash
# ★ 2026-11-12 佇列裡的「等卡空」關卡：卡上用量連續 HOLD 秒低於 FREE_MIB 才結束（rc 0）。
#   用途：單槽 runner 用高門檻共用卡跑訓練（外部行程佔 20 GB）時，把要整張卡的工作（換塊評測、離線拆解的 held-out）排在這行後面。
#   用法（佇列行）：bash scripts/lab/wait_gpu_free.sh [FREE_MIB=1500] [HOLD=300]
FREE=${1:-1500}; HOLD=${2:-300}; since=0; t0=$(date +%s); last=0
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  now=$(date +%s)
  if [ -n "$used" ] && [ "$used" -lt "$FREE" ]; then
    [ "$since" -eq 0 ] && since=$now
    [ $((now - since)) -ge "$HOLD" ] && { echo "[wait-gpu] ✅ 卡空 ${HOLD}s（目前 ${used} MiB；等了 $(( (now - t0) / 60 )) 分）"; exit 0; }
  else
    since=0
  fi
  [ $((now - last)) -ge 1800 ] && { echo "[wait-gpu] $(date '+%m-%d %H:%M') 卡上 ${used:-?} MiB（要 < ${FREE} 連續 ${HOLD}s）"; last=$now; }
  sleep 20
done
