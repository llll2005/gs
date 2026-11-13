#!/bin/bash
# ★ 2026-11-12 佇列裡的「等卡空」關卡：卡上用量連續 HOLD 秒低於 FREE_MIB 才結束（rc 0）。
#   用途：單槽 runner 用高門檻共用卡跑訓練（外部行程佔 20 GB）時，把要整張卡的工作（換塊評測、離線拆解的 held-out）排在這行後面。
#   用法（佇列行）：bash scripts/lab/wait_gpu_free.sh [FREE_MIB=1500] [HOLD=300]
#   ★ 11-14 佇列裡的 HOLD 由 60 改 15（使用者「等卡關卡縮短，只是對我們方便」；每塊省約 1 分；只影響排程、不影響訓練）。
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
  sleep $(( HOLD < 60 ? 5 : 20 ))   # ★ 11-14 短 HOLD（佇列改用 15 s）時每 5 s 看一次，否則關卡本身就要 20 s 以上
done
