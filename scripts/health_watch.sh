#!/usr/bin/env bash
# 雙機健康監看 —— 只在「要動手的事」發生時輸出一行（正常時 0 輸出 = 0 token）。
#
# 為什麼放在 repo 而不是 scratchpad：scratchpad 在 /tmp，而本機的 /tmp 是 **tmpfs**，
#   2026-09-13 重開機（修 nvidia 驅動版本不合）後 watch4.sh 與 labcheck.py 一起消失，
#   監控以 exit 127 靜默死掉。
# 這不是排程器（排程只有 scripts/runner.sh）；它只讀狀態、只印字、不啟動任何東西。
#
# 已修過的五個誤判（2026-09-13，全部實際發生過）：
#  ① 判「runner 死了」只看 runner.log 有沒有成長 —— 長跑次期間它本來就不寫 => 要同時看步數
#  ② 用 pgrep 判存活 —— 會匹配到檢查自己的 shell（本專案鐵律禁止）
#  ③ nvidia-smi 失敗時把錯誤訊息（寫在 **stdout**）數成「2 個行程」=> 看 rc，不看行數
#  ④ 已完賽跑次的 status 停在 `N/N (100%)` 被當成停滯 => 排除 100% 的檔
#  ⑤ 讀不到 GPU 時誤判成「卡空閒」=> 讀不到 = 不知道，兩個分支都不進
#  ⑥ lab 端把 `.aborted_*` 裡的死檔算進「最大步數」=> 誤報停滯；且全體最大步數抓不到單一跑次卡住
#     => 改用 lab 端探針 scripts/lab/hw_probe.sh，逐跑次看 train_status 多久沒更新
#
# 用法（token 只走環境變數，**不可寫進任何檔案**）：
#   JTOK=<token> bash scripts/health_watch.sh
set -u
cd "$(dirname "$0")/.." || exit 1
: "${JTOK:?需要環境變數 JTOK}"
INTERVAL=${WATCH_INTERVAL:-300}

L_max=-1; L_stall=0; L_idle=0; L_nosmi=0
B_fail=""; B_unreach=0; B_done=0; B_idle=0; B_staleflag=""

local_max_step() {   # 近 30 分鐘有更新、且**尚未完賽**的跑次中最大的步數
  local m=0 s f
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    grep -qE '^step .*\(100%\)' "$f" 2>/dev/null && continue
    s=$(grep -oE '^step [0-9,]+' "$f" 2>/dev/null | head -1 | tr -cd '0-9')
    [ -n "$s" ] && [ "$s" -gt "$m" ] && m="$s"
  done < <(find outputs -name train_status.txt -newermt '-30 minutes' -not -path '*/outputs/lab/*' 2>/dev/null)
  echo "$m"
}

while true; do
  # ── 本機 ──
  smi=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); rc=$?
  nq=$(awk 'NF && $1 !~ /^#/' scripts/queue.txt 2>/dev/null | grep -c .)
  if [ "$rc" -ne 0 ]; then
    L_stall=0; L_idle=0; L_max=-1
    [ "$L_nosmi" -eq 0 ] && echo "⛔ 本機：讀不到 nvidia-smi（rc=$rc）=> GPU 工作全停，可能要重開機"
    L_nosmi=1
  else
    L_nosmi=0
    napp=$(printf '%s' "$smi" | grep -c .)
    if [ "$napp" -gt 0 ]; then
      L_idle=0; lmax=$(local_max_step)
      if [ "$lmax" -gt 0 ] && [ "$lmax" -le "$L_max" ]; then
        L_stall=$((L_stall+1))
        [ "$L_stall" -eq 4 ] && echo "⚠ 本機：GPU 上有 $napp 個行程但步數 $((4*INTERVAL/60)) 分鐘沒前進（最大 $lmax）"
      else
        L_stall=0
      fi
      L_max="$lmax"
    else
      L_max=-1; L_stall=0
      if [ "${nq:-0}" -eq 0 ]; then
        L_idle=$((L_idle+1))
        [ "$L_idle" -eq 2 ] && echo "⚠ 本機：GPU 空著而佇列也空了 => 要補任務"
      else
        L_idle=0
      fi
    fi
  fi

  # ── lab ── 探針是 lab 端的 scripts/lab/hw_probe.sh（只讀；理由與兩個已修誤判寫在它的檔頭）
  out=$(timeout 90 python scripts/setup/lab.py run "cd /workspace/data/hdd/11213/gs && bash scripts/lab/hw_probe.sh" 2>/dev/null \
        | grep -oE '@HW fails=[0-9]+ pending=[0-9]+ procs=[0-9]+ active=[0-9]+ stale=[0-9]+ name=[^[:space:]]+' | tail -1)
  if [ -z "$out" ]; then
    B_unreach=$((B_unreach+1))
    [ "$B_unreach" -eq 2 ] && echo "⚠ lab：連續兩次連不上 API（或探針沒回應）"
  else
    B_unreach=0
    f=$(sed -E 's/.*fails=([0-9]+).*/\1/' <<<"$out")
    p=$(sed -E 's/.*pending=([0-9]+).*/\1/' <<<"$out")
    n=$(sed -E 's/.*procs=([0-9]+).*/\1/' <<<"$out")
    a=$(sed -E 's/.*active=([0-9]+).*/\1/' <<<"$out")
    st=$(sed -E 's/.*stale=([0-9]+).*/\1/' <<<"$out")
    nm=$(sed -E 's/.*name=([^[:space:]]+).*/\1/' <<<"$out")
    [ -z "$B_fail" ] && B_fail="$f"
    if [ "$f" -gt "$B_fail" ]; then
      echo "⛔ lab：新的 FAIL（$B_fail -> $f，待執行 $p）—— 看 logs/runner.log 與 runner.slot*.log"
      B_fail="$f"
    fi
    # 單一訓練卡住：它自己的 train_status 超過 25 分鐘沒更新（同一個跑次只報一次）
    if [ "$a" -gt 0 ] && [ "$st" -ge 25 ]; then
      [ "$B_staleflag" != "$nm" ] && echo "⚠ lab：$nm 的 train_status 已 $st 分鐘沒更新（活躍訓練 $a 個）=> 可能卡住或靜默死掉"
      B_staleflag="$nm"
    else
      B_staleflag=""
    fi
    # 佇列有東西但**沒有任何任務行程**：runner 可能停了（看行程，不看 status —— 評測類任務不寫 status）
    if [ "$n" -eq 0 ] && [ "$p" -gt 0 ]; then
      B_idle=$((B_idle+1))
      [ "$B_idle" -eq 2 ] && echo "⚠ lab：佇列還有 $p 個但連續兩輪沒有任何任務在跑 => runner 可能停了"
    else
      B_idle=0
    fi
    if [ "$p" -eq 0 ] && [ "$n" -eq 0 ] && [ "$B_done" -eq 0 ]; then
      echo "✅ lab：佇列跑完了（FAIL 累計 $f）"; B_done=1
    fi
  fi
  [ "${WATCH_ONCE:-0}" = "1" ] && break
  sleep "$INTERVAL"
done
