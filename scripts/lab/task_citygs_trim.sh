#!/bin/bash
# 官方 CityGaussianV2 的 **trim 階段**（吃 coarse、全域不分塊），以及它的資源統計。
#
# ⚠⚠ 2026-09-17 查到的關鍵事實：官方的 trim 階段**其實沒有 trim**。
#   `--model.initialize_from <.ckpt>` 會把 renderer **整個從 ckpt 反序列化回來**
#   （`internal/gaussian_splatting.py`，官方原始碼同一行也是這樣），而 coarse 設了
#   `diable_trimming: true` => 微調階段一次都沒 trim；**resolved config 仍記 false**，查 config 看不出來。
#   本機舊跑次 official_ft_blk5 跑完 60k 步，log 裡 `Trimming` 出現 **0 次**，是同一個坑。
#   => `CITYGS_KEEP_CFG_RENDERER=1`（本次新增，預設關）保留 config 的 renderer，trim 才會真的跑。
#
# 三個臂（全部吃同一個 coarse、同一份 config、全域不分塊，與使用者 09-15/09-16 手動跑的一致）：
#   asis     使用者已跑完的 outputs/citygsv2_mc_aerial_sh2_trim（沒 trim；test PSNR 25.62）
#   trimon   同一份 config + KEEP_CFG_RENDERER => trim 真的跑（官方 config 的原意）
#   trimvpc  trimon + 我方 v/c 判準（**唯一變數**）
#
# 用法：
#   task_citygs_trim.sh res <跑次名>      只做資源統計（離線 Load／逐步時間／N／ckpt 大小）
#   task_citygs_trim.sh gate              1,100 步閘門：驗 trim 真的觸發、v/c 首次觸發
#   task_citygs_trim.sh trimon|trimvpc    全長 60k（約 23 小時，建議 [solo]），跑完自動接資源統計
set -u
# ★ 不鎖 6GB：這條線要答的是「官方數字長什麼樣」，與 coarse 那支同一個立場（使用者 2026-09-13 確認）。
#   也不設 max_split_size_mb，與使用者手動跑的那兩次一致。
export CITYGS_VRAM_CAP_GB=
export CITYGS_ALLOC_CONF=
source "$(dirname "$0")/_common.sh"

MODE=${1:?用法: task_citygs_trim.sh res <run> | gate | trimon | trimvpc}
CFG_TRIM=configs/citygsv2_mc_aerial_sh2_trim.yaml
COARSE_DIR=outputs/citygsv2_mc_aerial_coarse_sh2

find_ckpt () {   # find_ckpt <跑次目錄> -> 步數最大的 ckpt
  find "$1/checkpoints" -maxdepth 1 -name '*step=*.ckpt' 2>/dev/null \
    | sed -E 's/.*step=([0-9]+)\.ckpt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

resource_report () {   # resource_report <跑次名>
  local run="$1" dir="outputs/$1"
  local R="logs/citygs_resource_$(printf '%s' "$1" | tr '/' '_').log"
  local ck; ck=$(find_ckpt "$dir")
  [ -n "$ck" ] || { echo "⛔ $run 沒有 ckpt"; return 2; }
  local step; step=$(echo "$ck" | sed -E 's/.*step=([0-9]+)\.ckpt$/\1/')
  # ⚠⚠ 2026-09-17 踩過：`cost_budget_calibrate.py` 當時**不在 lab 上**，報告照樣往下跑，
  #   而結束碼只反映最後一個指令 => 台帳寫 rc=0 DONE，離線 Load 那一欄整個沒量到。
  #   => 先擋工具是否存在，再逐一檢查結束碼，最後用 exit 把狀態帶出管線（同 task_load_compare.sh）。
  local t
  for t in tools/cost_budget_calibrate.py tools/step_breakdown.py; do
    [ -f "$t" ] || { echo "⛔ 缺工具 $t（本機有、lab 沒有？先 put 上來）"; return 3; }
  done
  {
    bad=0
    echo "════════ 資源統計：$run（$(basename "$ck")）════════"
    echo "-- ckpt 大小 --"; du -h "$ck" | cut -f1
    echo "-- 台帳（N／it-s／VRAM 峰值） --"
    grep -F "outputs/$run" logs/quad_progress.log | tail -4
    echo "-- trim 是否真的跑過（本腳本啟動的跑次才有 log） --"
    local L="logs/$(printf '%s' "$run" | tr '/' '_').log"
    if [ -f "$L" ]; then
      echo "Trimming 次數 = $(grep -c 'Trimming\.\.\.' "$L")；v/c 首次觸發 = $(grep -c 'trim-vpc\] ✅' "$L")"
      grep -m2 -E '\[init-renderer\]' "$L"
    else
      echo "（$L 不存在：這個跑次不是本腳本啟動的，只能看台帳與離線量測）"
    fi
    echo "-- 離線 Load（同工具、同相機；成本那一欄的標準量法） --"
    conda run -n gspl --no-capture-output python tools/cost_budget_calibrate.py --ckpt "$ck" --max-cam "${CITYGS_MAXCAM:-600}" \
      || { echo "⛔ 離線 Load 失敗（成本那一欄沒量到）"; bad=1; }
    echo "-- 穩態每步成本（同一 process、CUDA event） --"
    conda run -n gspl --no-capture-output python tools/step_breakdown.py --run "$run" --step "$step" --repeat 20 \
      || { echo "⛔ 逐步計時失敗"; bad=1; }
    exit "$bad"
  } 2>&1 | grep -vE 'pkg_resources|declare_namespace|找不到深度圖|caching images' | tee "$R"
  local st=${PIPESTATUS[0]}
  echo "報告：$R"
  return "$st"
}

fit_arm () {   # fit_arm <跑次名> <最大步數> [額外參數...]
  local name="$1" steps="$2"; shift 2
  local out="outputs/$name"
  [ -d "$out" ] && { mv "$out" "${out}.aborted_$(date +%m%d_%H%M%S)" || return 4; echo "  舊輸出已搬開"; }
  local ck; ck=$(find_ckpt "$COARSE_DIR")
  [ -n "$ck" ] || { echo "⛔ 找不到 coarse ckpt（$COARSE_DIR/checkpoints）"; return 2; }
  echo "=== $name / 吃 coarse $(basename "$ck") / 最大步數 $steps / $(date) ==="
  local L="logs/$(printf '%s' "$name" | tr '/' '_').log"
  # ★ KEEP_CFG_RENDERER=1：保留 config 的 renderer（trim 才會真的跑）；不設就是舊行為
  CITYGS_KEEP_CFG_RENDERER=1 conda run -n gspl --no-capture-output python -u main.py fit \
    --config "$CFG_TRIM" \
    --model.initialize_from "$ck" \
    --trainer.max_steps "$steps" \
    --data.train_max_num_images_to_cache 1024 \
    -n "$name" "$@" 2>&1 | tee "$L"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "❌ $name 失敗 rc=$rc"; return "$rc"; }
  # ⚠ 可觀測性：沒觸發就不要留下「跑完了」的假象（記憶 observability_saves_runs）
  local nt; nt=$(grep -c 'Trimming\.\.\.' "$L")
  echo "[檢查] Trimming 次數 = $nt"
  [ "$nt" -ge 1 ] || { echo "⛔⛔ trim 一次都沒跑（renderer 仍被 ckpt 覆蓋？）=> 這個跑次不可用"; return 5; }
  if [ "$name" != "${name%vpc}" ]; then
    grep -q 'trim-vpc\] ✅' "$L" || { echo "⛔⛔ v/c 沒有首次觸發 => 這個跑次等於基準，不可用"; return 6; }
  fi
  return 0
}

case "$MODE" in
  res)
    RUN=${2:?用法: task_citygs_trim.sh res <跑次名>}
    resource_report "$RUN" ;;
  gate)
    # 最嚴格的那個臂跑 1,100 步：trim 從 step 1,000 起、每 500 步一次 => 1,000 會觸發
    fit_arm "${RUN_PREFIX}citygs_gate" 1100 \
      --model.renderer.init_args.trim_by_value_per_cost true || exit $?
    resource_report "${RUN_PREFIX}citygs_gate" || exit $?
    : > logs/citygs_trim_gate.ok
    echo "✅ 閘門通過：trim 會跑、v/c 會觸發、資源工具讀得到 => 允許排全長" ;;
  trimon|trimvpc)
    [ -f logs/citygs_trim_gate.ok ] || { echo "⛔ 還沒過閘門（先跑 task_citygs_trim.sh gate）"; exit 2; }
    EXTRA=()
    [ "$MODE" = trimvpc ] && EXTRA=(--model.renderer.init_args.trim_by_value_per_cost true)
    NAME="${RUN_PREFIX}citygs_${MODE}"
    fit_arm "$NAME" 60000 ${EXTRA[@]+"${EXTRA[@]}"} || exit $?
    resource_report "$NAME" ;;
  *) echo "⛔ 不認得的模式：$MODE（res|gate|trimon|trimvpc）"; exit 2 ;;
esac
