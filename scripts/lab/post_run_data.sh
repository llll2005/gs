#!/bin/bash
# 圖表資料包：每個訓練跑完，把 `紀錄/實驗分析/` 各圖要用的量測一次量齊（2026-10-03 使用者：
#   「之後的跑次 包括當前 lab 上的 都得輸出所有圖表要求的數據 這樣要用的時候才不用重跑」）。
#   run_fit（scripts/lab/_common.sh）在訓練成功後自動呼叫；舊跑次用 backfill 補。
#
# 輸出：outputs/<跑次>/blocks/block_<塊>/chart_data/<項目>.txt —— 已存在就跳過（可重跑、可中斷續做）。
#   彙整成表：python tools/chart_data_summary.py（畫圖前用它取數字）
# 項目（終點 ckpt；帶 1499 的是 @1,499，給 `12` 的「@1,499 → @60k」）：
#   load        離線 Load（代理與精確 Σtiles，全部相機）               `04`、`05`、`07`   tools/cost_budget_calibrate.py
#   heldout     塊內官方 held-out（PSNR／SSIM／LPIPS／紋理比／ms 每幀）  `12` §6          tools/eval_official_test.py --block
#   geom        渲染深度 vs SfM 可見點 slope／corr＋懸空統計            `12` §1          tools/measure_depth_bias.py、audit_geometry.py
#   storage     ckpt 組成與跑次目錄                                     `03`             tools/run_storage_audit.py
#   heldout1499、geom1499、failure、tau   只對 >= 60,000 步的跑次（`12` 的圖只用 60k）
#     failure   失敗 tile（先存 val 渲染圖）                            `12` §3          task_savepics.sh＋tools/failure_map.py
#     tau       訓練後 tile 覆蓋 tau                                    `12` §5          tools/measure_tau.py
#   timing      forward／fwd+bwd ms 與 VRAM 分項 —— **只有獨佔才準**（`04`、`05`）       tools/step_breakdown.py
#               => 量測當下 GPU 上沒有別的行程（[solo] 訓練跑完）就直接量；否則登記到 logs/chart_timing_pending.txt，
#                  由 solo_batch（[solo]）一次量完。每次都寫 context.txt（當下其他 GPU 行程數）供判斷時間欄能不能用
#   全場景（block null）只量 load／storage（timing 登記），塊相關項目跳過
#
# 用法：
#   bash scripts/lab/post_run_data.sh run <跑次（含 lab/ 前綴）> <塊>     一個跑次（run_fit 自動呼叫）
#   bash scripts/lab/post_run_data.sh backfill [最少步數=20000]            補量 outputs/<前綴>* 底下所有跑完的跑次（缺的才量）
#   [solo] bash scripts/lab/post_run_data.sh solo_batch                    量完登記中的計時
#   環境變數：CITYGS_POSTDATA_SKIP="failure tau"（跳過指定項目）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial/train/block_all
PEND=logs/chart_timing_pending.txt
PY="conda run -n gspl --no-capture-output python"
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
FILT='pkg_resources|declare_namespace|caching images|找不到深度圖|appearance group|loading colmap|down sample|colmap dataparser|depth maps'
mkdir -p logs

final_ck () {   # final_ck <塊目錄> => 步數最大的 ckpt（不含 culldust）
  find "$1/checkpoints" -maxdepth 1 -name '*step=*.ckpt' ! -name '*culldust*' 2>/dev/null \
    | sed -E 's/.*step=([0-9]+)\.ckpt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}
step_of () { echo "$1" | sed -E 's/.*step=([0-9]+)\.ckpt$/\1/'; }
skip () { case " ${CITYGS_POSTDATA_SKIP:-} " in *" $1 "*) return 0 ;; esac; return 1; }

one () {   # one <跑次> <塊>
  local run="$1" b="$2" bd ck st out dim x y
  bd="outputs/$run/blocks/block_$b"
  ck=$(final_ck "$bd"); [ -n "$ck" ] || { echo "（$run block $b：沒有 ckpt，略過）"; return 0; }
  st=$(step_of "$ck"); out="$bd/chart_data"; mkdir -p "$out"
  # 切塊網格：讀 resolved config 的 data.parser.block_dim（沒寫＝5x5）
  dim=$($PY - "$bd" <<'PYEOF' 2>/dev/null | tail -1
import glob, sys, yaml
c = sorted(glob.glob(sys.argv[1] + "/lightning_logs/version_*/config.yaml"))
d = (yaml.safe_load(open(c[-1])) or {}).get("data", {}).get("parser", {}).get("init_args", {}) if c else {}
bd = d.get("block_dim") or [5, 5]
print(int(bd[0]), int(bd[1]))
PYEOF
)
  [ -n "$dim" ] || dim="5 5"
  set -- $dim; x=$1; y=$2
  # 量測當下 GPU 上有沒有別的行程：0 ＝獨佔 => 時間類數字（計時、held-out 的 ms／幀）可用；記進 context.txt
  local others mem; others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  #   容器外的佔用（lab 出現過 10~16 GB 的間歇佔用）不會列在行程清單 => 顯存已用 > 1.5 GB 也當非獨佔
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "$others" = 0 ] && [ "${mem:-0}" -gt 1500 ] 2>/dev/null && others="外部佔用${mem}MiB"
  printf '量測時間 %s\n量測當下 GPU 上其他行程數 %s（0＝獨佔：計時與 ms／幀可用）\n' "$(date '+%F %T')" "$others" > "$out/context.txt"
  echo "════ 圖表資料包：$run block $b（step $st、${x}x${y}；其他 GPU 行程 $others）→ $out"
  _do () {   # _do <項目> <指令...>：已有就跳過；失敗不留半個檔
    local it="$1"; shift
    skip "$it" && { echo "  略過 $it（CITYGS_POSTDATA_SKIP）"; return 0; }
    [ -s "$out/$it.txt" ] && { echo "  已有 $it"; return 0; }
    echo "  量 $it ..."
    if "$@" > "$out/$it.tmp" 2>&1; then
      grep -vE "$FILT" "$out/$it.tmp" > "$out/$it.txt"; rm -f "$out/$it.tmp"; echo "  ✅ $it"
    else
      mv "$out/$it.tmp" "$out/$it.failed"; echo "  ⛔ $it 失敗（見 $out/$it.failed）"
    fi
  }
  _do load $PY tools/cost_budget_calibrate.py --ckpt "$ck" --max-cam 100000
  _do storage $PY tools/run_storage_audit.py "$bd" "$ck"
  _heldout () { $PY tools/eval_official_test.py --ckpt "$1" --block "$b" --block_dim "$x" "$y" | grep -vE "\.\.\.[0-9]+/[0-9]+ +PSNR"; }
  _geom () { $PY tools/measure_depth_bias.py --ckpt "$1" --block "$b" --block_dim "$x" "$y" --content_bounds; [ "$2" = last ] && $PY tools/audit_geometry.py "$run" --block "$b" | tail -6; return 0; }
  # 全場景（block null，例：coarse）沒有塊視角 => 只量 load／storage／timing
  case "$b" in ''|*[!0-9]*) BLOCKWISE=0 ;; *) BLOCKWISE=1 ;; esac
  [ "$BLOCKWISE" = 1 ] && { _do heldout _heldout "$ck"; _do geom _geom "$ck" last; }
  if [ "$BLOCKWISE" = 1 ] && [ "$st" -ge 60000 ]; then
    local c1499; c1499=$(ls "$bd"/checkpoints/*step=1499.ckpt 2>/dev/null | head -1)
    if [ -n "$c1499" ]; then
      _do heldout1499 _heldout "$c1499"
      _do geom1499 _geom "$c1499" early
    fi
    _fail () {
      ls -d "$bd/test/"*/ >/dev/null 2>&1 || bash scripts/lab/task_savepics.sh "$run" "$b" | tail -3
      $PY tools/failure_map.py "$run" --blk "$b" --block-dim "$x" "$y"
    }
    _do failure _fail
    _bl="$D/partition/partitions-dim_${x}_${y}_visibility_0.08/$(printf '%03d_%03d' $((b % x)) $((b / x))).txt"
    [ -f "$_bl" ] && _do tau $PY tools/measure_tau.py --run "$run/blocks/block_$b" --block-list "$_bl"
  fi
  # 計時只在獨佔下才準：此刻獨佔（others=0，例如 [solo] 訓練跑完接著量）就直接量；否則登記等 solo_batch
  #   （4x4 的 res 模式自己會逐塊量，不重複）
  if ! skip timing && [ ! -s "$out/timing.txt" ]; then
    case "$run" in
      *full44*) ;;
      *) if [ "$others" = 0 ] && [ "$BLOCKWISE" = 1 ]; then
           _do timing $PY tools/step_breakdown.py --run "$run" --block "$b" --repeat 20
         else
           grep -qxF "$run $b" "$PEND" 2>/dev/null || echo "$run $b" >> "$PEND"; echo "  登記 timing（非獨佔或全場景 => 等 [solo] solo_batch）"
         fi ;;
    esac
  fi
  return 0
}

case "$1" in
  run) one "${2:?跑次}" "${3:?塊}" ;;
  backfill)
    MIN=${2:-20000}
    for bd in outputs/${PFX}*/blocks/block_*; do
      case "$bd" in *.aborted_*|*aborted_blocks*) continue ;; esac
      [ -d "$bd/checkpoints" ] || continue
      ck=$(final_ck "$bd"); [ -n "$ck" ] || continue
      [ "$(step_of "$ck")" -ge "$MIN" ] || continue
      r=${bd#outputs/}; r=${r%/blocks/*}; b=${bd##*/block_}
      case "$b" in null) continue ;; esac   # 全場景（coarse）沒有塊視角可量
      one "$r" "$b"
    done ;;
  solo_batch)
    [ -s "$PEND" ] || { echo "（沒有登記中的計時）"; exit 0; }
    cp "$PEND" "$PEND.work"
    while read -r r b; do
      [ -n "$r" ] || continue
      out="outputs/$r/blocks/block_$b/chart_data"; mkdir -p "$out"
      if [ ! -s "$out/timing.txt" ]; then
        echo "════ 計時 $r block $b"
        case "$b" in ''|*[!0-9]*) BA=() ;; *) BA=(--block "$b") ;; esac   # 全場景（block null）不帶 --block
        $PY tools/step_breakdown.py --run "$r" "${BA[@]}" --repeat 20 2>&1 | grep -vE "$FILT" > "$out/timing.tmp" \
          && mv "$out/timing.tmp" "$out/timing.txt" || mv "$out/timing.tmp" "$out/timing.failed"
      fi
      grep -vxF "$r $b" "$PEND" > "$PEND.new"; mv "$PEND.new" "$PEND"
    done < "$PEND.work"
    rm -f "$PEND.work" ;;
  *) echo "⛔ 不認得的模式：$1（run／backfill／solo_batch）"; exit 2 ;;
esac
