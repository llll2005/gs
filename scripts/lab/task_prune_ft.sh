#!/bin/bash
# 2026-10-01：「逐 tile 前 K 名剪完再短暫 fine-tune」（使用者要求；事後剪枝沒重訓時同成本下已是最好，看重訓能補回多少）。
#   基準模型：lab/cs60_conic b6（現行預設、60k、N 2.34M）。剪法：tk{K}＝逐 tile 前 K 名；op{K}＝同 N、opacity 最高；
#   base＝不剪（對照「多跑 5k 步」本身的效果）。fine-tune＝用 Lightning **接續**（--ckpt_path）60k -> 65k：
#   學習率停在最終值、增生與 trim 在 30k 後本來就停 => 純低學習率 fine-tune（見 tools/prune_resume_ckpt.py 檔頭）。
# 2026-10-10（使用者：06(b) 加上 v 與 v/c 判準）：makev 只產生 v{K}／vc{K}（同 N、與 tk{K} 同顆數；不覆寫既有 tk／op／base）；
#   eval 自動把有 65000 ckpt 的 v／vc 臂一起評。
# 用法：task_prune_ft.sh make ｜ makev ｜ ft <arm> ｜ eval
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
source "$(dirname "$0")/_common.sh"
BASE=outputs/lab/cs60_conic/blocks/block_6
BCK=$BASE/checkpoints/epoch=110-step=60000.ckpt
BCFG=$(ls $BASE/lightning_logs/version_*/config.yaml | sort -V | tail -1)
OUT=outputs/lab/pft_b6_src
MODE=${1:?用法: task_prune_ft.sh make|makev|ft <arm>|eval}
case "$MODE" in
  make)
    conda run -n gspl --no-capture-output python tools/prune_resume_ckpt.py make --ckpt "$BCK" --out "$OUT" --K 32 64 ;;
  makev)
    conda run -n gspl --no-capture-output python tools/prune_resume_ckpt.py make --ckpt "$BCK" --out "$OUT" --K 32 64 --arms v vc ;;
  ft)
    ARM=${2:?缺 arm（base|tk32|op32|tk64|op64|v32|vc32|v64|vc64）}
    CK="$OUT/$ARM/checkpoints/$(basename "$BCK")"
    [ -f "$CK" ] || { echo "⛔ 缺 $CK（先跑 make）"; exit 2; }
    NAME=${RUN_PREFIX}pft_b6_$ARM
    [ -d "outputs/$NAME" ] && mv "outputs/$NAME" "outputs/$NAME.aborted_$(date +%m%d_%H%M%S)"
    echo "=== $NAME：從 $CK 接續 60k -> 65k（config＝$BCFG）$(date) ==="
    conda run -n gspl --no-capture-output python -u main.py fit --config "$BCFG" --ckpt_path "$CK" \
      -n "$NAME" --trainer.max_steps 65000 ;;
  eval)
    L=logs/prune_ft_eval_$(date +%m%d_%H%M).log
    cks=("$BCK")
    for arm in tk32 op32 tk64 op64 v32 vc32 v64 vc64; do f="$OUT/$arm/checkpoints/$(basename "$BCK")"; [ -f "$f" ] && cks+=("$f"); done
    for arm in base tk32 op32 tk64 op64 v32 vc32 v64 vc64; do
      c=$(find outputs/${RUN_PREFIX}pft_b6_$arm -name '*step=65000*.ckpt' 2>/dev/null | head -1)
      [ -n "$c" ] && cks+=("$c") || echo "⚠ pft_b6_$arm 沒有 65000 步的 ckpt"
    done
    conda run -n gspl --no-capture-output python tools/prune_resume_ckpt.py eval "${cks[@]}" 2>&1 \
      | grep -vE 'pkg_resources|declare_namespace|caching|depth scale|appearance|colmap|down sample|depth maps' | tee "$L"
    exit "${PIPESTATUS[0]}" ;;
  *) echo "⛔ 不認得的模式 $MODE"; exit 2 ;;
esac
