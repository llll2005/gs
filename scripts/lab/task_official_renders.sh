#!/bin/bash
# 官方合併模型在我方各塊 test 視角上的渲染圖（使用者 10-15：compare_server 打開預設顯示官方，可改）
#   相機／GT 取自 g44_best1 該塊（沒有就 full44_best）的 test 圖；渲染用我方光柵器（官方光柵器有暗幀，09 §0.1）。
#   輸出 outputs/lab/official_release／official_paper/blocks/block_<B>/test/epoch=0-step=60000/（同 --save_val 格式）
# 用法：bash scripts/lab/task_official_renders.sh 6 12 10        之後本機 lab.py renders <塊> official_release official_paper
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
OR="outputs/${RUN_PREFIX%/}"; OR=${OR%/}
REL=$(ls ../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/checkpoints/*.ckpt 2>/dev/null | tail -1)
PAP=$(ls ../cityGS_origin_trimfix/outputs/citygsv2_mc_aerial_sh2_paper/checkpoints/*.ckpt 2>/dev/null | tail -1)
for B in "$@"; do
  LIKE=""
  for r in g44_best1 full44_best; do
    d="$OR/$r/blocks/block_$B"; ls -d "$d"/test/epoch=* >/dev/null 2>&1 && { LIKE=$d; break; }
  done
  [ -n "$LIKE" ] || { echo "⛔ block $B 沒有可對照的 test 圖"; continue; }
  for pair in "official_release:$REL" "official_paper:$PAP"; do
    nm=${pair%%:*}; ck=${pair#*:}
    [ -n "$ck" ] || { echo "⛔ $nm 沒有合併 ckpt"; continue; }
    echo "════ $nm block $B（視角＝$LIKE）$(date)"
    conda run -n gspl --no-capture-output python -u tools/render_like_run.py --ckpt "$ck" --like "$LIKE" --out_run "$nm" --out_root "$OR" 2>&1 \
      | grep -vE 'pkg_resources|declare_namespace|Warn|找不到深度圖'
  done
done
