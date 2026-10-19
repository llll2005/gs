#!/bin/bash
# ★★★★★ 使用者 10-18 目視 g44s15k_best1sh1_inbox：黑色變透明、地底大高斯、邊緣擬合差 => tools/inbox_diag.py 逐一量，
#   並和 best1 60k、SH1 60k、單目深度／normal＋深度／深度失真 100 三個幾何正則臂、官方 release 同塊比（10-18 加：馬路靠地底面元撐的比例＝[5]）（相機與分區盒一律取自我方 15k＋SH1 的同塊 ckpt）。
# 用法（佇列）：[solo] bash scripts/lab/task_inbox_diag.sh <塊> [跑次...]
#   給了跑次（outputs/lab/<跑次>；official＝官方 release）就只評那些（各取最大步數 ckpt），並加 --hidden_all（全部訓練相機量從未被混合的顆粒＝建築是否中空）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
B=$1
REF=$(ls outputs/lab/g44s15k_best1sh1/blocks/block_$B/checkpoints/*step=15000.ckpt | head -1)
O=../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim
L=logs/inbox_diag_b${B}_$(date +%m%d_%H%M).log
bad=0
if [ $# -gt 1 ]; then
  shift; XA=(--hidden_all)
  for r in "$@"; do
    if [ "$r" = official ]; then c=$(ls $O/blocks/block_$B/checkpoints/*step=60000.ckpt | head -1)
    else c=$(ls outputs/lab/$r/blocks/block_$B/checkpoints/*.ckpt 2>/dev/null | sort -t= -k3 -n | tail -1); fi
    echo "== $r：$c" | tee -a "$L"
    conda run -n gspl --no-capture-output python tools/inbox_diag.py --ckpt "$c" --ref_ckpt "$REF" --views 24 "${XA[@]}" 2>&1 \
      | grep -E '^N |^ +|^\[|Error|Traceback' | tee -a "$L"
    [ "${PIPESTATUS[0]}" = 0 ] || bad=1
  done
  exit "$bad"
fi
for c in "$REF" \
         "$(ls outputs/lab/g44_best1/blocks/block_$B/checkpoints/*step=60000.ckpt | head -1)" \
         "$(ls outputs/lab/g44_best1sh1/blocks/block_$B/checkpoints/*step=60000.ckpt | head -1)" \
         "$(ls outputs/lab/g44_best1depth/blocks/block_$B/checkpoints/*step=60000.ckpt | head -1)" \
         "$(ls outputs/lab/g44_best1geo/blocks/block_$B/checkpoints/*step=60000.ckpt | head -1)" \
         "$(ls outputs/lab/g44_best1dist100/blocks/block_$B/checkpoints/*step=60000.ckpt | head -1)" \
         "$(ls $O/blocks/block_$B/checkpoints/*step=60000.ckpt | head -1)"; do
  echo "== $c" | tee -a "$L"
  conda run -n gspl --no-capture-output python tools/inbox_diag.py --ckpt "$c" --ref_ckpt "$REF" --views 24 2>&1 \
    | grep -E '^N |^ +|^\[|Error|Traceback' | tee -a "$L"
  [ "${PIPESTATUS[0]}" = 0 ] || bad=1
done
exit "$bad"
