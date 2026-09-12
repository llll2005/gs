#!/bin/bash
# ★★★★★★★ 失敗區在**修正後資料**上的逐 tile 判定（純 CPU，但排在 GPU 序列裡以保證順序）
#
# 要答的問題：§16.14 的影像↔姿態錯位修好之後，那個「左半邊糊掉」的區域還在嗎？
# ⚠ PSNR 答不了 —— 失敗區只佔畫面一部分，全幅 PSNR 會被正常區稀釋
#   （eval_protocol 記過：逐塊評官方 test 卡在 ~15dB 與好壞無關，同一個稀釋效應）。
#   所以要看**糊掉 tile 的位置與比例**，而不是分數。
# ⚠ 判準用**絕對**對比門檻（tools/failure_map.py 的 --min-std）：
#   相對門檻（GT std > quantile(sg, 0.40)）會把平坦 tile 也收進來，那裡的 corr 是噪音
#   ⇒ 會把 28% 講成 46%。這個汙染 2026-09-13 量過。
#
# 對照：b12（舊資料上有失敗區）vs b13（沒觀察到）—— 同一組判準，兩塊互為對照。
set -u
cd "$(dirname "$0")/.." || exit 1
RC=0
run_one () {   # run_one <run> <blk>
  local R="$1" B="$2"
  local d
  d=$(ls -d "outputs/$R/blocks/block_$B"/test/*/ 2>/dev/null | tail -1)
  if [ -z "$d" ]; then
    echo "⚠ $R / block $B 沒有 test/ 影像（要先 main.py test --save_val）=> 跳過"
    RC=1; return
  fi
  echo "════════ $R / block $B   影像 $(ls "$d"*.png 2>/dev/null | wc -l) 張 ════════"
  conda run -n gspl python tools/failure_map.py "$R" --blk "$B" 2>&1 \
    | grep -vE "pkg_resources|declare|down sample|loading|appearance" | tail -30
}
run_one gate15000 12
run_one speed3 13
echo
echo "════════ 逐 tile 糊掉遮罩（兩塊同判準）════════"
conda run -n gspl python tools/blur_persistence.py gate15000 --blk 12 2>&1 | tail -12
conda run -n gspl python tools/blur_persistence.py speed3 --blk 13 2>&1 | tail -12
exit "$RC"
