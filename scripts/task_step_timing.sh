#!/bin/bash
# ★★★★★ 穩態每步成本（本機、同一 process、CUDA event 計時）—— 成本感知結果的「時間」那一欄。
#
# 為什麼不用 lab 的 it/s：lab 三槽平行，吞吐量取決於鄰居在做什麼；
#   而跨跑次比 wall time 的噪音本來就有 ~5%（agd vs agd2 差 5.2%，計算量完全相同）。
#   tools/step_breakdown.py 在**同一個 process** 內量、每段重複取中位數、不含一次性事件。
# 為什麼現在要量：cs_trimvpc 同顆數、同品質，離線 binning Load **-63%**
#   => 若 Load 真的是渲染成本，forward/backward 時間應該看得到下降；看不到就要重新理解 Load。
#
# 用法：task_step_timing.sh <step> <run> [<run> ...]
#   跑次名要能被 outputs/<run>/**/*step=<step>.ckpt 找到 => lab 拉回來的要帶 lab/
set -u
cd "$(dirname "$0")/.." || exit 1
STEP=${1:?用法: task_step_timing.sh <step> <run> [<run> ...]}; shift
[ $# -ge 1 ] || { echo "⛔ 至少要一個跑次"; exit 2; }
R=logs/step_timing_$(date +%m%d_%H%M).log
{
  bad=0
  for r in "$@"; do
    echo "════════ $r @ $STEP ════════"
    conda run -n gspl --no-capture-output python tools/step_breakdown.py --run "$r" --step "$STEP" --repeat 20 || bad=1
  done
  exit "$bad"
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|找不到深度圖|caching images' | tee "$R"
# ⚠ 2026-09-13：管線的結束碼是 `tee` 的 => python 崩潰時台帳照樣記 `rc=0 DONE`
#   （稽核清單「看起來正常但沒作用」同型）。取大括號那一段自己的結束碼。
st=${PIPESTATUS[0]}
echo "報告：$R"
[ "$st" -eq 0 ] || echo "⛔ 至少一個跑次失敗（rc=$st）—— 台帳的 DONE 不代表成功，看報告"
exit "$st"
