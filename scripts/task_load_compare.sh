#!/bin/bash
# ★★★★★ 同一支工具、同一組相機，量多個跑次**終點模型**的 Load —— 成本那一欄的標準量法。
#
# 為什麼要有這支（2026-09-13）：
#   訓練中的 `[cost-budget]` 打印與離線工具**不是同一個數**：cb25cost 同一個終點模型，
#   訓練中打印 8,541,161、離線量 5,618,320（差約 1.5 倍）。訓練中是「一個區間內的最壞視角」，
#   離線是「終點模型 x 全部相機」。=> **跨量測路徑不可比**，要比成本就用這支統一量。
#   而命題的收益在**成本**那一欄（PSNR 平手 + 成本下降就是勝），所以每個對比都該配一次。
# 前一版 `scripts/task_cb25_load.sh` 是寫死兩個跑次的一次性版本，保留作為 cb25 判決的出處。
#
# 用法：task_load_compare.sh <block> <run> [<run> ...]
#   跑次名**不帶** lab/；ckpt 須先拉回本機：lab.py get lab/<run> <block> <step>
#   每個跑次自動取該塊**步數最大**的 ckpt。
set -u
cd "$(dirname "$0")/.." || exit 1
BLK=${1:?用法: task_load_compare.sh <block> <run> [<run> ...]}; shift
[ $# -ge 1 ] || { echo "⛔ 至少要一個跑次"; exit 2; }
R=logs/load_compare_b${BLK}_$(date +%m%d_%H%M).log
{
  bad=0
  for r in "$@"; do
    dir="outputs/lab/$r/blocks/block_$BLK/checkpoints"
    ck=$(find "$dir" -maxdepth 1 -name '*step=*.ckpt' 2>/dev/null \
         | sed -E 's/.*step=([0-9]+)\.ckpt$/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -z "$ck" ]; then
      echo "⛔ $r：本機沒有 block_$BLK 的 ckpt（先 lab.py get lab/$r $BLK <step>）"; bad=1; continue
    fi
    echo "════════ $r  ($(basename "$ck")) ════════"
    conda run -n gspl --no-capture-output python tools/cost_budget_calibrate.py --ckpt "$ck" --max-cam 100000 || bad=1
  done
  exit "$bad"
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|找不到深度圖|caching images' | tee "$R"
# ⚠ 2026-09-13：管線的結束碼是 `tee` 的 => python 崩潰時台帳照樣記 `rc=0 DONE`
#   （稽核清單「看起來正常但沒作用」同型）。取大括號那一段自己的結束碼。
st=${PIPESTATUS[0]}
echo "報告：$R"
[ "$st" -eq 0 ] || echo "⛔ 至少一個跑次失敗（rc=$st）—— 台帳的 DONE 不代表成功，看報告"
exit "$st"
