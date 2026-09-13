#!/bin/bash
# ★★★★★★ 那道 OOM 牆是隨機的 —— 用重複試驗數 OOM 率，不要再用單點下結論
#
# 2026-09-13 同一天內同一個設定給出兩個相反結果：
#   09:33  不開 max_split，N=2.6M，ballast 0.80G  ⛔ OOM（retries 2, ooms 1）
#   09:45  不開 max_split，N=2.6M，ballast 0.80G  ✔ 活著（retries 1, ooms 0）
# ⇒ 碎片失敗是**隨機事件**（哪一步剛好要不到一塊連續記憶體是機率問題）。
#   我今天已經因此三次收回關於 max_split 的說法 ⇒ 改成統計判準。
#   這也回頭解釋了 lab ~14,000 死 vs 本機 23,802 死 —— 很可能是同一現象的不同抽樣。
#
# 設計：同一個 ballast 點各跑 N 次，數 num_ooms。判準是**比例**，不是單次。
#   兩臂的 OOM 率差要大到與樣本數相稱才算「max_split 真的買到餘裕」。
# ★ 只能在本機做：真實卡的餘裕 + 單租戶（lab 上的鄰居會讓實體碎片狀況隨機變動）。
set -u
cd "$(dirname "$0")/.." || exit 1
REPEAT=${REPEAT:-5}
BALLAST=${BALLAST:-0.8}
NN=${NN:-2.6}
R=logs/oom_wall_repeats_$(date +%m%d_%H%M).log
{
echo "N=${NN}M  ballast=${BALLAST}G  每臂 $REPEAT 次"
for AC in "" "max_split_size_mb:128"; do
  oom=0; ok=0
  echo
  echo "===== PYTORCH_CUDA_ALLOC_CONF = ${AC:-（不設）} ====="
  for i in $(seq 1 "$REPEAT"); do
    if [ -z "$AC" ]; then
      out=$(env -u PYTORCH_CUDA_ALLOC_CONF conda run -n gspl python tools/vram_pressure.py \
              --arm b --n "$NN" --ballast "$BALLAST" 2>&1)
    else
      out=$(env PYTORCH_CUDA_ALLOC_CONF="$AC" conda run -n gspl python tools/vram_pressure.py \
              --arm b --n "$NN" --ballast "$BALLAST" 2>&1)
    fi
    line=$(printf '%s' "$out" | grep -E "^ +${BALLAST}0?G" | tail -1)
    if printf '%s' "$line" | grep -q 'OOM'; then
      oom=$((oom+1)); echo "  第 $i 次 ⛔ OOM"
    else
      ok=$((ok+1)); echo "  第 $i 次 ✔ $(printf '%s' "$line" | awk '{print $2" ms  retries="$4}')"
    fi
  done
  echo "  => OOM $oom / $REPEAT"
done
} 2>&1 | grep -vE "pkg_resources|declare_namespace|找不到深度圖" | tee "$R"
echo
echo "報告：$R"
echo "判準：兩臂的 OOM 率差要與樣本數相稱；差不出來就**不能宣稱 max_split 買到餘裕**，"
echo "      那 17.4% 的時間代價就該收回（改走降 cap，實測 2.0M 比 2.6M+max_split 快 31%）。"
