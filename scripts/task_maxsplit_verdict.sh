#!/bin/bash
# ★★★★ max_split_size_mb:128 該不該繼續付？（~15 分）
#
# 2026-09-11 量到：它固定貴 **6.4%**（383.0 -> 407.4 ms），換到約 **+0.6 GB** 可用餘裕
# （無壓力測試在 0.90G ballast OOM，設了之後撐到 1.50G）。所有 task_*.sh 都寫死了它，
# 但從沒對照過。6.4% 對一個 8.8h 的跑次 = **34 分鐘**。
#
# 決策要問的不是「哪個快」，而是「**不設它，真實峰值撐不撐得住**」：
#   真實峰值不在 N=2.34M 的收割期，而在 densify 期 N 第一次頂到 cap 2.6M（約 step 20,900），
#   而且那時還要跑 trim pass（全部 284 台相機）。
#   ⇒ 這裡用 N=**2.6M** + ballast 模擬 Lightning/dataloader 那幾百 MB 的額外常駐。
#
# 判準：
#   不設也能在 ballast >= 0.3G 存活 => 那 6.4% 是白付的 => 排一次真實無旗標跑次確認
#   不設在 ballast < 0.3G 就 OOM    => 6.4% 是保險費，**繼續付**（或改用降 cap 換速度）
# ⚠ 配置器設定不改數值運算 ⇒ 這是純粹的速度/餘裕取捨，不需要做品質消融。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片 ⇒ **樂觀**，真實訓練只會更早撞牆。
set -u
cd "$(dirname "$0")/.." || exit 1
# ⚠ 2026-09-13 新資料實測：ballast 只掃到 0.6G **兩臂都沒 OOM** => 沒測到邊界，
#   而 max_split 換到的正是「大塊連續請求不會失敗」，那只在邊界才看得出來。
#   => ballast 列表改成可覆寫，用來找邊界：
#        BALLAST="0.6 0.8 1.0 1.2 1.5" bash scripts/task_maxsplit_verdict.sh
BALLAST=${BALLAST:-"0.0 0.2 0.3 0.4 0.6"}
R=logs/maxsplit_verdict_$(date +%m%d_%H%M).log
{
  for CONF in "" "max_split_size_mb:128"; do
    echo
    echo "===== PYTORCH_CUDA_ALLOC_CONF = ${CONF:-（完全不設）} ／ N = 2.6M（真實峰值）====="
    if [ -z "$CONF" ]; then
      env -u PYTORCH_CUDA_ALLOC_CONF conda run -n gspl --no-capture-output \
        python tools/vram_pressure.py --arm b --n 2.6 --ballast $BALLAST
    else
      PYTORCH_CUDA_ALLOC_CONF="$CONF" conda run -n gspl --no-capture-output \
        python tools/vram_pressure.py --arm b --n 2.6 --ballast $BALLAST
    fi
  done
} 2>&1 | grep -vE "pkg_resources|declare_namespace" | tee "$R"
echo
echo "報告：$R"
