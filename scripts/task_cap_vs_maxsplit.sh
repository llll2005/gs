#!/bin/bash
# ★★★★★ 兩個換餘裕的方式，哪個划算：**付 14~17% 開 max_split** vs **降 cap_max**
#
# 2026-09-13 已量到（本機、N=2.60M、計算量逐字相同）：
#   不設 max_split        264.6 ms   撐到 ballast 0.60G，0.80G ⛔ OOM
#   max_split_size_mb:128 310.4 ms   撐到 ballast 1.20G，1.50G ⛔ OOM
#   => +0.6 GB 餘裕是**真的**（與記憶 vram_wall_is_hard 一致），但時間代價是
#      **14~17%**，不是記憶裡記的 6.4%（那是舊資料、284 台相機時量的）。
#      14~17% 對一個 8.8h 跑次 = 75~90 分鐘。
#
# 但「換餘裕」不只一條路：降 cap_max 也能換，而且它換的是**計算量本身**（更快）。
# 所以要問的是：在同樣的餘裕下，哪一個操作點交付更多品質/小時？
#   路徑 A  cap 2.6M + max_split      = 慢 14~17%，顆數滿
#   路徑 B  cap 小一點 + 不開 max_split = 快，但顆數少
# 這支腳本量 B 的可行邊界：不開 max_split 時，N 要降到多少才撐得住 ballast 0.8G。
# ⚠ 品質那一側不在這裡量（顆數→PSNR 的斜率是舊年代的 +0.541 dB/加倍，跨年代不可引用）
#   ⇒ 本腳本只交付「餘裕與速度」兩軸，品質要另外排跑次。
# ★ 只能在本機做：要的是單租戶的時間與真實卡的餘裕。
set -u
cd "$(dirname "$0")/.." || exit 1
R=logs/cap_vs_maxsplit_$(date +%m%d_%H%M).log
{
  echo "===== 路徑 B：不開 max_split，掃 N 找撐得住 ballast 0.8G 的上限 ====="
  env -u PYTORCH_CUDA_ALLOC_CONF conda run -n gspl --no-capture-output \
    python tools/vram_pressure.py --arm b --n 2.6 --ballast 0.8
  for N in 2.4 2.2 2.0; do
    echo
    echo "--------- N = ${N}M ---------"
    env -u PYTORCH_CUDA_ALLOC_CONF conda run -n gspl --no-capture-output \
      python tools/vram_pressure.py --arm b --n "$N" --ballast 0.8
  done
  echo
  echo "===== 對照：路徑 A（cap 2.6M + max_split）在同一個 ballast ====="
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 conda run -n gspl --no-capture-output \
    python tools/vram_pressure.py --arm b --n 2.6 --ballast 0.8
} 2>&1 | grep -vE "pkg_resources|declare_namespace|找不到深度圖" | tee "$R"
echo
echo "報告：$R"
echo "判讀：若 B 在某個 N 下**又快又活**，而該 N 的顆數損失小於 14~17% 的時間收益，"
echo "      就該改走 B（降 cap）而不是付 max_split 的稅。"
