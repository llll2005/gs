#!/bin/bash
# ★★★★★★ cb25 最終 Load：同預算下 cb25cost 贏 0.66 dB，是「配得更聰明」還是「花得更多」？
#
# 背景（2026-09-13，lab b6 / 21,920 步）：
#   cs_cb25      26.580 / 0.758 / 0.308 / 0.588   N=0.22M  峰值 1.31 GB
#   cs_cb25cost  27.240 / 0.787 / 0.264 / 0.634   N=0.38M  峰值 1.40 GB   <- 四項全贏
# 但比較是**不對稱**的：cb25 沒開 cost_budget_report，看不到它的 Load；
# 而 cb25cost 的 Load 在增生停止後漏到預算的 **190%**（8,541,161 vs 4,496,883）。
#   => 若 cb25 的最終 Load 也在 ~190%  ：同成本，+0.66 dB 是**配置**贏的 => 命題成立
#   => 若 cb25 的最終 Load 明顯較低    ：cb25cost 只是**花得多** => 不能算命題的勝利
# ⚠ 兩個 ckpt 用**同一支工具、同一組相機**（工具取前 n 台，同塊同 parser => 同一組）。
set -u
cd "$(dirname "$0")/.." || exit 1
R=logs/cb25_load_$(date +%m%d_%H%M).log
{
  for r in cs_cb25 cs_cb25cost; do
    echo "════════ $r ════════"
    conda run -n gspl --no-capture-output python tools/cost_budget_calibrate.py \
      --ckpt "outputs/lab/$r/blocks/block_6/checkpoints/epoch=40-step=21920.ckpt" --max-cam 100000
  done
  echo "預算（兩者相同）= B0/4 = 4,496,883"
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|找不到深度圖|caching images' | tee "$R"
echo "報告：$R"
