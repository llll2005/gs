#!/bin/bash
# v1 config 於【修正後的程式碼】重跑。⚠ 2026-08-06 晚間更新目的：
#
# 原本要量「P7 害了多少」——**那個目的已失效**。除了 P7 的覆蓋率 gate，同一天還修了
# 貢獻度剪枝的 top-K（四處，見 紀錄/研究總覽.md §7.4），一次變兩個東西，差值無法歸因。
#
# 現在它是兩件事：
#   1. **修正後程式碼的第一個乾淨基準**（24k 是最快的載具，6.7h vs 60k 的 11h）
#   2. ★**「少而準」的正確測法**：v1 排程自然衰減到小顆數，且 densify 全程正常運作
#      （不像 cap80k 把 N 釘在 cap 導致 `num_gs = max(0, min(cap,1.05N)-N) = 0`，densify 永久關閉）
#      判準＝建築區紋理比，這是**絕對量測、不需要參照組**。
#
# ⚠ top-K 修正後排序相關係數 0.9157、集合重疊 86~95%，所以軌跡不會劇變，
#   但終點顆數不保證仍是 77,616。
#
# 【先驗】aggr24k_b12（被汙染版）：
#     5,679  20.644 / .582 / .764
#    11,359  21.142 / .593 / .732
#    17,039  21.588 / .598 / .714
#    22,719  21.907 / .604 / .698
#    24,000  22.070 / .604 / .695     N=77,616   建築紋理比 0.170
#    ⛔ 該跑次 2.4% 的步數帶著 d_reg 1e5~1e6 的壞梯度（1141/48091），**且用的是壞的 top-K**
#       ⇒ 只能當數量級參考，不是對照組。見 紀錄/研究總覽.md §7.3 與 §7.4
#
# 【這一臂同時回答兩件事】
#  1. P7 害了多少：乾淨版 vs 22.070 的差
#  2. ★「少而準」的正確測法：v1 的排程自然衰減到 77,616 顆，而且 densify 全程正常運作
#     （不像 cap80k 把 N 釘在 cap 導致 `num_gs = max(0, min(cap,1.05N)-N) = 0`，densify 永久關閉）。
#     ⇒ 這是一個「密度控制正常、訓練到 7.7 萬顆」的模型。
#     判準＝建築區紋理比：對照 oreg 900k 的 0.372、被汙染 v1 的 0.170。
#     ⚠ 事後剪枝測不到這件事：剪枝不重新擬合，剩下的高斯 scale/opacity 是在別人存在的前提下
#       調出來的，刪掉九成後個別權重不足。那 4.5dB 掉分大部分是「沒重擬合」的代價。
set -euo pipefail
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=v1clean_24k_b12
rm -rf "outputs/$NAME"
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_24k_aggr_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
