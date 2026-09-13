#!/bin/bash
# ★★★★★ 重量逐塊 tau —— v2 §0「待重測」的第一項
#
# 為什麼要重量（而不是作廢）：使用者 2026-09-13 問「檔名自述作廢，會不會也是 GT 問題」，
# 回查發現 `block_taus_..._已作廢.csv` 的作廢理由是「量於 EXACT_SUPPORT **之前**、
# 用 gsplat 前向估（非 trim）」—— **不是** GT 問題。
# tau（最壞視角的逐點 tile 覆蓋）是**渲染幾何統計量，不經過 GT 影像↔姿態的配對**
# => 量本身倖存，只是要用現行 renderer 在**現行資料的成品模型**上重量。
# （measure_tau.py 的檔頭也自述「GT 無關的記憶體量測」。）
#
# 它補的缺口：
#   ① 「舊 cap 表用初始化期 tau」—— N_max 公式的 tau 一直是 init 期的，偏樂觀
#      （freeze_b12_K2 預期 2.5M 卻在 1.8M 就 OOM，因為訓練中 tau 從 73 漲到約 120）
#   ② 「用 coarse 顆數定 cap」那個提案的前提
#   ③ EXACT_SUPPORT 之後實際 binding 相交數已從 314M 降到 240M => 舊表偏高，幅度未知
set -u
cd "$(dirname "$0")/.." || exit 1
Q='pkg_resources|declare|找不到深度圖|dataparser|down sample|loading|appearance'
P=data/matrix_city/aerial/train/block_all/partition/partitions-dim_5_5_visibility_0.08

echo "=== b6（lab/speed3 60k 成品，N=2.34M）==="
conda run -n gspl python tools/measure_tau.py --run lab/speed3 \
  --block-list "$P/001_001.txt" 2>&1 | grep -vE "$Q" | tail -20

echo
echo "=== b12（gate15000 15k 成品，N=2.34M）==="
conda run -n gspl python tools/measure_tau.py --run gate15000 \
  --block-list "$P/002_002.txt" 2>&1 | grep -vE "$Q" | tail -20

echo
echo "=== 對照：init 期的 tau（sfmfill 的 PLY，量的是起點幾何）==="
conda run -n gspl python tools/measure_tau.py \
  --ply data/matrix_city/aerial/train/block_all/sfmfill_init/block_12.ply \
  --block-list "$P/002_002.txt" 2>&1 | grep -vE "$Q" | tail -16
