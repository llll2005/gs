#!/bin/bash
# 幾分鐘的檢驗，但需要獨佔 GPU（要載入 2.34M 顆的模型）。
# A：輔助圖（depth_to_normal / rend_dist / view_normal）在最終配方下已無人使用，成本多少？
# B：rend_dist 是否是免費的誤差偵測器，還是只是在重複 alpha/覆蓋已經給的資訊？
set -u
cd "$(dirname "$0")/.." || exit 1
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/audit_byproducts.py --run dssim05_b12
