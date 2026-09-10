#!/usr/bin/env bash
# SfM-init 在 step 1500 與 60000 的「真表面覆蓋率」有沒有改善（使用者 2026-09-11 指定）
# 量法＝ novel_view_coverage.py：只算落在 SfM 點雲 1~99 百分位盒內的粒子，
# 因為 rend_alpha 會把場外的巨大 floater 算成覆蓋（arm B 0.9919 -> 0.4744 就是這樣騙到的）。
set -euo pipefail
cd "$(dirname "$0")/.."
R=logs/coverage_1500_$(date +%m%d_%H%M).log
{
  echo "===== step 1499（1500 步當下） ====="
  conda run -n gspl python tools/novel_view_coverage.py \
    --arms speed3_b12 speed3_sfminit_b12 agd2_b12 --step 1499 --max-cam 12
  echo
  echo "===== step 14999（densify 中段，看幕布有沒有被清掉） ====="
  conda run -n gspl python tools/novel_view_coverage.py \
    --arms speed3_b12 speed3_sfminit_b12 --step 14999 --max-cam 12
} 2>&1 | tee "$R"
echo "報告：$R"
