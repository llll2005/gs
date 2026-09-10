#!/usr/bin/env bash
# 場外幕布會不會活到 60k？以及濾掉它之後，糊掉區到底有沒有幾何？
# 2026-09-11：`novel_view_coverage.py` 抓到 arm B 的 rend_alpha 0.9919 有一半來自
# 場外的巨大 floater（只算 SfM 盒內剩 0.4744）。1500 步的四臂結論因此作廢。
# 成熟跑次 agd2_b12 盒外只佔 1.73% => 幕布**看起來**是早期現象，但 SfM-init 起點不同，
# 必須各自量過才算數。
set -euo pipefail
cd "$(dirname "$0")/.."
R=logs/inbox_coverage_$(date +%m%d_%H%M).log
{
  echo "===== 1) 場外比例與真表面覆蓋（60k 成熟跑次） ====="
  conda run -n gspl python tools/novel_view_coverage.py \
    --arms speed3_b12 speed3_sfminit_b12 agd2_b12 --step 60000 --max-cam 12
  echo
  echo "===== 2) 濾掉場外後：糊掉 tile 到底有沒有幾何（現行最佳） ====="
  conda run -n gspl python tools/coverage_vs_blur.py --arm speed3_b12 --step 60000 --inbox 1
  echo
  echo "===== 3) 同上，SfM-init ====="
  conda run -n gspl python tools/coverage_vs_blur.py --arm speed3_sfminit_b12 --step 60000 --inbox 1
} 2>&1 | tee "$R"
echo "報告：$R"
