#!/bin/bash
# ★★★★★ binning 外接盒浪費 + trim 成本感知判準（兩個診斷，一次 render loop，~6 分）
# 判準寫在 tools/binning_and_trim_probe.py 的 docstring 與輸出末尾。
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
R=logs/binning_trim_$(date +%m%d_%H%M).log
{
  for RUN in speed3_b12 sfminit2_b12; do
    conda run -n gspl python tools/binning_and_trim_probe.py --run $RUN --step 60000
  done
} 2>&1 | grep -vE "pkg_resources|declare_namespace|appearance|dataparser|down sample|loading|found " | tee "$R"
echo "報告：$R"
