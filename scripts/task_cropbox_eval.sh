#!/bin/bash
# ★★★★★ 裁掉 SfM 盒外的粒子：畫面代價 vs binning 省下多少（~8 分，純診斷不介入）
# 判準與完整推理見 tools/crop_box_eval.py 的 docstring。
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
R=logs/cropbox_$(date +%m%d_%H%M).log
conda run -n gspl python tools/crop_box_eval.py --runs speed3_b12 sfminit2_b12 2>&1 \
  | grep -vE "pkg_resources|declare_namespace|appearance|dataparser|down sample|loading|found " | tee "$R"
echo "報告：$R"
