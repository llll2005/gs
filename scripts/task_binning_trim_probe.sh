#!/bin/bash
# ★★★★★ binning 外接盒浪費 + trim 成本感知判準（兩個診斷，一次 render loop）
# 判準寫在 tools/binning_and_trim_probe.py 的 docstring 與輸出末尾。
#
# ⚠⚠ 2026-09-13 改為**可指定跑次/步數**。原本寫死 `speed3_b12 / sfminit2_b12 @60000`，
#   而那正是 09-12 得出錯誤結論的原因：
#     ① 那兩個跑次屬於**影像↔姿態錯位**年代（第三次分界，訓練結果全作廢）
#     ② step=60000 在 trim 的作用窗口**之外**（densify_until_iter=30000 就停了）
#     ③ 該時點零貢獻粒子 >= 剪枝比例 => 兩個判準完全並列 => 遮罩重疊**恆為 100%**
#   工具已加退化偵測；這裡把預設改成修正後年代、且落在窗口內的跑次。
# 用法: bash scripts/task_binning_trim_probe.sh [run:step ...]
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
PAIRS=("$@")
[ ${#PAIRS[@]} -eq 0 ] && PAIRS=(gate15000:15000 cs_trimvpc:2000)
R=logs/binning_trim_$(date +%m%d_%H%M).log
{
  for P in "${PAIRS[@]}"; do
    RUN=${P%%:*}; STEP=${P##*:}
    echo "════════ $RUN @ $STEP ════════"
    conda run -n gspl python tools/binning_and_trim_probe.py --run "$RUN" --step "$STEP"
  done
} 2>&1 | grep -vE "pkg_resources|declare_namespace|appearance|dataparser|down sample|loading|found " | tee "$R"
echo "報告：$R"
