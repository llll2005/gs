#!/bin/bash
# ADMM 統一框架 P1 探針（使用者 10-14；紀錄/實驗分析/14 §P1）：共識約束是不是活的＋用它的成本。只讀 full44_best 16 塊 ckpt，不訓練。
#   A 24 對相鄰區塊在共同視角上的深度不一致（依表面點位置分組：v 盒內／u 盒內／接縫帶／其他）
#   B 同一張影像在多個區塊裡擬合出的單目深度尺度 (a, b) 一致嗎（vs 現行 SfM 一次擬合）
#   C 成本：幾何通道＋深度共識項的每步時間／峰值、每輪 A_v 全視角渲染、ckpt 讀寫、z／u 記憶體
# 用法（佇列）：[solo] bash scripts/lab/task_admm_probe.sh
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=outputs/lab/admm_probe
mkdir -p "$OUT"
conda run -n gspl --no-capture-output python -u tools/admm_probe.py --run outputs/lab/full44_best --out "$OUT" "$@" 2>&1 \
  | grep -v -i -E 'warning|declare_namespace' | tee "$OUT/probe.log"
exit "${PIPESTATUS[0]}"
