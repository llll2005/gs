#!/bin/bash
# 2026-10-10 使用者：「排 塊外成本量測」—— 單塊訓練時合併會丟掉的塊外顆粒佔了多少時間／VRAM／渲染成本（tools/outbox_cost.py）。
#   對象：4x4 的 full44_best（opacity_reg 0.002）與 g44_best0（oreg0；合併時只留 27~35%）b6／b12 的 60k ckpt。
#   變體：全部／盒內（merge 規則）／盒內＋10% margin。只量成本（上界：塊外成本降到 0 時能省多少），不量品質。計時 => [solo]。
# 用法：[solo] bash scripts/lab/task_outbox_cost.sh [跑次...]（預設 full44_best g44_best0）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
RS=("$@"); [ ${#RS[@]} -gt 0 ] || RS=(full44_best g44_best0)
cks=()
for r in "${RS[@]}"; do
  for b in 6 12; do
    c=$(ls outputs/lab/$r/blocks/block_$b/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
    [ -n "$c" ] && cks+=("$c") || echo "⚠ $r block $b 沒有 60k ckpt，跳過"
  done
done
[ ${#cks[@]} -gt 0 ] || { echo "⛔ 沒有可量的 ckpt"; exit 2; }
L=logs/outbox_cost_$(date +%m%d_%H%M).log
conda run -n gspl --no-capture-output python tools/outbox_cost.py --ckpt "${cks[@]}" --margins 0 0.1 --ncam 24 2>&1 \
  | grep -vE "pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample" | tee "$L"
exit "${PIPESTATUS[0]}"
