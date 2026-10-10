#!/bin/bash
# 2026-10-10 塊外池（internal/utils/outside_pool.py）上 lab 前的行為檢查：本機（舊光柵器）只驗得到一般路徑，
#   這裡在 lab 的光柵器上把 lean 路徑（lean＋record_reduce＋backward 飽和跳過／先加總）也驗一次（tools/check_outside_pool.py）。
#   對象：4x4 g44_best0（沒有就用 full44_best）的 block 6 60k ckpt。判定：兩條路徑都印「全部通過」才 rc=0。
# 用法：[solo] bash scripts/lab/task_pool_check.sh
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
CK=$(ls outputs/lab/g44_best0/blocks/block_6/checkpoints/*step=60000.ckpt outputs/lab/full44_best/blocks/block_6/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "⛔ 找不到 4x4 block 6 的 60k ckpt"; exit 2; }
L=logs/pool_check_$(date +%m%d_%H%M).log
bad=0
for M in "" --lean; do
  echo "════ ${M:-一般路徑} ：$CK"
  conda run -n gspl --no-capture-output python tools/check_outside_pool.py --ckpt "$CK" $M 2>&1 \
    | grep -vE "pkg_resources|declare_namespace|WARNING|找不到深度圖" | tail -20
  [ "${PIPESTATUS[0]}" = 0 ] || bad=1
done 2>&1 | tee "$L"
grep -c "全部通過" "$L" | grep -qx 2 || { echo "⛔ 沒有兩條路徑都通過（見 $L）"; exit 1; }
echo "✅ 兩條路徑都通過"
