#!/bin/bash
# 等一個檔案出現（背景 CPU 工作的完成標記；lab.py bg 跑的前置工作寫在 ../labrun/<名稱>.done）。逾時 => rc 3（後面依賴它的任務會自己報缺檔）
# 用法（佇列）：[cpu] bash scripts/lab/wait_marker.sh <檔案> [逾時秒數，預設 7200]
case "${1:-}" in -h|--help|"") sed -n 2,3p "$0"; exit 2 ;; esac
f=$1; t=${2:-7200}; s=0
while [ ! -f "$f" ]; do sleep 20; s=$((s + 20)); [ "$s" -ge "$t" ] && { echo "⛔ 等 $f 逾時 ${t}s"; exit 3; }; done
echo "✅ $f 已出現（等了 ${s}s）"; cat "$f"
