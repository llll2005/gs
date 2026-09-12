#!/bin/bash
# ★ 把本機的 speed3 / block 13 從 outputs/lab/ 搬回正確的命名空間
#
# 為什麼會跑錯地方：scripts/lab/_common.sh 的 RUN_PREFIX 預設寫死 `lab/`，
# 2026-09-13 01:16 在本機啟動時沒帶 `RUN_PREFIX=` ⇒ 跑次名變成 `lab/speed3`。
# 而 outputs/lab/ 是「outputs/ 所有 ckpt 都這台 6GB 跑的」鐵律的**唯一例外區**，
# 來源混進去就再也分不出是哪台機器 ⇒ 分數作廢級的汙染。
# （判準已改成從機器判，不會再發生；但這一個跑次已經在裡面了。）
#
# ⚠ 跑動中**不能搬**（internal/cli.py 會 `Parent directory does not exist`，
#   2026-09-12 已經這樣弄死過一個 lab 跑次）⇒ 本腳本自己驗完賽才動手，
#   而且這一行**沒有 [cpu] 前綴**，所以 runner 會先等 GPU 空出來（＝訓練結束）。
set -u
cd "$(dirname "$0")/.." || exit 1
SRC=outputs/lab/speed3/blocks/block_13
DST=outputs/speed3/blocks/block_13
[ -d "$SRC" ] || { echo "⚠ $SRC 不存在（也許已經搬過了）"; exit 0; }
[ -e "$DST" ] && { echo "⛔ $DST 已存在，不覆蓋。請人工確認"; exit 1; }

# 驗完賽①：有 step>=59000 的 ckpt
LAST=$(ls "$SRC"/checkpoints/*.ckpt 2>/dev/null | sed 's/.*step=//;s/\.ckpt//' | sort -n | tail -1)
[ -z "$LAST" ] && { echo "⛔ 找不到任何 ckpt"; exit 1; }
if [ "$LAST" -lt 59000 ]; then
  echo "⛔ 最後的 ckpt 只到 step=$LAST（<59000）=> 跑次沒完賽，不搬"
  exit 1
fi
# 驗完賽②：train_status.txt 已經 300 秒沒動（＝沒有行程還在寫）
AGE=$(( $(date +%s) - $(stat -c %Y "$SRC/train_status.txt" 2>/dev/null || echo 0) ))
if [ "$AGE" -lt 300 ]; then
  echo "⛔ train_status.txt 只有 ${AGE}s 前才更新 => 可能還有行程在寫，不搬"
  exit 1
fi
echo "完賽確認：最後 ckpt step=$LAST，status 已靜止 ${AGE}s"

mkdir -p outputs/speed3/blocks || exit 1
mv "$SRC" "$DST" || { echo "⛔ 搬移失敗"; exit 1; }
echo "✔ $SRC -> $DST"
rmdir outputs/lab/speed3/blocks outputs/lab/speed3 2>/dev/null   # 只有空目錄才會成功
echo "=== 搬完後的狀態 ==="
conda run -n gspl python tools/run_status.py 2>&1 | grep -iE 'speed3|step|psnr' | head -10
echo "=== results.txt ==="
cat "$DST/results.txt" 2>/dev/null | head -12
