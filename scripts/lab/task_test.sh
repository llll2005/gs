#!/bin/bash
# 官方 held-out 評測（741 幀裡落在該塊訓練相機範圍內的那些）。
# 用法：task_test.sh <run_name> <block_id>
# ⚠ 官方協定評的是**合併後**的 25 塊（§16.12.1）；單塊是**下界**，只能互相比。
source "$(dirname "$0")/_common.sh"
RUN=${1:?用法: task_test.sh <run> <blk>}; BLK=${2:?}
CK=$(ls -t "outputs/$RUN/blocks/block_$BLK/checkpoints/"*.ckpt 2>/dev/null | head -1)
[ -z "$CK" ] && { echo "❌ 找不到 $RUN block $BLK 的 ckpt"; exit 1; }
echo "=== held-out: $RUN block $BLK  ($(basename "$CK")) ==="
conda run -n gspl python tools/eval_official_test.py --ckpt "$CK" --block "$BLK" 2>&1 | tail -20
