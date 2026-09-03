#!/bin/bash
# main.py test --save_val --test_speed on outputs/$1/blocks/block_$2 (block defaults to 12).
# results.txt / best_val.txt are backed up and restored: they hold the TRAINING-TIME val numbers
# quoted in 紀錄, and a test-split score is a different quantity. An interrupted test would
# otherwise leave the wrong numbers in place (happened to cap4m_b12 on 2026-08-07).
set -u
cd "$(dirname "$0")/.." || exit 1
NAME=$1; BLK=${2:-}
# ⚠ 2026-08-30：原本寫死 `BLK=${2:-12}`，`task_test.sh agd2_b7`（漏第二個參數）
# 就去找不存在的 block_12 => `no config` + exit 1，**0 秒失敗**，而 runner 只記 rc=1
# 往下跑，很容易被當成「跑過了」。現在缺參數就自己推斷（只有一個區塊時）。
if [ -z "$BLK" ]; then
  cand=$(find "outputs/$NAME/blocks" -maxdepth 1 -type d -name 'block_*' 2>/dev/null)
  if [ "$(printf '%s\n' "$cand" | grep -c .)" = "1" ]; then
    BLK=$(basename "$cand" | sed 's/^block_//')
  else
    BLK=12
  fi
fi
D=outputs/$NAME/blocks/block_$BLK
CFG=$(ls -t $D/lightning_logs/version_*/config.yaml 2>/dev/null | head -1)
[ -z "$CFG" ] && { echo "no config for $NAME"; exit 1; }
cp -f $D/results.txt $D/results.txt.bak 2>/dev/null
cp -f $D/best_val.txt $D/best_val.txt.bak 2>/dev/null
conda run -n gspl --no-capture-output python -u main.py test --config "$CFG" --save_val --test_speed
rc=$?
cp -f $D/results.txt $D/results_test.txt 2>/dev/null   # ★ 先把 test 分數另存，再還原訓練期的
mv -f $D/results.txt.bak $D/results.txt 2>/dev/null
mv -f $D/best_val.txt.bak $D/best_val.txt 2>/dev/null
echo "圖 $(find $D/test -name '*.png' 2>/dev/null | wc -l) 張"

# Geometry-side numbers the photometric metrics cannot see. Removing the two geometric priors
# bought PSNR and paid in floaters (0.94% -> 1.19% -> 1.45%), and nothing in results.txt showed it.
# CPU only, a few seconds, so every run gets it for free.
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/geometry_health.py \
  --runs "$NAME" --block "$BLK" || true

exit $rc
