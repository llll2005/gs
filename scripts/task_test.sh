#!/bin/bash
# main.py test --save_val --test_speed on outputs/$1/blocks/block_$2 (block defaults to 12).
# results.txt / best_val.txt are backed up and restored: they hold the TRAINING-TIME val numbers
# quoted in 紀錄, and a test-split score is a different quantity. An interrupted test would
# otherwise leave the wrong numbers in place (happened to cap4m_b12 on 2026-08-07).
set -u
cd "$(dirname "$0")/.." || exit 1
NAME=$1; BLK=${2:-12}
D=outputs/$NAME/blocks/block_$BLK
CFG=$(ls -t $D/lightning_logs/version_*/config.yaml 2>/dev/null | head -1)
[ -z "$CFG" ] && { echo "no config for $NAME"; exit 1; }
cp -f $D/results.txt $D/results.txt.bak 2>/dev/null
cp -f $D/best_val.txt $D/best_val.txt.bak 2>/dev/null
conda run -n gspl --no-capture-output python -u main.py test --config "$CFG" --save_val --test_speed
rc=$?
mv -f $D/results.txt.bak $D/results.txt 2>/dev/null
mv -f $D/best_val.txt.bak $D/best_val.txt 2>/dev/null
echo "圖 $(find $D/test -name '*.png' 2>/dev/null | wc -l) 張"
exit $rc
