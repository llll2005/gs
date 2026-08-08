#!/bin/bash
# `main.py test --save_val --test_speed` on lrfloor_b12 and blursplit_b12, once training is done.
#
# results.txt / best_val.txt are backed up and restored: they hold the TRAINING-TIME val numbers
# already quoted in 紀錄 (lrfloor 24.465 / texratio 0.3618), and a test-split score is a different
# quantity. An interrupted run would otherwise leave the wrong numbers in place -- which is exactly
# what happened on 2026-08-07 when cap4m's test was killed mid-way.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1

while pgrep -f "python -u main.py fit" > /dev/null; do sleep 120; done
sleep 60

for NAME in lrfloor_b12 blursplit_b12; do
  D=outputs/$NAME/blocks/block_12
  CFG=$D/lightning_logs/version_0/config.yaml
  if [ ! -f "$CFG" ]; then
    echo "[test] $NAME 無 config，跳過 $(date +%F_%T)" >> logs/quad_progress.log
    continue
  fi
  cp -f $D/results.txt  $D/results.txt.bak  2>/dev/null
  cp -f $D/best_val.txt $D/best_val.txt.bak 2>/dev/null
  conda run -n gspl --no-capture-output python -u main.py test \
    --config "$CFG" --save_val --test_speed > logs/test_$NAME.log 2>&1
  rc=$?
  mv -f $D/results.txt.bak  $D/results.txt  2>/dev/null
  mv -f $D/best_val.txt.bak $D/best_val.txt 2>/dev/null
  echo "[test] $NAME rc=$rc 圖在 $D/test/ $(date +%F_%T)" >> logs/quad_progress.log
done
