#!/bin/bash
# 1. blursplit's test (it failed on 2026-08-08 with "Option 'blur_split_weight' is not accepted"
#    -- renaming the field broke configs already written to disk; a deprecated alias now restores
#    loading). results.txt is backed up and restored: it holds the training-time val numbers.
# 2. trimstop: which half of the churn suppresses quality (see run_trimstop_b12.sh).
# 3. blurlas: blur budget + long-axis spread, only after blurbudget has reported alone.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1

while pgrep -f "python -u main.py (fit|test)" > /dev/null; do sleep 120; done
sleep 45

D=outputs/blursplit_b12/blocks/block_12
cp -f $D/results.txt $D/results.txt.bak 2>/dev/null
conda run -n gspl --no-capture-output python -u main.py test \
  --config $D/lightning_logs/version_0/config.yaml --save_val --test_speed \
  > logs/test_blursplit_b12.log 2>&1
rc=$?
mv -f $D/results.txt.bak $D/results.txt 2>/dev/null
n=$(find $D/test -name "*.png" 2>/dev/null | wc -l)
echo "[test] blursplit_b12 rc=$rc 圖 $n 張 $(date +%F_%T)" >> logs/quad_progress.log

bash scripts/run_trimstop_b12.sh
bash scripts/run_blurlas_b12.sh
