#!/bin/bash
# Score existing checkpoints on the OFFICIAL MatrixCity held-out test set (741 views, prepared and
# unused since April). Every number we have so far comes from split_mode=reconstruction, where val
# is a SUBSET OF TRAIN -- so none of them are held-out and all of them flatter us. This is the first
# honest measurement, and it recalibrates the whole table.
#
# Inference only: loads a checkpoint, renders, scores. No training, no optimizer. Peak is whatever a
# single forward costs (~1.5-2 GiB at 1.8M primitives), so it is short, but it still needs the card
# and therefore goes through the queue like everything else.
#
# The runs picked are the ones that answer a question:
#   b12 x3 -- the opacity_reg sweep on the current binary. On val (val subset of train) reg=0 and
#             reg=0.002 look equal and reg=0.007 looks clearly worse; held-out may disagree, since
#             opacity_reg's whole job is suppressing floaters that val cannot see.
#   b7  x2 -- our best run vs the self-run CityGSV2 baseline, same block, same data. On val we lead
#             by 2.7 dB / -0.375 LPIPS. If that lead survives held-out it is a real result; if it
#             collapses, the lead was val-overfitting and we need to know now.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
OUT=紀錄/official_test_results.txt
{
  echo "===== 官方 held-out test 評測  $(date '+%F %T') ====="
  echo "說明：單塊模型只在「該塊訓練相機 AABB 內的測試視角」上評分，是下界"
  echo "（官方協定評的是合併後的 25 塊模型，這裡的塊邊界外內容根本不存在）"
} > "$OUT"

run () {   # run <name> <ckpt> <block>
  echo "" | tee -a "$OUT"
  echo "---------- $1 (block $3) ----------" | tee -a "$OUT"
  conda run --no-capture-output -n gspl python tools/eval_official_test.py \
    --ckpt "$2" --block "$3" --save_dir "outputs/$1/official_test_vis" 2>&1 \
    | grep -vE "^\s*\.\.\.|Warning|warn" | tee -a "$OUT"
}

run b12_cap2m_reg000_exact       outputs/b12_cap2m_reg000_exact/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt 12
run oreg_0p0_b12                 outputs/oreg_0p0_b12/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt 12
run oreg_0p007_b12               outputs/oreg_0p007_b12/blocks/block_12/checkpoints/epoch=212-step=60000.ckpt 12
run mcmc_60k_sh3_aggr17_prune_b7 outputs/mcmc_60k_sh3_aggr17_prune_b7/blocks/block_7/checkpoints/epoch=313-step=60000.ckpt 7
run citygsv2_b7_faithful         outputs/citygsv2_b7_faithful/blocks/block_7/checkpoints/epoch=313-step=60000.ckpt 7

echo "" >> "$OUT"
echo "[彙整] $OUT"
