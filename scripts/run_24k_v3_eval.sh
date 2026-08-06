#!/bin/bash
# Evaluate a 24k-aggressive block against the 60k reference on the same block.
# Usage: run_24k_aggr_eval.sh <block_id>
#
# Separate from run_24k_aggr_block.sh on purpose: that script is already executing when this gets
# written, and bash reads scripts incrementally, so editing a running one is a good way to get
# undefined behaviour. The queue runs tasks in order, so a follow-up task is the safe way to append.
#
# Three numbers, in increasing order of how much they can be trusted:
#   1. main.py test  -- val PSNR/SSIM/LPIPS. split_mode=reconstruction means val is a SUBSET OF
#      TRAIN, so this is not held-out and cannot go in a paper. It is still the yardstick every
#      earlier run was measured with, so it is what makes "24k vs 60k on block 12" comparable.
#   2. rescore_by_content -- splits views into water-heavy and building-heavy by GT gradient. b12 is
#      roughly half flat water, where a near-uniform render still scores 32-40 dB, and that dilution
#      is what hid a 4x texture difference behind a flat LPIPS line for two months. The building
#      group is the one that means anything.
#   3. depth bias -- already reported by the fit script.
#
# The reference to beat, same block, 60k, cap 2M: val PSNR 23.39 / SSIM 0.667 / LPIPS 0.491,
# slope 1.082, corr 0.903.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
BLK=${1:?用法: run_24k_aggr_eval.sh <block_id>}
NAME=aggr24k_v3_b${BLK}
CFG=$(find outputs/$NAME -name config.yaml | head -1)
[ -n "$CFG" ] || { echo "找不到 $NAME 的 config，fit 可能沒跑完"; exit 1; }

echo "===== 1. val 指標（val ⊂ train，僅供與 60k 內部比較）====="
conda run --no-capture-output -n gspl python main.py test --config "$CFG" --save_val 2>&1 \
  | grep -vE "pkg_resources|__import__|Warning" | tail -20

echo ""
echo "===== 2. 內容分組（水面 / 建築）＋紋理比 ====="
conda run --no-capture-output -n gspl python tools/rescore_by_content.py --block "$BLK" \
  --runs $NAME aggr24k_b12 b12_cap2m_reg000_exact oreg_0p002_b12 \
  --known_bad mcmc_sb_b12_cap1m_condense 2>&1 \
  | grep -vE "pkg_resources|__import__|Warning"

printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "aggr24k-eval" "BLOCK $BLK done" >> logs/quad_progress.log
