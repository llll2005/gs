#!/bin/bash
# Merge whatever 3x3 blocks exist and score the merged scene on the OFFICIAL held-out test set.
#
# Runs twice in the queue: once after the cross (blocks 1,3,5,7 + the existing 4) so a contiguous
# five-block region gives an early answer roughly two days sooner, and again after the corners.
# Merging is cheap and non-destructive -- the per-block checkpoints are untouched, so re-merging
# with different post-processing costs nothing and never requires retraining.
#
# cull_dust is applied to a COPY. It was verified bit-identical at 7.4x pruning, and the merged
# scene is where dust matters most: opacity_reg=0 suppresses nothing, so each block contributes its
# own floaters and they accumulate. If the merged render is fog, compare the two scores below
# before blaming the recipe.
set -u
# pipefail: every eval below is piped through grep|tee, and a pipeline's exit status is its LAST
# command's. Without this the 2026-08-03 run reported rc=0 to the queue while BOTH evals had died
# on a traceback -- a silent failure is worse than a loud one, because the queue moved on and the
# next block occupied the card for eight hours before anyone looked.
set -o pipefail
rc_any=0
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
NAME=b3x3_full
BLK4=outputs/b3x3_blk4_reg000/blocks/block_4

# Block 4 trained before this queue existed, into its own directory. Link it in so the merge sees
# all blocks; a symlink keeps the original run intact and re-runnable.
mkdir -p outputs/$NAME/blocks
[ -e "outputs/$NAME/blocks/block_4" ] || [ ! -d "$BLK4" ] || \
  ln -s "$(realpath $BLK4)" "outputs/$NAME/blocks/block_4"

echo "===== 合併 $(ls -d outputs/$NAME/blocks/block_*/ 2>/dev/null | wc -l) 塊  $(date '+%F %T') ====="
ls -d outputs/$NAME/blocks/block_*/ 2>/dev/null | xargs -n1 basename
conda run --no-capture-output -n gspl python utils/merge_citygs_ckpts.py outputs/$NAME || exit 1

MERGED=$(find outputs/$NAME -maxdepth 2 -name "*merged*.ckpt" 2>/dev/null | head -1)
[ -z "$MERGED" ] && MERGED=$(find outputs/$NAME/checkpoints -name "*.ckpt" 2>/dev/null | head -1)
[ -z "$MERGED" ] && { echo "找不到合併後的 ckpt"; exit 1; }
echo "合併結果: $MERGED"

OUT=紀錄/official_test_merged.txt
{
  echo "===== 合併模型 官方 held-out test（741 幀全集）  $(date '+%F %T') ====="
  echo "區塊: $(ls -d outputs/$NAME/blocks/block_*/ | xargs -n1 basename | tr '\n' ' ')"
} >> $OUT

echo "---- 合併模型（未清塵）----" | tee -a $OUT
conda run --no-capture-output -n gspl python tools/eval_official_test.py \
  --ckpt "$MERGED" --offset 0 --save_dir outputs/$NAME/official_test_vis 2>&1 \
  | grep -vE "^\s*\.\.\.|Warning|warn" | tee -a $OUT || rc_any=1

CULLED="${MERGED%.ckpt}_culled.ckpt"
conda run --no-capture-output -n gspl python tools/cull_dust.py "$MERGED" "$CULLED" 2>&1 | tail -3
if [ -f "$CULLED" ]; then
  echo "---- 合併模型（cull_dust 後）----" | tee -a $OUT
  conda run --no-capture-output -n gspl python tools/eval_official_test.py \
    --ckpt "$CULLED" --offset 0 2>&1 | grep -vE "^\s*\.\.\.|Warning|warn" | tee -a $OUT || rc_any=1
fi
echo "[彙整] $OUT"
[ "$rc_any" -eq 0 ] || { echo "⚠ 至少一次評測失敗（見上方 traceback）"; exit 1; }
