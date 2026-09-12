#!/bin/bash
# Global coarse pretraining with the UPSTREAM config, as a geometry diagnostic.
#
# The question: our per-block models place the rendered surface at ~0.75x the true depth -- a
# rooftop-height blanket that never resolves the streets -- and the bias is fully formed by step 499
# and barely moves over 60k. Every other candidate explanation has been eliminated by measurement
# (poses, GT-image pairing, exposure, floater occlusion, opacity_reg, merge, colour capacity,
# primitive count, depth-loss weight, depth supervision accuracy, DGD-which-is-not-implemented).
# What remains is that we replaced the upstream global coarse pretraining with per-block depth-init.
#
# So: does globally joint optimisation avoid the blanket on this hardware?
#
# This stops at coarse. Measuring the coarse model's own depth bias answers the question -- if the
# global model's surface sits at the right depth, the blanket comes from our depth-init + per-block
# path; if it does not, the blanket is a property of the budget or the scene, and the project's
# premise needs rethinking. Per-block fine-tuning is skipped, which also removes the stage that
# would actually OOM (uncapped grad-densify, 60k steps, denser gradient per unit area).
#
# NOT a reproduction of the published 27.23. Do not quote its numbers against the paper.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
NAME=official_coarse_sh2
rm -rf outputs/$NAME
conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/_official_citygsv2_mc_aerial_coarse_sh2.yaml \
  -n $NAME > logs/$NAME.log 2>&1
rc=$?
printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "official-coarse" "DONE rc=$rc" >> logs/quad_progress.log
exit $rc
