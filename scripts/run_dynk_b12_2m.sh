#!/bin/bash
# Validate predictive dynamic-K: b12 SB @ cap 2.0M with dynamic_strips=on.
# Predicted behavior (CPU-verified in tools/strip_cameras.predict_num_strips):
#   safe phase (<=1.24M): K=1 -> fast; monster regime (~1.9M/424M load): K=5-6
#   -> survives (B' died at fixed K=2). No quality change (K is loss-invariant).
# Same screen_size_prune_px=300 as the dead B' (fair comparison, isolates dynamic K).
# Verdict: (1) reaches 60k with NO OOM (dynamic K tames the monster), (2) faster
# than a hypothetical fixed K=8 (most steps run K=1), (3) PSNR = the count arm the
# ablation needs. Waits for the GPU.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

while pgrep -f "python -u main.py fit" > /dev/null; do sleep 300; done
sleep 30

echo "[dynk] b12 SB@2M dynamic-K start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sb_dynk_b12_cap2m
conda run -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from $PLY \
  --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 2000000 \
  --model.dynamic_strips true \
  --model.strip_max 8 \
  -n mcmc_sb_dynk_b12_cap2m > logs/mcmc_sb_dynk_b12_cap2m.log 2>&1
rc=$?
psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_sb_dynk_b12_cap2m/blocks/block_12/results.txt 2>/dev/null | head -1)
kmax=$(grep -oE '\-> K=[0-9]+' logs/mcmc_sb_dynk_b12_cap2m.log 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
echo "[dynk] done rc=$rc $psnr maxK=$kmax $(date +%F_%T)" >> $PROG
