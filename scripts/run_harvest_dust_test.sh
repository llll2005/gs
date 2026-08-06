#!/bin/bash
# harvest_dust_trim_interval validation: same config as A' (SB@1.0M, K=1, reconciled
# binary) + periodic dust cull during the harvest phase (>42k). Anchor: A' = 22.12.
# Judgment: (1) PSNR ≈ 22.12 (dust is non-rendering, cull should be lossless),
# (2) harvest-phase N drops (dust removed), (3) it/s up + VRAM down in harvest.
# Dust = (opacity<0.05 AND sub-pixel), pruned every 2000 steps after densify stops.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply
while pgrep -f "python -u main.py fit" > /dev/null; do sleep 300; done
sleep 30
echo "[harvest-dust] b12 SB@1.0M +trim2000 start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sb_b12_cap1m_harvestdust
conda run -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from $PLY \
  --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 1000000 \
  --model.density.init_args.harvest_dust_trim_interval 2000 \
  -n mcmc_sb_b12_cap1m_harvestdust > logs/mcmc_sb_b12_cap1m_harvestdust.log 2>&1
rc=$?
psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_sb_b12_cap1m_harvestdust/blocks/block_12/results.txt 2>/dev/null | head -1)
nfinal=$(grep -oE '\-> N=[0-9]+' logs/mcmc_sb_b12_cap1m_harvestdust.log 2>/dev/null | tail -1)
echo "[harvest-dust] done rc=$rc $psnr $nfinal (A' anchor 22.12) $(date +%F_%T)" >> $PROG
