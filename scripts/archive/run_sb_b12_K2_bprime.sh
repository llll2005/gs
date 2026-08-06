#!/bin/bash
# B' — completes the same-kernel count ablation that arm B's OOM interrupted:
# trim+SB @ cap 2.0M with K=2 strip training (the budget formula's supply lever,
# now on the main-line kernel). Arm B died at ~30k / 5.15G in backward; K=2 bounds
# the rasterizer/activation footprint to ~1/2. Money shot if it survives: formula
# predicted the K=1 ceiling (~1.4-1.5M practical), K=2 rescues 2.0M on the kernel
# that holds our best numbers. Verdict vs A (SB@1.0M = 21.99).
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
PLY=data/matrix_city/aerial/train/block_all/depth_init/block_12.ply

while pgrep -f "main.py fit" > /dev/null; do sleep 300; done
sleep 30

echo "[bprime] SB@2.0M K=2 start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sb_60k_aggr17_b12_cap2m_K2
conda run -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml \
  --model.initialize_from $PLY \
  --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 2000000 \
  --model.train_strips 2 \
  -n mcmc_sb_60k_aggr17_b12_cap2m_K2 > logs/mcmc_sb_60k_aggr17_b12_cap2m_K2.log 2>&1
rc=$?
psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_sb_60k_aggr17_b12_cap2m_K2/blocks/block_12/results.txt 2>/dev/null | head -1)
echo "[bprime] done rc=$rc $psnr (vs A 21.99) $(date +%F_%T)" >> $PROG
