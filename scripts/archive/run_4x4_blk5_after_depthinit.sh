#!/bin/bash
# Waits for the 4x4 depth-init to finish, then trains block 5 of that partition for 60k.
#
# Why 4x4: it is the official CityGS layout, so 16 bigger blocks instead of our 25. Bigger blocks
# carry more images and more content, which is also the open OOM question -- b12 at 1.80M peaked
# at 5.63/6.1G, so the budget is no longer the binding constraint it was, but a 4x4 block is a
# different shape of load.
#
# Recipe is the current best: opacity_reg=0 (reg000, +1.04 dB over the L1 recipe -- MCMC's opacity
# L1 exists to feed relocation, and our depth-init already solves the initialisation problem that
# relocation is there to fix) plus the lossless half of EXACT_SUPPORT. cap 1.5M sits between
# reg000's 0.90M and the 1.80M that peaked at 5.63G.
#
# cap 2M (2026-07-29): b12 of the 5x5 grid reached 1.80M at 5.63/6.1G, and a 4x4 block holds more
# content than a 5x5 one, so this may not fit. That is the test -- if it OOMs, the ledger records
# the step and N it died at, which is the number worth having.
#
# Waits on the PROCESS, not just on block_5.ply: depth-init writes PLYs one at a time, so the file
# can appear well before the run is done.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
DATA=data/matrix_city/aerial/train/block_all
D4=$DATA/depth_init_4x4
BLK=5
log () { printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "$1" "$2" >> logs/quad_progress.log; }

while pgrep -f "depth_init_blocks.py" > /dev/null; do sleep 60; done

if [ ! -f "$D4/block_${BLK}.ply" ]; then
  log "4x4-blk${BLK}" "QUEUE 取消：$D4/block_${BLK}.ply 不存在（產出 $(ls "$D4" 2>/dev/null | wc -l) 個 PLY，看 logs/depth_init_4x4.log）"
  exit 1
fi
log "4x4-blk${BLK}" "QUEUE depth-init 完成（$(ls "$D4" | wc -l) 個 PLY），開始訓練"

while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)" -ge 1500 ]; do sleep 120; done
sleep 20

rm -rf outputs/b4x4_blk${BLK}_reg000
conda run --no-capture-output -n gspl python -u main.py fit \
  --config configs/mcmc_2dgs_sb_60k_aggr17_aerial_4x4.yaml \
  --model.initialize_from "$D4/block_${BLK}.ply" --data.parser.block_id $BLK \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.cap_max 2000000 \
  --model.metric.init_args.opacity_reg 0.0 \
  -n b4x4_blk${BLK}_reg000 > logs/b4x4_blk${BLK}_reg000.log 2>&1
