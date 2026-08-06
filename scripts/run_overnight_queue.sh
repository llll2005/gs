#!/bin/bash
# Overnight queue, 2026-07-29. Runs unattended; every stage logs and none of them clobbers data.
#
#   1. densify-blindness diagnostic (2k steps, ~10 min)
#        Measures only -- err_guided_densify stays 0, behaviour is unchanged. MCMC picks split
#        sites with probs = opacity and no error term at all, so the question is how much that
#        costs. Two numbers come out per densify event:
#          err@opacity-sampled / mean   ~1.00 means opacity is blind to error
#          ceiling(top5%) / mean        how much a perfect sampler could win at all
#        The ceiling matters more than the blindness: if error is spread evenly over the frame,
#        no sampler can win much and the mechanism is not worth building.
#
#   2a. build the 4x4 partition (utils/partition_citygs.py)
#        REQUIRED and easy to miss: neither depth_init_blocks.py nor the dataparser creates the
#        partition -- both only READ <dataset>/partition/partitions-dim_<BX>_<BY>_visibility_<t>/.
#        The 2026-07-29 run failed here ("Partition directory not found") because this step was
#        assumed to happen implicitly at data-load time. It does not.
#
#   2b. depth-init for a 4x4 partition (~30 min, CPU/RAM heavy, peak ~11GB RAM)
#        --output_dir is REQUIRED here: the script defaults to <dataset>/depth_init, which
#        already holds the 5x5 PLYs every current experiment initialises from. Writing 4x4 there
#        would silently destroy them.
#        estimated_depths/ is per-image and partition-independent (5621 files, the expensive
#        Depth-Anything pass) -- not regenerated.
#
#   3. one 4x4 block, full 60k
#        4x4 gives 16 bigger blocks instead of our 25, which is the official CityGS layout and
#        therefore the more comparable one. Bigger blocks carry more images and more content, so
#        this is also the OOM question: b12 at 2M peaked at 5.63/6.1G, i.e. the budget is no
#        longer the binding constraint it was, but a 4x4 block is a different shape of load.
#        Recipe is the current best: opacity_reg=0 (reg000, +1.04dB over the L1 recipe) plus the
#        lossless half of EXACT_SUPPORT. cap 1.5M sits between reg000's 0.90M and the 1.80M that
#        peaked at 5.63G, leaving headroom for a block that may hold more content.
#
# The partition itself needs no separate step: block_dim lives in the dataparser config, so
# pointing at the 4x4 config makes it build (and cache) partitions-dim_4_4_visibility_0.08.
#
# START/DONE/DIED per run are written by the training process (internal/callbacks.py); this
# script only records queue-level events. See 紀錄/完整指令手冊.md.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
DATA=data/matrix_city/aerial/train/block_all
CFG4=configs/mcmc_2dgs_sb_60k_aggr17_aerial_4x4.yaml
CFG5=configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml
D4=$DATA/depth_init_4x4
log () { printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "$1" "$2" >> logs/quad_progress.log; }

wait_gpu () {   # nvidia-smi, not pgrep: conda run rewrites cmdline so pgrep never matches
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)" -ge 1500 ]; do
    sleep 120
  done
  sleep 20
}

log "overnight" "QUEUE BATCH START — densify 診斷 → 4x4 depth-init → 4x4 單塊 60k"

# ---- 1. densify-blindness diagnostic — DONE 2026-07-29, mechanism rejected ----
# err@opacity-sampled/mean = 1.000-1.005 (MCMC is indeed blind to error), but
# ceiling(top5%)/mean = 1.18-1.27: the error is spread evenly over the frame, so even a perfect
# sampler could only reach ~20% more error than average. Not worth building. Skipped.

# ---- 2a. build the 4x4 partition -------------------------------------------
P4=$DATA/partition/partitions-dim_4_4_visibility_0.08
if [ -f "$P4/001_001.txt" ]; then
  log "overnight" "QUEUE 4x4 partition 已存在，跳過"
else
  log "overnight" "QUEUE 4x4 partition 建置中"
  conda run --no-capture-output -n gspl python utils/partition_citygs.py \
    --config_path $CFG4 --block_dim 4 4 --force > logs/partition_4x4.log 2>&1
  log "overnight" "QUEUE 4x4 partition rc=$? 產出=$(ls "$P4" 2>/dev/null | wc -l) 個檔"
fi

# ---- 2b. depth-init for the 4x4 partition ----------------------------------
if [ -d "$D4" ] && [ "$(ls "$D4" 2>/dev/null | wc -l)" -ge 16 ]; then
  log "overnight" "QUEUE 4x4 depth-init 已存在（$D4），跳過"
else
  log "overnight" "QUEUE 4x4 depth-init 開始 → $D4（不碰 5x5 的 depth_init/）"
  conda run --no-capture-output -n gspl python utils/depth_init_blocks.py "$DATA" \
    --block_dim 4 4 --voxel_min 0.03 --voxel_max 0.7 --chunk_size 75 \
    --output_dir "$D4" > logs/depth_init_4x4.log 2>&1
  log "overnight" "QUEUE 4x4 depth-init rc=$? 產出=$(ls "$D4" 2>/dev/null | wc -l) 個 PLY"
fi

# ---- 3. one 4x4 block, full 60k --------------------------------------------
# Block 5 is an interior cell of the 4x4 grid (rows/cols 1,1), so it carries real content rather
# than a sparse edge cell -- the useful case to test. If its PLY is missing, stage 2 failed and
# there is nothing to run.
BLK=5
if [ -f "$D4/block_${BLK}.ply" ]; then
  wait_gpu
  rm -rf outputs/b4x4_blk${BLK}_reg000
  conda run --no-capture-output -n gspl python -u main.py fit --config $CFG4 \
    --model.initialize_from "$D4/block_${BLK}.ply" --data.parser.block_id $BLK \
    --model.density.init_args.screen_size_prune_px 300 \
    --model.density.init_args.cap_max 1500000 \
    --model.metric.init_args.opacity_reg 0.0 \
    -n b4x4_blk${BLK}_reg000 > logs/b4x4_blk${BLK}_reg000.log 2>&1
else
  log "overnight" "QUEUE 跳過 4x4 訓練：$D4/block_${BLK}.ply 不存在（depth-init 失敗，看 logs/depth_init_4x4.log）"
fi

log "overnight" "QUEUE BATCH DONE"
