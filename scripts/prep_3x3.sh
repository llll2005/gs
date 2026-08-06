#!/bin/bash
# Prepare a 3x3 partition + depth-init while a training run is in flight.
#
# Safe to run concurrently: partition_from_colmap.py pins device to CPU (line 165) and
# depth_init_blocks.py never touches CUDA at all. Depth-init is RAM-heavy though (~11GB peak of
# our 46GB), which is why it is worth doing now rather than alongside a second trainer.
#
# Why 3x3 and not 2x2: VRAM tracks primitive count, and count needed tracks area. A 4x4 block
# currently sits at 1.89M / 3.66G. A 3x3 block covers 16/9 = 1.78x that area, so ~3.4M primitives
# at equal density -- right at the measured ceiling (~2.5M on b12-like content, more on lighter
# blocks). A 2x2 block covers 4x the area, ~7.6M, which is 2-3x past the ceiling. Not reachable.
#
# The real prize for fewer blocks is not VRAM but SEAMS: merged quality measured 7.0 dB below
# single-block (consolidation heals ~51%). 25 blocks -> 16 -> 9 cuts the seam count from 40 to 24
# to 12, and that shows up in the global number the north-star comparison actually uses.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
DATA=data/matrix_city/aerial/train/block_all
log () { printf '%s | %-24s | %s\n' "$(date '+%m-%d %H:%M')" "prep-3x3" "$1" >> logs/quad_progress.log; }

log "QUEUE START — 3x3 partition + depth-init（純 CPU：partition 的 device 寫死 cpu，depth-init 無 cuda 字樣）"

conda run --no-capture-output -n gspl python utils/partition_from_colmap.py "$DATA" \
  --block_dim 3 3 --content_threshold 0.08 --force > logs/partition_3x3.log 2>&1
rc=$?
P3=$DATA/partition/partitions-dim_3_3_visibility_0.08
log "QUEUE partition rc=$rc 產出=$(ls "$P3" 2>/dev/null | wc -l) 個檔"
[ $rc -ne 0 ] && { log "QUEUE 中止：partition 失敗，看 logs/partition_3x3.log"; exit 1; }


conda run --no-capture-output -n gspl python utils/depth_init_blocks.py "$DATA" \
  --block_dim 3 3 --voxel_min 0.03 --voxel_max 0.7 --chunk_size 75 \
  --output_dir "$DATA/depth_init_3x3" > logs/depth_init_3x3.log 2>&1
log "QUEUE depth-init rc=$? 產出=$(ls "$DATA/depth_init_3x3" 2>/dev/null | wc -l)/9 個 PLY（GPU 現為 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)MiB）"
