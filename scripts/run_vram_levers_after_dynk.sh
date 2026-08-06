#!/bin/bash
# Wait for dynk (or any fit) to free the GPU, then measure the 3-lever VRAM table.
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
while pgrep -f "python -u main.py fit" >/dev/null; do sleep 300; done
sleep 30
echo "[vram-levers] start $(date +%F_%T)" >> logs/quad_progress.log
conda run -n gspl python -u tools/measure_vram_levers.py > logs/vram_levers.log 2>&1
echo "[vram-levers] done $(date +%F_%T)" >> logs/quad_progress.log
grep -E "config|peak|K=|ckpt|OOM" logs/vram_levers.log | tail -8 >> logs/quad_progress.log
