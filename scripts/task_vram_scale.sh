#!/usr/bin/env bash
# cap_max 到底能設多高：把粒子複製到目標 N，跑真實訓練步 + trim pass，讓它自己撞牆。
# 起因：vram_gap 量到 allocated 2.42 vs reserved 4.87 GB，而台帳記的是 reserved
# => cap_max 長年照著一個含約 40% 碎片的數字調。
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/vram_scale.py \
  --run agd2_b12 --targets 2.34 2.8 3.2 3.6 4.0 4.4
