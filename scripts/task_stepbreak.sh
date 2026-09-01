#!/bin/bash
# ★★★ 穩態每步成本拆解（~2 分鐘）。不訓練、不寫 outputs，只載 ckpt 做微基準。
set -u
cd "$(dirname "$0")/.." || exit 1
conda run -n gspl --no-capture-output python tools/step_breakdown.py --run agd2_b12 --repeat 20
