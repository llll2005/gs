#!/usr/bin/env bash
# 兩個「冗餘」候選各值多少 ms/step（CUDA event 精確歸因，~3 分鐘）
#   1 `_accumulate_error_score`：after_backward 裡唯一**無守衛、每步跑**的；消費者現行全為 0
#   2 MCMC 噪音稀疏化：gate 對高 opacity 本來就 ~0，實測可跳過 69%（60k ckpt）
#   3 順帶量 `fast_noise`（已實作但從沒開過）
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/knob_cost.py --run agd2_b12
