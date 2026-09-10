#!/usr/bin/env bash
# 代換 1 的前置（~8 分）：成本**代理**（_max_radii2D²）vs 光柵器的**精確** c_i
# （num_covered_pixels，trim 已在算但丟掉）。
# ⚠ 這個量測有獨立價值：代理若不準，先前所有用代理得到的成本結論（§11.102 的 82% 重疊、
#   §11.28 的 rho(v,c)、退化定理前提）都要打折。
set -euo pipefail
cd "$(dirname "$0")/.."
conda run -n gspl --no-capture-output python -u tools/cost_proxy_quality.py \
  --run agd2_b12 --step 60000 --max-cam 96
