#!/bin/bash
# init 方法的單變數 A/B。對照組＝outputs/official_ft_blk5（同 config + coarse init，23.342）。
# 設計理由與對照數據全部寫在 configs/_official_citygsv2_depthinit_sh2_trim.yaml 檔頭。
set -euo pipefail
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=official_depthinit_blk5
rm -rf "outputs/$NAME"
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/_official_citygsv2_depthinit_sh2_trim.yaml \
  --data.parser.block_id 5 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
