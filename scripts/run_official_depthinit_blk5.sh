#!/bin/bash
# init 方法的單變數 A/B。對照組＝outputs/official_ft_blk5（同 config + coarse init，23.342）。
# 設計理由與對照數據全部寫在 configs/_official_citygsv2_depthinit_sh2_trim.yaml 檔頭。
set -euo pipefail
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
NAME=official_depthinit_blk5
rm -rf "outputs/$NAME"
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/_official_citygsv2_depthinit_sh2_trim.yaml \
  --data.parser.block_id 5 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
