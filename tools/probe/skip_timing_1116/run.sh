#!/bin/bash
# 11-16 A-1／A-3 正式版計時（光柵器的 skip_box／pair_bit_warp 已進 submodule；用裝在 env 裡的那份，靠 renderer 開關切換、同一個 .so 對照）：
#   新底 5 b1／b4 終點，一般步（增生期／收割期）與完整路徑步，只渲染＋backward、不訓練。
#   （早先的 scratch 補丁流程＝patch_skip.py＋time_skip.py＋CITYGS_SKIPF，留著當紀錄；原始碼已含補丁，不能再套一次）
# 用法（lab 佇列，[solo]）：bash tools/probe/skip_timing_1116/run.sh
case "${1:-}" in -h|--help) sed -n 2,5p "$0"; exit 0 ;; esac
set -u
cd "$(dirname "$0")/../../.." || exit 1
T=tools/probe/skip_timing_1116
G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
"$G/bin/python" -c "import diff_trim_surfel_rasterization as m; F=m.GaussianRasterizationSettings._fields; assert 'skip_box' in F and 'pair_bit_warp' in F, '光柵器沒重編（缺 skip_box／pair_bit_warp）'; print('光柵器', m.__file__)" || exit 2
B=outputs/lab/g44tn_s17g70np16tricap0p5cmflattrimasklowresgeo8log0densecapfc_best1maskdistrfed10b4/blocks
for b in 1 4; do
  CK=$(ls $B/block_$b/checkpoints/*.ckpt | grep -v xyz | tail -1)
  echo "######## b$b $CK"
  "$G/bin/python" -u "$T/time_prod.py" --ckpt "$CK" --nview 24 --rep 3 2>&1 | grep -E "══|forward|N=|Error|error"
done
echo "DONE skip-timing"
