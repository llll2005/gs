#!/bin/bash
# 11-16 Fable A-1／A-3 計時版（使用者 11-16「要改 cuda 的都同意」）：複製光柵器到 scratch 目錄、套補丁、在那裡編（不動 env 裡裝的光柵器），
# 用新底 5 的 b1／b4 終點量 forward／backward（只渲染＋backward、不訓練）。nf＝backward 近平面判準改成跟 forward 一樣（BWD_NEAR_LIKE_FWD 1）。
# 用法（lab 佇列，[solo]）：bash tools/probe/skip_timing_1116/run.sh [nf]
case "${1:-}" in -h|--help) sed -n 2,4p "$0"; exit 0 ;; esac
set -u
cd "$(dirname "$0")/../../.." || exit 1
R=$PWD; T=$R/tools/probe/skip_timing_1116
W=${SKIPT_DIR:-$R/outputs/lab/_scratch/rast_skip${1:+_$1}}
G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
ARCH=${TORCH_CUDA_ARCH_LIST:-$(conda run -n gspl python -c "import torch;c=torch.cuda.get_device_capability();print(f'{c[0]}.{c[1]}')" 2>/dev/null | tail -1)}
echo "[skip-timing] scratch $W｜env $G｜arch $ARCH"
rm -rf "$W"; mkdir -p "$(dirname "$W")"; cp -r submodules/diff-surfel-rasterization-trim-pp "$W"; rm -rf "$W/build" "$W"/*.egg-info
(cd "$W/cuda_rasterizer" && "$G/bin/python" "$T/patch_skip.py") || exit 2
[ "${1:-}" = nf ] && sed -i 's/^#define BWD_NEAR_LIKE_FWD 0/#define BWD_NEAR_LIKE_FWD 1/' "$W/cuda_rasterizer/backward.cu"
(cd "$W" && env CUDA_HOME="$G" CUDA_PATH="$G" PATH="$G/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" "$G/bin/python" setup.py build_ext --inplace > build.log 2>&1) \
  || { echo "⛔ 編譯失敗（$W/build.log）"; tail -20 "$W/build.log"; exit 3; }
B=outputs/lab/g44tn_s17g70np16tricap0p5cmflattrimasklowresgeo8log0densecapfc_best1maskdistrfed10b4/blocks
for b in 1 4; do
  CK=$(ls $B/block_$b/checkpoints/*.ckpt | grep -v xyz | tail -1)
  echo "######## b$b $CK"
  PYTHONPATH="$W" "$G/bin/python" -u "$T/time_skip.py" --ckpt "$CK" --nview 24 --rep 3 2>&1 | grep -E "══|forward|N=|光柵器|Error|error"
done
CK=$(ls $B/block_1/checkpoints/*.ckpt | grep -v xyz | tail -1)
echo "######## 梯度差（b1，一個視角）"
PYTHONPATH="$W" "$G/bin/python" -u "$T/grad_diff.py" "$CK" 300 2>&1 | grep -E "｜|Error"
echo "DONE skip-timing"
