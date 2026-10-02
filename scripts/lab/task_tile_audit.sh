#!/bin/bash
# ★★★★★ 逐配對工作量的上限量測（2026-10-02 使用者同意；不訓練）：tools/tile_budget_audit.py
#   ① tile 精確剔除（StopThePop）能省多少配對 ② 遮擋尾巴 ③ 誤差引導 tile 抽樣的工作量上限 ④ 完美 tile 之後還完美嗎
#   光柵器 audit kernel（forward.cu auditCUDA）只在另外裝的 logs/rast_audit_pkg 裡，用 PYTHONPATH 指過去 => 不動共用環境。
# 用法：[solo] bash scripts/lab/task_tile_audit.sh [塊=6] [跑次=cs60_conic]
set -u
cd "$(dirname "$0")/../.." || exit 1
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
B=${1:-6}; RUN=${2:-cs60_conic}
D=outputs/${PFX}$RUN/blocks/block_$B/checkpoints
CKS=()
for s in 14999 29999 60000; do c=$(ls $D/*step=$s.ckpt 2>/dev/null | grep -v culldust | head -1); [ -n "$c" ] && CKS+=("$c"); done
[ "${#CKS[@]}" -ge 1 ] || { echo "⛔ $D 沒有 14999/29999/60000 的 ckpt"; exit 2; }
G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
T=logs/rast_audit_pkg; S=logs/rast_audit_src
L=logs/tile_audit_b${B}_${RUN}_$(date +%m%d_%H%M).log
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
{ echo "════ 建 audit 光柵器到 $T（不動共用環境）$(date)"
  rm -rf "$T" "$S"; cp -r submodules/diff-surfel-rasterization-trim-pp "$S"; rm -rf "$S/build"
  conda run -n gspl --no-capture-output env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 \
    pip install --no-build-isolation --no-deps --target "$T" "$S" 2>&1 | tail -2
  echo "════ ckpt：${CKS[*]}"
  conda run -n gspl --no-capture-output env PYTHONPATH="$T" python tools/tile_budget_audit.py --ckpts "${CKS[@]}" --nview 60
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$L"
st=${PIPESTATUS[0]}
echo "報告：$L"
exit "$st"
