#!/bin/bash
# ★ 2026-11-21 把目前 repo 的光柵器編到「另一個目錄」，不動 gspl 環境裡共用的那份（紀錄/實驗分析/39 的 lab 驗證用）。
#   用法：bash scripts/lab/build_rast_alt.sh <標籤>        => $HOME/rast_alt/<標籤>/diff_trim_surfel_rasterization
#   用的時候在指令前加 PYTHONPATH=$HOME/rast_alt/<標籤>（PYTHONPATH 排在 site-packages 前面；conda run 會沿用）。
#   為什麼不直接重編共用環境：佇列裡還有要「獨佔量時間」互比的跑次（全場景消融、CityGS 計時），
#     它們前半段是舊 .so 跑的 => 中途換 .so 會讓時間不可比（旗標關時渲染逐位元相同，但 kernel 多了共用記憶體與分支）。
#   ⚠ 重編要強制 CUDA_HOME（lab 容器預設指向系統 nvcc 12.9，gspl 的 torch 是 cu118）；build/ 先刪（distutils 只看 .cu 的 mtime）。
#   之後正式用 lab_pull.sh 同步時，.lab_rast_built 還是舊的 => 會照常重編共用環境（這支不寫那個檔）。
set -u
cd "$(dirname "$0")/../.." || exit 1
case "${1:-}" in -h|--help|"") sed -n 2,9p "$0"; exit 0 ;; esac
TAG=$1
RAST=submodules/diff-surfel-rasterization-trim-pp
DST=$HOME/rast_alt/$TAG
G=${CONDA_GSPL:-/root/miniconda3/envs/gspl}
rm -rf "$RAST/build" "$DST"; mkdir -p "$DST"
echo "[build_rast_alt] $(date +%T) 編 $(git log --oneline -1 | cut -c1-60) => $DST"
# ★ 11-21 晚：RAST_F32_CONST=1 bash scripts/lab/build_rast_alt.sh <標籤> => 編 float 常數版（auxiliary.h；紀錄/實驗分析/39 #12）
echo "[build_rast_alt] RAST_F32_CONST=${RAST_F32_CONST:-0}"
conda run -n gspl --no-capture-output env CUDA_HOME=$G CUDA_PATH=$G TORCH_CUDA_ARCH_LIST=8.6 RAST_F32_CONST=${RAST_F32_CONST:-0} \
  pip install --no-build-isolation --no-deps --target "$DST" "$RAST" 2>&1 | tail -3
rm -rf "$RAST/build"
# 驗：用 PYTHONPATH 載到的是這一份、而且認得新欄位；過了才寫 .built_from（佇列裡用它當「編好了」的前提）
PYTHONPATH=$DST conda run -n gspl --no-capture-output python -c "
import diff_trim_surfel_rasterization as m, os
from diff_trim_surfel_rasterization import GaussianRasterizationSettings as S
p = os.path.dirname(m.__file__); ok = p.startswith('$DST')
print('[build_rast_alt] 載入', p, '｜新欄位', all(f in S._fields for f in ('inv_homog', 'exact_strip_cull')))
raise SystemExit(0 if ok and all(f in S._fields for f in ('inv_homog', 'exact_strip_cull')) else 1)" \
  && { git rev-parse HEAD; echo "RAST_F32_CONST=${RAST_F32_CONST:-0}"; } > "$DST/.built_from" && echo "[build_rast_alt] ✅ 完成（$DST/.built_from）"
