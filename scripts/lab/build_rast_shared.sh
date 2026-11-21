#!/bin/bash
# ★ 2026-11-21 在佇列裡重編 gspl 共用環境的光柵器（＝lab_pull.sh 第 3 步，但排在佇列裡、等前一個獨佔任務跑完才動）。
#   用法（佇列）：[solo] bash scripts/lab/build_rast_shared.sh
#   為什麼不用 lab_pull.sh 直接編：lab 正在跑要量時間的獨佔任務時，現場編譯會吃 CPU、干擾那個任務的計時。
#   ⚠ 強制 CUDA_HOME（lab 容器預設指向系統 nvcc 12.9，gspl 的 torch 是 cu118）；build/ 先刪（distutils 只看 .cu 的 mtime）；
#     --force-reinstall（同版號的本機路徑 pip 會靜默跳過）。驗過「載入的是 site-packages 那份＋認得新欄位」才寫 .lab_rast_built。
set -u
cd "$(dirname "$0")/../.." || exit 1
case "${1:-}" in -h|--help) sed -n 2,6p "$0"; exit 0 ;; esac
RAST=submodules/diff-surfel-rasterization-trim-pp
G=${CONDA_GSPL:-/root/miniconda3/envs/gspl}
NEW=$(git rev-parse HEAD)
echo "[build_rast_shared] $(date +%T) 編 $(git log --oneline -1 | cut -c1-60)（上次重編 $(cut -c1-9 .lab_rast_built 2>/dev/null)）"
rm -rf "$RAST/build"
conda run -n gspl --no-capture-output env CUDA_HOME=$G CUDA_PATH=$G TORCH_CUDA_ARCH_LIST=8.6 \
  pip install --no-build-isolation --no-deps --force-reinstall "$RAST" 2>&1 | tail -2
rm -rf "$RAST/build"
env -u PYTHONPATH conda run -n gspl --no-capture-output python -c "
import diff_trim_surfel_rasterization as m, os, time
from diff_trim_surfel_rasterization import GaussianRasterizationSettings as S
p = os.path.dirname(m.__file__); so = [x for x in os.listdir(p) if x.endswith('.so')][0]
ok = 'site-packages' in p and all(f in S._fields for f in ('inv_homog', 'exact_strip_cull'))
print('[build_rast_shared] 載入', p, so, time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(os.path.join(p, so)))), '｜新欄位', ok)
raise SystemExit(0 if ok else 1)" \
  && echo "$NEW" > .lab_rast_built && echo "[build_rast_shared] ✅ 完成（.lab_rast_built = ${NEW:0:9}）"
