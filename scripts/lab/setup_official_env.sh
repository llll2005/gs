#!/usr/bin/env bash
# 官方參考線的**乾淨**環境 gspl_official（2026-09-24，使用者要求）。
#
# 規則（使用者 2026-09-24 原話）：「使用官方要求的/製作所有程式與submodule，
#   但我方使用的log/計數器/需要資料的紀錄等等不在此限」。
#
# ⇒ 套件**完全照** cityGS_origin/doc/installation.md 的三行：
#      pip install -r requirements/pyt201_cu118.txt
#      pip install -r requirements.txt
#      pip install -r requirements/CityGS.txt
#   Depth-Anything-V2 照 doc/data_preparation.md：git clone 官方 repo ＋ 從 HF 下載 vitl 權重。
#
# 為什麼不沿用 gspl_origin：它是用**我方** bootstrap.sh 建的 ⇒ 套件版本來自我方 lock 檔，
#   diff_gaussian_rasterization / simple_knn 是從我方 submodules/ 編的；只有光柵器換成官方。
#   而 09-22 的 coarse 甚至是跑在 gspl（lab 從沒 pull 到切換環境的 commit）。
#
# 只加了「讓官方指令能在這個容器裡執行」的基礎建設，**都不改任何套件的原始碼或版本**：
#   - nvcc 11.8 與 gcc 11 裝進 env（官方假設系統就有 CUDA 11.8；lab 系統是 12.9 / gcc 13）
#   - activate.d 清掉容器塞進 LD_LIBRARY_PATH 的系統 torch（否則 torch._C 載錯 .so）
#   - pip 的 --no-build-isolation（git+ 的 CUDA 擴充要看得到已裝的 torch 才編得起來）
#   - TORCH_CUDA_ARCH_LIST 設成這張卡（容器預設含 10.0，torch 2.0.1 不認得）
#
# 用法：bash scripts/lab/setup_official_env.sh     （可重跑；完成後寫 logs/citygs_official_env.txt）
set -uo pipefail
ENV_NAME=${CITYGS_OFF_ENV:-gspl_official}
ORIG=/workspace/data/hdd/11213/cityGS_origin
GS=/workspace/data/hdd/11213/gs
REPORT=$GS/logs/citygs_official_env.txt
say()  { printf '\n=== %s ===\n' "$*"; }
die()  { echo "⛔ $*"; exit 1; }

export CONDA_NO_PLUGINS=true CONDA_SOLVER=classic
unset PIP_CONSTRAINT
_out=""; IFS=: read -ra _ps <<< "${LD_LIBRARY_PATH:-}"
for _p in "${_ps[@]:-}"; do case "$_p" in *dist-packages/torch/lib|*dist-packages/torch_tensorrt/lib|"") ;; *) _out="${_out:+$_out:}$_p";; esac; done
export LD_LIBRARY_PATH="$_out"

cd "$ORIG" || die "找不到 $ORIG"
say "0. cityGS_origin 必須是未修改的官方原始碼"
git fetch -q origin 2>/dev/null
echo "HEAD $(git rev-parse --short HEAD)   origin/main $(git rev-parse --short origin/main 2>/dev/null)"
DIRTY=$(git status --porcelain --untracked-files=no)
[ -z "$DIRTY" ] || { echo "$DIRTY"; die "cityGS_origin 有被改過的追蹤檔"; }
echo "✅ 追蹤檔零改動（未追蹤：$(git status --porcelain | grep '^??' | awk '{print $2}' | tr '\n' ' ')）"

say "1. conda env $ENV_NAME（python 3.9，照 doc/installation.md）"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "已存在，沿用（要全新重來：conda env remove -n $ENV_NAME）"
else
  conda create -yn "$ENV_NAME" python=3.9 pip || die "建環境失敗"
fi
PREFIX=$(conda run -n "$ENV_NAME" python -c "import sys;print(sys.prefix)" | tr -d '\r')
[ -n "$PREFIX" ] || die "拿不到 prefix"
mkdir -p "$PREFIX/etc/conda/activate.d"
cat > "$PREFIX/etc/conda/activate.d/00_sanitize_ld.sh" <<'EOF'
_out=""; IFS=: read -ra _ps <<< "${LD_LIBRARY_PATH:-}"
for _p in "${_ps[@]:-}"; do case "$_p" in *dist-packages/torch/lib|*dist-packages/torch_tensorrt/lib|"") ;; *) _out="${_out:+$_out:}$_p";; esac; done
export LD_LIBRARY_PATH="$_out"; unset _out _ps _p
EOF

say "1b. 建置工具鏈（nvcc 11.8、gcc 11；只用來編官方指定的 CUDA 擴充）"
[ -x "$PREFIX/bin/nvcc" ] || conda install -y -n "$ENV_NAME" -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 || die "nvcc 11.8 裝不起來"
[ -x "$PREFIX/bin/nvcc" ] || die "沒有 $PREFIX/bin/nvcc"
ls "$PREFIX"/bin/*-linux-gnu-g++ >/dev/null 2>&1 || conda install -y -n "$ENV_NAME" -c conda-forge gcc_linux-64=11 gxx_linux-64=11 || die "gcc 11 裝不起來"
CC=$(ls "$PREFIX"/bin/*-linux-gnu-gcc | head -1); CXX=$(ls "$PREFIX"/bin/*-linux-gnu-g++ | head -1)
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' \r')
BUILD_ENV="CUDA_HOME=$PREFIX CUDA_PATH=$PREFIX TORCH_CUDA_ARCH_LIST=$ARCH CC=$CC CXX=$CXX NVCC_PREPEND_FLAGS=-ccbin=$CXX"
echo "$BUILD_ENV"
pipi () { conda run -n "$ENV_NAME" --no-capture-output env $BUILD_ENV pip install --no-build-isolation "$@"; }

say "2. 官方三行 requirements（順序照 doc/installation.md）"
conda run -n "$ENV_NAME" --no-capture-output pip install -r requirements/pyt201_cu118.txt || die "pyt201_cu118 失敗"
pipi -r requirements.txt || die "requirements.txt 失敗"
pipi -r requirements/CityGS.txt || die "CityGS.txt 失敗"
# ⚠ 2026-09-24 第一次建環境撞到：官方 requirements **沒有釘 numpy** => 今天 pip 解出 numpy 2.0.2，
#   而 torch 2.0.1 是對 numpy 1.x 編的：`Failed to initialize NumPy: _ARRAY_API not found`、
#   torch.from_numpy 直接失敗 => 官方程式碼跑不起來。釘 numpy<2（1.26.4）同樣滿足官方所有 requirements
#   （官方沒指定版本），只是選了 torch 2.0.1 能用的那一個；官方發布時 numpy 2 尚未存在。
conda run -n "$ENV_NAME" --no-capture-output pip install "numpy<2" || die "numpy<2 失敗"
# 官方 requirements 自己同時列了兩個 diff-gaussian-rasterization（lightning23 -> common.txt 的 graphdeco@59f5f77、
# CityGS.txt 的 DekuLiuTesla@82c31d7），後裝的蓋前面 ⇒ 最終是 DekuLiuTesla 那份，這也是照官方指令的結果。

say "3. Depth-Anything-V2（照 doc/data_preparation.md）"
DA=utils/Depth-Anything-V2
if [ -L "$DA" ]; then rm -f "$DA"; echo "移除先前指向 gs 的符號連結"; fi
[ -d "$DA/.git" ] || git clone https://github.com/DepthAnything/Depth-Anything-V2 "$DA" || die "clone DA-V2 失敗"
mkdir -p "$DA/checkpoints"
W="$DA/checkpoints/depth_anything_v2_vitl.pth"
if [ ! -s "$W" ]; then
  wget -q -O "$W" "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" \
    || { rm -f "$W"; echo "⚠ HF 下載失敗 => 複製 gs 裡那份（同一個官方權重檔，sha256 會寫進報告）"; \
         cp "$(find "$GS/utils/Depth-Anything-V2/checkpoints" -name 'depth_anything_v2_vitl.pth' | head -1)" "$W" || die "沒有權重"; }
fi

say "4. OpenCV 在 headless 容器"
if ! conda run -n "$ENV_NAME" python -c "import cv2" >/dev/null 2>&1; then
  echo "⚠ import cv2 失敗 => 官方 common.txt 指定的是 opencv-python-headless，把被別的套件帶進來的 opencv-python 移掉"
  conda run -n "$ENV_NAME" pip uninstall -y opencv-python opencv-python-headless >/dev/null 2>&1
  conda run -n "$ENV_NAME" pip install -q --force-reinstall --no-deps "opencv-python-headless==4.10.*"
fi

say "5. 驗證（裝好 != 裝對）"
conda run -n "$ENV_NAME" --no-capture-output python - <<'PY' || die "驗證失敗"
import importlib, sys
bad = 0
# 必要：官方 coarse／微調／分區／merge／test／深度工具真的會 import 的
for m in ["torch", "lightning", "numpy", "diff_trim_surfel_rasterization", "diff_gaussian_rasterization",
          "simple_knn._C", "torch_scatter", "cv2", "plyfile", "PIL"]:
    try:
        x = importlib.import_module(m)
        f = getattr(x, "__file__", None) or "?"
        flag = "⛔ 來自 gs/" if "/11213/gs/" in f else ""
        bad += bool(flag)
        print(f"  {m:32s} {getattr(x, '__version__', '')}  {f} {flag}")
    except Exception as e:
        bad += 1; print(f"  ⛔ {m}: {e!r}")
# 只顯示：官方 requirements 有列，但 CityGS 訓練流程沒用到（bnb 官方程式碼零引用；open3d 只在 mesh 萃取）
for m in ["bitsandbytes", "open3d"]:
    try:
        importlib.import_module(m); print(f"  （資訊）{m} 可匯入")
    except Exception as e:
        print(f"  （資訊，不擋）{m} 無法匯入：{type(e).__name__}（CityGS 訓練流程不用它）")
import numpy, torch
assert torch.__version__.startswith("2.0.1"), torch.__version__
assert numpy.__version__.startswith("1."), numpy.__version__
assert torch.cuda.is_available()
torch.from_numpy(numpy.zeros(3))
from simple_knn._C import distCUDA2
print("  distCUDA2:", distCUDA2(torch.rand(100, 3).cuda()).shape)
from diff_trim_surfel_rasterization import GaussianRasterizationSettings as S
ours = {"pp_shifty", "exact_conic_aabb", "return_tiles"} & set(S._fields)
print("  光柵器欄位:", S._fields)
print("  我方自訂欄位:", ours or "無 => 官方版 ✅")
bad += bool(ours)
import os
w = "utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"
print("  DA-V2 權重:", os.path.getsize(w) if os.path.exists(w) else "⛔ 缺")
bad += not os.path.exists(w) or os.path.getsize(w) < 1e8
sys.exit(1 if bad else 0)
PY

say "6. 寫環境報告 $REPORT"
{ echo "# gspl_official 環境報告 $(date '+%F %T')"
  echo "cityGS_origin HEAD: $(git rev-parse HEAD)"
  echo "Depth-Anything-V2 HEAD: $(git -C "$DA" rev-parse HEAD)"
  echo "vitl 權重 sha256: $(sha256sum "$W" | cut -d' ' -f1)"
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader)"
  echo "nvcc: $("$PREFIX/bin/nvcc" --version | tail -2 | head -1)"
  echo "## pip freeze"; conda run -n "$ENV_NAME" pip freeze
} > "$REPORT"
touch "$PREFIX/.citygs_official_env_ok"
echo "✅ 完成：$ENV_NAME 可用（標記 $PREFIX/.citygs_official_env_ok）"
