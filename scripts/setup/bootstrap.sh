#!/usr/bin/env bash
# 從 git clone 到「可以開始訓練」的一鍵腳本。
#
# 用法:
#   bash scripts/setup/bootstrap.sh                 # 全部做
#   bash scripts/setup/bootstrap.sh --env myenv     # 換 conda 環境名（預設 gspl）
#   bash scripts/setup/bootstrap.sh --skip-env      # 環境已存在，只編光柵器 + 驗證
#   bash scripts/setup/bootstrap.sh --verify-only   # 什麼都不裝，只跑驗證
#
# 設計原則：每一步都**驗證它真的做到了**，而不是「指令回傳 0 就算過」。
# 本專案最貴的教訓是七個「看起來正常但沒作用」的機制，全都不報錯；環境層同一個形狀。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

# ── 容器環境的兩個坑（2026-09-12 在 lab 的 NVIDIA NGC 容器上實測撞到）────────────
#  ① `PIP_CONSTRAINT` 指向容器的 constraint 檔，把 torch 釘在 2.7.0a0+...nv25.4
#     => `pip install torch==2.0.1` 直接 ResolutionImpossible
#  ② `LD_LIBRARY_PATH` 第一項是 `/usr/local/lib/python3.12/dist-packages/torch/lib`
#     => torch 2.0.1 **裝成功了**，但 `torch._C` 載到系統 torch 的 libtorch_python.so
#        `ImportError: undefined symbol: _PyThreadState_GetCurrent`
#     兩個都不會在安裝階段報錯，是「看起來裝好了但不能用」的形狀。
export PIP_CONSTRAINT=""
unset PIP_CONSTRAINT
# ⚠ lab 的 conda 26.5.3 / CPython 3.14 / libmamba 在**任何** conda 子指令上都可能丟
#   「An unexpected error has occurred」並自己建議關插件 => 整支腳本都關掉。
#   ⚠⚠ 用**環境變數**而不是 `--no-plugins` 旗標：那個旗標必須放在**子指令之前**
#   （`conda --no-plugins install`），放成 `conda install --no-plugins` 會變成
#   「unrecognized arguments」直接失敗 —— 2026-09-12 我這樣寫，害 cuda-toolkit 與
#   gcc 11 兩個安裝都「失敗」而錯誤訊息被 2>/dev/null 吞掉，追了兩輪。
export CONDA_NO_PLUGINS=true
# ⚠ 關掉插件會連 **libmamba solver 也關掉**（它本身是插件），而設定檔還要求用它 =>
#   `CondaValueError: You have chosen a non-default solver backend (libmamba) but it was
#    not recognized. Choose one of: classic` ⇒ 必須同時指定 classic。
#   ⚠ classic solver 在 conda-forge 上**很慢**（實測 >12 分鐘），不是卡住。
export CONDA_SOLVER=classic
_sanitize_ld() {
  local out="" p
  IFS=: read -ra _ps <<< "${LD_LIBRARY_PATH:-}"
  for p in "${_ps[@]:-}"; do
    case "$p" in
      *dist-packages/torch/lib|*dist-packages/torch_tensorrt/lib) ;;   # 丟掉系統 torch
      "") ;;
      *) out="${out:+$out:}$p";;
    esac
  done
  export LD_LIBRARY_PATH="$out"
}
_sanitize_ld
ROOT="$PWD"
ENV_NAME=gspl
DO_ENV=1; DO_BUILD=1; VERIFY_ONLY=0
LOCK=requirements/lock_gspl_2026-09-11.txt
RAST=submodules/diff-surfel-rasterization-trim-pp

while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2;;
    --skip-env) DO_ENV=0; shift;;
    --skip-build) DO_BUILD=0; shift;;
    --verify-only) VERIFY_ONLY=1; DO_ENV=0; DO_BUILD=0; shift;;
    -h|--help) sed -n '2,12p' "$0"; exit 0;;
    *) echo "不認得的參數: $1"; exit 2;;
  esac
done

say()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '  ✅ %s\n' "$*"; }
warn() { printf '  ⚠  %s\n' "$*"; }
die()  { printf '  ❌ %s\n' "$*"; exit 1; }

# ── 0. 前置條件 ───────────────────────────────────────────────────────────────
say "0. 前置條件"
command -v conda >/dev/null || die "找不到 conda。先裝 miniconda/anaconda。"
ok "conda $(conda --version | awk '{print $2}')"
command -v git >/dev/null || die "找不到 git。"
if command -v nvidia-smi >/dev/null; then
  ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  warn "沒有 nvidia-smi —— 只能做 CPU 端的診斷工具，不能訓練。"
fi
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 40 ] && ok "磁碟可用 ${FREE_GB}G" \
  || warn "磁碟只剩 ${FREE_GB}G。資料集 + outputs 很吃空間（單一 60k 跑次的 ckpt 約 1~2G）。"

# ── 1. conda 環境 ─────────────────────────────────────────────────────────────
if [ "$DO_ENV" = 1 ]; then
  say "1. conda 環境 $ENV_NAME（python 3.9）"
  if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    ok "已存在，沿用（要重來先 conda env remove -n $ENV_NAME）"
  else
    conda create -yn "$ENV_NAME" python=3.9 pip || die "建環境失敗"
    ok "建好了"
  fi

  # ── 1b. CUDA 11.8 工具鏈（編光柵器用）──────────────────────────────────────
  say "1b. CUDA 11.8 工具鏈（進 conda env，不動系統）"
  # ⚠ 2026-09-12 補：原本漏了這步。我方的 nvcc 11.8 是從 conda 的
  #   `nvidia/label/cuda-11.8.0` 頻道來的（本機 `conda list` 可見 cuda-compiler 11.8.0 等）。
  #   只做 `conda create python=3.9 pip` + pip 安裝**不會**帶 nvcc
  #   => 在系統 CUDA 是 12.x 的機器上（例如 lab 主機是 12.9），
  #      步驟 5 的光柵器編譯會拿到 12.x 或找不到 nvcc 而失敗。
  #   torch 是 cu118，所以工具鏈必須也是 11.8，不能用系統的 12.x。
  if conda run -n "$ENV_NAME" bash -c '[ -x "$CONDA_PREFIX/bin/nvcc" ]' 2>/dev/null; then
    ok "env 裡已有 nvcc（$(conda run -n "$ENV_NAME" bash -c '$CONDA_PREFIX/bin/nvcc --version' 2>/dev/null | tail -2 | head -1 | sed 's/.*release //;s/,.*//'))"
  else
    # ⚠ 2026-09-12：lab 的容器是 conda 26.5.3 / CPython 3.14 / libmamba，
    #   這行會丟「unexpected error」並自己建議 --no-plugins => 直接帶上。
    conda install -y -n "$ENV_NAME" -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 \
      || die "CUDA 11.8 工具鏈安裝失敗（沒有它就編不出光柵器）"
    # 後援：conda 整條掛掉時用 pip 的 nvcc 輪子（只要 nvcc 能跑就夠編光柵器）
    if ! conda run -n "$ENV_NAME" bash -lc 'command -v nvcc' >/dev/null 2>&1; then
      echo "  conda 裝不起來 => 改用 pip 的 nvidia-cuda-nvcc-cu11"
      conda run -n "$ENV_NAME" --no-capture-output pip install -q nvidia-cuda-nvcc-cu11 || true
    fi
    conda run -n "$ENV_NAME" bash -c '[ -x "$CONDA_PREFIX/bin/nvcc" ]' \
      || die "裝完還是沒有 $CONDA_PREFIX/bin/nvcc"
    ok "nvcc 11.8 就位"
  fi

  # ── 2. torch 必須先裝，且走 CUDA 專屬索引 ──────────────────────────────────
  say "2. torch 2.0.1 + cu118（走 PyTorch 專屬索引，不能從 lock 檔裝）"
  conda run -n "$ENV_NAME" --no-capture-output pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 || die "torch 安裝失敗"
  conda run -n "$ENV_NAME" python -c "import torch;assert torch.__version__.startswith('2.0.1'),torch.__version__" \
    || die "torch 版本不對"
  ok "torch 2.0.1+cu118"

  # ── 3. 其餘套件 ────────────────────────────────────────────────────────────
  say "3. 其餘套件（版本鎖 = 這個環境實際跑出過所有 outputs/ 的那一組）"
  [ -f "$LOCK" ] || die "找不到 $LOCK"
  # ⚠ 鎖檔是 `pip freeze` 產的，會包含**不在 PyPI 上的**條目（從 git/本地裝的套件，
  #   freeze 只留 `name==version`、URL 丟了）。2026-09-12 在 lab 上撞到兩個：
  #     dataclasses==0.8（Python 3.7+ 的標準庫，已直接從鎖檔移除）
  #     plas==0.1（PyPI 上沒有；本機是從 git 裝的）
  #   ⇒ 不能讓一個裝不到的條目擋住整個部署。改成**自我修復**：
  #     解析 pip 的 "No matching distribution found for X"，把 X 移出後重試，
  #     最後把跳過的清單印出來（讓使用者知道少了什麼，而不是靜默少裝）。
  _WORK=$(mktemp); cp "$LOCK" "$_WORK"; _SKIPPED=""
  for _try in $(seq 1 12); do
    _LOG=$(mktemp)
    if conda run -n "$ENV_NAME" --no-capture-output pip install -r "$_WORK" 2>&1 | tee "$_LOG"; then
      rm -f "$_LOG"; break
    fi
    _MISS=$(grep -oE "No matching distribution found for [A-Za-z0-9_.-]+" "$_LOG" \
            | tail -1 | awk '{print $NF}' | sed 's/[=<>!].*//')
    rm -f "$_LOG"
    if [ -z "$_MISS" ]; then die "套件安裝失敗（不是「找不到套件」，看上面的錯誤）"; fi
    echo "  ⚠ PyPI 上沒有 $_MISS => 移出鎖檔後重試（第 $_try 次）"
    _SKIPPED="$_SKIPPED $_MISS"
    grep -viE "^${_MISS}([=<>!\[]|$)" "$_WORK" > "$_WORK.n" && mv "$_WORK.n" "$_WORK"
  done
  rm -f "$_WORK"
  # 坑四（2026-09-12 lab）：容器是 headless，沒有 libGL.so.1，而鎖檔同時裝了
  #   opencv-python 與 opencv-python-headless —— 前者遮住後者 =>
  #   `import cv2` 丟 `ImportError: libGL.so.1: cannot open shared object file`。
  #   移掉非 headless 的那個即可（不需要動系統套件）。
  if ! conda run -n "$ENV_NAME" python -c "import cv2" >/dev/null 2>&1; then
    if conda run -n "$ENV_NAME" python -c "import cv2" 2>&1 | grep -q "libGL"; then
      warn "cv2 缺 libGL（headless 容器）=> 移除 opencv-python，保留 opencv-python-headless"
      conda run -n "$ENV_NAME" --no-capture-output pip uninstall -y opencv-python >/dev/null 2>&1
      conda run -n "$ENV_NAME" python -c "import cv2;print('  cv2', cv2.__version__)" \
        && ok "cv2 可用" || warn "cv2 仍不可用 —— 手動 pip install opencv-python-headless"
    fi
  fi
  if [ -n "$_SKIPPED" ]; then
    echo "  ⚠⚠ 這些套件**沒有裝**（PyPI 上找不到，鎖檔是 pip freeze 產的所以丟了來源）："
    for _m in $_SKIPPED; do echo "       $_m"; done
    echo "     若之後 import 失敗就是它們 —— 到原專案找安裝來源。"
  fi

  # 讓「清掉系統 torch 的 lib」對之後每一次 `conda run -n gspl` 都生效，
  # 否則使用者自己敲指令時會再撞一次同樣的 ImportError（而它不在安裝階段報錯）。
  _ACT="$(conda run -n "$ENV_NAME" python -c 'import sys,os;print(os.path.dirname(os.path.dirname(sys.executable)))')/etc/conda/activate.d"
  mkdir -p "$_ACT"
  cat > "$_ACT/00_sanitize_ld.sh" <<'ACTEOF'
# 由 scripts/setup/bootstrap.sh 產生。容器（NVIDIA NGC）把系統 torch 的 lib 目錄
# 放在 LD_LIBRARY_PATH 最前面，會讓本環境的 torch._C 載到錯的 libtorch_python.so。
_out=""; IFS=: read -ra _ps <<< "${LD_LIBRARY_PATH:-}"
for _p in "${_ps[@]:-}"; do
  case "$_p" in
    *dist-packages/torch/lib|*dist-packages/torch_tensorrt/lib|"") ;;
    *) _out="${_out:+$_out:}$_p";;
  esac
done
export LD_LIBRARY_PATH="$_out"
unset _out _ps _p
ACTEOF
  echo "  已寫入 $_ACT/00_sanitize_ld.sh"
  ok "裝完 $(grep -c '==' "$LOCK") 個套件"
  warn "repo 裡的 requirements.txt / requirements/lightning23.txt 會裝 lightning 2.3 + bitsandbytes 0.45，"
  warn "  那**不是**能跑的組合（0.45 的優化器 state key 與 density controller 對不上）=> 用 lock 檔。"
fi

# ── 4. 光柵器原始碼 ───────────────────────────────────────────────────────────
say "4. Trim 光柵器原始碼"
if [ ! -f "$RAST/cuda_rasterizer/auxiliary.h" ]; then
  echo ""
  echo "  ❌ $RAST 是空的 / 沒有原始碼。"
  echo ""
  echo "  這是**上傳端**的問題，不是這台機器的問題："
  echo "    我方改過的光柵器（ABSGRAD、EXACT_SUPPORT、transmittance/num_covered_pixels）"
  echo "    在原 repo 裡是一個 **gitlink**，而且 .gitmodules **沒有它的條目**，"
  echo "    它的 origin 又指向上游（我們推不上去）且與我方歷史**沒有共同祖先**。"
  echo "    => 純 git clone 拿不到那 20 個檔，整個 6GB 配方無法重現。"
  echo ""
  echo "  修法（在**原始那台機器**上執行一次，然後重新 push）："
  echo "    bash scripts/setup/preflight_upload.sh --fix-rasterizer"
  echo ""
  die "缺少光柵器原始碼，無法繼續"
fi
ok "原始碼在（$(ls "$RAST"/cuda_rasterizer/*.cu "$RAST"/cuda_rasterizer/*.h 2>/dev/null | wc -l) 個 cuda_rasterizer 檔）"
# ⚠ 2026-09-12：原本這裡是 `A || B || C` 的鏈，而 **A 成功但什麼都沒做** ——
#   `.gitmodules` 裡沒有光柵器的條目（見記憶 repo_not_clone_ready），
#   `git submodule update` 就回傳 0 => `||` 短路 => clone 從來沒被執行 => 建置必失敗。
#   改成**每一步都驗標頭檔在不在**（回傳碼不可信，這是本專案最貴的那類 bug 的形狀）。
_GLM="$RAST/third_party/glm"
if [ ! -f "$_GLM/glm/glm.hpp" ]; then
  warn "third_party/glm 是空的，抓它（建置必需）"
  git submodule update --init --recursive "$RAST" >/dev/null 2>&1 || true
  if [ ! -f "$_GLM/glm/glm.hpp" ]; then
    git -C "$RAST" submodule update --init --recursive >/dev/null 2>&1 || true
  fi
  if [ ! -f "$_GLM/glm/glm.hpp" ]; then
    warn "submodule 抓不到（.gitmodules 沒條目）=> 直接 clone glm"
    rm -rf "$_GLM"; mkdir -p "$(dirname "$_GLM")"
    git clone --depth 1 https://github.com/g-truc/glm.git "$_GLM" 2>&1 | tail -2
  fi
fi
[ -f "$RAST/third_party/glm/glm/glm.hpp" ] && ok "glm 在" || die "glm 還是缺 —— 手動 clone https://github.com/g-truc/glm.git 到 $RAST/third_party/glm"

# ── 5. 編譯（三個踩過的坑都在這裡） ───────────────────────────────────────────
if [ "$DO_BUILD" = 1 ]; then
  say "5. 編譯 Trim 光柵器"
  # 坑一：distutils **只看 .cu 的 mtime**，改了 .h 不會觸發重編 => 一律先砍 build
  rm -rf "$RAST/build"
  ok "清掉 build/（distutils 只看 .cu 的 mtime，改 .h 不會重編）"
  # 坑二：shell 匯出的 CUDA_PATH 可能指向系統較新的 CUDA，torch 會優先用它然後拒編
  #       => 強制指到 conda 環境裡那份 11.8
  PREFIX=$(conda run -n "$ENV_NAME" python -c "import sys,os;print(sys.prefix)" 2>/dev/null | tr -d '\r')
  [ -n "$PREFIX" ] || die "拿不到 conda prefix"
  ok "CUDA_HOME 強制指向 $PREFIX"
  # 坑二點五（2026-09-12 在 lab 撞到）：容器把 TORCH_CUDA_ARCH_LIST 設成含 Blackwell（10.0），
  #   而 torch 2.0.1 不認識 => `ValueError: Unknown CUDA arch (10.0) or GPU not supported`。
  #   不要猜，直接問**這台機器的 GPU**（在任何裝置上都對）。
  _ARCH=$(conda run -n "$ENV_NAME" python -c \
    "import torch;print('.'.join(map(str,torch.cuda.get_device_capability(0))))" 2>/dev/null | tr -d '\r')
  if [ -n "$_ARCH" ]; then
    export TORCH_CUDA_ARCH_LIST="$_ARCH"
    ok "TORCH_CUDA_ARCH_LIST=$_ARCH（問 GPU 問出來的，覆蓋容器設的值）"
  else
    warn "問不到 GPU 的 compute capability => 沿用環境變數 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-未設}"
  fi
  # 坑二點七五（2026-09-12 在 lab 撞到）：容器是 Ubuntu 24.04，gcc 13；
  #   而 **CUDA 11.8 的 nvcc 只支援 gcc <= 11**：
  #     crt/host_config.h:132: error: unsupported GNU version! gcc versions later than 11
  #   ⛔ 不用 nvcc 的 `-allow-unsupported-compiler`：它自己警告
  #      "may cause incorrect run time execution" —— 這是算梯度的 kernel，
  #      不能冒「編得過但數值錯」的風險（本專案最貴的 bug 全是這種形狀）。
  #   正解：把 gcc 11 裝進 conda 環境，用 -ccbin 指給 nvcc。
  _GCCV=$(gcc -dumpfullversion 2>/dev/null | cut -d. -f1)
  if [ -n "$_GCCV" ] && [ "$_GCCV" -gt 11 ] 2>/dev/null; then
    warn "系統 gcc 是 $_GCCV，CUDA 11.8 只吃 <=11 => 裝 conda-forge 的 gcc 11 到環境裡"
    conda install -y -n "$ENV_NAME" -c conda-forge \
      gcc_linux-64=11 gxx_linux-64=11 >/dev/null 2>&1 || warn "conda 裝 gcc 11 失敗"
    _CC=$(ls "$PREFIX"/bin/*-linux-gnu-gcc 2>/dev/null | head -1)
    _CXX=$(ls "$PREFIX"/bin/*-linux-gnu-g++ 2>/dev/null | head -1)
    if [ -n "$_CC" ] && [ -n "$_CXX" ]; then
      export CC="$_CC" CXX="$_CXX"
      export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -ccbin $_CXX"
      ok "CC=$(basename "$_CC") $("$_CC" -dumpfullversion)  且 nvcc -ccbin 已指給它"
    else
      die "環境裡找不到 gcc 11（$PREFIX/bin/*-linux-gnu-gcc）—— 手動 conda install -c conda-forge gxx_linux-64=11"
    fi
  fi
  # 坑三：pip 會**靜默跳過**同版本的本地路徑 => 一定要 --force-reinstall
  conda run -n "$ENV_NAME" --no-capture-output env CUDA_HOME="$PREFIX" CUDA_PATH="$PREFIX" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" \
    CC="${CC:-cc}" CXX="${CXX:-c++}" NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}" \
    pip install --no-build-isolation --no-deps --force-reinstall "$RAST" || die "編譯失敗"
  ok "編好了（--force-reinstall 是必要的，pip 會靜默跳過同版本的本地路徑）"
fi

# ── 6. 驗證 ───────────────────────────────────────────────────────────────────
say "6. 驗證（裝好 != 裝對）"
conda run -n "$ENV_NAME" --no-capture-output python scripts/setup/verify_env.py
VRC=$?

# ── 7. 單元測試 ───────────────────────────────────────────────────────────────
say "7. 單元測試"
echo "  ⚠ -p 不能省：檔名是 <name>_test.py（後綴），unittest 預設樣式是 test*.py（前綴），"
echo "    少了 -p 會**發現 0 個測試然後回報 OK**。"
conda run -n "$ENV_NAME" python -m unittest discover -s tests -p "*_test.py" 2>&1 | grep -E "^(OK|FAILED|Ran )" | tail -3
echo "  （基準 2026-09-11：86 collected / 81 過 / 5 error —— 5 個全是缺選用相依的上游測試：\n    tinycudann、gsplat._torch_impl、deformable model，不是我方程式）"

# ── 8. 資料 ───────────────────────────────────────────────────────────────────
say "8. 資料（**不在 git 裡**，要自己放）"
cat <<'TXT'
  data/ 在 .gitignore 裡（資料集數百 GB）。需要的東西：

    data/matrix_city/aerial/train/block_all/
      input/                  訓練影像
      sparse/0/               COLMAP 稀疏重建（SfM-init 與姿態都吃它）
      estimated_depths/       Depth Anything V2 的輸出 <- 最貴的一步（5621 張）
      depth_init/block_*.ply  depth-init 起點
      partition/              分塊結果
    data/matrix_city/aerial/test/block_all_test/     官方 741 幀 held-out

  後三者可以自己重生（照 紀錄/完整指令手冊.md §2，三行指令）：
    python utils/estimate_dataset_depths.py data/matrix_city/aerial/train/block_all --encoder vitl
    python utils/partition_from_colmap.py data/matrix_city/aerial/train/block_all --block_dim 5 5 --content_threshold 0.08 --force
    python utils/depth_init_blocks.py data/matrix_city/aerial/train/block_all --block_dim 5 5 --voxel_min 0.03 --voxel_max 0.7 --chunk_size 75

  ⚠ estimate_dataset_depths.py 需要 Depth-Anything-V2 的權重與程式碼；本機是用
    depth_anything_v2 這個符號連結指到外部 clone，**連結不會跟著 git 過來**。
  ⚠ 換了資料就是換了世代：SfM 重生會換座標系，舊 ckpt 不可當 init（紀錄/研究總覽.md 的鐵律）。
TXT

say "完成"
echo "  下一步："
echo "    conda activate $ENV_NAME"
echo "    python tools/queue_status.py                 # 排程器現況"
echo "    bash scripts/task_speed3.sh                  # 現行最佳配方（b12，約 8.8h）"
echo "  文件：紀錄/_ctx.md（開場必讀）-> 紀錄/完整指令手冊.md -> 紀錄/研究總覽.md §13"
exit $VRC
