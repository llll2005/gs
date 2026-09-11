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
  conda run -n "$ENV_NAME" --no-capture-output pip install -r "$LOCK" || die "套件安裝失敗"
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
if [ ! -f "$RAST/third_party/glm/glm/glm.hpp" ]; then
  warn "third_party/glm 是空的，抓它（建置必需）"
  git submodule update --init --recursive "$RAST" 2>/dev/null \
    || git -C "$RAST" submodule update --init --recursive 2>/dev/null \
    || git clone --depth 1 https://github.com/g-truc/glm.git "$RAST/third_party/glm"
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
  # 坑三：pip 會**靜默跳過**同版本的本地路徑 => 一定要 --force-reinstall
  conda run -n "$ENV_NAME" --no-capture-output env CUDA_HOME="$PREFIX" CUDA_PATH="$PREFIX" \
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
echo "  （基準：55 過 / 7 個因缺選用相依而 error —— 那 7 個全是上游測試，不是我方程式）"

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
