#!/bin/bash
# ★★★★★ lean_render（剃除沒人用的光柵器計算）驗證＋計時（2026-10-02 使用者：「安排一些測試 剃除無用計算」）
#
# 讀碼找到的浪費（現行配方 normal/dist/depth 權重全為 0）：
#   訓練 forward   每像素累積深度／法線／中位深度／distortion 6 個通道，訓練中沒有任何消費者
#   訓練 backward  每個 (像素, 顆粒) 配對 3 個加 0 的法線 atomicAdd＋深度/distortion 梯度算術（貢獻恰為 0）
#   trim pass      record 模式照算顏色與 7 個通道，呼叫端只取 T*alpha 與覆蓋數（增生期每 500 步掃全部相機）
# 實作：光柵器加模板參數（舊 kernel 原始碼一字不動）＋ renderer `lean_train`（預設關）。
#
# ⚠ 新光柵器裝到 logs/rast_lean_pkg（pip --target），用 PYTHONPATH 指過去 => **共用環境的 .so 不動**，
#   佇列裡其他跑次照舊。驗證通過後才另外決定要不要裝進環境。
# 三組比對（tools/check_lean_render.py，同 ckpt、同 6 台相機、同目標影像）：
#   A 舊 .so 兩次          => 噪音底（浮點 atomicAdd 順序本來就不固定）
#   B 舊 .so vs 新 .so 舊路徑 => 新 .so 沒改到舊路徑（渲染/radii/覆蓋數必須逐位元相同，梯度在噪音底內）
#   C 新 .so lean 關 vs 開  => lean 等價 + 光柵器層級計時
# 再用真實訓練迴圈（逐段計時 _StepProfiler，同 task_stepcost2.sh）lean 關／開各 1,200 步 => 每步真實省多少。
#
# 用法：[solo] bash scripts/lab/task_lean_check.sh            驗證＋計時（不動共用環境）
#       [solo] bash scripts/lab/task_lean_check.sh install    驗證通過後才用：把新光柵器裝進 gspl（舊路徑已驗證不變）
#       [solo] bash scripts/lab/task_lean_check.sh profile_runs <塊> <跑次...>   各跑次 60k ckpt 的 kernel 拆解
set -u
cd "$(dirname "$0")/../.." || exit 1
if [ "${1:-}" = profile_runs ]; then      # profile_runs <塊> <跑次...>：對各跑次的 60k ckpt 做 kernel 拆解（共用環境的 .so）
  shift; B=$1; shift
  if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
  L=logs/profile_runs_b${B}_$(date +%m%d_%H%M).log
  for r in "$@"; do
    c=$(ls outputs/${PFX}$r/blocks/block_$B/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
    [ -n "$c" ] || { echo "（$r 沒有 60k ckpt，略過）"; continue; }
    echo "════ $r"
    conda run -n gspl --no-capture-output python tools/check_lean_render.py profile --ckpt "$c" --lean 0
  done 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$L"
  exit 0
fi
if [ "${1:-}" = install ]; then
  G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
  RAST=submodules/diff-surfel-rasterization-trim-pp
  rm -rf $RAST/build
  conda run -n gspl env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --force-reinstall $RAST 2>&1 | tail -2
  conda run -n gspl python -c "import diff_trim_surfel_rasterization as m,os,time;p=os.path.dirname(m.__file__);f=[x for x in os.listdir(p) if x.endswith('.so')][0];print('so:',f,time.strftime('%m-%d %H:%M',time.localtime(os.path.getmtime(os.path.join(p,f)))));assert 'lean_render' in m.GaussianRasterizationSettings._fields;print('✅ lean_render 欄位在')"
  exit $?
fi
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
T=logs/rast_lean_pkg; S=logs/rast_lean_src; O=logs/lean_check
CK=$(ls outputs/${PFX}cs60_conic/blocks/block_6/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
CK2=$(ls outputs/${PFX}cs60_conic/blocks/block_6/checkpoints/*step=14999.ckpt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "⛔ 找不到 cs60_conic b6 60k ckpt"; exit 2; }
L=logs/lean_check_$(date +%m%d_%H%M).log
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
PY="conda run -n gspl --no-capture-output"
{ bad=0
  echo "════ 建新光柵器到 $T（不動共用環境；CUDA_HOME=$G）$(date)"
  rm -rf "$T" "$S"; cp -r submodules/diff-surfel-rasterization-trim-pp "$S"; rm -rf "$S/build"
  $PY env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --target "$T" "$S" 2>&1 | tail -2
  $PY env PYTHONPATH="$T" python -c "import diff_trim_surfel_rasterization as m; print('新:', m.__file__); assert 'lean_render' in m.GaussianRasterizationSettings._fields" || { echo "⛔ 新光柵器沒裝好"; exit 3; }
  $PY python -c "import diff_trim_surfel_rasterization as m; print('舊（共用環境）:', m.__file__, 'lean 欄位' , 'lean_render' in m.GaussianRasterizationSettings._fields)"
  mkdir -p "$O"
  echo "════ dump（ckpt $CK）"
  $PY python tools/check_lean_render.py dump --ckpt "$CK" --out "$O/old1.pt" --lean 0 || bad=1
  $PY python tools/check_lean_render.py dump --ckpt "$CK" --out "$O/old2.pt" --lean 0 || bad=1
  $PY env PYTHONPATH="$T" python tools/check_lean_render.py dump --ckpt "$CK" --out "$O/new0.pt" --lean 0 || bad=1
  $PY env PYTHONPATH="$T" python tools/check_lean_render.py dump --ckpt "$CK" --out "$O/new1.pt" --lean 1 || bad=1
  echo "════ 比對"
  $PY python tools/check_lean_render.py compare "$O/old1.pt" "$O/old2.pt" --label "A 噪音底：舊 .so 跑兩次" || bad=1
  $PY python tools/check_lean_render.py compare "$O/old1.pt" "$O/new0.pt" --label "B 新 .so 的舊路徑 vs 舊 .so" || bad=1
  $PY python tools/check_lean_render.py compare "$O/new0.pt" "$O/new1.pt" --label "C lean 關 vs 開（同一個新 .so）" || bad=1
  rm -f "$O"/*.pt        # 每個 ~0.6 GB，結論在 log 裡
  # ★ kernel 層級拆解（使用者 10-02：「梯度計算是不是運算大宗」）：一個訓練步（forward／loss／backward／Adam）
  #   ＋一次 trim record pass 的每個 kernel 時間，分成「逐配對／逐顆／逐像素／逐參數」=> 決定下一步打哪裡
  echo "════ kernel 拆解（torch.profiler，每步換一台相機，12 步）"
  if [ -n "$CK2" ]; then
    $PY env PYTHONPATH="$T" python tools/check_lean_render.py profile --ckpt "$CK2" --lean 0 || bad=1
    $PY env PYTHONPATH="$T" python tools/check_lean_render.py profile --ckpt "$CK2" --lean 1 || bad=1
  fi
  $PY env PYTHONPATH="$T" python tools/check_lean_render.py profile --ckpt "$CK" --lean 0 || bad=1
  # ★ fused Adam（optimizer.step 佔每步 9~13%）：等價（同梯度 1/50 步）＋計時，再看 lean＋fused 的整步拆解
  echo "════ fused Adam"
  $PY python tools/check_lean_render.py adamcheck --ckpt "$CK" || bad=1
  [ -n "$CK2" ] && { $PY env PYTHONPATH="$T" python tools/check_lean_render.py profile --ckpt "$CK2" --lean 1 --fused 1 || bad=1; }
  if [ -n "$CK2" ]; then
    echo "════ 真實訓練迴圈逐段計時：從 $CK2（增生期，含 trim pass）各 1,200 步，lean 關／開 $(date)"
    for lean in false true; do
      n="${PFX}leanloop_${lean}"
      [ -d "outputs/$n" ] && mv "outputs/$n" "outputs/$n.aborted_$(date +%m%d_%H%M%S)"
      ( unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
        $PY env PYTHONPATH="$T" CITYGS_STEP_PROFILE=1 CITYGS_KEEP_CFG_RENDERER=1 python -u main.py fit \
          --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6 -n "$n" \
          --model.initialize_from "$CK2" --trainer.max_steps 1200 \
          --data.image_uint8 true --data.skip_unused_depth true \
          --model.density.init_args.cap_max 2600000 --model.density.init_args.densify_until_iter 30000 \
          --model.density.init_args.absgrad_densify 2.0 --model.density.init_args.fast_noise true \
          --model.density.init_args.noise_gate_eps 0.001 --model.metric.init_args.opacity_reg 0.002 \
          --model.metric.init_args.lambda_normal 0.0 --model.metric.init_args.depth_loss_weight.init 0.0 \
          --model.renderer.init_args.exact_conic_aabb true --model.renderer.init_args.lean_train "$lean" ) 2>&1 \
        | grep -E "\[lean\]|Trimming done|Error|error|⛔" | head -8
      echo "── step_cost（lean=$lean）"
      cat "outputs/$n/blocks/block_6/step_cost.txt" 2>/dev/null || { echo "⛔ 沒有 step_cost.txt"; bad=1; }
    done
  else
    echo "（找不到 cs60_conic b6 的 step=14999 ckpt，略過真實迴圈計時）"
  fi
  exit "$bad"; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$L"
st=${PIPESTATUS[0]}
echo "報告：$L"
exit "$st"
