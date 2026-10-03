#!/bin/bash
# ★★★★★ 第二批剃除運算（2026-10-03 使用者：「都做 排在官方驗證前」）：等價驗證＋計時
#   record_reduce  trim pass 的 record 在 block 內先加總、每個 (tile,顆粒) 一次 global atomic（forward.cu recordReduceCUDA）
#                  lean 後 trim 約佔增生期每步 37%，而 lean 對它只省 1~2%（瓶頸是每個被評估配對 2 個 global atomic）
#   tile_cull      非 record 呼叫只綁「可能 alpha >= 1/255」的 tile（rasterizer_impl.cu tileMayContribute；保守上界）
#                  tile 上限量測：30~42% 的配對整個 tile 都沒有 alpha >= 1/255 的像素
# 新光柵器裝到 logs/rast_speed2_pkg，用 PYTHONPATH 指過去 => **共用環境不動**（現在裝的是 lean 版）。
# 比對（tools/check_lean_render.py；cs60_conic b6 60k；6 台相機）：
#   B   共用環境（lean 版）vs 新 .so，lean 關       => 新 .so 的舊路徑沒變（渲染／radii／覆蓋數逐位元相同）
#   N   新 .so lean 開跑兩次                       => 噪音底
#   R   lean vs lean＋record_reduce               => 覆蓋數逐位元相同、T*alpha 浮點級、trim 遮罩重疊
#   T   lean vs lean＋record_reduce＋tile_cull     => 渲染逐位元相同、binning 配對減少多少、梯度在噪音底內
# 真實迴圈（@14,999 起 1,200 步，含 1 次 trim）：lean／lean+rr／lean+rr+tc 三組逐段計時
# 用法：[solo] bash scripts/lab/task_speed2_check.sh          等價驗證＋真實迴圈計時（不動共用環境）
#       [solo] bash scripts/lab/task_speed2_check.sh gate     依事先寫好的規則判定（tools/speed2_gate.py）；正確性全過且有變快才把新光柵器裝進 gspl
#                                                            結果寫 logs/speed2_gate.json（task_full44.sh prep 讀它決定開不開 4x4、開哪些旗標）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
if [ "${1:-}" = gate ]; then
  # ★ 2026-10-03 使用者：「等速度第二批裝好再開 4x4」—— 判定要能在佇列裡自己跑（lab 由使用者看，Claude 不輪詢）。
  #   [solo]：安裝時不能有別的訓練在用共用環境。正確性沒過 => exit 1（不裝，4x4 prep 也會拒絕開）。
  conda run -n gspl --no-capture-output python tools/speed2_gate.py 2>&1 | grep -vE 'pkg_resources|declare_namespace' || exit 1
  st=$(python3 -c "import json;print(json.load(open('logs/speed2_gate.json'))['status'])" 2>/dev/null)
  [ "$st" = fail ] || [ -z "$st" ] && { echo "⛔ 判定沒過（status=[$st]）=> 不安裝"; exit 1; }
  if [ "$st" = install ]; then
    G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
    RAST=submodules/diff-surfel-rasterization-trim-pp
    rm -rf $RAST/build
    conda run -n gspl env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --force-reinstall $RAST 2>&1 | tail -2
    conda run -n gspl python -c "import diff_trim_surfel_rasterization as m,os,time;p=os.path.dirname(m.__file__);f=[x for x in os.listdir(p) if x.endswith('.so')][0];print('so:',f,time.strftime('%m-%d %H:%M',time.localtime(os.path.getmtime(os.path.join(p,f)))));F=m.GaussianRasterizationSettings._fields;assert all(k in F for k in ('lean_render','record_reduce','tile_cull'));print('✅ lean_render／record_reduce／tile_cull 欄位都在')" \
      || { echo "⛔ 安裝後驗證失敗"; exit 1; }
    python3 -c "import json;d=json.load(open('logs/speed2_gate.json'));d['status']='installed';json.dump(d,open('logs/speed2_gate.json','w'),ensure_ascii=False,indent=1)"
    echo "✅ 已裝進 gspl；4x4 會開：$(python3 -c "import json;print('＋'.join(json.load(open('logs/speed2_gate.json'))['flags']))")"
  else
    echo "（正確但沒有變快 => 不裝；4x4 只開 lean）"
  fi
  exit 0
fi
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
T=logs/rast_speed2_pkg; S=logs/rast_speed2_src; O=logs/speed2_check
CK=$(ls outputs/${PFX}cs60_conic/blocks/block_6/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
CK2=$(ls outputs/${PFX}cs60_conic/blocks/block_6/checkpoints/*step=14999.ckpt 2>/dev/null | head -1)
[ -n "$CK" ] && [ -n "$CK2" ] || { echo "⛔ 找不到 cs60_conic b6 的 60k／14999 ckpt"; exit 2; }
L=logs/speed2_check_$(date +%m%d_%H%M).log
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
PY="conda run -n gspl --no-capture-output"
{ bad=0
  echo "════ 建新光柵器到 $T（不動共用環境）$(date)"
  rm -rf "$T" "$S"; cp -r submodules/diff-surfel-rasterization-trim-pp "$S"; rm -rf "$S/build"
  $PY env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --target "$T" "$S" 2>&1 | tail -1
  $PY env PYTHONPATH="$T" python -c "import diff_trim_surfel_rasterization as m; f=m.GaussianRasterizationSettings._fields; assert 'record_reduce' in f and 'tile_cull' in f; print('新:', m.__file__)" || { echo "⛔ 新光柵器沒裝好"; exit 3; }
  mkdir -p "$O"
  D="python tools/check_lean_render.py dump --ckpt $CK"
  $PY python tools/check_lean_render.py dump --ckpt "$CK" --out "$O/inst0.pt" --lean 0 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/new0.pt" --lean 0 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/L1.pt" --lean 1 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/L2.pt" --lean 1 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/LR.pt" --lean 1 --record-reduce 1 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/LRT.pt" --lean 1 --record-reduce 1 --tile-cull 1 || bad=1
  C="python tools/check_lean_render.py compare"
  $PY $C "$O/inst0.pt" "$O/new0.pt" --label "B 共用環境 vs 新 .so（lean 關）：舊路徑沒變" || bad=1
  $PY $C "$O/L1.pt" "$O/L2.pt" --label "N 噪音底：新 .so lean 跑兩次" || bad=1
  $PY $C "$O/L1.pt" "$O/LR.pt" --label "R lean vs lean+record_reduce" || bad=1
  $PY $C "$O/L1.pt" "$O/LRT.pt" --label "T lean vs lean+record_reduce+tile_cull" || bad=1
  rm -f "$O"/*.pt
  for cfg in L LR LRT; do
    case $cfg in
      L)   XA=() ;;
      LR)  XA=(--model.renderer.init_args.record_reduce true) ;;
      LRT) XA=(--model.renderer.init_args.record_reduce true --model.renderer.init_args.tile_cull true) ;;
    esac
    n="${PFX}speed2loop_$cfg"
    [ -d "outputs/$n" ] && mv "outputs/$n" "outputs/$n.aborted_$(date +%m%d_%H%M%S)"
    echo "════ 真實迴圈 $cfg（@14,999 起 1,200 步）$(date)"
    ( unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
      $PY env PYTHONPATH="$T" CITYGS_STEP_PROFILE=1 CITYGS_KEEP_CFG_RENDERER=1 python -u main.py fit \
        --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6 -n "$n" \
        --model.initialize_from "$CK2" --trainer.max_steps 1200 \
        --data.image_uint8 true --data.skip_unused_depth true \
        --model.density.init_args.cap_max 2600000 --model.density.init_args.densify_until_iter 30000 \
        --model.density.init_args.absgrad_densify 2.0 --model.density.init_args.fast_noise true \
        --model.density.init_args.noise_gate_eps 0.001 --model.metric.init_args.opacity_reg 0.002 \
        --model.metric.init_args.lambda_normal 0.0 --model.metric.init_args.depth_loss_weight.init 0.0 \
        --model.renderer.init_args.exact_conic_aabb true --model.renderer.init_args.lean_train true "${XA[@]}" ) 2>&1 \
      | grep -E "\[lean\]|Trimming done|Error|error|⛔" | head -8
    echo "── step_cost（$cfg）"
    cat "outputs/$n/blocks/block_6/step_cost.txt" 2>/dev/null | sed -n 1,22p || { echo "⛔ 沒有 step_cost.txt"; bad=1; }
  done
  exit "$bad"; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample|colmap dataparser|depth maps' | tee "$L"
st=${PIPESTATUS[0]}
echo "報告：$L（通過後：bash scripts/lab/task_lean_check.sh install 會把**目前原始碼**的光柵器裝進 gspl）"
exit "$st"
