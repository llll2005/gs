#!/bin/bash
# 2026-10-09 backward 最佳化（使用者：「照你說的」）：光柵器 backward.cu renderCUDA_opt（只在 lean 路徑生效，原 kernel 一字不動）
#   bwd_sat_skip  從 tile 內最大的最後貢獻者開始 => 飽和後的配對不再整批載入（處理的配對與順序完全相同）
#   bwd_reduce    梯度先 warp 內 cg::reduce、再加到 shared memory，每批每顆一次全域 atomic（record_reduce 的模式）
# 驗證（新光柵器裝到 logs/rast_bwdopt_pkg，用 PYTHONPATH 指過去 => 共用環境不動；cs60_sfmdup4 b6 @14,999，6 台相機，lean＋record_reduce）：
#   B   共用環境 vs 新 .so（旗標全關）    => 舊路徑沒變：渲染／radii／覆蓋數逐位元相同
#   N   新 .so 旗標全關跑兩次             => 梯度噪音底
#   S／R／SR  關 vs sat／red／兩者        => 渲染逐位元相同；梯度相對差 <= 10 x 噪音底
# 計時：單台相機 backward 中位（dump，20 次）＋真實迴圈逐段（@14,999 起 1,200 步，含 trim；base／sat／red／sat+red）
# 判定寫 logs/bwdopt_gate.json。
# 用法：[solo] bash scripts/lab/task_bwdopt_check.sh            驗證＋計時（不動共用環境）
#       [solo] bash scripts/lab/task_bwdopt_check.sh install    判定 pass 才把目前原始碼的光柵器裝進 gspl
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
RAST=submodules/diff-surfel-rasterization-trim-pp
J=logs/bwdopt_gate.json
if [ "${1:-}" = install ]; then
  st=$(python3 -c "import json;print(json.load(open('$J'))['status'])" 2>/dev/null)
  [ "$st" = pass ] || { echo "⛔ 判定不是 pass（status=[$st]）=> 不安裝"; exit 1; }
  rm -rf $RAST/build
  conda run -n gspl env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --force-reinstall $RAST 2>&1 | tail -2
  conda run -n gspl python -c "import diff_trim_surfel_rasterization as m;F=m.GaussianRasterizationSettings._fields;assert all(k in F for k in ('bwd_sat_skip','bwd_reduce','absgrad','record_reduce'));print('✅ 欄位都在')" \
    || { echo "⛔ 安裝後驗證失敗"; exit 1; }
  python3 -c "import json;d=json.load(open('$J'));d['status']='installed';json.dump(d,open('$J','w'),ensure_ascii=False,indent=1)"
  echo "✅ 已裝進 gspl（預設行為不變；要用就在 renderer 開 bwd_sat_skip／bwd_reduce）"
  exit 0
fi
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
T=logs/rast_bwdopt_pkg; S=logs/rast_bwdopt_src; O=logs/bwdopt_dump
CK=$(ls outputs/${PFX}cs60_sfmdup4/blocks/block_6/checkpoints/*step=14999.ckpt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "⛔ 找不到 cs60_sfmdup4 b6 的 14999 ckpt"; exit 2; }
L=logs/bwdopt_check_$(date +%m%d_%H%M).log
PY="conda run -n gspl --no-capture-output"
{ bad=0
  echo "════ 建新光柵器到 $T（不動共用環境）$(date)"
  rm -rf "$T" "$S"; cp -r $RAST "$S"; rm -rf "$S/build"
  $PY env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --target "$T" "$S" 2>&1 | tail -1
  $PY env PYTHONPATH="$T" python -c "import diff_trim_surfel_rasterization as m; F=m.GaussianRasterizationSettings._fields; assert 'bwd_sat_skip' in F and 'bwd_reduce' in F; print('新:', m.__file__)" || { echo "⛔ 新光柵器沒裝好"; exit 3; }
  mkdir -p "$O"
  D="python tools/check_lean_render.py dump --ckpt $CK --lean 1 --record-reduce 1 --repeat 20"
  $PY $D --out "$O/inst.pt" || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/n1.pt" || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/n2.pt" || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/s.pt" --bwd-sat 1 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/r.pt" --bwd-red 1 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/sr.pt" --bwd-sat 1 --bwd-red 1 || bad=1
  C="python tools/check_lean_render.py compare"
  $PY $C "$O/inst.pt" "$O/n1.pt" --label "B 共用環境 vs 新 .so（旗標全關）：舊路徑沒變" || bad=1
  $PY $C "$O/n1.pt" "$O/n2.pt" --label "N 噪音底：新 .so 旗標全關跑兩次" || bad=1
  $PY $C "$O/n1.pt" "$O/s.pt" --label "S 關 vs 飽和跳過" || bad=1
  $PY $C "$O/n1.pt" "$O/r.pt" --label "R 關 vs block 內先加總" || bad=1
  $PY $C "$O/n1.pt" "$O/sr.pt" --label "SR 關 vs 兩者" || bad=1
  rm -f "$O"/*.pt
  CK2=$CK
  for cfg in base sat red satred; do
    case $cfg in
      base)   XA=() ;;
      sat)    XA=(--model.renderer.init_args.bwd_sat_skip true) ;;
      red)    XA=(--model.renderer.init_args.bwd_reduce true) ;;
      satred) XA=(--model.renderer.init_args.bwd_sat_skip true --model.renderer.init_args.bwd_reduce true) ;;
    esac
    n="${PFX}bwdoptloop_$cfg"
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
        --model.renderer.init_args.exact_conic_aabb true --model.renderer.init_args.lean_train true \
        --model.renderer.init_args.record_reduce true "${XA[@]}" ) 2>&1 \
      | grep -E "\[lean\]|Trimming done|Error|error|⛔" | head -8
    echo "── step_cost（$cfg）"
    cat "outputs/$n/blocks/block_6/step_cost.txt" 2>/dev/null | sed -n 1,22p || { echo "⛔ 沒有 step_cost.txt"; bad=1; }
  done
  echo "bad=$bad"
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample|colmap dataparser|depth maps' | tee "$L"
# 判定（寫在結果出來前）：B、S、R、SR 渲染逐位元相同（✅ 行）；S／R／SR 參數梯度相對差 <= 10 x N
python3 - "$L" "$J" <<'PY'
import json, re, sys
t = open(sys.argv[1], encoding="utf-8").read()
def sec(k):
    m = t.split("══ " + k, 1)
    return m[1].split("══ ", 1)[0] if len(m) > 1 else ""
rel = lambda s: [float(x) for x in re.findall(r"相對(?:（÷最大幅度）)? *([0-9.]+e[-+][0-9]+)", s)]
N = rel(sec("N "))
res = {}
for k in ("S ", "R ", "SR "):
    s = sec(k); g = rel(s)
    bw = re.search(r"backward ([0-9.]+) -> ([0-9.]+)（([-+0-9.]+)%）", s)
    res[k.strip()] = {"render_bitident": "✅" in s,
                      "grads_within_noise": bool(N) and bool(g) and all(a <= 10 * max(b, 1e-7) for a, b in zip(g, N)),
                      "backward_ms": [float(bw.group(1)), float(bw.group(2))] if bw else None,
                      "backward_delta_pct": float(bw.group(3)) if bw else None}
ok = "✅" in sec("B ") and all(v["render_bitident"] and v["grads_within_noise"] for v in res.values()) and "bad=0" in t
d = {"status": "pass" if ok else "fail", "old_path_unchanged": "✅" in sec("B "), "arms": res, "log": sys.argv[1]}
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("判定:", json.dumps(d, ensure_ascii=False))
PY
