#!/bin/bash
# 2026-10-09 absgrad_gate（使用者：「排」）：光柵器 backward 的 absgrad atomicAdd 改成執行期開關，
#   renderer.absgrad_gate=true 時只在有人讀的步驟才算（增生期且 absgrad_densify>0；收割期與 absgrad 關時省掉）。
#   等價性：其他梯度不受影響（dL_dmean2D.z 從沒被讀）；舊呼叫端（bool）行為不變（bit1=0 => absgrad 照開）。
# 驗證（新光柵器裝到 logs/rast_absgate_pkg，用 PYTHONPATH 指過去 => 共用環境不動；cs60_sfmdup4 b6 @14,999，6 台相機）：
#   B  共用環境 vs 新 .so（absgrad 不碰、lean 開）  => 新 .so 的舊路徑沒變：渲染／radii／覆蓋數逐位元相同
#   N  新 .so absgrad 開跑兩次                     => 梯度噪音底
#   G  新 .so absgrad 開 vs 關                     => 渲染逐位元相同、xy 與參數梯度在噪音底內（≤ 10 倍）、關的一邊 z 全為 0；backward 時間
# 判定寫 logs/absgrad_gate.json（pass/fail＋backward ms）。
# 用法：[solo] bash scripts/lab/task_absgrad_gate.sh            驗證（不動共用環境）
#       [solo] bash scripts/lab/task_absgrad_gate.sh install    判定 pass 才把新光柵器裝進 gspl（要在比較家族開跑前，避免同家族混 .so）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
G=$(conda run -n gspl python -c "import sys;print(sys.prefix)" 2>/dev/null | tail -1)
RAST=submodules/diff-surfel-rasterization-trim-pp
J=logs/absgrad_gate.json
if [ "${1:-}" = install ]; then
  st=$(python3 -c "import json;print(json.load(open('$J'))['status'])" 2>/dev/null)
  [ "$st" = pass ] || { echo "⛔ 判定不是 pass（status=[$st]）=> 不安裝"; exit 1; }
  rm -rf $RAST/build
  conda run -n gspl env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --force-reinstall $RAST 2>&1 | tail -2
  conda run -n gspl python -c "import diff_trim_surfel_rasterization as m,os,time;p=os.path.dirname(m.__file__);f=[x for x in os.listdir(p) if x.endswith('.so')][0];print('so:',f,time.strftime('%m-%d %H:%M',time.localtime(os.path.getmtime(os.path.join(p,f)))));F=m.GaussianRasterizationSettings._fields;assert all(k in F for k in ('lean_render','record_reduce','absgrad'));print('✅ absgrad 欄位在')" \
    || { echo "⛔ 安裝後驗證失敗"; exit 1; }
  python3 -c "import json;d=json.load(open('$J'));d['status']='installed';json.dump(d,open('$J','w'),ensure_ascii=False,indent=1)"
  echo "✅ 已裝進 gspl（預設行為不變；要省就在 renderer 開 absgrad_gate）"
  exit 0
fi
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
T=logs/rast_absgate_pkg; S=logs/rast_absgate_src; O=logs/absgrad_gate_dump
CK=$(ls outputs/${PFX}cs60_sfmdup4/blocks/block_6/checkpoints/*step=14999.ckpt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "⛔ 找不到 cs60_sfmdup4 b6 的 14999 ckpt"; exit 2; }
L=logs/absgrad_gate_$(date +%m%d_%H%M).log
PY="conda run -n gspl --no-capture-output"
{ bad=0
  echo "════ 建新光柵器到 $T（不動共用環境）$(date)"
  rm -rf "$T" "$S"; cp -r $RAST "$S"; rm -rf "$S/build"
  $PY env CUDA_HOME="$G" CUDA_PATH="$G" TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --target "$T" "$S" 2>&1 | tail -1
  $PY env PYTHONPATH="$T" python -c "import diff_trim_surfel_rasterization as m; assert 'absgrad' in m.GaussianRasterizationSettings._fields; print('新:', m.__file__)" || { echo "⛔ 新光柵器沒裝好"; exit 3; }
  mkdir -p "$O"
  D="python tools/check_lean_render.py dump --ckpt $CK --lean 1 --repeat 20"
  $PY $D --out "$O/inst.pt" || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/new.pt" || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/A1.pt" --absgrad 1 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/A2.pt" --absgrad 1 || bad=1
  $PY env PYTHONPATH="$T" $D --out "$O/A0.pt" --absgrad 0 || bad=1
  C="python tools/check_lean_render.py compare"
  $PY $C "$O/inst.pt" "$O/new.pt" --label "B 共用環境 vs 新 .so（absgrad 不碰）：舊路徑沒變" || bad=1
  $PY $C "$O/A1.pt" "$O/A2.pt" --label "N 噪音底：新 .so absgrad 開跑兩次" || bad=1
  $PY $C "$O/A1.pt" "$O/A0.pt" --label "G absgrad 開 vs 關" || bad=1
  rm -f "$O"/*.pt
  echo "bad=$bad"
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$L"
# 判定（寫在結果出來前）：B 與 G 渲染逐位元相同（✅ 行）、G 關的一邊 z 全 0 而開的一邊不是、G 的參數與 xy 梯度相對差 <= 10 x N 的
python3 - "$L" "$J" <<'PY'
import json, re, sys
t = open(sys.argv[1], encoding="utf-8").read()
sec = {k: t.split("══ " + k, 1)[1].split("══ ", 1)[0] if ("══ " + k) in t else "" for k in ("B ", "N ", "G ")}
ok_render = all("✅" in sec[k] for k in ("B ", "G "))
rel = lambda s: [float(x) for x in re.findall(r"相對(?:（÷最大幅度）)? *([0-9.]+e[-+][0-9]+)", s)]
m = re.search(r"absgrad z +A 非零 ([\d,]+) +B 非零 ([\d,]+)", sec["G "])
z_ok = bool(m) and int(m.group(1).replace(",", "")) > 0 and int(m.group(2).replace(",", "")) == 0
gN, gG = rel(sec["N "]), rel(sec["G "])
# G 的 viewspace(+absgrad) 那一行含 z（本來就不同）=> 去掉第 len(params)+1 項；這裡保守：取參數梯度（除了最後兩行 viewspace）比
grad_ok = bool(gN) and bool(gG) and max(gG[:-2] or [0]) <= 10 * max(max(gN[:-2] or [0]), 1e-7) and gG[-1] <= 10 * max(gN[-1], 1e-7)
bw = re.search(r"backward ([0-9.]+) -> ([0-9.]+)（([-+0-9.]+)%）", sec["G "])
d = {"status": "pass" if (ok_render and z_ok and grad_ok and "bad=0" in t) else "fail",
     "render_bitident": ok_render, "z_zero_when_off": z_ok, "grads_within_noise": grad_ok,
     "backward_ms_on": float(bw.group(1)) if bw else None, "backward_ms_off": float(bw.group(2)) if bw else None,
     "backward_delta_pct": float(bw.group(3)) if bw else None, "log": sys.argv[1]}
json.dump(d, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("判定:", json.dumps(d, ensure_ascii=False))
PY
