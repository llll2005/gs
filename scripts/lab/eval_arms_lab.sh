#!/bin/bash
# ★ 2026-11-09 候選臂的全面向評估（本機 scratchpad eval_arms_1108.sh 的 lab 版；使用者「換成現行配方」＋「所有面向」）
#   held-out（塊內 MatrixCity test 子集）、幾何 F1、地面／地下／地上層、貼地近樓／遠離樓、立面貼牆、CityGS 立面朝向依高度、海面上方、路燈與車、點精靈。
#   時間＝tfevents 首尾；11-09 起候選臂全部 [solo] => 與同塊現行配方（也是 [solo]）可比，看相對差（判準：本機輕量版的時間預算，換成 lab 上「不比基準慢超過雜訊」）。
#   CityGS 參考＝../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/blocks/block_<B>（與本機 outputs/official_release 是同一批模型；本機那份裁到塊內，量測本身也只看塊內）。
# 用法（佇列）：[solo] bash scripts/lab/eval_arms_lab.sh <塊> <標籤> 名稱=跑次 ...      （跑次帶臂名，例 g44s30ksfm10nuPF2DB2_best1maskdistrfed10b4；第一個當基準）
#   輸出 logs/eval_arms_<標籤>.log；EVAL_ONLY=f1,ground,...（逗號分隔的節名）只跑那幾節
cd "$(dirname "$0")/../.." || exit 1
B=${1:?塊}; TAG=${2:?標籤}; shift 2
# ★ 11-14 EVAL_ROOT：跑次所在的根（預設 outputs/lab；本機跑次用 outputs/local，使用者「lab 等太久，抓資料在本機跑」）
ROOT=${EVAL_ROOT:-outputs/lab}
PY="conda run -n gspl --no-capture-output python -u"
F='Warn\|warn\|pkg_res\|declare_namespace\|rank_zero'
CGK=${CITYGS_REF:-$(ls ../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/blocks/block_$B/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)}
[ -n "$CGK" ] || { echo "⛔ 找不到 CityGS b$B 參考 ckpt"; exit 2; }
CG="CityGS=$CGK"
M=(); RUNS=()
for x in "$@"; do
  c=$(ls $ROOT/${x#*=}/blocks/block_$B/checkpoints/*step=30000.ckpt 2>/dev/null | head -1)
  # 11-09 coarse 排程臂（20k／15k）沒有 30k ckpt => 取步數最大的那個（＝該跑次的終點；同 task_full44.sh swap）
  [ -n "$c" ] || c=$(ls $ROOT/${x#*=}/blocks/block_$B/checkpoints/*step=*.ckpt 2>/dev/null | grep -v xyz_rgb | awk -F'step=' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2)
  if [ -n "$c" ]; then M+=("${x%%=*}=$c"); RUNS+=("$x"); else echo "⚠ 缺 $x"; fi
done
[ ${#M[@]} -ge 1 ] || { echo "⛔ 沒有可評估的 ckpt"; exit 2; }
# ★ 11-09 lab 容器沒有 X：open3d（F1 用）載入時缺 libX11／libGLX => 只對 F1 這一步把 conda 套件快取裡現成的 X11／GL 函式庫加進 LD_LIBRARY_PATH（不安裝任何東西）。
#   11-09 第一次評估的 F1 因此整節空白（grep 只留 F1 行，錯誤被濾掉）。
pk() { ls -d /root/miniconda3/pkgs/$1-[0-9]*/lib 2>/dev/null | tail -1; }
O3D_LP=$(for k in xorg-libx11 libxcb xorg-libxau xorg-libxdmcp xorg-libxext libgl libglx libopengl libglvnd libegl; do pk $k; done | tr '\n' ':')
want() { [ -z "$EVAL_ONLY" ] || [[ ",$EVAL_ONLY," == *",$1,"* ]]; }
mkdir -p logs; LOG=logs/eval_arms_$TAG.log
{
printf '%s\n' "$CG" "${M[@]}"
if want time; then
echo "### 時間（tfevents 首尾；[solo] 跑次之間可比；★ 11-14 有 run_summary.json 就用它的純訓練＝第一步 → 最後一步，demo 模式沒有 tfevents）"
EVAL_ROOT=$ROOT $PY - "$B" "${RUNS[@]}" <<'PYEOF' 2>&1 | grep -v "$F"
import glob, sys, os, json
ROOT = os.environ.get("EVAL_ROOT", "outputs/lab")
from tensorboard.backend.event_processing import event_accumulator as ea
B = sys.argv[1]
for x in sys.argv[2:]:
    lab, r = x.split("=", 1)
    js = f"{ROOT}/{r}/blocks/block_{B}/run_summary.json"
    if os.path.exists(js):
        s = json.load(open(js, encoding="utf-8"))
        print(f"{lab:16} 純訓練（run_summary 第一步→最後一步）{(s.get('train_s') or 0) / 60:.1f} 分｜啟動 {s.get('startup_s')} s｜整個任務 {(s.get('task_total_s') or 0) / 60:.1f} 分｜最後一步 {s.get('steps')}｜N {s.get('n_final')}"); continue
    fs = sorted(glob.glob(f"{ROOT}/{r}/blocks/block_{B}/lightning_logs/*/events*"))
    if not fs: continue
    a = ea.EventAccumulator(fs[-1], size_guidance={ea.SCALARS: 0}); a.Reload(); ev = a.Scalars("train/loss")
    print(f"{lab:16} 純訓練（tfevents 首尾）{(ev[-1].wall_time - ev[0].wall_time) / 60:.1f} 分｜最後一步 {ev[-1].step}")
PYEOF
fi
want heldout && { echo "### heldout"; for m in "${M[@]}"; do echo "== ${m%%=*}"; $PY tools/eval_official_test.py --ckpt "${m#*=}" --block $B --block_dim 4 4 2>&1 | grep -E '^\[模型\]|^PSNR|^SSIM|^LPIPS'; done; }
want f1 && { echo "### F1"; C=(); N=(); for m in "${M[@]}"; do N+=("${m%%=*}"); C+=("${m#*=}"); done
LD_LIBRARY_PATH=$O3D_LP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} $PY tools/block_f1.py --block $B --merged none --stride 2 --cand "${C[@]}" --names "${N[@]}" --out $ROOT/f1_$TAG 2>&1 | grep -E 'F1 [0-9]|Error|error' | grep -v -i warn; }
want ground && { echo "### ground"; $PY tools/probe/ground_layer_1103.py $B "$CG" "${M[@]}" 2>&1 | grep -v "$F"; }
want ground_split && { echo "### ground split"; $PY tools/probe/ground_split_1109.py $B "${M[@]}" 2>&1 | grep -v "$F"; }
want facade && { echo "### facade"; for m in "${M[@]}"; do echo "== ${m%%=*}"; $PY tools/facade_orient.py --block $B --ckpt "${m#*=}" 2>&1 | grep -E '貼牆中位' | sed 's/.*塊內不透明/塊內不透明/'; done; }
want orient && { echo "### orient bands"; CITYGS_CKPT=$CGK $PY tools/probe/orient_band_1109.py $B "${M[@]}" 2>&1 | grep -v "$F"; }
want sea && { echo "### sea"; SEA_BLOCK=$B $PY tools/probe/sea_opacity_nosfm_1106.py "$CG" "${M[@]}" 2>&1 | grep -v "$F\|塊外" || echo "（海面量測失敗：這塊可能沒有海面格）"; }
want smallobj && { echo "### smallobj"; $PY tools/probe/smallobj_cpu_1109.py --block $B --out $ROOT/smallobj_$TAG --models "$CG" "${M[@]}" 2>&1 | grep -v "$F"; }
want sprite && { echo "### sprite"; SPRITE_BLOCK=$B $PY tools/probe/sprite_bands_1107.py "$CG" "${M[@]}" 2>&1 | grep -v "$F"; }
echo DONE
} 2>&1 | tee "$LOG"
