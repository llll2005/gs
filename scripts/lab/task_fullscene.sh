#!/bin/bash
# ★★★★★★ best1 系列的 4x4 全場景（2026-10-17 使用者：自適應 cap 的提案「可以排」；`紀錄/實驗分析/14` §3.3）
#   task_full44.sh 寫死舊配方（conic、full44_best、60000 步）；這支給任何 g44 家族跑次用：
#   每塊就是 `task_cmp.sh <塊> <臂>`（CITYGS_GRID=44）=> 16 塊都落在 outputs/lab/<跑次>/blocks/block_<B>，與單變數臂同一份配方。
# 用法（佇列）：
#   [solo] bash scripts/lab/task_fullscene.sh merge <跑次>          16 塊都跑到同一步數才合併（utils/merge_citygs_ckpts.py）
#   [solo] bash scripts/lab/task_fullscene.sh test <跑次>           官方 741 幀 held-out（tools/eval_official_test.py；與 full44／官方同一支工具）
#   [solo] bash scripts/lab/task_fullscene.sh halftest <跑次>       部署端半精度（tools/half_ckpt.py，輸出 deploy/merged_fp16.ckpt）＋官方 741 幀
#   [solo] bash scripts/lab/task_fullscene.sh alloc <探針跑次> <frac> <STEPS>
#          自適應 cap ①：tools/price_alloc.py 用共同價格 lambda（V/C 門檻）把 frac x 探針總顆數分給 16 塊，
#          再把判決實驗 ② 插到佇列最前：同 Σcap 的「統一 cap」與「價格分配 cap」各 16 塊＋merge＋test。
#          兩臂起始份數同一條規則：dup 取 <= 4 中起始 <= 0.9 cap 的最大值（缺的 PLY 現場產）。
#   [cpu]  bash scripts/lab/task_fullscene.sh maskplan <STEPS> <cap...>
#          ★ 10-22 遮罩臂的 4x4 全場景（使用者：遮罩＋dist100＋30k 的 2.5M 與 1M，各加融合 Adam＋累加式 trim，都跑 4x4）：
#          補產缺的塊外遮罩（tools/block_mask44.py，CPU 每塊約 3 分）＋逐塊 init 跟 cap 走（scripts/lab/pick_dup.sh <塊> <cap> 1）
#          => 每個 cap 16 塊＋merge＋test 插到佇列最前；臂＝MASKPLAN_ARM（預設 best1maskdistrf）、跑次 g44s<k>km<cap/1e5>_<臂>
#          佇列註解：MASKPLAN_NOTE（訓練行）／MASKPLAN_TAG（合併、測試行）
#          多個 cap 一次給（同一個行程依序做，避免兩個前處理同時產遮罩／搶 make_sfm_fill_init 的鎖）
#   環境變數：FULLSCENE_ARM（預設 best1run）、FULLSCENE_FLAGS（每塊額外旗標；預設 record_reduce＋absgrad_gate，與單變數臂相同）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u -o pipefail
cd "$(dirname "$0")/../.." || exit 1
MODE=$1; shift
D=data/matrix_city/aerial/train/block_all
INIT=$D/sfmfill_sweep44
PY="conda run -n gspl --no-capture-output python -u"
ARM=${FULLSCENE_ARM:-best1run}
FLAGS=${FULLSCENE_FLAGS:---model.renderer.init_args.record_reduce true --model.renderer.init_args.absgrad_gate true}
mkdir -p logs
npts () { python3 - "$1" <<'EOF'
import sys
with open(sys.argv[1], "rb") as f:
    for ln in f:
        if ln.startswith(b"element vertex"):
            print(int(ln.split()[2])); break
EOF
}
maxstep () { ls "$1"/checkpoints/*.ckpt 2>/dev/null | grep -oE 'step=[0-9]+' | cut -d= -f2 | sort -n | tail -1; }

case "$MODE" in
  merge)
    R=${1:?跑次}; OUT=outputs/lab/$R
    st=; bad=
    for b in $(seq 0 15); do
      s=$(maxstep "$OUT/blocks/block_$b"); [ -n "$s" ] || { bad="$bad $b"; continue; }
      [ -z "$st" ] && st=$s; [ "$s" = "$st" ] || bad="$bad $b(step $s)"
    done
    [ -z "$bad" ] || { echo "⛔ $R：16 塊沒有都跑到同一步數（缺／不符：$bad）"; exit 2; }
    mkdir -p "$OUT/aborted_blocks"
    for d in "$OUT"/blocks/*.aborted_*; do [ -e "$d" ] && mv "$d" "$OUT/aborted_blocks/"; done
    $PY utils/merge_citygs_ckpts.py "$OUT" 2>&1 | grep -vE 'pkg_resources|declare_namespace' | tail -5; rc=${PIPESTATUS[0]}
    ls -la "$OUT/checkpoints/" 2>/dev/null; exit "$rc" ;;

  test)
    R=${1:?跑次}; OUT=outputs/lab/$R
    CK=$(ls "$OUT/checkpoints/"*.ckpt 2>/dev/null | tail -1)
    [ -n "$CK" ] || { echo "⛔ $R 沒有合併 ckpt（先 merge）"; exit 2; }
    L=logs/fullscene_test_${R}_$(date +%m%d_%H%M).log
    { echo "════ $R 合併模型：$CK（$(du -h "$CK" | cut -f1)）════"
      $PY tools/eval_official_test.py --ckpt "$CK" --save_dir "$OUT/test_official"
    } 2>&1 | grep -vE 'pkg_resources|declare_namespace' | tee "$L"
    exit "${PIPESTATUS[0]}" ;;

  halftest)
    # ★ 2026-10-24 部署端半精度（tools/half_ckpt.py；紀錄/加速論文對照.md C8）：合併模型 means 以外轉 float16、丟 Adam => 官方 741 幀
    #   ⚠ 輸出放 deploy/（不放 checkpoints/：test 模式用 `ls checkpoints/*.ckpt | tail -1` 挑檔，merged_fp16 會排在 merged 後面被誤用）
    R=${1:?跑次}; OUT=outputs/lab/$R
    CK=$(ls "$OUT/checkpoints/"*.ckpt 2>/dev/null | tail -1)
    [ -n "$CK" ] || { echo "⛔ $R 沒有合併 ckpt（先 merge）"; exit 2; }
    mkdir -p "$OUT/deploy"; HK="$OUT/deploy/merged_fp16.ckpt"
    L=logs/fullscene_halftest_${R}_$(date +%m%d_%H%M).log
    { echo "════ $R 半精度部署：$CK -> $HK ════"
      $PY tools/half_ckpt.py "$CK" "$HK" || exit 2
      $PY tools/eval_official_test.py --ckpt "$HK" --save_dir "$OUT/test_official_fp16"
    } 2>&1 | grep -vE 'pkg_resources|declare_namespace' | tee "$L"
    exit "${PIPESTATUS[0]}" ;;

  alloc)
    R=${1:?探針跑次}; FR=${2:?frac}; ST=${3:?STEPS}
    PCT=$(python3 -c "print(int(round($FR*100)))"); K=$((ST / 1000))
    J=logs/price_alloc_${R}_f${PCT}.json
    $PY tools/price_alloc.py --run "outputs/lab/$R" --frac "$FR" --out "$J" 2>&1 | grep -vE 'pkg_resources|declare_namespace|WARNING depth' \
      | tee "logs/price_alloc_${R}_f${PCT}.log"
    [ "${PIPESTATUS[0]}" = 0 ] && [ -f "$J" ] || { echo "⛔ price_alloc 失敗"; exit 2; }
    UC=$(python3 -c "import json;print(json.load(open('$J'))['uniform_cap'])")
    dupfor () {   # dupfor <塊> <cap> => 印出份數（<= 4 中起始 <= 0.9cap 的最大值；缺 PLY 就產）
      local b=$1 cap=$2 k F n
      for k in 4 3 2 1; do
        F="$INIT/dup$k/block_$b.ply"
        if [ ! -f "$F" ]; then
          mkdir -p "$INIT/dup$k"
          $PY tools/make_sfm_fill_init.py "$D" --blocks "$b" --block_dim 4 4 --out-dir "$INIT/dup$k" \
            --fill-ratio 0.35 --fill-voxel 0.35 --dup "$k" >/dev/null 2>&1 || { echo "⛔ 產 $F 失敗" >&2; return 1; }
        fi
        n=$(npts "$F"); [ "${n:-0}" -le $((cap * 9 / 10)) ] && { echo "$k"; return 0; }
      done
      echo 1; echo "⚠ block $b：dup 1 起始 $(npts "$INIT/dup1/block_$b.ply") 仍 > 0.9 x $cap（起始即超額 regime）" >&2
    }
    INS=$(mktemp)
    for arm in u p; do
      FAM="g44s${K}k${arm}${PCT}_"
      for b in $(seq 0 15); do
        if [ "$arm" = u ]; then cap=$UC; lbl="統一 cap $UC"; else cap=$(python3 -c "import json;print(json.load(open('$J'))['caps']['$b'])"); lbl="價格分配 cap $cap"; fi
        k=$(dupfor "$b" "$cap") || exit 2
        echo "# ★★★★★★ 使用者 10-17 自適應 cap ②：${lbl}（Σcap 同；frac $FR、${K}k）b$b dup$k" >> "$INS"
        echo "[solo] CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=$ST CITYGS_CAP=$cap CITYGS_DUP44=$k CITYGS_FAMILY=$FAM bash scripts/lab/task_cmp.sh $b $ARM $FLAGS" >> "$INS"
      done
    done
    for arm in u p; do
      RN="g44s${K}k${arm}${PCT}_${ARM}"
      printf '# ★★★★★★ 自適應 cap ②：合併 %s\n[solo] bash scripts/lab/task_fullscene.sh merge %s\n# ★★★★★★ 自適應 cap ②：官方 741 幀 %s\n[solo] bash scripts/lab/task_fullscene.sh test %s\n' "$RN" "$RN" "$RN" "$RN" >> "$INS"
    done
    Q=scripts/queue.txt; cp "$Q" "scripts/queue.bak_alloc_$(date +%m%d_%H%M)"
    python3 - "$Q" "$INS" <<'EOF'
import sys
q, ins = sys.argv[1], open(sys.argv[2]).read()
L = open(q).read().splitlines(True)
i = next((k for k, l in enumerate(L) if l.strip() and not l.startswith('#')), len(L))
while i > 0 and L[i - 1].startswith('#'):      # 連同上方的註解塊一起往後推
    i -= 1
open(q, 'w').write(''.join(L[:i]) + ins + ''.join(L[i:]))
EOF
    echo "=== 已插到佇列最前：$(grep -c '^\[solo\]' "$INS") 行 ==="; grep -v '^#' "$INS" | cut -c1-160
    rm -f "$INS" ;;

  maskplan)
    ST=${1:?STEPS}; shift; [ $# -ge 1 ] || { echo "⛔ 至少給一個 cap"; exit 2; }
    A=${MASKPLAN_ARM:-best1maskdistrf}; K=$((ST / 1000))
    miss=; for b in $(seq 0 15); do [ -f "$D/block_masks44/block_$b/stats.json" ] || miss="$miss $b"; done
    if [ -n "$miss" ]; then
      echo "=== 產塊外遮罩：block$miss $(date) ==="
      $PY tools/block_mask44.py --blocks $miss --check 2>&1 | grep -vE 'pkg_resources|declare_namespace' | grep -E 'block [0-9]+：[0-9]+ 張|SfM 檢查|⛔|Error|Traceback' \
        | tee "logs/block_mask44_fullscene_$(date +%m%d_%H%M).log"
    fi
    for b in $(seq 0 15); do [ -f "$D/block_masks44/block_$b/stats.json" ] || { echo "⛔ block $b 沒有遮罩"; exit 2; }; done
    INS=logs/maskplan_ins_$$.txt; : > "$INS"
    for CAP in "$@"; do
      CM=$(python3 -c "print(f'{int(round($CAP / 1e5)):02d}')"); FAM="g44s${K}km${CM}_"; RN="${FAM}${A}"
      for b in $(seq 0 15); do
        k=$(bash scripts/lab/pick_dup.sh "$b" "$CAP" 1 | tail -1)
        case "$k" in ''|*[!0-9]*) echo "⛔ block $b cap $CAP 算不出 dup（$k）"; rm -f "$INS"; exit 2 ;; esac
        echo "  cap $CAP block $b => 裁盒 dup$k"
        echo "# ★★★★★★ ${MASKPLAN_NOTE:-使用者 10-22：遮罩＋dist100＋融合 Adam＋累加式 trim} ${K}k cap $CAP 全場景 b$b dup$k" >> "$INS"
        echo "[solo] CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=$ST CITYGS_CAP=$CAP CITYGS_DUP44=$k CITYGS_FAMILY=$FAM bash scripts/lab/task_cmp.sh $b $A $FLAGS" >> "$INS"
      done
      printf '# ★★★★★★ %s：合併 %s\n[solo] bash scripts/lab/task_fullscene.sh merge %s\n# ★★★★★★ %s：官方 741 幀 %s\n[solo] bash scripts/lab/task_fullscene.sh test %s\n' "${MASKPLAN_TAG:-使用者 10-22}" "$RN" "$RN" "${MASKPLAN_TAG:-使用者 10-22}" "$RN" "$RN" >> "$INS"
    done
    Q=scripts/queue.txt; cp "$Q" "scripts/queue.bak_maskplan_$(date +%m%d_%H%M)"
    python3 - "$Q" "$INS" <<'PYINS'
import sys
q, ins = sys.argv[1], open(sys.argv[2]).read()
L = open(q).read().splitlines(True)
i = next((k for k, l in enumerate(L) if l.strip() and not l.startswith('#')), len(L))
while i > 0 and L[i - 1].startswith('#'):      # 連同上方的註解塊一起往後推
    i -= 1
open(q, 'w').write(''.join(L[:i]) + ins + ''.join(L[i:]))
PYINS
    echo "=== 已插到佇列最前：$(grep -c '^\[solo\]' "$INS") 行 ==="; grep -v '^#' "$INS" | cut -c1-170
    rm -f "$INS" ;;

  *) echo "未知模式 $MODE"; exit 2 ;;
esac
