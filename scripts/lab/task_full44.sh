#!/bin/bash
# ★★★★★★ 我方現行最佳解的 4x4 全場景（2026-10-02 使用者要求）：與官方 CityGSV2（同為 16 塊）逐項比較，順便量 held-out。
#
# 配方 = cs60_sfmdup4 那一份（conic 開 + dup4 init + speed3 其餘設定，60k），**直接呼叫 task_cmp.sh 的 conic 臂**
#   => 與 60k 比較組逐項相同，不會因為另抄一份旗標而漂移。只多三項：
#     --data.parser.block_dim [4,4]       切塊換成 4x4（我方 partition_from_colmap，不需 coarse；與 5x5 分開存，不會覆寫）
#     --data.parser.points_from ply ...   本塊的 dup init
#     --model.density.init_args.churn_report true   純打印（每次 densify 印 dead->relocate／新增），不改訓練行為
#   ⚠ cap 維持 2.6M／塊：這是 6GB 信封決定的（N 主導 VRAM），不是隨塊面積調的參數 => 4x4 每塊面積是 5x5 的 1.56 倍，
#     單位面積顆數較少；這正是「6GB 上做 16 塊」的真實條件。
#   ⚠ dup 份數**自適應**：dup4 在 5x5 塊的起始顆數 ~2.3M（剛好在 0.9cap 下）；4x4 塊 SfM 點更多，dup4 可能超過 cap，
#     而起始 > cap 時 add_new_gs 加 0、N 只被 trim 衰減（那是 sfmdup5 在測的另一個 regime）。
#     => 每塊從 4 份往下試，取起始 <= 0.9 x cap 的最大份數（記在 logs/full44_init.tsv）。dup4 的機制是「起始就接近滿額」，
#        這樣做保留的是機制，不是字面上的 4。
#
# 時間：訓練走 lab 三槽平行（品質／離線 Load／峰值 VRAM 與鄰居無關）；**時間與 FPS 一律用 [solo] 量**：
#   res  逐塊 step_breakdown（同官方 res 模式用的工具）＋合併模型離線 Load（共同量尺）＋儲存
#   test 官方 held-out 741 幀（同一份 block_all_test_official2、同一支評分工具），並用同一支工具**重評官方合併模型**
#        當交叉驗證：工具對官方模型要重現官方 main.py test 的 25.79/0.833/0.176，我方數字才可與之並列。
#
# ★ 2026-10-03 使用者：「等速度第二批裝好再開」，且排在單變數 60k 臂之前 =>
#   prep 先讀 logs/speed2_gate.json（task_speed2_check.sh gate 寫的）：installed／no_gain 才開，fail 或沒有判定就拒絕（exit 3）；
#   16 個 block 行插到**佇列最前**，每行帶 CITYGS_LEAN=1 與判定通過的旗標（寫在佇列行上 => resolved config 看得到）。
#   要跳過這道閘門：FULL44_SKIP_SPEED2=1（只開 lean）。
# 用法（佇列）：
#   [cpu] bash scripts/lab/task_full44.sh prep
#   CITYGS_LEAN=1 bash scripts/lab/task_full44.sh block <0..15> [額外 CLI 覆寫...]
#   [solo] bash scripts/lab/task_full44.sh merge
#   [solo] bash scripts/lab/task_full44.sh test
#   [solo] bash scripts/lab/task_full44.sh res
set -u
cd "$(dirname "$0")/../.." || exit 1
MODE=${1:?用法: task_full44.sh prep | block <id> | merge | test | res}
GATE=logs/speed2_gate.json
D=data/matrix_city/aerial/train/block_all
PART=$D/partition/partitions-dim_4_4_visibility_0.08
INIT=$D/sfmfill_sweep44
CAP=2600000
START_MAX=$((CAP * 9 / 10))
TSV=logs/full44_init.tsv
mkdir -p logs
if [ -f .lab_machine ]; then PFX=lab/; else case "$(pwd)" in */hdd/11213/*) PFX=lab/ ;; *) PFX= ;; esac; fi
NAME="${RUN_PREFIX-$PFX}full44_best"
OUT="outputs/$NAME"
OFF_CK=$(ls ../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/checkpoints/*.ckpt 2>/dev/null | tail -1)
npts () { python3 - "$1" <<'EOF'
import sys
with open(sys.argv[1], "rb") as f:
    for ln in f:
        if ln.startswith(b"element vertex"):
            print(int(ln.split()[2])); break
EOF
}

case "$MODE" in
  prep)
    FL=
    if [ "${FULL44_SKIP_SPEED2:-0}" != 1 ]; then
      st=$(python3 -c "import json;print(json.load(open('$GATE'))['status'])" 2>/dev/null)
      case "$st" in
        installed|no_gain) ;;
        *) echo "⛔ 速度第二批還沒裝好（$GATE status=[${st:-沒有判定}]）=> 不開 4x4"
           echo "   （使用者 10-03：等速度第二批裝好再開；看 logs/speed2_check_*.log；要只開 lean 硬開：FULL44_SKIP_SPEED2=1）"; exit 3 ;;
      esac
      FL=$(python3 -c "import json;d=json.load(open('$GATE'));print(' '.join('--model.renderer.init_args.%s true' % f for f in d['flags']) if d['status']=='installed' else '')")
      echo "速度第二批判定：$st；4x4 每塊開 lean${FL:+ ＋ $FL}"
    fi
    if [ ! -f "$PART/partitions.pt" ]; then
      echo "=== 4x4 切塊（我方 partition_from_colmap；輸出 $PART）==="
      conda run -n gspl --no-capture-output python utils/partition_from_colmap.py "$D" --block_dim 4 4 --content_threshold 0.08 || exit $?
    else echo "略過切塊（$PART 已存在）"; fi
    for b in $(seq 0 15); do
      x=$((b % 4)); y=$((b / 4))
      n=$(grep -cv '^\s*$' "$PART/$(printf '%03d_%03d' $x $y).txt" 2>/dev/null || echo 0)
      echo "block $b：$n 台相機"
    done
    printf 'block\tdup\tN_start\n' > "$TSV"
    for b in $(seq 0 15); do
      ok=
      for k in 4 3 2 1; do
        mkdir -p "$INIT/dup$k"
        F="$INIT/dup$k/block_$b.ply"
        if [ ! -f "$F" ]; then
          conda run -n gspl --no-capture-output python tools/make_sfm_fill_init.py "$D" --blocks "$b" --block_dim 4 4 \
            --out-dir "$INIT/dup$k" --fill-ratio 0.35 --fill-voxel 0.35 --dup "$k" | tail -2 || exit $?
        fi
        n=$(npts "$F")
        if [ "${n:-0}" -le "$START_MAX" ]; then
          printf '%s\t%s\t%s\n' "$b" "$k" "$n" >> "$TSV"; ok=1
          echo "✅ block $b：dup $k => 起始 $n（<= 0.9cap $START_MAX）"; break
        fi
        echo "   block $b：dup $k 起始 $n > $START_MAX，降一份"
      done
      [ -n "$ok" ] || { echo "⛔ block $b dup 1 仍超過 0.9cap"; exit 3; }
    done
    echo "=== 起始顆數表 $TSV ==="; cat "$TSV"
    # ★ 由 prep 自己把 16 個 block 行＋merge/test/res 插進佇列（同 task_sfmfill_sweep2.sh gen 的做法）：
    #   預先排的話，三槽平行下 block 行可能在 PLY 產完前就被空槽搶走 => rc=2 白白消耗。
    #   插在第一個 [solo] 行之前 => 與其他訓練行連成一批平行跑；merge 是 [solo]，會等 16 塊都跑完。
    Q=scripts/queue.txt
    if grep -q "task_full44.sh block" "$Q" 2>/dev/null; then echo "（佇列已有 full44 block 行，不重複插入）"; exit 0; fi
    _ins=$(mktemp)
    {
      for b in $(seq 0 15); do
        echo "# ★★★★★★ 我方最佳解 4x4 全場景（conic＋dup init，60k；對照官方 CityGSV2 16 塊）：block $b"
        echo "CITYGS_LEAN=1 bash scripts/lab/task_full44.sh block $b${FL:+ $FL}"
      done
      echo "# ★★★★★★ 我方 4x4：合併 16 塊"
      echo "[solo] bash scripts/lab/task_full44.sh merge"
      echo "# ★★★★★★ 我方 4x4：官方 held-out 741 幀（＋同工具重評官方模型當交叉驗證；FPS 要獨佔）"
      echo "[solo] bash scripts/lab/task_full44.sh test"
      echo "# ★★★★★★ 我方 4x4：資源（逐塊逐步計時、合併模型離線 Load、儲存）"
      echo "[solo] bash scripts/lab/task_full44.sh res"
    } > "$_ins"
    # ★ 2026-10-03 使用者：排在單變數 60k 臂之前 => 插到佇列**最前面**（prep 本身是 [cpu]，平行模式下啟動時已從佇列移除）
    #   （舊錨點是第一個 [solo] Load 比較行＝排在所有單變數臂之後，已不符使用者的順序）
    _ln=1
    if [ -s "$Q" ]; then
      while [ "$_ln" -gt 1 ] && [ "$(sed -n "$((_ln-1))p" "$Q" | cut -c1)" = "#" ]; do _ln=$((_ln-1)); done
      _tmp=$(mktemp)
      { head -n $((_ln-1)) "$Q"; cat "$_ins"; tail -n +"$_ln" "$Q"; } > "$_tmp"
      mv "$_tmp" "$Q"
      echo "✅ 已把 16 個 block 行＋merge/test/res 插到佇列最前"
    else
      cat "$_ins" >> "$Q"
      echo "✅ 已把 16 個 block 行＋merge/test/res 追加到佇列尾"
    fi
    rm -f "$_ins" ;;

  block)
    B=${2:?block id}; shift 2   # 其餘參數（例：record_reduce／tile_cull 旗標）原樣轉給 task_cmp.sh（最後者勝）
    [ -f "$TSV" ] || { echo "⛔ 缺 $TSV（先跑 prep）"; exit 2; }
    K=$(awk -v b="$B" '$1==b{print $2}' "$TSV")
    [ -n "$K" ] || { echo "⛔ $TSV 沒有 block $B"; exit 2; }
    P="sfmfill_sweep44/dup$K/block_$B.ply"
    [ -f "$D/$P" ] || { echo "⛔ 缺 $D/$P"; exit 2; }
    echo "block $B：init $P（dup $K）"
    # NCAM 只用來算預設 STEPS 與印出；STEPS 已明給 => 不影響配方
    NCAM=1 STEPS=60000 CITYGS_RUN_NAME="$NAME" CITYGS_FAMILY=full44_ \
      bash scripts/lab/task_cmp.sh "$B" conic \
        --data.parser.block_dim "[4,4]" \
        --data.parser.points_from ply --data.parser.ply_file "$P" \
        --model.density.init_args.churn_report true "$@" ;;

  merge)
    n=0; bad=
    for b in $(seq 0 15); do
      if ls "$OUT/blocks/block_$b/checkpoints/"*step=60000.ckpt >/dev/null 2>&1; then n=$((n + 1)); else bad="$bad $b"; fi
    done
    [ "$n" = 16 ] || { echo "⛔ 只有 $n/16 塊跑到 60000（缺：$bad）"; exit 2; }
    # 失敗殘留（*.aborted_*）不是 block_N 名稱，但 merge 會掃 blocks/ 底下所有目錄 => 先搬開
    mkdir -p "$OUT/aborted_blocks"
    for d in "$OUT"/blocks/*.aborted_*; do [ -e "$d" ] && mv "$d" "$OUT/aborted_blocks/"; done
    conda run -n gspl --no-capture-output python utils/merge_citygs_ckpts.py "$OUT" || exit $?
    ls -la "$OUT/checkpoints/" ;;

  test)
    CK=$(ls "$OUT/checkpoints/"*.ckpt 2>/dev/null | tail -1)
    [ -n "$CK" ] || { echo "⛔ 沒有合併 ckpt（先 merge）"; exit 2; }
    L=logs/full44_test_$(date +%m%d_%H%M).log
    {
      echo "════ 我方 4x4 合併模型：$CK ════"
      conda run -n gspl --no-capture-output python tools/eval_official_test.py --ckpt "$CK" --save_dir "$OUT/test_official" || exit $?
      # 交叉驗證只做一次（官方模型不會變）
      if [ -n "$OFF_CK" ] && ! ls logs/full44_test_offcheck.log >/dev/null 2>&1; then
        echo "════ 交叉驗證：同一支工具評官方合併模型 $OFF_CK（官方 main.py test = 25.79 / 0.833 / 0.176）════"
        conda run -n gspl --no-capture-output python tools/eval_official_test.py --ckpt "$OFF_CK" 2>&1 | tee logs/full44_test_offcheck.log
      fi
    } 2>&1 | grep -vE 'pkg_resources|declare_namespace' | tee "$L"
    exit "${PIPESTATUS[0]}" ;;

  res)
    CK=$(ls "$OUT/checkpoints/"*.ckpt 2>/dev/null | tail -1)
    [ -n "$CK" ] || { echo "⛔ 沒有合併 ckpt（先 merge）"; exit 2; }
    L=logs/full44_res_$(date +%m%d_%H%M).log
    { bad=0
      echo "════ 儲存 ════"
      echo "合併 ckpt：$(du -h "$CK" | cut -f1)  $CK"
      for b in $(seq 0 15); do
        c=$(ls "$OUT/blocks/block_$b/checkpoints/"*step=60000.ckpt 2>/dev/null | head -1)
        p=$(ls "$OUT/blocks/block_$b/checkpoints/"*step=60000-xyz_rgb.ply 2>/dev/null | head -1)
        echo "block $b：ckpt $(du -h "$c" 2>/dev/null | cut -f1)  ply $(du -h "$p" 2>/dev/null | cut -f1)"
      done
      echo "════ 訓練台帳（START/DONE：步數／N／it/s／峰值 VRAM；⚠ it/s 與牆鐘是三槽平行下的，不可與官方 solo 比）════"
      grep -F "$NAME" logs/quad_progress.log | tail -40
      echo "════ 合併模型離線 Load（共同量尺：同工具、同光柵器量雙方）════"
      conda run -n gspl --no-capture-output python tools/cost_budget_calibrate.py --ckpt "$CK" --max-cam "${CITYGS_MAXCAM:-600}" || { echo "⛔ 離線 Load 失敗"; bad=1; }
      for b in $(seq 0 15); do
        echo "── 逐步計時（[solo]）：我方 block $b ──"
        conda run -n gspl --no-capture-output python tools/step_breakdown.py --run "$NAME" --block "$b" --repeat 20 \
          || { echo "⛔ 逐步計時失敗（block $b）"; bad=1; }
      done
      exit "$bad"; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$L"
    exit "${PIPESTATUS[0]}" ;;
  *) echo "⛔ 不認得的模式：$MODE"; exit 2 ;;
esac
