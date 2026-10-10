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
#   [solo] bash scripts/lab/task_full44.sh offheldout [塊...]  官方逐塊模型在**我方同一批塊視角**上的 held-out（同塊對照；預設 0~3）
#          CITYGS_OFF_LINE=paper 改評照論文設定那條線（trim 執行＋ω 0.9、prune 0.025）=> official_heldout_paper.txt
#   [solo] bash scripts/lab/task_full44.sh stepprof [塊...]   每塊兩段真實迴圈逐段計時 => 推算獨佔下整趟 60k 要多久（見該模式註解）
#   [solo] bash scripts/lab/task_full44.sh failheld [ours|release|paper ...]  三個合併模型的 741 幀 held-out＋失敗 tile，按 4x4 塊拆開（12）
#   bash scripts/lab/task_full44.sh swap <塊> [4x4 跑次...]   換塊評測：合併模型只換第 B 塊（＋自我檢查），741 幀中該塊視角（定案用；tools/swap_block_eval.py）
#   [solo] bash scripts/lab/task_full44.sh prunecurve [ours|release|paper ...]  合併模型依 opacity 只留 100/90/75/50/25%，官方 741 幀 held-out（同一支工具；09g）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
MODE=${1:?用法: task_full44.sh prep | block <id> | merge | test | res | offheldout | stepprof | prunecurve}
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
    # ★ 2026-10-03：速度旗標在**開跑時**讀判定（logs/speed2_gate.json），不只看 prep 當時寫進佇列行的。
    #   原因：第一版判定器漏算 trim（判成 no_gain），修正重判時 prep 已經在跑、佇列行已定 => 開跑時讀才跟得上。
    #   旗標仍會出現在 resolved config（renderer.init_args），可追溯。
    GF=$(python3 -c "import json;d=json.load(open('$GATE'));print(' '.join('--model.renderer.init_args.%s true' % f for f in d['flags']) if d['status']=='installed' else '')" 2>/dev/null)
    [ -n "$GF" ] && echo "速度旗標（$GATE）：$GF"
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
        --model.density.init_args.churn_report true "$@" $GF ;;

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
  failheld)
    # ★ 2026-10-10 使用者（圖 12）：官方的同一失敗區統計。val 兩邊不是同一批影像 => 改用官方 741 幀 held-out。
    #   ⚠ 改版（同日）：原本評**單塊**模型，但單塊 held-out 主要量到「帶了多少塊外內容」（b12：我方單塊 22.55／合併 27.82，
    #     官方單塊 26.39／合併 27.10；`紀錄/實驗分析/09` §0.2b）=> 改評三個**合併**模型的全部 741 幀（逐幀 CSV 含失敗 tile 計數），
    #     再用 tools/merged_heldout_by_block.py 按 4x4 塊拆開。判準同 failure_map：48px、GT std>=0.10、corr<0.6。
    #   ⚠ 分母是全部 tile（held-out 影像不在 SfM 裡）=> 只在本表內互比，不和 `12` 的 val 失敗率比。
    shift; WL=("$@"); [ ${#WL[@]} -gt 0 ] || WL=(ours release paper)
    bad=0; CS=()
    for w in "${WL[@]}"; do
      case "$w" in
        ours)    ck=$(ls "$OUT/checkpoints/"*.ckpt 2>/dev/null | tail -1) ;;
        release) ck=$OFF_CK ;;
        paper)   ck=$(ls ../cityGS_origin_trimfix/outputs/citygsv2_mc_aerial_sh2_paper/checkpoints/*.ckpt 2>/dev/null | tail -1) ;;
        *) echo "⛔ 不認得 $w（ours|release|paper）"; bad=1; continue ;;
      esac
      [ -n "$ck" ] || { echo "⛔ $w 沒有合併 ckpt"; bad=1; continue; }
      C=logs/full44_fail_${w}.csv
      echo "════ $w：$ck"
      conda run -n gspl --no-capture-output python tools/eval_official_test.py --ckpt "$ck" --per_image_csv "$C" 2>&1 \
        | grep -vE "\.\.\.[0-9]+/[0-9]+ +PSNR|pkg_resources|declare_namespace" | grep -E "^PSNR|^SSIM|^LPIPS|^失敗 tile|⛔|Error"
      [ "$(wc -l < "$C" 2>/dev/null)" -ge 742 ] && CS+=("$w=$C") || { echo "⛔ $w 逐幀 CSV 不完整"; bad=1; }
    done
    echo "════ 按 4x4 塊拆開"
    conda run -n gspl --no-capture-output python tools/merged_heldout_by_block.py --csv "${CS[@]}" 2>&1 | grep -v pkg_resources | tee logs/full44_failheld.txt
    exit "$bad" ;;
  swap)
    # ★ 2026-10-10 使用者（「被淘汰的方案會不會只是塊外表現差」）：單塊 val／held-out 都混了塊外內容，合併會丟掉 =>
    #   定案用換塊評測：本合併模型（full44_best）只把第 B 塊換成候選（4x4 訓練）的塊內顆粒，其他 15 塊不變。--sanity 把自己的塊換回去當檢查。
    shift; B=${1:?缺塊}; shift
    cks=(); ns=()
    for r in "$@"; do
      c=$(ls outputs/${PFX}$r/blocks/block_$B/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
      [ -n "$c" ] || { echo "⛔ $r block $B 沒有 60k ckpt"; exit 2; }
      cks+=("$c"); ns+=("$r")
    done
    L=logs/swap_b${B}_$(date +%m%d_%H%M).log
    XA=(); [ ${#cks[@]} -gt 0 ] && XA=(--cand "${cks[@]}" --names "${ns[@]}")
    conda run -n gspl --no-capture-output python tools/swap_block_eval.py --merged_run "$OUT" --block "$B" --sanity "${XA[@]}" 2>&1 | grep -vE "pkg_resources|declare_namespace" | tee "$L"
    exit "${PIPESTATUS[0]}" ;;
  prunecurve)
    # ★ 2026-10-10 使用者（圖 09g）：官方剪枝曲線旁邊加我方的同類統計。舊的官方曲線用官方 main.py test（含暗幀偏差、絕對值低約 1.5 dB），
    #   這裡三個合併模型都改用**我方評分工具**（eval_official_test.py --opacity_keep，載入一次、逐比例剪完評分；與 official_prune_ckpt.py 同規則：
    #   依 sigmoid 後 opacity 由高到低保留、並列依索引）=> 同一把尺。結果：logs/full44_prunecurve_<線>.tsv（keep N psnr ssim lpips）。
    shift; WL=("$@"); [ ${#WL[@]} -gt 0 ] || WL=(ours release paper)
    bad=0
    for w in "${WL[@]}"; do
      case "$w" in
        ours)    ck=$(ls "$OUT/checkpoints/"*.ckpt 2>/dev/null | tail -1) ;;
        release) ck=$OFF_CK ;;
        paper)   ck=$(ls ../cityGS_origin_trimfix/outputs/citygsv2_mc_aerial_sh2_paper/checkpoints/*.ckpt 2>/dev/null | tail -1) ;;
        *) echo "⛔ 不認得 $w（ours|release|paper）"; bad=1; continue ;;
      esac
      [ -n "$ck" ] || { echo "⛔ $w 沒有合併 ckpt"; bad=1; continue; }
      L=logs/full44_prunecurve_${w}_$(date +%m%d_%H%M).log
      echo "════ $w：$ck"
      conda run -n gspl --no-capture-output python tools/eval_official_test.py --ckpt "$ck" --opacity_keep 1.0 0.9 0.75 0.5 0.25 2>&1 \
        | grep -vE "\.\.\.[0-9]+/[0-9]+ +PSNR|pkg_resources|declare_namespace" > "$L"
      grep "^\[剪枝曲線\]" "$L" | sed 's/^\[剪枝曲線\] //' > "logs/full44_prunecurve_${w}.tsv"
      cat "logs/full44_prunecurve_${w}.tsv"
      [ "$(wc -l < "logs/full44_prunecurve_${w}.tsv")" -ge 6 ] || { echo "⛔ $w 曲線不完整（見 $L）"; bad=1; }
    done
    exit "$bad" ;;
  offheldout)
    # ★ 2026-10-04 使用者：同樣的塊在原版的 val／塊內 held-out／峰值／獨佔時間，作個對照。
    #   兩邊同格位的訓練相機只重疊 52~70%（我方 visibility 0.08、官方 0.05，定界方式也不同）=> val 的評分影像不同、只能參考；
    #   held-out 才公平：同一支工具、**同一批視角**（eval_official_test --block 用我方 block_all 的 4x4 分區選視角），
    #   評官方逐塊模型（未修改原始碼、trim 從未執行的原版）。結果寫進我方該塊 chart_data/official_heldout.txt（論文設定線 official_heldout_paper.txt）。
    shift; BL=("$@"); [ ${#BL[@]} -gt 0 ] || BL=(0 1 2 3)
    #   CITYGS_OFF_LINE=paper（10-05）：改評「照論文設定」那條線（trim 真的執行＋ω 0.9、prune 0.025；task_citygs_origin.sh blockpaper），
    #   寫進 official_heldout_paper.txt；預設 orig＝官方原版。
    case "${CITYGS_OFF_LINE:-orig}" in
      orig)  OB=../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/blocks; OHF=official_heldout.txt; OLBL=官方原版 ;;
      paper) OB=../cityGS_origin_trimfix/outputs/citygsv2_mc_aerial_sh2_paper/blocks; OHF=official_heldout_paper.txt; OLBL=論文設定 ;;
      *) echo "⛔ CITYGS_OFF_LINE 只能是 orig 或 paper"; exit 2 ;;
    esac
    bad=0
    for B in "${BL[@]}"; do
      ck=$(ls "$OB/block_$B"/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
      [ -n "$ck" ] || { echo "⛔ $OLBL block $B 沒有 60k ckpt"; bad=1; continue; }
      mkdir -p "$OUT/blocks/block_$B/chart_data"
      echo "════ $OLBL block $B：$ck"
      conda run -n gspl --no-capture-output python tools/eval_official_test.py --ckpt "$ck" --block "$B" --block_dim 4 4 2>&1 \
        | grep -vE "\.\.\.[0-9]+/[0-9]+ +PSNR|pkg_resources|declare_namespace" | tee "$OUT/blocks/block_$B/chart_data/$OHF" \
        | grep -E "選視角|模型|^PSNR|^SSIM|^LPIPS|^紋理比|^渲染|⛔"
      grep -q "^PSNR" "$OUT/blocks/block_$B/chart_data/$OHF" || bad=1
    done
    exit "$bad" ;;
  stepprof)
    # ★ 2026-10-03 使用者：「當前 lab 在跑當前最佳解 你沒設 solo 嗎 這樣怎麼測時間」
    #   4x4 的 16 塊是三槽平行訓練（品質／Load／峰值 VRAM 與鄰居無關），**牆鐘被鄰居拉長、不可當訓練時間**；
    #   res 模式的 step_breakdown 只量單一相機的 forward／backward，不含 trim／optimizer／資料載入。
    #   => 每塊從自己的 ckpt 接著跑兩段真實迴圈（_StepProfiler，每標記點同步；[solo]、不鎖 VRAM，同官方 stepprof）：
    #        grow     @14,999 起 1,200 步、densify 開（含週期 trim）  => 增生期每步＝真實步＋每次 trim ÷ 500
    #        harvest  @41,999 起 1,200 步、densify 關（無 trim）     => 收割期每步
    #      推算整趟 ≈ 30,000 x 增生期每步 + 30,000 x 收割期每步（不含 val／存 ckpt；每標記點同步會略高估）
    #   結果寫進該塊 chart_data/stepprof_{grow,harvest}.txt，最後印逐塊推算表；量完刪掉暫存跑次（ckpt 每個約 1.5 GB）。
    shift; BL=("$@"); [ ${#BL[@]} -gt 0 ] || BL=($(seq 0 15))
    GF=$(python3 -c "import json;d=json.load(open('$GATE'));print(' '.join('--model.renderer.init_args.%s true' % f for f in d['flags']) if d['status']=='installed' else '')" 2>/dev/null)
    bad=0
    for B in "${BL[@]}"; do
      bd="$OUT/blocks/block_$B"; mkdir -p "$bd/chart_data"
      for ph in grow harvest; do
        [ -s "$bd/chart_data/stepprof_$ph.txt" ] && { echo "（block $B $ph 已量過）"; continue; }
        if [ "$ph" = grow ]; then ck=$(ls "$bd"/checkpoints/*step=14999.ckpt 2>/dev/null | head -1); DU=30000
        else ck=$(ls "$bd"/checkpoints/*step=41999.ckpt 2>/dev/null | head -1); DU=0; fi
        [ -n "$ck" ] || { echo "⛔ block $B 沒有 $ph 用的 ckpt"; bad=1; continue; }
        n="${NAME}_prof_${ph}"; rm -rf "outputs/$n/blocks/block_$B"
        echo "════ block $B $ph（$ck，1,200 步）$(date)"
        ( unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
          CITYGS_STEP_PROFILE=1 CITYGS_KEEP_CFG_RENDERER=1 conda run -n gspl --no-capture-output python -u main.py fit \
            --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id "$B" --data.parser.block_dim "[4,4]" -n "$n" \
            --model.initialize_from "$ck" --trainer.max_steps 1200 \
            --data.image_uint8 true --data.skip_unused_depth true \
            --model.density.init_args.cap_max 2600000 --model.density.init_args.densify_until_iter $DU \
            --model.density.init_args.absgrad_densify 2.0 --model.density.init_args.fast_noise true \
            --model.density.init_args.noise_gate_eps 0.001 --model.metric.init_args.opacity_reg 0.002 \
            --model.metric.init_args.lambda_normal 0.0 --model.metric.init_args.depth_loss_weight.init 0.0 \
            --model.renderer.init_args.exact_conic_aabb true --model.renderer.init_args.lean_train true $GF ) 2>&1 \
          | grep -E "Trimming done|Error|error|⛔" | head -4
        if [ -s "outputs/$n/blocks/block_$B/step_cost.txt" ]; then
          cp "outputs/$n/blocks/block_$B/step_cost.txt" "$bd/chart_data/stepprof_$ph.txt"
        else echo "⛔ block $B $ph 沒有 step_cost.txt"; bad=1; fi
        rm -rf "outputs/$n/blocks/block_$B"
      done
    done
    echo "════ 逐塊推算：獨佔下整趟 60k 的訓練時間（不含 val／存 ckpt）"
    python3 - "$OUT" <<'PYEOF'
import glob, os, re, sys
out = sys.argv[1]
REAL = re.compile(r"（真實步時間·含重疊）\s+([\d.]+)")
TRIM = re.compile(r"^11 週期性 trim.*?\s{2,}[\d.]+\s+[\d.]+\s+[\d,]+\s+([\d.]+)\s", re.M)
tot = []
print(f"{'塊':>4} {'增生期 ms/步':>12} {'收割期 ms/步':>12} {'推算 h':>8}")
for b in range(16):
    cd = f"{out}/blocks/block_{b}/chart_data"
    try:
        g = open(f"{cd}/stepprof_grow.txt", encoding="utf-8").read(); h = open(f"{cd}/stepprof_harvest.txt", encoding="utf-8").read()
    except OSError:
        continue
    gr = float(REAL.search(g).group(1)); m = TRIM.search(g); tr = float(m.group(1)) / 500 if m else 0.0
    hv = float(REAL.search(h).group(1))
    hours = (30000 * (gr + tr) + 30000 * hv) / 3.6e6
    tot.append(hours)
    print(f"{b:>4} {gr + tr:>12.1f} {hv:>12.1f} {hours:>8.2f}")
if tot:
    print(f"合計 {sum(tot):.1f} h（逐塊序列）／平均每塊 {sum(tot) / len(tot):.2f} h —— 對照官方逐塊獨佔牆鐘 3.3~4.9 h（含 val／存檔）")
PYEOF
    exit "$bad" ;;
  *) echo "⛔ 不認得的模式：$MODE"; exit 2 ;;
esac
