#!/bin/bash
# sfmfill 初始化的參數比較（**更正版**，2026-09-17）：20k、零機制，沿用 task_initcmp.sh 單變數設計。
#
# ⚠⚠ 取代 task_sfmfill_sweep.sh：那一版以為 lab/init_sfmfill 用的是工具預設（fill-ratio 0.1／fill-voxel 0.15），
#   但 v2 §4.3 記著實際是 **--fill-ratio 0.35 --fill-voxel 0.35**（b6 635,627 點＝現有 PLY 標頭）
#   => 舊版每個變體和對照差 2~3 個參數，**不是單變數**。這一版以實際基準為準，一次只動一個參數。
#
# 對照（lab 已跑完，不重跑）：lab/init_sfmfill（基準）、lab/init_sfm
# 變體（其餘參數同基準：dup 1、jitter 0.5、pct 2.5）：
#   dup2 / dup4       SfM 點複製 2 / 4 份（「增加倍數」），複製份加 0.5x 最近鄰抖動
#   fill01 / fill10   空區補點密度 0.1 / 1.0（基準 0.35）
#   vox015            空區判定體素 0.15（基準 0.35；更細 => SfM 點之間的縫也算空、也補）
#   pct05             補點範圍百分位 0.5（基準 2.5；更寬 => 周邊也補）
#   def               基準參數重產，**逐位元比對**現有 sfmfill_init/ 那份；對不上就不准跑（基準不明）
#
# 用法：task_sfmfill_sweep2.sh gen              產生 PLY（純 CPU；排 [solo] 讓 run 等它）
#       task_sfmfill_sweep2.sh run <塊> <變體>   訓練（task_initcmp.sh 的 sfmsweep_<變體> 臂）
set -u
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial/train/block_all
OUT=$D/sfmfill_sweep
OKF=logs/sfmsweep_provenance.ok
VARIANTS="def dup2 dup4 fill01 fill10 vox015 pct05"
# ⚠⚠ 2026-09-17 第一次 gen 的 b13 來源核對沒過（點數同 670,761 但位元不同，b6 相同）=> 根因：
#   make_sfm_fill_init.py 的 rng = np.random.default_rng(42) 建在**塊迴圈之外**，每個塊輪流消耗
#   => 產出**取決於 --blocks 的清單與順序**。歷史基準（lab/init_sfmfill 的 PLY）是 --blocks 6 12 13 產的，
#      b12 先抽掉一段亂數 => 只跑 --blocks 6 13 時 b13 的補點位置就不同（b6 是第一個所以不受影響）。
#   ⇒ 一律用歷史清單產生：def 才對得上基準，各變體之間的亂數推進也一致（只是多產一份 b12，約 10 MB）。
GENBLOCKS="6 12 13"
MAXPTS=3000000   # cap 2.6M、lab 鎖 5.66GB；pct05 可能框進 z 離群點 => 補點暴增
args_of () {
  case "$1" in
    def)    echo "--fill-ratio 0.35 --fill-voxel 0.35" ;;
    dup2)   echo "--fill-ratio 0.35 --fill-voxel 0.35 --dup 2" ;;
    dup4)   echo "--fill-ratio 0.35 --fill-voxel 0.35 --dup 4" ;;
    fill01) echo "--fill-ratio 0.1 --fill-voxel 0.35" ;;
    fill10) echo "--fill-ratio 1.0 --fill-voxel 0.35" ;;
    vox015) echo "--fill-ratio 0.35 --fill-voxel 0.15" ;;
    pct05)  echo "--fill-ratio 0.35 --fill-voxel 0.35 --pct 0.5" ;;
    *) return 1 ;;
  esac
}
npts () { head -c 400 "$1" | grep -a -m1 'element vertex' | awk '{print $3}'; }
MODE=${1:?用法: task_sfmfill_sweep2.sh gen | run <塊> <變體>}
case "$MODE" in
  gen)
    rm -f "$OKF"
    for v in $VARIANTS; do
      if [ -f "$OUT/$v/block_6.ply" ] && [ -f "$OUT/$v/block_13.ply" ]; then echo "略過 $v（已存在）"; continue; fi
      a=$(args_of "$v")
      echo "=== 產生 $v：$a $(date) ==="
      # shellcheck disable=SC2086
      conda run -n gspl --no-capture-output python tools/make_sfm_fill_init.py "$D" --blocks $GENBLOCKS \
        --out-dir "$OUT/$v" $a || exit $?
    done
    echo "=== 來源核對：基準參數重產 vs lab/init_sfmfill 用的那份 ==="
    ok=1
    for b in 6 13; do
      x=$(md5sum < "$OUT/def/block_$b.ply" | cut -c1-12); y=$(md5sum < "$D/sfmfill_init/block_$b.ply" | cut -c1-12)
      if [ "$x" = "$y" ]; then echo "✅ block_$b 逐位元相同（$x）=> 基準確認"
      else ok=0; echo "⛔ block_$b 不同（重產 $x $(npts "$OUT/def/block_$b.ply") 點／現有 $y $(npts "$D/sfmfill_init/block_$b.ply") 點）"; fi
    done
    if [ "$ok" = 1 ]; then
      : > "$OKF"
      # ★ 2026-09-17：通過核對後**由 gen 自己把 12 個訓練行追加到佇列**。
      #   為什麼不預先排：gen 是 [cpu] 但仍佔一槽，而訓練行可能在 PLY 還沒產完時就被空出的槽搶走 => rc=2 白白消耗。
      #   這樣不需要輪詢、不會有空轉的槽，順序也保證正確（runner 每次取任務前會重讀 queue.txt）。
      if grep -q "task_sfmfill_sweep2.sh run" scripts/queue.txt 2>/dev/null; then
        echo "（佇列已有 run 行，不重複追加）"
      else
        # 插到官方參考線的 block 行**之前**（使用者 2026-09-18：先等 sfmfill 結果）；找不到就附加到尾巴
        _ins=$(mktemp)
        {
          echo ""
          for v in dup2 dup4 fill01 fill10 vox015 pct05; do
            case "$v" in
              dup2)   d="SfM 點複製 2 份（純顆數對照）" ;;
              dup4)   d="SfM 點複製 4 份（純顆數對照；起始已近 cap 2.6M，幾乎沒有成長空間）" ;;
              fill01) d="空區補點密度 0.1（基準 0.35）" ;;
              fill10) d="空區補點密度 1.0" ;;
              vox015) d="空區判定體素 0.15（基準 0.35，更細 => 點間縫隙也補）" ;;
              pct05)  d="補點範圍百分位 0.5（基準 2.5，更寬 => 周邊也補）" ;;
            esac
            for b in 6 13; do
              echo "# ★★★★★ sfmfill 參數比較（20k 零機制；對照 lab/init_sfmfill 28.683／28.204、lab/init_sfm 28.518／28.017）：$v＝$d／b$b"
              echo "bash scripts/lab/task_sfmfill_sweep2.sh run $b $v"
            done
          done
        } > "$_ins"
        if grep -qE "^bash scripts/lab/task_citygs_origin\.sh block " scripts/queue.txt; then
          _ln=$(grep -nE "^bash scripts/lab/task_citygs_origin\.sh block " scripts/queue.txt | head -1 | cut -d: -f1)
          while [ "$_ln" -gt 1 ] && [ "$(sed -n "$((_ln-1))p" scripts/queue.txt | cut -c1)" = "#" ]; do _ln=$((_ln-1)); done
          _tmp=$(mktemp)
          { head -n $((_ln-1)) scripts/queue.txt; cat "$_ins"; tail -n +"$_ln" scripts/queue.txt; } > "$_tmp"
          mv "$_tmp" scripts/queue.txt
          echo "✅ 已把 12 個訓練行插到官方參考線之前（第 $_ln 行）"
        else
          cat "$_ins" >> scripts/queue.txt
          echo "✅ 已把 12 個訓練行追加到佇列尾（找不到官方參考線的 block 行）"
        fi
        rm -f "$_ins"
      fi
    else
      echo "⛔⛔ 基準不明 => 不追加訓練行，所有 run 也會拒絕開跑（要先查 lab/init_sfmfill 的真實參數）"
    fi
    echo "=== 點數總表（> $MAXPTS 的變體在 run 時會被擋下）==="
    for v in $VARIANTS; do for b in 6 13; do echo "$(npts "$OUT/$v/block_$b.ply")  $v/b$b"; done; done ;;
  run)
    BLK=${2:?用法: task_sfmfill_sweep2.sh run <塊> <變體>}; V=${3:?同上}
    args_of "$V" >/dev/null || { echo "⛔ 不認得的變體：$V（$VARIANTS）"; exit 2; }
    [ -f "$OKF" ] || { echo "⛔ 來源核對沒通過或還沒跑 gen => 基準不明，不跑"; exit 2; }
    F="$OUT/$V/block_$BLK.ply"
    [ -f "$F" ] || { echo "⛔ 缺 $F（先跑 gen）"; exit 2; }
    n=$(npts "$F")
    [ "${n:-0}" -le "$MAXPTS" ] || { echo "⛔ $V／b$BLK 有 $n 點 > $MAXPTS => 不跑，避免起步 OOM"; exit 2; }
    echo "[sweep2] 變體 $V（$(args_of "$V")）／block $BLK／起始點數 $n"
    export CITYGS_DIFF_VS=lab/init_sfmfill CITYGS_DIFF_EXPECT=1   # 與基準只該差 ply_file
    exec bash scripts/lab/task_initcmp.sh "$BLK" "sfmsweep_$V" ;;
  *) echo "⛔ 不認得的模式：$MODE"; exit 2 ;;
esac
