#!/bin/bash
# sfmfill 初始化的參數比較（使用者 2026-09-17）：20k、零機制，沿用 task_initcmp.sh 的單變數設計。
#
# 對照（lab 已跑完，**不重跑**）：lab/init_sfmfill、lab/init_sfm
#   預設 = dup 1、jitter 0.5、fill-ratio 0.1、fill-voxel 0.15、pct 2.5（工具固定亂數種子 42 => 結果確定）
# 一次只動一個參數：
#   dup2 / dup4      Tier A（SfM 點）複製份數＝「增加倍數」；複製份加 0.5x 最近鄰抖動
#   fill03 / fill10  空區補點密度：SfM 區密度的 0.3 / 1.0 倍（預設 0.1）
#   vox075           「空區」判定體素 0.15 -> 0.075：更細 => SfM 點之間的縫也算空、也補
#   pct05            補點範圍百分位 2.5 -> 0.5：更寬 => 周邊也補（「空區域覆蓋」）
#   def              預設參數重產一份，**只用來核對** lab/init_sfmfill 那份 PLY 是不是預設參數產的
#
# 用法：task_sfmfill_sweep.sh gen              產生全部變體的 PLY（純 CPU；排 [solo] 讓 run 等它）
#       task_sfmfill_sweep.sh run <塊> <變體>   訓練（呼叫 task_initcmp.sh 的 sfmfill_<變體> 臂）
set -u
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial/train/block_all
VARIANTS="def dup2 dup4 fill03 fill10 vox075 pct05"
# ⚠ 上限：cap 2.6M、lab 鎖 5.66GB。pct05 放寬範圍可能把 z 離群點框進來 => 補點數暴增、起步就 OOM
MAXPTS=3000000
args_of () {
  case "$1" in
    def) echo "" ;; dup2) echo "--dup 2" ;; dup4) echo "--dup 4" ;;
    fill03) echo "--fill-ratio 0.3" ;; fill10) echo "--fill-ratio 1.0" ;;
    vox075) echo "--fill-voxel 0.075" ;; pct05) echo "--pct 0.5" ;;
    *) return 1 ;;
  esac
}
npts () { head -c 400 "$1" | grep -a -m1 'element vertex' | awk '{print $3}'; }
MODE=${1:?用法: task_sfmfill_sweep.sh gen | run <塊> <變體>}
case "$MODE" in
  gen)
    for v in $VARIANTS; do
      out="$D/sfmfill_init_$v"
      if [ -f "$out/block_6.ply" ] && [ -f "$out/block_13.ply" ]; then echo "略過 $v（已存在）"; continue; fi
      a=$(args_of "$v")
      echo "=== 產生 $v：${a:-（預設參數）} $(date) ==="
      # shellcheck disable=SC2086
      conda run -n gspl --no-capture-output python tools/make_sfm_fill_init.py "$D" --blocks 6 13 \
        --out-dir "$out" $a || exit $?
    done
    echo "=== 來源核對：預設參數重產 vs lab/init_sfmfill 用的那份 ==="
    for b in 6 13; do
      x=$(md5sum < "$D/sfmfill_init_def/block_$b.ply" | cut -c1-12)
      y=$(md5sum < "$D/sfmfill_init/block_$b.ply" | cut -c1-12)
      if [ "$x" = "$y" ]; then
        echo "✅ block_$b 逐位元相同 => lab/init_sfmfill 就是預設參數，直接當基準、不必重跑"
      else
        echo "⚠⚠ block_$b 不同（重產 $x／現有 $y；點數 $(npts "$D/sfmfill_init_def/block_$b.ply") vs $(npts "$D/sfmfill_init/block_$b.ply")）"
        echo "    => lab/init_sfmfill 可能不是預設參數，需要補跑 def 當基準"
      fi
    done
    echo "=== 點數總表（> $MAXPTS 的變體在 run 時會被擋下）==="
    for f in "$D"/sfmfill_init*/block_*.ply; do echo "$(npts "$f")  $f"; done ;;
  run)
    BLK=${2:?用法: task_sfmfill_sweep.sh run <塊> <變體>}; V=${3:?同上}
    args_of "$V" >/dev/null || { echo "⛔ 不認得的變體：$V（$VARIANTS）"; exit 2; }
    F="$D/sfmfill_init_$V/block_$BLK.ply"
    [ -f "$F" ] || { echo "⛔ 缺 $F（先跑 gen）"; exit 2; }
    n=$(npts "$F")
    [ "${n:-0}" -le "$MAXPTS" ] || { echo "⛔ $V／b$BLK 有 $n 點 > $MAXPTS（cap 2.6M、5.66GB）=> 不跑，避免起步 OOM"; exit 2; }
    echo "[sweep] 變體 $V／block $BLK／起始點數 $n"
    # 跑完自動 diff resolved config：與 lab/init_sfmfill 應只差 ply_file 這 1 項
    export CITYGS_DIFF_VS=lab/init_sfmfill CITYGS_DIFF_EXPECT=1
    exec bash scripts/lab/task_initcmp.sh "$BLK" "sfmfill_$V" ;;
  *) echo "⛔ 不認得的模式：$MODE"; exit 2 ;;
esac
