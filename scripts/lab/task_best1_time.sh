#!/bin/bash
# best1 的乾淨計時（使用者 10-15：報告要「用幾分 PSNR 換到多少速度」；10-14 的 g44_best1 訓練期間卡上有外部行程 => 牆鐘不可用）
#   loop <塊>   真實訓練迴圈逐段時間（CITYGS_STEP_PROFILE，同 task_bwdopt_check verify2 的量法）：
#               三組配方 x 兩個階段，各 1,200 步（initialize_from 該跑次自己的 ckpt；步數從 0 起算）
#                 best0      g44_best0 的 ckpt，best0 旗標＋lean＋record_reduce（加速旗標關）
#                 best0fast  同 ckpt，再開 backward 飽和跳過＋先加總（隔離兩個加速旗標）
#                 best1      g44_best1 的 ckpt，best1 全部旗標（含兩個加速旗標）
#               增生期＝@14,999 起（densify 與每 500 步 trim 照常）；收割期＝@41,999 起、densify_until_iter 0（不增生、不 trim）
#               => 一塊 60k 的時間 ≈ 30k x 增生期每步 ＋ 30k x 收割期每步（另加驗證／存檔）
#   full <塊>   best1 完整 60k 獨佔重跑（跑次 g44_best1time，不跑圖表資料包 => 台帳 START/DONE 就是純訓練牆鐘）
# 每段開始前後印整卡已用 VRAM 與其他行程數；量測期間每 30 s 記一次到 <輸出>/gpu_watch.txt（> 1500 MiB 的外部佔用 => 該段時間不可用）
# 用法（佇列）：[solo] bash scripts/lab/task_best1_time.sh loop 6
#               [solo] bash scripts/lab/task_best1_time.sh full 6
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
MODE=$1; B=${2:?塊}
# runner 的 [solo] 只等自己佇列裡的任務，不看卡上的其他行程（10-14 的 best1 就是被外部行程污染）
#   => 量之前等到卡上沒有任何計算行程（最多 2 h；逾時照量，log 標 ⚠）
for i in $(seq 1 240); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" -eq 0 ] && break
  [ "$i" = 1 ] && echo "⏳ 卡上還有 $n 個計算行程 => 等它們結束再量（每 30 s 查一次，最多 2 h）$(date)"
  sleep 30
done
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && echo "✅ 卡上無其他行程，開始量 $(date)" || echo "⚠⚠ 等滿 2 h 仍有其他行程 => 這次時間可能被污染"
F=(--model.renderer.init_args.record_reduce true --model.renderer.init_args.absgrad_gate true)
gpu () { echo "（整卡已用 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB、計算行程 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) 個）"; }
watch_on () { ( while true; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 30; done ) >> "$1" 2>&1 & WPID=$!; }
watch_off () { kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null; }

if [ "$MODE" = full ]; then
  NAME="${RUN_PREFIX}g44_best1time"
  mkdir -p "outputs/$NAME"; W="outputs/$NAME/gpu_watch_b$B.txt"; : > "$W"
  echo "=== best1 完整 60k 獨佔重跑：$NAME block $B $(date) ==="; gpu
  watch_on "$W"
  CITYGS_POSTDATA=0 CITYGS_RUN_NAME="$NAME" CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=60000 CITYGS_FAMILY=g44_ \
    bash scripts/lab/task_cmp.sh "$B" best1 "${F[@]}"; rc=$?
  watch_off; gpu
  echo "外部佔用取樣：$(awk '{n++; k=gsub(/;/,";"); if(k>1)h++} END{printf "%d/%d 次卡上有 >1 個計算行程", h, n}' "$W")（不鎖 VRAM 時我方自己的保留量就會 > 6 GB，所以看行程數）"
  grep -E "g44_best1time/blocks/block_$B\$" logs/quad_progress.log | tail -2
  exit $rc
fi

[ "$MODE" = loop ] || { echo "⛔ 模式只有 loop／full"; exit 2; }
L=logs/best1_time_b${B}_$(date +%m%d_%H%M).log
{
  for ph in grow harvest; do
    case $ph in grow) S=14999; XP=() ;; harvest) S=41999; XP=(--model.density.init_args.densify_until_iter 0) ;; esac
    for arm in best0 best0fast best1; do
      case $arm in
        best0)     SRC=g44_best0; A=best0; XA=() ;;
        best0fast) SRC=g44_best0; A=best0; XA=(--model.renderer.init_args.bwd_sat_skip true --model.renderer.init_args.bwd_reduce true) ;;
        best1)     SRC=g44_best1; A=best1; XA=() ;;
      esac
      CK=$(ls outputs/${RUN_PREFIX}$SRC/blocks/block_$B/checkpoints/*step=$S.ckpt 2>/dev/null | head -1)
      [ -n "$CK" ] || { echo "⛔ 沒有 $SRC block $B @$S"; continue; }
      NAME="${RUN_PREFIX}time_${arm}_${ph}"
      OUTD="outputs/$NAME/blocks/block_$B"
      echo "════ $arm／$ph（$SRC @$S 起 1,200 步）$(date)"; gpu
      mkdir -p "outputs/$NAME"; W="outputs/$NAME/gpu_watch_b$B.txt"; : > "$W"; watch_on "$W"
      CITYGS_STEP_PROFILE=1 CITYGS_KEEP_CFG_RENDERER=1 CITYGS_POSTDATA=0 CITYGS_RUN_NAME="$NAME" CITYGS_GRID=44 CITYGS_LEAN=1 \
        STEPS=60000 CITYGS_FAMILY=g44_ bash scripts/lab/task_cmp.sh "$B" "$A" "${F[@]}" "${XA[@]}" "${XP[@]}" \
        --model.initialize_from "$CK" --trainer.max_steps 1200 2>&1 | grep -E "\[lean\]|Trimming done|Error|error|⛔|❌" | head -8
      watch_off; gpu
      echo "外部佔用取樣：$(awk '{n++; k=gsub(/;/,";"); if(k>1)h++} END{printf "%d/%d 次卡上有 >1 個計算行程", h, n}' "$W")"
      echo "── step_cost（$arm／$ph）"; sed -n 1,24p "$OUTD/step_cost.txt" 2>/dev/null || echo "⛔ 沒有 step_cost.txt"
    done
  done
} 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample|colmap dataparser|depth maps' | tee "$L"
echo "log：$L"
