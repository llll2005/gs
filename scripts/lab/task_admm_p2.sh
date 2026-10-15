#!/bin/bash
# ADMM 統一框架 P2（使用者 10-14 往前排；紀錄/實驗分析/14 §11）：best1 ＋ 每輪的共識深度，Jacobi 式輪次、接續訓練。
#   第 1 輪＝best1 本身（0 -> T）；之後每輪：tools/admm_consensus.py 做 z-更新（TSDF 融合 -> raycast）與對偶更新，
#   下一輪用現有深度 loss 拉向目標 z - u（權重 rho 固定、不衰減），從上一輪的 ckpt 接續（CITYGS_RESUME=1）。
#   變體 self＝P2-a 自我共識（z 只來自本塊、多視角 TSDF 一致性）；nb＝P2-b 另把 full44 的鄰塊模型融合進 z。
#   需要幾何通道 => 不開 lean。輸出 outputs/lab/g44_best1admm<變體>/blocks/block_<塊>/（共識深度在 admm/round_<r>/）。
# 用法（佇列）：[solo] bash scripts/lab/task_admm_p2.sh <塊> [self|nb] [rho=0.2]
#   本機功能測試（10-14 通過：b10 3 輪 x 200 步，接續步數接上、共識深度載入、d_reg 非 0）：
#     CITYGS_ADMM_FLAGS="--model.renderer.init_args.bwd_sat_skip false --model.renderer.init_args.bwd_reduce false" CITYGS_ADMM_ROUNDS=3 STEPS=600 CITYGS_ADMM_STRIDE=10
#   CITYGS_ADMM_ROUNDS（預設 4）、STEPS（預設 60000）、CITYGS_ADMM_STRIDE（z-更新相機間隔，功能測試用）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u -o pipefail
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
export LD_LIBRARY_PATH=/workspace/data/hdd/11213/o3dlibs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
B=$1; VAR=${2:-self}; RHO=${3:-0.2}
R=${CITYGS_ADMM_ROUNDS:-4}; STEPS=${STEPS:-60000}; T=$((STEPS / R))
NAME="${RUN_PREFIX}g44_best1admm${VAR}"
OUT="outputs/$NAME/blocks/block_$B"
# ⚠ 10-15 修：原本 F=(${CITYGS_ADMM_FLAGS-"..."})，未設環境變數時雙引號裡的預設值**不會被拆字**，整串變成一個參數 => main.py 報
#   「argument --model.renderer: expected one argument」、10 秒 rc=2（本機測試有設環境變數所以沒踩到；log 印出的指令用空白接起來看不出來）
if [ -n "${CITYGS_ADMM_FLAGS+x}" ]; then read -r -a F <<< "$CITYGS_ADMM_FLAGS"   # 本機光柵器沒有那兩個欄位 => 功能測試設 CITYGS_ADMM_FLAGS=""
else F=(--model.renderer.init_args.record_reduce true --model.renderer.init_args.absgrad_gate true); fi
NB=()
if [ "$VAR" = nb ]; then
  for nb in $(python3 -c "b=$B; x,y=b%4,b//4; print(' '.join(str(yy*4+xx) for xx,yy in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)) if 0<=xx<4 and 0<=yy<4))"); do
    c=$(ls outputs/lab/full44_best/blocks/block_$nb/checkpoints/*step=60000.ckpt 2>/dev/null | head -1); [ -n "$c" ] && NB+=("$c")
  done
  echo "鄰塊（full44）：${#NB[@]} 個"
fi
echo "=== ADMM P2：$NAME block $B，$R 輪 x $T 步，rho $RHO ==="
for r in $(seq 1 "$R"); do
  end=$((r * T)); post=0; [ "$r" = "$R" ] && post=1
  if [ "$r" = 1 ]; then
    CITYGS_POSTDATA=$post CITYGS_RUN_NAME="$NAME" CITYGS_GRID=44 STEPS=$STEPS CITYGS_FAMILY=g44_ \
      bash scripts/lab/task_cmp.sh "$B" best1 "${F[@]}" --trainer.max_steps "$end" || exit $?
  else
    D="$(pwd)/$OUT/admm/round_$r"
    CITYGS_RESUME=1 CITYGS_POSTDATA=$post CITYGS_RUN_NAME="$NAME" CITYGS_GRID=44 STEPS=$STEPS CITYGS_FAMILY=g44_ \
      bash scripts/lab/task_cmp.sh "$B" best1 "${F[@]}" --trainer.max_steps "$end" \
      --model.metric.init_args.depth_loss_weight.init "$RHO" --model.metric.init_args.depth_loss_weight.final_factor 1.0 \
      --data.parser.depth_dir "$D" --data.parser.depth_scale_name "$D/consensus_scales" || exit $?
  fi
  [ "$r" = "$R" ] && break
  ck=$(ls -t "$OUT"/checkpoints/*.ckpt 2>/dev/null | grep -v -E '_inbox' | head -1)
  echo "=== 第 $r 輪結束（$ck）=> z-更新與對偶更新 -> round_$((r + 1)) ==="
  mkdir -p "$OUT/admm"
  prev=(); [ "$r" -gt 1 ] && prev=(--prev "$OUT/admm/round_$r")
  conda run -n gspl --no-capture-output python -u tools/admm_consensus.py --ckpt "$ck" --out "$OUT/admm/round_$((r + 1))" \
    "${prev[@]}" ${NB[@]:+--neighbors "${NB[@]}"} --stride "${CITYGS_ADMM_STRIDE:-1}" 2>&1 \
    | grep -v -i -E 'warn|declare|找不到深度圖' | tee "$OUT/admm/round_$((r + 1)).log" || exit $?
done
echo "=== ADMM P2 完成：$NAME block $B ==="
