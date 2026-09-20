#!/bin/bash
# 現行最佳配方在**修正後資料**上的基準線。用法：task_speed3.sh <block_id>
# ⚠ 與舊的 26.6957 等數字**完全不可比**：資料修正了（§16.14）、相機數也變了
#   （b12 284 -> 653）。這是新年代的第一個基準。
source "$(dirname "$0")/_common.sh"
BLK=${1:?用法: task_speed3.sh <block_id> [後綴] [覆蓋參數...]}
# 用法（2026-09-14 改成「配方 + 後綴 + 覆蓋參數」，使用者要求與本機下指令的習慣一致）：
#   task_speed3.sh 6                                      跑次 speed3            —— 與最初版本逐位元相同
#   task_speed3.sh 6 trimvpc                              跑次 speed3_trimvpc    —— 預設組：+ trim 判準 v/c（向後相容）
#   task_speed3.sh 6 <後綴> --model.xxx 值 [--model.yyy 值 ...]
#                                                         跑次 speed3_<後綴>；覆蓋參數原樣接在配方旗標**後面**
#                                                         （jsonargparse：後面的覆蓋前面的）
# ⚠ 自訂後綴**必須**帶覆蓋參數，否則會跑出「名字不同、內容等於 speed3」的跑次。
# ⚠ 後綴只准 [a-z0-9_]（它是跑次名的一部分）。佇列行不可含 $(...) 或反斜線。
SUFFIX=${2:-}
if [ $# -ge 2 ]; then shift 2; else shift 1; fi
case "$SUFFIX" in
  "")      NAME="${RUN_PREFIX}speed3";         EXTRA=() ;;
  trimvpc) NAME="${RUN_PREFIX}speed3_trimvpc"; EXTRA=(--model.renderer.init_args.trim_by_value_per_cost true) ;;
  *)       [[ "$SUFFIX" =~ ^[a-z0-9_]+$ ]] || { echo "❌ 後綴只准 [a-z0-9_]：$SUFFIX"; exit 2; }
           [ $# -gt 0 ] || { echo "❌ 後綴 $SUFFIX 沒帶任何覆蓋參數（會跑出內容等於 speed3 的跑次）"; exit 2; }
           NAME="${RUN_PREFIX}speed3_${SUFFIX}"; EXTRA=() ;;
esac
EXTRA+=("$@")
echo "[task_speed3] 跑次 $NAME／覆蓋參數：${EXTRA[*]:-（無）}"
# ★ 2026-09-18：`CITYGS_RUN_NAME` 覆蓋跑次名。用途＝全場景 25 塊收在同一個 run 目錄下
#   （`outputs/lab/<name>/blocks/block_N/`，正是 utils/merge_citygs_ckpts.py 期望的結構），
#   同時避免 run_fit 把既有的 lab/speed3 b6／b13 搬走。不設就是原本的名字。
run_fit "${CITYGS_RUN_NAME:-$NAME}" "$BLK" \
  --model.initialize_from null \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  ${EXTRA[@]+"${EXTRA[@]}"}
