#!/bin/bash
# ★★★★★★★ speed3 配方截斷到 N 步的「閘門」跑次。用法：task_gate.sh <block_id> <steps>
#   跑次名 = gate<steps>  =>  outputs/[lab/]gate<steps>/blocks/block_<N>/
#
# 為什麼值得：本機一個完整 60k 要 ~7 小時（違反「本機一個 block <=3 小時」），
# 而 config 的 save_iterations 已含 **500 / 1500 / 15000**
# ⇒ 一次 15,000 步的跑次**同時**交付 1500 步的探針與一個紮實的觀察點，不必分兩次。
#
# ★ 首要用途（使用者反覆強調的最高優先）：**失敗區在修正後資料上還在嗎**。
#   舊資料的 normal_b12 在 **1500 步**就看得出「正常區重建成功、問題區塊一樣在左半邊糊掉」
#   ⇒ 這個現象不需要 60k 才看得見，1500 步是最便宜的驗法。
#
# ⚠ 一律 SfM init（`initialize_from null`）：`depth_init/` 那批 PLY 是**舊配對**算的
#   （每顆點用錯幀的深度圖擺位置，§16.14）⇒ 在新資料上不可用，要等 depthprep 重生。
# ⚠ 下面的旗標必須與 scripts/lab/task_speed3.sh 完全一致（唯一差別是 max_steps）。
#   目前是手抄的 —— 改配方時兩支都要改。（沒抽成共用陣列，是因為 task_speed3.sh
#   當時正在被執行，不能動它：見 scripts/_noedit.sh 檔頭。）
source "$(dirname "$0")/_common.sh"
BLK=${1:?用法: task_gate.sh <block_id> <steps>}
STEPS=${2:?用法: task_gate.sh <block_id> <steps>}
NAME="${RUN_PREFIX}gate${STEPS}"
run_fit "$NAME" "$BLK" \
  --model.initialize_from null \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --trainer.max_steps "$STEPS"
rc=$?
[ "$rc" -ne 0 ] && exit "$rc"
# 存 val 影像 —— tools/failure_map.py / blur_persistence.py 讀的是
# outputs/<run>/blocks/block_N/test/step=*/  底下的 png（tools/veil_detect.py:29）
C=$(ls "outputs/$NAME/blocks/block_$BLK"/lightning_logs/version_*/config.yaml 2>/dev/null | tail -1)
if [ -n "$C" ]; then
  echo "=== 存 val 影像（給逐 tile 分析用）==="
  conda run -n gspl --no-capture-output python -u main.py test --config "$C" --save_val
else
  echo "⚠ 找不到 resolved config，跳過存圖"
fi
