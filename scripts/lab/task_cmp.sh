#!/bin/bash
# ★★★★★★★ 對比實驗的**標準骨架**（使用者 2026-09-13 定調：一律以 2 萬多步為基準長度）
# 用法：task_cmp.sh <block_id> <arm>
#
# ## 為什麼是「2 萬多步」而不是 60k
# 60k 一趟 7.6h（lab）/ 8.8h（本機）。而判準是**四個指標的正負號模式**
# （例如「紋理比↑ 而光度↓」那個虛假細節簽名還在不在），不是誰分數高
# —— 名次要 ~51k 步才穩，但符號模式好分辨得多。⇒ 用 1/3 的時間問對的問題。
#
# ## ★ 排程必須**跟著全長縮放**，否則比較本身是歪的
# 直接把 60k 的配方截斷到 2 萬步，會得到一個**不同的 regime**：
# ```
#   densify_until_iter 30000  >  全長 21,920  =>  整趟都在增生，**沒有收割期**
#   means_lr max_steps 60000                  =>  跑完時 LR 才衰減到 36%
# ```
# 所以本腳本按全長重新配置，而且用的是**已驗證過的比例**而不是隨手挑的數字：
# ```
#   trainer.max_steps        = STEPS
#   means_lr max_steps       = STEPS            （LR 積分在跑完時剛好走完）
#   densify_until_iter       = STEPS / 2        （sched30 驗證過的 50%，兩塊都顯著改善）
# ```
# **刻意不縮放的兩項，以及為什麼（這關係到「不偏向任何選項」）**：
# ```
#   densify_from_iter 1000   暖身，不是全長的比例。而且 `cost_add_densify` 需要
#                            densify 晚於第一次 trim（step 500）=> 縮成 ~365 會讓成本臂
#                            **直接失效** ⇒ 縮放它就是偏向對照組。
#   cap_max 2.6M             容量上限，不是排程。所有臂相同即可。
#   densification_interval   縮放它會改變 densify 事件數，而沒有任何「已驗證的比例」
#                            可依循 ⇒ 隨手挑一個就是引入偏向。固定 = 所有臂相同。
# ```
# ⚠⚠ 連帶後果：這個排程的跑次**不能**與 `lab/speed3`（60k 排程）的 10,960/21,920 直接比
#   —— 所以 arm `base` 存在，它就是這個排程下的基準。**比較一律在同排程的臂之間做。**
#
# ## 步數為什麼不是一個固定數字
# val 的節奏是 `check_val_every_n_epoch: 20` ⇒ 每 20×相機數 步一次。各塊相機數不同：
# ```
#   b6  548 台 -> val 每 10,960 -> 兩期 = 21,920
#   b12 653 台 -> val 每 13,060 -> 兩期 = 26,120
#   b13 667 台 -> val 每 13,340 -> 兩期 = 26,680
# ```
# 取「不少於 20,000 的最小整數期」=> 每塊都剛好停在 val 點上，兩個觀測點可直接對齊。
# 要覆寫用 `STEPS=<n>`。
#
# ## ★ w 的校準值（2026-09-13 在**新資料**上實測，兩塊一致）
# 目標：令 `1 + w*ceiling(訊號)` = 25.8，與 absgrad（已證有效那個）同動態範圍。
# ```
#            ceiling(top5%)/mean      w = 24.8/ceiling
#   1/c      b12 6.32 / b13 6.14      3.92 / 4.04   => 取 **4.0**
#   c        b12 8.97 / b13 8.58      2.76 / 2.89   => 取 **2.8**
# ```
# ⚠ 旗標 docstring 建議的 **2.278** 是用**舊資料**（ceiling(1/ĉ)=10.89x）算的
#   ⇒ 在新資料上只有目標強度的 0.57 倍。而「強度沒對齊」正是我指認為原始否證主因的東西
#   ⇒ 沿用舊值等於用相反方向重蹈同一個錯。**校準值要跟著資料重算。**
set -u
source "$(dirname "$0")/_common.sh"
BLK=${1:?用法: task_cmp.sh <block_id> <arm>}
ARM=${2:?同上}

# ★ B0 = 當前操作點的 Load（`max_view Σ (2r/16)^2`），由 tools/cost_budget_calibrate.py
#   在**已訓練好的 ckpt** 上一次前向量出來（不需要訓練，見該工具檔頭）。
#   b6  speed3@14999   N≈2.60M  B0 = 17,987,534  (B/N 6.92)
#   b12 gate15000@15000 N=2.34M B0 = 14,177,821  (B/N 6.06)
#   ⚠ 不可沿用 tools/cost_budget_probe.py 的 23.1M —— 那支用解析投影，而生效的是光柵器 radii。
case "$BLK" in
  6)  NCAM=548; B0=17987534 ;;
  12) NCAM=653; B0=14177821 ;;
  13) NCAM=667; B0=0 ;;
  *)  NCAM=${NCAM:?未知的 block，請用 NCAM=<相機數> 指定}; B0=${B0:-0} ;;
esac
PERIOD=$((NCAM * 20))
STEPS=${STEPS:-$(( ( (20000 + PERIOD - 1) / PERIOD ) * PERIOD ))}
HALF=$((STEPS / 2))
echo "block $BLK：$NCAM 台相機 / val 每 $PERIOD 步 => 全長 $STEPS 步、densify_until $HALF"

case "$ARM" in
  base)       EXTRA=();                                                        EXP=0 ;;
  # ⚠⚠ 2026-09-13 時序：`cs_costdir` 這個**已經在跑的**跑次是 03:15 啟動的，當時腳本寫的還是
  #   w=2.278（旗標 docstring 依**舊資料** ceiling=10.89x 算的）。我在 04:2x 才依新資料重算成 4.0。
  #   ⇒ **磁碟上的 `cs_costdir` = w 2.278**，不是這一行現在寫的值。
  #   不重跑它，改成把兩個強度都留著當**強度掃描**（記憶 observability_saves_runs 記過
  #   「加權過猛」害過一次：權重實現 148.7x 而非設計 25.8x）：
  #     cs_costdir      w 2.278  = 校準目標的 0.57 倍（已跑）
  #     cs_costdir_cal  w 4.0    = 依新資料校準（ceiling(1/c) b12 6.32 / b13 6.14 => 24.8/c）
  costdir)    EXTRA=(--model.density.init_args.cost_add_densify 2.278);         EXP=1 ;;
  costdir_cal) EXTRA=(--model.density.init_args.cost_add_densify 4.0);          EXP=1 ;;
  costtaming) EXTRA=(--model.density.init_args.cost_add_densify -2.8);          EXP=1 ;;
  # ★★ 把**週期 trim 的剪枝判準**從價值 v 換成每單位成本的價值 v/c（零額外成本，c 本來就在算）
  #   依據：rho(v,c)≈0（三個模型）=> 兩種排序選出**不同**的一批（top10% 重疊 54~67%）；
  #        按 v/c 貪婪在同成本預算下多拿到 +48~57%（10% 預算）；
  #        而按 contribution 排序剪枝只比**隨機**好 1.27 倍 => 現行判準本身很弱。
  #   ⚠ 這是「破壞集合」那一側；vpc_prune_frac 的 -2.34 dB 是「把低 v/c 的粒子**搬走**」，
  #     移除與搬移不同，但先驗要謹慎 => 判準看四個指標的正負號模式，不是單看 PSNR。
  trimvpc)    EXTRA=(--model.renderer.init_args.trim_by_value_per_cost true);      EXP=1 ;;
  dssim05)    EXTRA=(--model.metric.init_args.lambda_dssim 0.5);                EXP=1 ;;
  both)       EXTRA=(--model.density.init_args.cost_add_densify 4.0
                     --model.metric.init_args.lambda_dssim 0.5);                EXP=2 ;;
  # ★★ 命題的**約束端**：把生長條件從顆數 `N<=cap_max` 換成渲染成本 `Load<=cost_budget`
  #   => N 不再是我設的定值，而是**從場景長出來**（使用者 2026-09-13 指出固定 N 會隨場景複雜度失準）。
  #   為什麼排在**緊預算**：tools/rho_value_cost.py 量到背包天花板是預算鬆緊的函數 ——
  #     預算佔總成本 10% => +57%／25% => +22~27%／50% => +6~9%／75% => +1~2%（b12/b13 一致）
  #   而被否證的三個成本感知變體跑在 cap 2.6M ＝**寬鬆端**，天花板只有 2~9%
  #   ⇒ 「它們輸」與「這方向沒用」是兩回事。
  #   ⚠ `cap_max` 保留當安全閥：VRAM = 逐顆儲存（只看 N）＋ binning（只看成本），
  #     本旗標只約束後者（旗標 docstring 明文要求）。
  cb50)       [ "${B0:-0}" -gt 0 ] || { echo "⛔ block $BLK 沒有標定過的 B0"; exit 2; }
              EXTRA=(--model.density.init_args.cost_budget $((B0 / 2)));         EXP=1 ;;
  cb25)       [ "${B0:-0}" -gt 0 ] || { echo "⛔ block $BLK 沒有標定過的 B0"; exit 2; }
              EXTRA=(--model.density.init_args.cost_budget $((B0 / 4)));         EXP=1 ;;
  *) echo "⛔ 未知的 arm：$ARM（base|costdir|costdir_cal|costtaming|dssim05|both|cb50|cb25|trimvpc）"; exit 2 ;;
esac
# run_fit 收尾會 diff resolved config；基準就是同排程的 `cs_base`
export CITYGS_DIFF_VS="${RUN_PREFIX}cs_base"
export CITYGS_DIFF_EXPECT=$EXP
run_fit "${RUN_PREFIX}cs_${ARM}" "$BLK" \
  --model.initialize_from null \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --trainer.max_steps "$STEPS" \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps "$STEPS" \
  --model.density.init_args.densify_until_iter "$HALF" \
  "${EXTRA[@]}"
