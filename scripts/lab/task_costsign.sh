#!/bin/bash
# ★★★★★★★ 成本感知的「形式與強度」重測 + 它與 SSIM 是否在做同一件事（2x2）
# 用法：task_costsign.sh <block_id> <costdir|costtaming|dssim05|both>
#
# ## 為什麼重開
# 29 小時 GPU 的否證做在**影像↔姿態錯位**的資料上（86.64% 的影像對到別的姿態）
# ⇒ 光度增益當時**不可能**達成，而否證的實證形狀正是「紋理比↑ 而光度↓」
# ⇒ 任何增加容量的機制都會呈現那個形狀 ⇒ 那是最可能是資料假象的一類結論。
#
# ## 為什麼是這四臂（2026-09-13 三個前提重量後的結果）
# ```
# ② 邊際報酬遞減              ✅ 仍成立
# ③ 代理 c vs 精確 c          ⚠ Spearman 0.825／top10% 重疊 74.4% = borderline，
#                               且代理把動態範圍高估 1.5~1.7 倍 ⇒ 先前用代理的結論不可沿用
#    ⇒ 被否證的冪次式跨度 0.083~25.0 = **300 倍**，而校準帶是 w≈2.28（對齊 absgrad 的 26x）
#      ⇒ 它們可能是**加權過猛**而不是方向錯 => 本腳本用 `cost_add_densify`
#        （加法式 + **精確** c，取自 trim pass 的 num_covered_pixels，不是 radii² 代理）
# ④ SSIM 是否隱含按足跡計價   ⛔ 假說被推翻，而且是**反向**的：固定總誤差量、只改攤開方式，
#    SSIM/L1 從半徑 2px 的 14.75 單調降到 64px 的 1.05（ΔL1 幾乎不動 ⇒ 斜率全屬於 SSIM）
#    ⇒ SSIM 按足跡的**反比**計價：誤差攤成小斑塊罰得比集中成一大塊重 14 倍
#    ⇒ 我方的 `1/c`（偏好小足跡）與 SSIM 已在做的事**同向** ⇒ 可能是冗餘的
#      ⇒ 可驗的預測：**提高 lambda_dssim 之後，1/c 的增益應該縮小或消失**
# ```
# ## 2x2（基準那一格已經有了，所以只要跑四臂中的三臂 + Taming 方向）
# ```
#                    cost_add=0              cost_add=+2.278
#   dssim 0.2        lab/speed3（已完賽）✓    costdir
#   dssim 0.5        dssim05                 both
#   額外：costtaming = cost_add **-2.278**（訊號變成 c，Taming 的方向）
# ```
# ## 為什麼是 21,920 步而不是 60k
# b6 有 548 台相機、`check_val_every_n_epoch: 20` ⇒ val 落在 10,960 / 21,920 / …
# 而基準 `lab/speed3` 已經有這兩點的值 ⇒ **21,920 步就能直接對齊比較**，每臂約 2.4h。
# ★ 判準是**四個指標的正負號模式**（「紋理比↑ 而光度↓」那個虛假細節簽名還在不在），
#   不是誰分數高 —— 名次要 ~51k 步才穩，而符號模式好分辨得多。
# ⚠ `cost_add_densify` 需要 trim 開著且 densify 晚於第一次 trim（現行配方 trim@500、
#   densify@1000 ⇒ 成立）；拿不到訊號它會**大聲印警告**，不會靜默 no-op。
source "$(dirname "$0")/_common.sh"
BLK=${1:?用法: task_costsign.sh <block_id> <costdir|costtaming|dssim05|both>}
ARM=${2:?同上}
case "$ARM" in
  costdir)    EXTRA=(--model.density.init_args.cost_add_densify 2.278);           EXP=2 ;;
  costtaming) EXTRA=(--model.density.init_args.cost_add_densify -2.278);          EXP=2 ;;
  dssim05)    EXTRA=(--model.metric.init_args.lambda_dssim 0.5);                  EXP=2 ;;
  both)       EXTRA=(--model.density.init_args.cost_add_densify 2.278
                     --model.metric.init_args.lambda_dssim 0.5);                  EXP=3 ;;
  *) echo "⛔ 未知的 arm：$ARM"; exit 2 ;;
esac
export CITYGS_DIFF_EXPECT=$EXP          # run_fit 收尾會 diff resolved config，超過就大聲講
run_fit "${RUN_PREFIX}${ARM}" "$BLK" \
  --model.initialize_from null \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 0.001 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --trainer.max_steps 21920 \
  "${EXTRA[@]}"
