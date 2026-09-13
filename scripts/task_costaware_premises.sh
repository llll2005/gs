#!/bin/bash
# ★★★★★★ 成本感知線的**前提**在修正後資料上還成立嗎（分鐘級，不訓練）
#
# 使用者 2026-09-13 問：能不能用本機便宜地驗證被廢棄的 cost-aware 系列，
# 還是只能跑完幾萬步才知道。
#
# 答案要分兩半：
#   ⛔ 便宜做不到的：**決定誰分數高**。名次要 ~51k 步才穩（記憶 shell_problem_status），
#      而「時間壓縮 proxy」早就試過並退役（CLAUDE.md：three time scales cannot all be preserved）。
#   ✅ 便宜做得到的：**重量那些前提**。29 小時 GPU 的否證不是只靠分數，它站在三個前提上，
#      而那三個前提全部量在**錯位的資料**上：
#        ① 退化定理 v_i ~ c_i（價值與成本都隨足跡放大 ⇒ 比值不帶訊號）
#        ② 邊際報酬遞減（(1-1/e) 保證與 CELF 加速都靠它）
#        ③ 代理 c 與精確 c 的一致性、動態範圍、w 校準
#      ★ 還有第四個，而且它**與資料無關**：SSIM 是不是本來就按足跡計價？
#        若是，顯性的 c 項在建構上就是冗餘的 —— 那會讓這條線不論資料如何都死。
#
# 全部打在**修正後**的 ckpt 上：gate15000/b12@15000、speed3/b13@14999。
set -u
cd "$(dirname "$0")/.." || exit 1
R=logs/costaware_premises_$(date +%m%d_%H%M).log
CK12=$(ls outputs/gate15000/blocks/block_12/checkpoints/*step=15000.ckpt 2>/dev/null | head -1)
CK13=$(ls outputs/speed3/blocks/block_13/checkpoints/*step=14999.ckpt 2>/dev/null | head -1)
CFG=configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml
{
echo "ckpt b12 = ${CK12:-（缺）}"
echo "ckpt b13 = ${CK13:-（缺）}"

echo; echo "════════════ 前提 ④ SSIM 是否本來就按足跡計價（不需模型、與資料無關）════════════"
echo "  斜率隨斑塊半徑上升 => SSIM 隱含的 c_i 是真的 => 顯性 c 項冗餘"
echo "  平線               => dssim05 的 floater 下降是巧合"
conda run -n gspl python tools/loss_footprint_pricing.py --run gate15000 2>&1 \
  | grep -vE "pkg_resources|declare|找不到深度圖|dataparser|down sample" | tail -25

echo; echo "════════════ 前提 ③ 代理 c vs 精確 c（b12，取 200 台相機）════════════"
conda run -n gspl python tools/cost_proxy_quality.py --run gate15000 --step 15000 --max-cam 200 2>&1 \
  | grep -vE "pkg_resources|declare|找不到深度圖|dataparser|down sample" | tail -30

echo; echo "════════════ 前提 ② 邊際報酬遞減（b12）════════════"
if [ -n "$CK12" ]; then
  conda run -n gspl python tools/measure_diminishing_returns.py --ckpt "$CK12" \
    --config "$CFG" --block 12 --rank_views 16 --eval_views 8 2>&1 \
    | grep -vE "pkg_resources|declare|找不到深度圖|dataparser|down sample" | tail -25
else
  echo "  ⚠ 缺 b12 ckpt，跳過"
fi

echo; echo "════════════ 同樣三項在 b13 上覆核（跨塊一致才算）════════════"
conda run -n gspl python tools/cost_proxy_quality.py --run speed3 --step 14999 --max-cam 200 2>&1 \
  | grep -vE "pkg_resources|declare|找不到深度圖|dataparser|down sample" | tail -20
} 2>&1 | tee "$R"
echo
echo "報告：$R"
echo "★ 判讀："
echo "  前提全部仍成立  => 子集選擇那一形式**不論資料如何都死**，省下 lab 的 60k 跑次"
echo "  有前提破了      => 那條線真的重開，才值得排 15k 的簽名 A/B 與 lab 的 60k"
