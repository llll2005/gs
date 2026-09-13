#!/bin/bash
# ★★★★★★ 修正後資料上第一個完整 60k（lab/speed3 b6，PSNR 30.45 / 2.34M 顆）的兩項分析
#
# 兩個問題都只有 60k 模型能答，而它剛落地本機：
# ```
# ① 失敗區在 60k 還是乾淨的嗎？
#    15,000 步時 b12 4.92% / b13 4.88%（舊資料同判準 25.5%），但那是**短跑**。
#    60k 讓 N 長到 2.34M，若失敗區在長跑後回來，那 15k 的乾淨就是假象。
# ② 背包的空間在 60k 還在嗎？
#    rho(v,c)≈0 與「天花板隨預算收緊而跳」是在 15k 的模型上量的。
#    cost_budget 的介入臂跑 21,920 步，但最終要推到 60k => 要知道這個量在 60k 會不會變。
# ```
# ⚠ 這是 lab 跑的（outputs/lab/ 底下），**不適用「outputs/ 都是這台 6GB 跑的」鐵律**。
#   但它是在 CITYGS_VRAM_CAP_GB=5.66 下跑的，所以仍在 6GB 信封內。
set -u
cd "$(dirname "$0")/.." || exit 1
R=outputs/lab/speed3/blocks/block_6
CK=$(ls "$R"/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "⛔ 找不到 b6 的 60k ckpt"; exit 1; }
CFG=$(ls "$R"/lightning_logs/version_*/config.yaml 2>/dev/null | tail -1)
[ -n "$CFG" ] || { echo "⛔ 找不到 resolved config（用 lab.py getfile 補抓）"; exit 1; }
Q='pkg_resources|declare|找不到深度圖|No depth maps found'

echo "=== ① 存 val 影像（逐 tile 失敗區分析要圖）==="
if ls -d "$R"/test/*/ >/dev/null 2>&1; then
  echo "  已有 test/ 影像，跳過"
else
  conda run -n gspl --no-capture-output python -u main.py test --config "$CFG" --save_val 2>&1 \
    | grep -vE "$Q" | tail -12
fi

echo
echo "=== ② 失敗區逐 tile 判定（絕對對比門檻）==="
conda run -n gspl python tools/failure_map.py lab/speed3 --blk 6 2>&1 | grep -vE "$Q" | tail -26

echo
echo "=== ③ 背包空間在 60k 還在嗎 ==="
conda run -n gspl python tools/rho_value_cost.py --ckpt "$CK" --max-cam 120 2>&1 \
  | grep -vE "$Q" | tail -24
