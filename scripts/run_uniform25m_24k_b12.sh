#!/bin/bash
# ★形狀 vs 覆蓋：用「完全沒有形狀」的初始化跑 v1 配方（會淨衰減的那個）。
#
# 【init】tools/make_uniform_init.py --n 2500000 --opacity 0.05
#   2,500,000 顆均勻填滿 b12 的內容體積（10.9×9.5×3.94=408 立方單位），間距 0.05465、
#   scale 0.02733、opacity 0.05、法向隨機。**不攜帶任何場景資訊。**
#   離參照表面 <1 個間距的只有 18.8% ⇒ 八成的點從空氣中出發，要靠 relocation 搬到表面。
#
# 【要問的兩件事】
#  1. MCMC 的初始化不變性到哪為止。論文（3dgs-mcmc.pdf §3）自承 relocation 只在**現存活點**
#     之間取樣宿主、SGLD 噪音走出 support region「irrecoverable」⇒ 預測是
#     **精度可以爛（relocation 會修），覆蓋不能缺（relocation 到不了空的地方）**。
#     均勻填充是這個預測的極端測法：覆蓋滿分、精度為零。
#  2. v1 配方會把顆數從 121 萬壓到 7.7 萬（interval 350 > 破平衡 231.5）。
#     從 250 萬無形狀的點出發，最後能不能用少量顆粒表現出高品質。
#
# 【對照】
#   aggr24k_b12（depth-init, 同 config）24,000 步 22.070 / .604 / .695，N 77,616
#     ⛔ 但該跑次有 P7 汙染 2.4%，只能當數量級參考
#   v1clean_24k_b12（同 config、修正後程式）跑中 → 那才是乾淨對照
#
# 【判讀】主要看**建築區紋理比**（tools/rescore_by_content.py），不是整體 PSNR
#   ——b12 一半是平坦水面，均勻色塊也拿 32-40 dB。
#   ⚠ 若 step 5,680 的 val 明顯低於 20.6，先懷疑 opacity=0.05 / scale 這兩個**沒驗證過的**
#     參數（照 3DGS 隨機初始化慣例挑的），而不是直接判「均勻 init 不行」。
#
# 【風險】2.5M > b12 標定的 N_max≈1.70M。但起始 trim 是 forward-only（no_grad）且發生在
#   第一次 backward 之前，culls 到約 36 萬，所以可能撐得住。若 OOM 只損失幾分鐘。
set -euo pipefail
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
NAME=uniform25m_24k_b12
rm -rf "outputs/$NAME"
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_24k_aggr_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/uniform25m_block_12.ply \
  --data.parser.block_id 12 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
