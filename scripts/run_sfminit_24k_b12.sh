#!/bin/bash
# S1：SfM-init 單臂 A/B（b12，24k）
#
# 【問的問題】floater 的病灶是不是在初始化？
#   紀錄/初始化溯源與SfM-init_2026-08-05.md：depth_init PLY 起點就離表面 20.2×（訓練只推到
#   29.2×，80% 偏差在第一步）；3DGS 論文 init 消融說 init 不良的 floater「cannot be removed by
#   optimization」——這解釋了為什麼貢獻度剪枝／opacity_reg／塵埃收割／多視角集中度／深度一致度
#   全部失敗：它們都在下游。
#
# 【對照組】aggr24k_b12（v1，同 config，唯一完整跑完的 24k）：
#     step  5,679  20.644 / .582 / .764
#     step 11,359  21.142 / .593 / .732
#     step 17,039  21.588 / .598 / .714
#     step 22,719  21.907 / .604 / .698
#     step 24,000  22.070 / .604 / .695
#
# 【唯一的變數】不傳 --model.initialize_from。config 裡該欄是 null，
#   gaussian_splatting.py:227 於是走 setup_from_pcd(datamodule.point_cloud)，
#   而 colmap_block_dataparser.py:150 讀 points3D 時已帶 selected_image_ids ⇒ 逐塊 SfM 點雲。
#   b12 實得 320,513 顆（depth_init 是 1,211,537）。
#   ⚠ 非純單變數：起始位置與起始顆數同時變（3.8× 少）。這是 init 方法本身的差異，拆不開，
#     報告要照實寫，不可說成「只改了位置」。
#
# 【怎麼判讀】⚠ 不要用單一早期 val 下判斷。實測 oreg_0p002_b12 在 step 5,680 落後
#   b12_cap2m_reg000_exact 0.40 dB、到 step 34,079（走完 57%）都還落後，最終反超 +0.46。
#   ⇒ 看的是五個點的整條曲線形狀，不是單點。
#
# cap_max 不用調：MCMC 每次 densify ×1.05，320k→3M 需 ln(9.4)/ln(1.05)≈46 次；
# interval 350 → 約 16,100 步，而 densify_until_iter=16000，剛好用滿。
set -euo pipefail
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

NAME=sfminit24k_b12
rm -rf "outputs/$NAME"

conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_24k_aggr_aerial.yaml \
  --data.parser.block_id 12 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
