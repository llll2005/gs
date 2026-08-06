#!/bin/bash
# Stage 0：信心分級初始化（Tier A 錨點 + Tier B 填充），b12，24k
#
# 【init 內容】tools/make_graded_init.py --block 12 --track_min 4
#     區域內 SfM 1,160,384 → track>=4 是 648,416 → 體素降採樣 @0.02456 → 錨點 203,916
#     Tier B 填充 1,112,307        總計 1,316,223（depth-init 1,211,537 的 1.09×，顆數對齊）
#     ⚠ 錨點只佔 15.5%——瓶頸是「有多少體素格裡有任何 SfM 點」，不是 track 門檻。
#
# 【對照組】aggr24k_b12（v1，完全相同的 config，只差 initialize_from）
#     step  5,679  20.644 / .582 / .764        N=  338,033 @499 → 292,798 @1499
#     step 11,359  21.142 / .593 / .732        N=  181,327 @5999
#     step 17,039  21.588 / .598 / .714
#     step 22,719  21.907 / .604 / .698
#     step 24,000  22.070 / .604 / .695        N=   77,616  ← v1 是淨衰減配方
#
# 【這一臂真正要量的東西】不是 PSNR。15.5% 的錨點在沒有 lr 保護（那是 Stage 1）下，
#   預期分數差異很小。要量的是兩件在 step 5,680 就能讀到、且決定 Stage 1 值不值得做的事：
#     (1) 漂移：錨點離它自己的初始（已驗證）位置跑了多遠
#         → 漂了 ⇒ 光度損失會把驗證過的點拉離表面，Stage 1 的 lr 保護是必需品
#         → 沒漂 ⇒ init 線比假設的弱，該轉向 RAIN-GS 的觀點（問題在優化不在 init）
#     (2) 存活率：v1 的 config 是淨衰減（interval 350 > break-even 231.5，每 3500 步 −22%），
#         所以貢獻度剪枝會大量處決。錨點的存活率若不高於填充點，就再一次證明
#         貢獻度剪枝盲於幾何——這是本專案第六個判準的獨立驗證。
set -euo pipefail
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

NAME=graded_k4_24k_b12
INIT=data/matrix_city/aerial/train/block_all/depth_init/graded_k4_block_12.ply
rm -rf "outputs/$NAME"

conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_sb_24k_aggr_aerial.yaml \
  --model.initialize_from "$INIT" \
  --data.parser.block_id 12 \
  -n "$NAME" > "logs/$NAME.log" 2>&1
