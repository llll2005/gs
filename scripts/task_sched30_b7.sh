set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/sched30_b7
# ★★ 配方轉移測試：現行最佳（noprior）換到 block_7（2026-08-16 排入）
#
# 動機（使用者提議）：整套配方都是在 b12 上調的，而**b12 在 25 塊裡內容密度排第 13、
# 低於中位數**（GT 梯度 0.01547），半面是水。b7 是**最重的一塊**（0.02189 = b12 的 1.42×），
# 而且是官方 CityGSV2 baseline 重跑用的那塊。同時是最硬的內容測試與有歷史對照的塊。
#
# 唯一變數是 block_id 與 init PLY，cap 刻意維持 2.6M 與 b12 完全一致 —— 換 cap 就失去可比性。
#
# ⛔ 撤回：我原本在這裡押注 b7 = 24.22，由既有 8 塊 run 擬的「內容密度 vs PSNR」外推。
#   擬完才發現**那 8 塊只有 4 塊真的跑到 60k**（block 2/3 死在 1499、block 7 死在 14999、
#   block 5 死在 29999），而 results.txt 照樣寫 PSNR 不寫步數。那條斜率其實在描述
#   「哪些塊會死」不是內容難度 => 押注與由它外推的合併分數推估一併作廢。
#   比較任何 PSNR 之前先跑 `python tools/run_status.py --runs <name>`。
#
# ⚠⚠ **b7 有前科**：aggr17 那次它就是死在 step 14,999 的那塊，而且它是內容最重的一塊
#   （0.02242 = b12 的 1.45x）。本臂 cap 2.6M 高於 aggr17 的 1.7M => OOM 風險是真的。
#   若 OOM，那**本身就是要找的隱藏問題**：VRAM 餘裕是 b12 特有的，全場景 25 塊會一路踩。
#   ledger 會記 DIED 與死在第幾步。不要為了避免它而偷偷調低 cap —— 換 cap 就失去可比性。
#
# 判讀（b12 noprior = 26.083）：
#   b7 落在 25~26      -> 配方會轉移，內容更重只是稍難，可以放心推全場景
#   b7 明顯更低(<24)   -> 配方過擬合到 b12 的低內容/水面，逐塊 cap 或參數要按內容調
#   b7 反而更高        -> b12 的水面在拖累它，真實的建築區能力被低估了
#   OOM                -> **這本身就是重要發現**：VRAM 餘裕是 b12 特有的，內容重的塊
#                         在同 cap 下 overdraw 更高，全場景 25 塊會一路踩雷。ledger 會記
#                         DIED 與死在第幾步，不算白跑。
#
# ⚠ b7 只有 192 視角（b12 是 284）=> epoch 數會不同，但 max_steps 都是 60,000，可比。
# ★★ 排程壓縮：densify_until 42,000 -> 30,000（2026-08-17 改寫理由）
#
# 這不是「多給步數」，是把**收斂預算變 3 倍**，而且是**回到參考實作的比例**。
#
# ① 我們偏離了參考值。上游 vanilla：densify_until 15,000 / max_steps 30,000 = **50%**。
#    我方 42,000 / 60,000 = **70%**。config 註解自承「30k -> 42k：主力積極槓桿」，
#    而該 config 日期 2026-06-25，**在 GT 修正(08-12)之前** => 那次調參的依據已作廢。
#    30,000/60,000 = 50%，正好回到參考比例。
#
# ② LR 積分（記憶 degeneracy_and_blur_split 的 "瓶頸是 B_opt 不是 B_mem"）。實測
#    lr(t)/lr0 = exp(-7.675e-5 t)（對 42k=3.98%、59.9k=1.01% 完全吻合）：
#      densify_until   收割期 LR 積分   起始 LR
#         42,000            388          4.0%
#         30,000           1173         10.0%   <- 3.02 倍
#         24,000           1935         15.8%   <- 但族群還沒長滿，不是免費的
#    顆數在 26,879(b7) / 28,399(b12) 撞 cap => **30,000 是族群建完後最早的切點**。
#
# ③ 收割期本來就是最大單筆增益來源：b12 +1.85 dB、b7 +1.43 dB，而且發生在 LR 只剩
#    1~4% 的時候、完全不動密度 => 那些分數族群本來就有，是被密度 churn 擋著收斂不了。
#
# 判讀（b12 noprior 26.083，best_val 26.149@56,800；b7 24.817，60,000 仍在爬）：
#   b12 微幅上升或打平 -> 合理，它本來就在 56.8k 飽和，多的預算沒地方去
#   **b7 明顯上升      -> 機制證實**：b7 是有收斂餘裕卻預算不夠的那個，效應該最大
#   兩邊都掉           -> 晚期 densify 真的在改善配置，churn 是必要探索，維持 42k
#
# ⚠ 不要同時改 max_steps，否則 LR 排程整條被拉伸，分不出是排程還是步數。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n sched30_b7
