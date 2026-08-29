set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/cap30_b7
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
# ★★ b7 的顆數天花板：cap 2.6M -> 3.0M（2026-08-21）
#
# 為什麼是現在：`scalereg_b7` 開獎 —— `scale_reg` 0.007->0.02（2.9 倍）只把相對尺度
# 從 1.95 推到 1.87（**−4%**），離 b12 的 1.46 差很遠，而且 PSNR −0.133(4.1x底)、
# SSIM −0.0018(2.0x)、LPIPS +0.0022 三項都退。
# ⇒ **正則壓不下尺度，是重建損失在頂回來** ⇒ 粒子需要那麼大才能覆蓋表面。
#
# 而 b7 的 SfM 取樣間距 0.00447 與 b12 的 0.00408 相近，粒子卻大 1.4 倍。最可能的解釋：
# **空拍 SfM 在屋頂密、在建築立面稀** => b7 的真實表面積被 SfM 低估 => 它需要更多粒子。
# **b7 的 cap 從來沒測過**（b12 在 2.3M 飽和，但那是半面水的塊）。
#
# 判讀（對照 `sched30_b7` 24.993 / 相對尺度 1.95 / 建築低頻 0.02195；**多指標，跑 audit_all.py**）：
#   相對尺度明顯降 + 指標不退 -> b7 是供給不足，逐塊 cap 該按內容調 => 這才是疊影的解
#   相對尺度不降            -> 尺度不是顆數決定的，疊影另有原因
#   指標退（如 b12 的 cap30）  -> b7 也已飽和 => 顆數線在兩塊都關閉
# ⚠ b12 的 cap30 是「PSNR 微退但其餘勝」，所以**不可只看 PSNR 判**（§12.30 的教訓）。
# ⚠ 開獎必須**看圖**確認疊影有沒有實際減少 —— 這問題是眼睛找到的。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.density.init_args.cap_max 3000000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n cap30_b7
