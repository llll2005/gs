set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/scrprune_b7
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
# ★★★ 打開一個從沒跑過的機制：screen-size prune（2026-08-21）
#
# ⛔ 發現：`--model.density.init_args.screen_size_prune_px 300` **在每個跑次都是死參數**。
#   `_max_radii2D` 只在 `screen_prune_emergency_px > 0` 的分支被填（mcmc_2dgs:296-299），
#   而它預設 **-1**、config 與所有 task 腳本都沒設 => `:353` 的 `_max_radii2D is not None`
#   永遠為假 => **剪枝從未執行**（runner.log 裡 "screen-prune" 出現 **0** 次）。
#   而 `:229` 的說明文字自己就寫了正確用法：「Set this ABOVE the normal screen_size_prune_px
#   (e.g. prune=300, emergency=600)」—— **要兩個一起設，我們只設了一個。**
#
# 為什麼這打在疊影上：使用者目視發現的疊影，成因量到是**粒子相對尺度過大**
#   （b7 2.06 vs b12 1.46，長軸/表面取樣間距，研究總覽 §12.28），
#   而 screen-size prune 正是**專門回收超大螢幕足跡粒子**的機制 —— 它從來沒跑過。
#   對照 §12.33：用 scale_reg 壓尺度只換到 4% 且三項指標都退；**剪掉**可能比**壓**有效。
#
# 本臂 = `sched30_b7` + `screen_prune_emergency_px 600`（讓 300 那個真的生效）。
#
# 判讀（對照 sched30_b7：PSNR 24.993 / SSIM 0.8009 / LPIPS 0.2554 / 相對尺度 1.95 / 建築低頻 0.02195）：
#   相對尺度顯著降 + 指標不退 -> 機制成立，且**所有既有跑次的 screen_size_prune 都是無效的**，
#                                 要重新檢視「大足跡粒子」相關的全部結論
#   指標退很多               -> 那些大粒子是必要的（承載大面積內容），剪掉會破洞
#   幾乎沒變                 -> 大粒子不多，疊影另有成因
# ⚠ **必須看圖**確認疊影是否實際減少 —— 這問題是眼睛找到的，指標全都沒抓到。
# ⚠ 多指標判讀（`python tools/audit_all.py`），不可只看 PSNR（§12.30 的教訓）。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.screen_prune_emergency_px 600 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n scrprune_b7
