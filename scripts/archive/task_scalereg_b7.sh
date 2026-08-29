set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/scalereg_b7
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
# ★★ 使用者目視發現的疊影：粒子相對尺度過大（2026-08-20）
#
# 使用者看 best_b7 的 test 圖，發現「重建細節清楚的照片會出現其他張照片的疊影」。查證屬實，
# 而且**這些 test 視角本來就在訓練集裡**（split_mode=reconstruction，192 張全訓練，每 8 張取 1
# 當 val）=> 模型連自己訓練過的影像都重建不出來，不是泛化失敗。
#
# 四個假設被逐一否證（全部 CPU，見研究總覽 §12.28）：
#   ❌ 收斂不足        —— sched30_b7 視覺上幾乎一樣
#   ❌ 視線多層粒子    —— 模型深度離散度只有 SfM 基準的 0.58 倍（b12 是 0.82），比內容更集中
#   ❌ 供給不足        —— b7 每個 SfM 點分到 3.05 顆（b12 只有 2.32），不透明度也更高
#   ❌ GT/相機配錯     —— GT 側銳利正確、建築位置一致
#
# ✅ 站得住的：**粒子相對尺度**（長軸 / 表面取樣間距，已對塊的空間尺度正規化）
#      best_b7 2.06   sched30_b7 1.95   sched30_b12 **1.47**
#   b7 的粒子橫跨兩個以上表面取樣點 => 在深度不連續處（高樓邊緣）架成半透明的膜，
#   把本來不相連的表面連起來 => 正是肉眼看到的疊影。b12 在 1.47 就沒有這個症狀。
#
# 本臂：`scale_reg` 0.007 -> 0.02（MCMC 的尺度 L1，直接壓縮尺度），其餘同 sched30_b7。
# 對照 `sched30_b7` 24.993 / 相對尺度 1.95 / 建築低頻 0.02195；噪音底 PSNR 0.0325、低頻 0.00028。
#
# 判讀（**相對尺度是主判準，PSNR 是護欄**）：
#   相對尺度 -> ~1.5 且 PSNR 不掉超過 3x 底 -> 機制成立，疊影應目視可見地減少 => 進配方
#   相對尺度降但 PSNR 大掉               -> 尺度是被內容需要的，壓它是用品質換乾淨
#   相對尺度沒降                          -> scale_reg 不是有效的旋鈕，改試 screen_size_prune_px 調低
# ⚠ 開獎必須**看圖**，不能只看數字 —— 這個問題是使用者用眼睛找到的，數字全都沒抓到。
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_7.ply \
  --data.parser.block_id 7 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.scale_reg 0.02 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n scalereg_b7
