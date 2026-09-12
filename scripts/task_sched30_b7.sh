set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
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
# ★★ 排程壓縮：densify_until 42,000 -> 30,000（2026-08-17，理由已被 CPU 驗證修正）
#
# ⛔ 撤回原本的理由。我原本寫「收割期 LR 積分變 3.02 倍」。CPU 實測（10 個修正後跑次的
#    收割段擬合，scratchpad/lrfit.py）：把自變數換成 LR 積分 vs 換成步數，**R² 幾乎一樣，
#    10 個裡 8 個是步數擬得更好**（例：sb4 0.9858 vs 0.9878）。收割期內 LR 是步數的平滑單調
#    函數 => 兩種參數化分不開，「LR 積分是資源」只是換座標軸重講同一條曲線，不可當機制。
#
# ✅ 站得住的是外推出來的收斂餘量（同一次擬合，A = 漸近值）：
#      b7                   A=26.051   餘量 **1.234 dB**   <- 差一整個 dB 沒收完
#      b12 各跑次(n=9)                 餘量 0.10 ~ 0.20 dB <- 已收斂
#    b12 在 56.8k 見頂、b7 到 60k 仍在爬(+0.107)，與此一致。
#    ⚠ 1.234 是**外推**不是量測，且 b7 的時間常數 c=0.0014 比 b12 群(0.0057~0.0095)小 5 倍、
#      是離群值 => 當方向與量級參考，不要當預測值引用。
#
# ★ 因此本臂的判準是可證偽的，分成兩個不同的宣稱：
#     sched30_b7 > 26.05  => **停 churn 提高了可達品質**（不只是收斂更快）=> 機制成立，進配方
#     24.8 < x < 26.05    => 只買到收斂速度（同樣有用，但宣稱不同：是省時間不是提品質）
#     x <= 24.82(基線)    => 晚期 densify 真的在改善配置，churn 是必要探索 => 維持 42k
#
# ★ 提早中止判準（省 GPU）：baseline b7 在 step 42,239 是 23.815。本臂在 42,000 時已經
#   收割了 12,000 步 => **若 42k 附近的 val 沒有明顯高於 23.815，機制就沒發生，可以直接砍掉**，
#   不必等到 60k（省約 4 小時）。用 `tail -f logs/runner.log` 看 val 即可。
#
# 附帶（與機制無關但成立）：30k/60k=50% 正好是上游 vanilla 的比例（15k/30k）。我方 42k/60k=70%
# 是 2026-06-25 設的（config 未追蹤、註解自承「主力積極槓桿」），**早於 GT 修正 => 依據已作廢**。
# 顆數 26,879(b7)/28,399(b12) 撞 cap => 30,000 是族群建完後最早的切點。
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
