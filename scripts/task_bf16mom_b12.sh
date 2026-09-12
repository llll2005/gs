#!/bin/bash
# ★★★★★ 壓縮**優化器狀態**（不是模型）：Adam 動量存 bf16 + 用省下的 VRAM 加 cap
#
# 為什麼是這條，而不是 DBS/SH2（研究總覽 §11.66/§11.79）：
#   VRAM 拆解  逐顆儲存 928 B（**83%**）／binning 191 B（17%）
#   儲存拆解   參數 232 + 梯度 232 + exp_avg 232 + exp_avg_sq 232
#              => **動量佔儲存一半，且完全不影響模型表達力**
#   已測且落敗的兩條（都是拿表達力換顆數）：
#     sh2 + cap3.06M（等 VRAM）  PSNR **-16.4sd** => -0.454 dB
#     SB-4 + 35% 顆粒            六項指標**全輸**
#   => 「同樣 6GB 該買更強的顆」已結案；剩下的只能從**純開銷**要。
#
# 算式：bf16 動量 => 232+232+116+116 = 696 B 儲存 +191 binning = 887 B/顆
#       vs 基準 1119 B/顆 => 同 VRAM 可裝 **1.262 倍** => cap 2.6M -> **3.2M**（保守取）
# ⚠⚠ 2026-09-10 修正判準：我原本用 +0.541 **dB**/加倍 估成「+0.16 dB」——**用錯指標**。
#   在這個邊際上 PSNR 是**平的**，增益出現在 SSIM/LPIPS（`cap30` 實測，**兩塊都複製**）：
#     cap 2.6M->3.0M（+15%）  b12 PSNR +1.6sd 平／SSIM **+30.6sd**／LPIPS **+4.5sd**
#                             b7  PSNR +1.4sd 平／SSIM **+41.0sd**／LPIPS **+6.6sd**
#   => 本臂 +26% 容量，線性外推 SSIM +0.005~0.007、LPIPS +0.007~0.010
#      （噪音底 3sd：SSIM 0.0015／LPIPS 0.0075）=> 即使打對折仍遠超 SSIM 門檻。
# ⚠ 代價：`cap30_b7` 的**最差 10% 視角掉 0.61** => 必須連 `tools/tail_analysis.py` 一起判。
# ⚠ 本臂**不會變快**（數學仍 fp32，只省 optimizer step 約 28% 的記憶體流量 ≈ 全步 0.6%，
#   低於 5% 的 wall-time 噪音底）=> 這是**純容量**操作，freed VRAM 必須拿去加 cap 才有價值。
# ★ 同時是 8-bit（+45% 容量）的前置測試：低精度動量若本身就傷品質，8-bit 只會更傷。
# ⚠ 8-bit 不是 drop-in：bitsandbytes 的 state 是 `state1/state2/absmax*`（blockwise），
#   而 density controller 的 cat/mask/relocation 三處都硬寫 `exp_avg`/`exp_avg_sq`
#   => 要為它重寫狀態手術，且 absmax 的 block 邊界會被 append 打亂（錯了不報錯）。
# ⚠⚠ **必須開 `stochastic_rounding`**（參考論文 2603.16731v1，低精度 EMA 的 stalling）：
#   v += (1-b2)(g²-v)，b2=0.999 => 相對變化 0.001·|g²/v-1|；bf16 捨入門檻 2^-8/2=0.00195
#   => 需 |g²/v-1| > 1.953，而 g²>=0 使**向下更新永遠跨不過** => **v 只增不減 = 單向棘輪**
#   => Adam 步長 lr·m/sqrt(v) 單調萎縮 => 模型逐漸凍結，而分數只會「看起來差」查不出原因。
#   實測（3,000 步梯度縮小 100 倍）：確定捨入 v 比值 **1.0000（完全凍結）**／
#                                  隨機捨入 **0.0500**／fp32 對照 0.0501（幾乎一致）
#   ⚠ 第一版隨機捨入用 `torch.nextafter` 是**靜默無效**的（fp32 的下一個值捨回同一個 bf16）；
#     正解是在截斷前加 16-bit 隨機數。**兩版給出完全相同的 v，靠這個測試才抓到。**
# ✅ 已驗：state 8.00 -> 4.00 B/元素；
#    densify 後 dtype **仍是 bf16**（`cat_tensors_to_optimizers_` 原本會靜默升回 fp32，已修）。
# ⚠ 非位元等價 => 判準看四指標；對照 speed3_b12（26.6957 / 0.7757 / 0.3446 / 0.4744）。
# ⚠ OOM 風險：先前 fp32 在 N=3.10M 爆過；bf16 的每顆儲存較低，同 N 佔用更少，但 binning 更多。
set -u
cd "$(dirname "$0")/.." || exit 1
# 2026-09-12 實測（tools/vram_pressure.py）：max_split_size_mb:128 固定貴 6.4~10.3%，
# 而在真實峰值 N=2.6M + 0.6G ballast 下「完全不設」也沒 OOM => 預設不開（使用者決定）。
# ⚠ 微基準沒有 Lightning/dataloader/長跑碎片，是樂觀估計 => 若真的 OOM，一個變數就復原：
#     CITYGS_ALLOC_CONF=max_split_size_mb:128 bash scripts/task_xxx.sh
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/bf16mom_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.gaussian.init_args.optimization.optimizer.class_path LowPrecMomentAdam \
  --model.gaussian.init_args.optimization.optimizer.init_args.moment_dtype bfloat16 \
  --model.gaussian.init_args.optimization.optimizer.init_args.stochastic_rounding true \
  --model.density.init_args.cap_max 3200000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 1e-3 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n bf16mom_b12
