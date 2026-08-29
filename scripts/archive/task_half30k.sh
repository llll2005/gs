#!/bin/bash
# ★★★★ 壓縮排程：總步數砍半 + 依演算法特性重調各階段（2026-08-24，使用者要求「謹慎但大膽」）
#
# 底座 = `coarseft_b12`（現行最佳 26.518）的完整組態，只改「時間」相關的東西。
# 對照組就是 coarseft_b12 本身：平均 26.510 / 中位 26.56 / 最差 17.41 / 建築低頻 0.02504。
# 噪音底 PSNR 0.0325。⚠ 判準多指標 + 最差 10% + 看圖。
#
# ── 逐項理由（不是等比例縮放，是按各機制的特性調）─────────────────────────────
#
# max_steps 60,000 -> 30,000
# means_lr_scheduler.max_steps 60,000 -> 30,000
#   ⚠ **這兩個必須一起改**。位置 LR 是 exp(-7.675e-5 t) 衰減到 max_steps 為止；
#     只改 trainer 不改 scheduler，LR 會停在初始的 10%，等於整段都在高 LR 亂跳。
#
# densification_interval 150 -> 100     （事件更密）
# add_ratio 1.05 -> 1.08                （每次變動量更大；本次才讓它可調）
# contribution_prune_interval 500 -> 250（trim 同步加密，維持破平衡餘裕）
#   實測破平衡點：舊組(1.05/500/0.1)=231.5，新組(1.08/250/0.1)=182.6，interval 100 仍在成長側。
#   淨成長率：舊組每 500 步 x1.059；新組每 250 步 x1.091 => **每步成長速度約 2 倍**，
#   正好補回步數砍半。
#
# densify_from_iter 1000 -> 500
# densify_until_iter 30,000 -> **10,000（佔 33%，比 sched30 的 50% 更激進）**
#   依據 §12.21/§12.25：densify 停止後的收割期是最大增益來源（+1.85 dB）且**強烈前傾**；
#   而撞 cap 之後到 densify_until 之間全是 churn（族群翻轉 5.2 倍卻淨零），實測是淨負的。
#   推估族群在 step ~5,800 撞 cap（起始 494,317 x 0.7 = 346,022，到 2.6M 是 7.5 倍，
#   ln(7.51)/ln(1.091) = 23.2 個 250 步）=> 留 ~4,200 步 churn 餘裕，其餘 20,000 步全給收割。
#
# start_prune_ratio 0.0 -> 0.3          （使用者要求「壓低初期的總顆數」）
#   起始 trim 從「只砍貢獻度最低的那些」變成「砍掉最低的 30%」=> 早期 VRAM 更低、
#   densify 從更乾淨的基底長起。⚠ §11.2 量到起始 trim 保留的是「前層」，砍 30% 可能
#   連對的一起砍 —— 這是本臂的主要風險。
#
# noise_lr 500,000 -> 1,000,000         （x2）
#   MCMC 的 SGLD 噪音正比於當前位置 LR，而 LR 隨 max_steps 衰減。步數砍半 =>
#   噪音的時間積分砍半 => 探索量砍半。x2 補回來。
#
# sh_degree_up_interval 1,000 -> 500
#   SH 0->3 需要 3 次遞增，原本 step 3,000 到位（佔 5%）；砍半後 step 1,500（同樣 5%）。
#
# cap_max 2,600,000 不變 —— 刻意不動，否則與 coarseft 的比較會多一個變數。
#   （§11.1/§12.27 已證實顆數在 2.34M 就飽和，往上只買到感知且傷尾巴。）
#
# ── 判讀 ───────────────────────────────────────────────────────────────────
#   接近 26.5 -> **一半的時間拿到同樣的品質**，這是「消費級硬體」敘事的實料
#   掉 0.1~0.3 -> 壓縮有代價，但可畫出「時間 vs 品質」的效率前緣（也是可用的結果）
#   掉很多     -> 60k 不是可壓的，收割期的絕對長度才是關鍵
#   OOM       -> 使用者已授權可接受；ledger 會記死在第幾步與當時顆數
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
[ -d outputs/coarse_fix ] || { echo "✘ outputs/coarse_fix 不存在 —— 先跑 task_coarse.sh"; exit 1; }
rm -rf outputs/half30k_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from outputs/coarse_fix \
  --data.parser.block_id 12 \
  --trainer.max_steps 30000 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 30000 \
  --model.gaussian.init_args.optimization.sh_degree_up_interval 500 \
  --model.renderer.init_args.start_prune_ratio 0.3 \
  --model.renderer.init_args.contribution_prune_interval 250 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.densify_from_iter 500 \
  --model.density.init_args.densify_until_iter 10000 \
  --model.density.init_args.densification_interval 100 \
  --model.density.init_args.add_ratio 1.08 \
  --model.density.init_args.noise_lr 1000000.0 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n half30k_b12
