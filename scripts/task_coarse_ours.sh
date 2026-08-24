#!/bin/bash
# ★★★ 按 coarse 的**用途**設計的 coarse（使用者提議，2026-08-24）
#
# 【它的用途只有一個】對我方而言 coarse **純粹是 init** —— 相機到塊的指派我們用
# `partition_from_colmap.py` 做（不需要 coarse），所以它不必是個好模型，只要提供**好的幾何**。
# ⇒ 依此設計，而不是照搬官方或照搬我方微調的組態：
#
# ① **sh_degree 0**（只有 DC）。視角相依的顏色是微調階段的事，而且是逐塊的；
#    coarse 給位置就好。sh0 每顆 13 floats vs sh3 的 58 => 同樣 VRAM 能放 4.5 倍的點。
#    ✅ 升維是支援的：`gaussian_splatting.py:190-204` 在 `cfg_sh > ckpt_sh` 時把
#      `shs_rest` 從 [N,0,3] 補零成 [N,15,3]（註解自承 "coarse trained at sh_degree=0,
#      fine-tuning at sh_degree=2"）。
#
# ② **用 trim 當壓縮器，而不是用 densify 當生成器** —— 這是本設計最不直觀的一步。
#    全域 SfM 有 **3,826,641** 顆，遠多於任何合理的 coarse。而 `_ctx` 陷阱 2 說
#    「cap 低於現有顆數 => densify 永久關閉」（`add_new_gs = max(0, min(cap,1.05N) - N)`）。
#    ⇒ **刻意讓 cap(1.2M) 低於 SfM(3.83M)**，densify 全程關閉，只剩 trim 每 500 步 x0.9
#      的單調衰減 —— 那正是 coarse 該做的事：**把 SfM 壓縮成可用的 init**。
#      （官方 coarse 也是壓縮：grad-densify 從 3.83M 收到 494,317。）
#
#    停在哪裡是算出來的：trim 從 `contribution_prune_from_iter`(1000) 到 `densify_until` 之間
#    每 500 步一次。**`densify_until 6,500` = 12 次 => 3,826,641 x 0.9^12 = 1,080,756 顆**。
#    而實測 b12 的 284 台相機只看得到全域 coarse 的 **31.9%**（`scratchpad/coarseshare.py`）
#    ⇒ 微調端起始 ~344,761 顆，**幾乎正好對齊 depth-init 的 338,033**
#    ⇒ `coarseft_ours` vs `sched30` 才是「同樣顆數、只差位置品質」的乾淨比較。
#    剩下 23,500 步是純最佳化（densify/trim/noise 都停）—— 收割期正是增益來源（§12.21）。
#
# ③ 其餘沿用我方配方（MCMC / opacity_reg 0.002 / lambda_normal 0 / depth_loss 0 /
#    densify_until 50% / trimming 啟用）。
#    ⚠ **`depth_loss 0` 是刻意不改的**：它是在單塊 b12 上為了最終影像品質調的，而 coarse 的
#      工作是幾何 —— 用它於 coarse 可能是類別錯誤。但 `depthre` 正在測「修正 scales 後
#      depth loss 到底有沒有用」，**等那個答案再決定要不要開，不要現在疊變數**。
#
# ★ 附帶價值：coarse 的 renderer 會被微調端整個繼承（ckpt 路徑，§11.11）。
#   用我方 config 訓練 => renderer 是我方的（trimming 啟用）
#   => `coarseft_ours` 與 `sched30` 的差別**只剩 init**，那才是「coarse-init 值多少」的乾淨答案。
#
# ⚠ **`train_max_num_images_to_cache 512` 必須設**：我方 config 沒設它，預設 -1 = 全部快取，
#   5,621 張 x ~23 MB ≈ 129 GB，會撐爆 46 GB RAM。（官方 coarse config 有設，我方沒有。）
# ⚠ `means_lr_scheduler.max_steps` 必須跟著 `max_steps` 一起改（half30k 的教訓）。
# ⚠ **step 1 的 VRAM 是本臂唯一的 OOM 風險**：起始 3.83M 顆 x sh0，位元組模型估 5.3 GB
#   （上限 6.1）。但 `down_sample_factor 2` 讓螢幕面積只有主線的 1/2.8，binning 緩衝跟著小
#   => 實際應該遠低於估計。使用者已授權 OOM 可接受，而且失敗會在幾分鐘內發生、代價很小。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/coarse_ours
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --data.parser.down_sample_factor 2 \
  --data.train_max_num_images_to_cache 512 \
  --trainer.max_steps 30000 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 30000 \
  --model.gaussian.init_args.sh_degree 0 \
  --model.density.init_args.cap_max 1200000 \
  --model.density.init_args.densify_until_iter 6500 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n coarse_ours
