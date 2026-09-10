#!/bin/bash
# ★★★★★ **我方命題的符號**：cost_aware_densify **+0.5** = 往**便宜**（小足跡）的地方增生
#
# 為什麼現在跑（2026-09-06，全部實測）：
#   w=-0.5（Taming 的符號，往貴的地方增生）實測 **輸**：
#     PSNR -4.8sd ／ LPIPS -4.1sd ／ floater 1.660% -> 1.713% ／ 紋理比 +4.0sd
#   而三個跑次呈完美單調：**floater ↑ ⟺ 紋理比 ↑ ⟺ PSNR ↓**
#     speed3 1.540%/0.4744/26.6957  ／ agd2 1.660%/0.4902/26.5602 ／ fpd 1.713%/0.5109/26.4260
#   ⇒ 紋理比在本工作點主要量到**塵埃**；而往貴的地方增生會製造塵埃。
#   ⇒ **反方向（我方命題的方向）從未被測過** —— 先前的跑次不是靜默 no-op 就是 w<0。
#
# 這一臂是論文命題 `max Q s.t. (1/K)Σc_i <= B` 的**取樣端**直接檢驗：
#   `probs /= (c/median)^w`，w>0 ⇒ 小足跡的粒子更容易被選為分裂父代。
# ⚠ 強度 +0.5 與 -0.5 對稱，好直接對比。
# ⚠ 判準：PSNR/LPIPS 為主（紋理比在本工作點不可當正面證據 —— 它跟著塵埃走）。
#   對照 agd2_b12 26.5602 / 0.7755 / 0.3467 ／ floater 1.660%。
# ★ 開跑後確認 log 有 `[cost-aware] ✅ 首次觸發`。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/cheap_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.cost_aware_densify 0.5 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  -n cheap_b12
