#!/bin/bash
# ★★★★★ 把 CityGaussianV2 有、我方關掉的幾何正則項加回來：偽深度監督（depth_loss_weight.init=0.5）
#
# 我方 vs V2 的四個差異：①lambda_normal→0 ②depth_loss→0 ③_densify_and_split（MCMC 完全沒有）
# ④DGD。③ 的 MCMC 替代品 blur_split 修正後已測且失敗；①② 從未在當前資料上公平測過。
# ⚠ depth_loss 是在 §12.6 關掉的，而那是用**汙染的 scales** 量的（§11.6 才修好），
#   且**只看平均與建築區、沒看尾巴** —— 紀錄裡早就標註「值得重測，判準看尾巴」，一直沒做。
# ★ 判準改了：**主看 tools/veil_detect.py 的糊掉% 與 corr中位**，不是平均 PSNR。
#   理由 §11.55：失效集中在 46% 的 tile，而平均被能用的那 54% 主導 =>
#   「修好壞 tile、代價是好 tile 稍軟」的機制在平均上必然淨負，會被誤殺。
# 對照 = agd2_b12（26.5602 / 糊掉 46.08% / corr 0.624）。唯一變數 = 這一個旗標。
# ⚠ 這**不是**「調大一定更好」——vpc 的教訓正好相反（強度 5% 直接崩 -2.34 dB）。
#   本臂就是要找出 w 的轉折點在 1 和 2 之間還是更後面。若 w=2 就開始掉，
#   代表 w=1 已在峰值附近，這條線收斂，不必再往上試。
# ⚠ 需要光柵器以 ABSGRAD=1 編譯（子模組分支 citygs-6gb，c58fb57）。
# ⚠ 只開這一個，不要同時開 cost_aware_densify（§11.49 已證它在顆數約束下無效）。
#
# 判準：`tools/cmp_runs.py agd_b12 agd2dl05_b12`（最後 4 個 val 點平均），
#       外加 `tools/veil_detect.py agd_b12:12 agd2dl05_b12:12` 看糊掉%（agd_b12 = 46.17%）。
#       ⚠ 對照是 **agd_b12（26.432）**，不是 sched30_b12 —— 問的是「更強有沒有更好」。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/agd2dl05_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.density.init_args.cap_max 2600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 30000 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.5 \
  -n agd2dl05_b12
