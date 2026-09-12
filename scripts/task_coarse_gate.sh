#!/bin/bash
# ★★★★★★ 殼層問題：coarse 幾何閘門（1M 顆 sh0，~2.5h，只看幾何）
#
# 這**不是**在測殼有沒有消失，是在測「coarse 的幾何值不值得當先驗」。
# 若閘門沒過，§15 的 ①③④ 整組放棄，不必再投基建。
#
# 為什麼是 sh0 且 1M：sh0 的 F=13 floats => 208 B/顆，1M 合計約 2.26 GB，很便宜。
#   （sh3 是 928 B/顆。coarse 只要幾何，不需要高階色彩。）
# ⚠ 顆數要快速上升，但**不可**讓 densification_interval 超過破平衡點 231.5
#   （p7_depth_coverage_bug：interval 350 會讓顆數**衰減**並製造覆蓋破洞）=> 用 150。
# ⚠ 用 depth-init 當起點會把偽深度的殼帶進來 => **用 SfM-init**（initialize_from null），
#   這樣 coarse 的幾何完全由影像決定，才是我們想測的東西。
# ⚠ 本機 3 小時上限（使用者 2026-09-12）=> 24,000 步（實測約 2.44 it/s 時約 2.7h）。
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/coarsegate_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from null \
  --data.parser.block_id 12 \
  --model.gaussian.init_args.sh_degree 0 \
  --model.density.init_args.cap_max 1100000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.densify_until_iter 12000 \
  --model.density.init_args.densification_interval 150 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 1e-3 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --trainer.max_steps 24000 \
  -n coarsegate_b12
echo "===== 閘門：只看幾何 ====="
conda run -n gspl python tools/audit_geometry.py coarsegate_b12 --block 12 2>&1 | tail -25
echo "===== 對照：現行最佳與 SfM-init 的同一個量 ====="
conda run -n gspl python tools/audit_geometry.py speed3_b12 --block 12 2>&1 | tail -12
