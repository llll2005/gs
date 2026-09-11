#!/bin/bash
# ★★★★★ 實機驗證（~6 分）：K-strip 的 viewspace-grad 跨條帶加總真的生效了嗎
#
# 2026-09-11 修的是：條帶路徑回傳 `dict(last_outputs)`，只有 radii/visibility_filter 被換掉，
# 而 `absgrad_densify`（現行最佳配方的核心）從 `outputs["viewspace_points"].grad[:,2]` 讀 |g|
# => K>1 時只看得到最後一條帶，沒出現在該帶的粒子 |g|=0、退化成 probs = o（等同 absgrad 沒開）。
#
# 判準：log 要出現
#   [K-strip] ✅ viewspace-grad 跨條帶加總首次觸發 K=2：|g| 非零粒子 最後一條帶 X -> 全幀 Y (>1.0x)
# 比值 ~1.0 表示兩者一樣 => 要嘛條帶沒真的切、要嘛加總沒生效，兩種都要查。
# ⚠ 這支只驗「訊號完整性」，不驗分數；train_strips=2 的分數不可與 K=1 的跑次比（SSIM 邊界
#   誤差在 K=2 已經是噪音底的 2.4 倍，見 tools/strip_loss_audit.py）。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
rm -rf outputs/stripgrad_check
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  --model.train_strips 2 \
  --model.density.init_args.cap_max 600000 \
  --model.density.init_args.absgrad_densify 2.0 \
  --model.density.init_args.absgrad_report 200 \
  --model.density.init_args.densify_until_iter 300 \
  --model.density.init_args.screen_size_prune_px 300 \
  --model.density.init_args.fast_noise true \
  --model.density.init_args.noise_gate_eps 1e-3 \
  --model.metric.init_args.opacity_reg 0.002 \
  --model.metric.init_args.lambda_normal 0.0 \
  --model.metric.init_args.depth_loss_weight.init 0.0 \
  --trainer.max_steps 400 \
  -n stripgrad_check 2>&1 | tee logs/stripgrad_check.log | grep -E "K-strip|absgrad|Error|Traceback"
echo "--- 判準檢查 ---"
grep -E "\[K-strip\]" logs/stripgrad_check.log || echo "❌ 沒印出 [K-strip] —— 加總沒觸發，或條帶沒走到"
