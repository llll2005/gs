#!/bin/bash
# ★★★ 量 `skip_surf_normal` 值多少（~3 分鐘）—— 與 08-26 21:22 那次 profile 直接可比
# 同 config（probe_ctrl）、同 300 步、同 N，只多加 `--model.renderer.init_args.skip_surf_normal true`。
# 對照：閘門前 0.22123 s → metric 端閘門後 **0.18206 s**（§11.35）。
# `depth_to_normal`（深度圖反投影成 3D 點 + 鄰域叉積，全幅逐像素且可微）是剩下的那塊。
# ⚠ 失敗模式是大聲的：lambda_normal>0 卻開此旗標 => gs2d_metrics 取 surf_normal 直接 KeyError。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
OUT=logs/profile_nosn_$(date +%m%d_%H%M).txt
rm -rf outputs/profile_nosn
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/proxy/probe_ctrl.yaml \
  --model.renderer.init_args.skip_surf_normal true \
  --trainer.profiler simple \
  --trainer.max_steps 300 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 300 \
  -n profile_nosn 2>&1 | tee "$OUT"
echo; echo "===== 對照 ====="
grep -a -A8 "Mean duration" "$OUT" | grep -E "run_training_batch"
echo "  閘門前 0.22123 / metric 閘門後 0.18206（§11.35）"
