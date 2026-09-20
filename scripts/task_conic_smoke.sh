#!/bin/bash
# ★★★★★ 精確圓錐外接盒（EXACT_CONIC_AABB）功能煙霧測試：本機 b6 短跑 1,200 步 x 2（開/關）
#   目的只有一個：確認「能正常跑」—— 不當機、loss 會降、起始 trim/週期 trim/增生事件都照常觸發。
#   1,200 步涵蓋 step 1 的起始 trim、step 1,000 的週期 trim 與多次增生（與 task_bitcheck_data.sh 同理由）。
#   ⚠ 這不是品質實驗：模型是用線性化盒訓練慣例來的，短跑 PSNR 不可拿來判斷盒子好壞。
set -u
cd "$(dirname "$0")/.." || exit 1
unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
STEPS=${STEPS:-1200}
AUX=submodules/diff-surfel-rasterization-trim-pp/cuda_rasterizer/auxiliary.h
R=logs/conic_smoke_$(date +%m%d_%H%M).log
COMMON=(--config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6
  --model.initialize_from null --trainer.max_steps "$STEPS"
  --data.image_uint8 true --data.skip_unused_depth true
  --model.density.init_args.cap_max 2600000 --model.density.init_args.densify_until_iter 30000
  --model.density.init_args.absgrad_densify 2.0 --model.density.init_args.fast_noise true
  --model.density.init_args.noise_gate_eps 0.001 --model.metric.init_args.opacity_reg 0.002
  --model.metric.init_args.lambda_normal 0.0 --model.metric.init_args.depth_loss_weight.init 0.0)

rebuild () {   # rebuild <0|1>
  sed -i "s|^#define EXACT_CONIC_AABB .*$|#define EXACT_CONIC_AABB $1|" "$AUX"
  rm -rf submodules/diff-surfel-rasterization-trim-pp/build
  conda run -n gspl env CUDA_HOME="$HOME/miniconda3/envs/gspl" CUDA_PATH="$HOME/miniconda3/envs/gspl" \
    pip install --no-build-isolation --no-deps --force-reinstall submodules/diff-surfel-rasterization-trim-pp \
    > logs/conic_build_$1.log 2>&1 || { echo "[build] ❌ EXACT_CONIC_AABB=$1 編譯失敗，見 logs/conic_build_$1.log"; return 1; }
  echo "[build] ✅ EXACT_CONIC_AABB=$1"
}
run1 () {   # run1 <跑次名>
  local name=$1
  [ -d "outputs/$name" ] && mv "outputs/$name" "outputs/$name.old_$(date +%m%d_%H%M%S)"
  local t0=$SECONDS
  conda run -n gspl --no-capture-output python -u main.py fit "${COMMON[@]}" -n "$name" > "logs/$name.log" 2>&1
  local rc=$?
  echo "##### $name  rc=$rc  牆鐘 $((SECONDS-t0)) 秒 #####"
  grep -E "Traceback|Error|nan|NaN" "logs/$name.log" | head -3
  grep -E "Trimming|\[churn\]|首次觸發" "logs/$name.log" | head -4
  grep -E "^val step|best_val_psnr" "logs/$name.log" | tail -3
  return $rc
}
{ bad=0
  echo "════ EXACT_CONIC_AABB = 1（精確圓錐不對稱盒）════"
  rebuild 1 && run1 conic_on || bad=1
  echo
  echo "════ EXACT_CONIC_AABB = 0（線性化，對照）════"
  rebuild 0 && run1 conic_off || bad=1
  echo
  echo "════ 台帳（牆鐘與顆數看這裡，it/s 是瞬時值不可比）════"
  grep -E "conic_on|conic_off" logs/quad_progress.log | tail -6
  echo "⚠ 結束時 EXACT_CONIC_AABB 已還原為 0（預設關）"
  exit $bad
} 2>&1 | tee "$R"
