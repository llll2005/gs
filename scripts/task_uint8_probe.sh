#!/bin/bash
# ★★★★★ uint8 在**高顆數**下會不會拖慢訓練（2026-09-19）
# 起因：lab 的 sfmfill 變體只有 0.42 it/s，而同配方的 init_sfmfill（09-14）是 2.75 it/s（慢 6.5 倍）。
#   GPU 0%／磁碟 0／CPU 單核 100% ⇒ 時間全在 CPU。config 只差 image_uint8＋skip_unused_depth。
#   ⚠ 本機 09-18 的驗證是在 N=0.57M 量的（4.35 vs 4.35 it/s，沒差）=> 這次補「高顆數」那一格。
# 做法：從本機 cs_refrep 的 ckpt（N=1.85M）起跑 400 步，uint8 關／開各一次，只比 it/s。
set -u
cd "$(dirname "$0")/.." || exit 1
unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
CK=outputs/cs_refrep/blocks/block_6/checkpoints/epoch=40-step=21920.ckpt
[ -f "$CK" ] || { echo "⛔ 找不到 $CK"; exit 2; }
R=logs/uint8_probe_$(date +%m%d_%H%M).log
run1 () {
  local name=$1 u=$2 d=$3
  [ -d "outputs/$name" ] && mv "outputs/$name" "outputs/$name.aborted_$(date +%m%d_%H%M%S)"
  echo "##### $name  image_uint8=$u skip_unused_depth=$d  $(date) #####"
  local t0=$(date +%s)
  conda run -n gspl --no-capture-output python -u main.py fit \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6 \
    --model.initialize_from "$CK" --trainer.max_steps 400 \
    --model.density.init_args.cap_max 2600000 \
    --model.density.init_args.densify_until_iter 10000 \
    --model.density.init_args.absgrad_densify 0.0 \
    --model.density.init_args.fast_noise false \
    --model.metric.init_args.opacity_reg 0.002 \
    --model.metric.init_args.lambda_normal 0.0 \
    --model.metric.init_args.depth_loss_weight.init 0.0 \
    --data.image_uint8 "$u" --data.skip_unused_depth "$d" \
    -n "$name" > "logs/$name.log" 2>&1
  local rc=$? t1=$(date +%s)
  local st=$(grep -aoE "step [0-9,]+/400[^│]*│[^│]*│[^│]*│[^│]*│ *[0-9.]+ it/s" "outputs/$name/blocks/block_6/train_status.txt" 2>/dev/null | tail -1)
  echo "[結果] $name rc=$rc 牆鐘 $((t1-t0))s  ${st:-（讀不到進度行）}"
  grep -aE "\[depth\]|Traceback|Error" "logs/$name.log" | head -2
}
{ run1 uint8probe_off false false
  run1 uint8probe_on  true  true ; } 2>&1 | grep -vE "pkg_resources|declare_namespace" | tee "$R"
echo "報告：$R"
