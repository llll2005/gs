#!/bin/bash
# ★★★★★ 判決：0.42 it/s 是我加的 image_uint8/skip_unused_depth 造成的，還是卡的問題？
# 09-14 基準 init_sfmfill b6 = 2.75 it/s；現在同工作量只有 0.44。顆數已排除（fill01 起始點數與基準相同、一樣慢）。
# 獨佔、同一個 PLY（fill01 b6）、各 500 步，只比 it/s。
set -u
cd "$(dirname "$0")/../.." 2>/dev/null || cd /workspace/data/hdd/11213/gs || exit 1
export CITYGS_VRAM_CAP_GB=5.66
P="sfmfill_sweep/fill01/block_6.ply"
[ -f "data/matrix_city/aerial/train/block_all/$P" ] || { echo "A ⛔ 缺 $P"; exit 2; }
run1 () {
  local name=$1 u=$2 d=$3
  [ -d "outputs/$name" ] && mv "outputs/$name" "outputs/$name.aborted_$(date +%m%d_%H%M%S)"
  echo "A ##### $name  image_uint8=$u skip_unused_depth=$d  $(date +%H:%M:%S) #####"
  local t0=$(date +%s)
  conda run -n gspl --no-capture-output python -u main.py fit \
    --config configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml --data.parser.block_id 6 \
    --model.initialize_from null --data.parser.points_from ply --data.parser.ply_file "$P" \
    --trainer.max_steps 500 \
    --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 20000 \
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
  local n=$(grep -aoE "init顆數 [0-9,]+" "outputs/$name/blocks/block_6/train_status.txt" 2>/dev/null | tail -1)
  echo "A [結果] $name rc=$rc 牆鐘 $((t1-t0))s => 500 步平均 $(python3 -c "print('%.2f'%(500/max($t1-$t0,1)))") it/s  $n"
  grep -aE "\[depth\]|Traceback|Error" "logs/$name.log" | head -2 | sed "s/^/A /"
}
run1 abtest_float false false
run1 abtest_uint8 true  true
echo "A ── 卡的狀態（空載）──"
nvidia-smi --query-gpu=power.draw,pstate,memory.used --format=csv,noheader | sed "s/^/A /"
