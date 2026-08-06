#!/bin/bash
# Queued single-block experiments (順位 3-5 in 待辦與半成品清單.md), run serially
# after the current GPU queue frees. Each waits for GPU. Kill this driver to stop.
set -u
cd /home/LnoArch/Projects/專題/CityGaussian || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROG=logs/quad_progress.log
SB=configs/mcmc_2dgs_sb_60k_aggr17_aerial.yaml
SH3=configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml
PLY_DIR=data/matrix_city/aerial/train/block_all/depth_init

# Wait until the GPU is actually free. NOTE (2026-07-25 bugfix): the old check
# `pgrep -f "python -u main.py fit"` never matched, because conda run rewrites the
# cmdline — so every queued run launched onto a busy GPU and died silently at CUDA
# init. Query the driver instead: it reports real per-process VRAM regardless of how
# the process was spawned.
wait_gpu () {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$used" ] && used=0
    [ "$used" -lt 1500 ] && break     # <1.5GB = no training resident
    sleep 300
  done
  sleep 30
}
# Sanity gate: a run that dies at init leaves no results.txt and a short log. Surface
# that instead of silently reporting an empty PSNR (2026-07-25: a whole queue was lost
# to silent init failures — wait_gpu pattern mismatch, then a RecursionError).
report () {  # name, block, anchor-note
  local psnr; psnr=$(grep -oE 'val/psnr: [0-9.]+' "outputs/$1/blocks/block_$2/results.txt" 2>/dev/null | head -1)
  if [ -z "$psnr" ]; then
    local errline; errline=$(grep -oE "(Error|error|Exception|OutOfMemory)[^\"]{0,80}" "logs/$1.log" 2>/dev/null | tail -1)
    echo "[queue] !! $1 FAILED (no results.txt). last error: ${errline:-<none, check logs/$1.log>} $(date +%F_%T)" >> $PROG
    return
  fi
  echo "[queue] $1 done $psnr ($3) $(date +%F_%T)" >> $PROG
}

# --- 2.5 ★凝結實驗: min_opacity 退火 (30k-42k 抬死亡線, 強制清霧/提早凝結) ---
# 動機: 霧(97% opacity~0.02)廢掉了 rasterizer 的 early ray termination → 慢+吃記憶體.
# 抬死亡線 0.005→0.05 (趁 densify relocation 還開, 輸家變探針不是殭屍) → 減霧 →
# early termination 救回. 判讀 vs A' 22.12: PSNR 平? harvest期 it/s 升(=速度紅利)? VRAM 降?
# ★若贏, 後續實驗(尤其 25 塊北極星)全吃增益 → 這是插在前面的理由.
wait_gpu; echo "[queue] condense-anneal (死亡線30k-42k→0.05) start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sb_b12_cap1m_condense
conda run -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY_DIR/block_12.ply --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 --model.density.init_args.cap_max 1000000 \
  --model.density.init_args.min_opacity_final 0.05 \
  --model.density.init_args.min_opacity_anneal_start_iter 30000 \
  --model.density.init_args.min_opacity_anneal_end_iter 42000 \
  -n mcmc_sb_b12_cap1m_condense > logs/mcmc_sb_b12_cap1m_condense.log 2>&1
rc=$?
psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_sb_b12_cap1m_condense/blocks/block_12/results.txt 2>/dev/null | head -1)
its=$(grep -oE '[0-9.]+ it/s' logs/mcmc_sb_b12_cap1m_condense.log 2>/dev/null | tail -3 | head -1)
echo "[queue] condense-anneal done rc=$rc $psnr harvest-speed=$its (vs A' 22.12; it/s 升=early-term 救回) $(date +%F_%T)" >> $PROG

# --- 2.7 ★★v/c 判準: 價值從 opacity → 多視角渲染貢獻 (理論收尾) ---
# 判別式兩側都 render-grounded: v=多視角Σ(T·α)/覆蓋px, c=螢幕足跡. 能抓 opacity 抓不到的
# 怪物(o=0.028 逃過死亡線但足跡爆大→v/c 崩) 且同時殺霧(binning 82%). 訊號來自現成
# record_transmittance, 免改 CUDA. 判讀 vs A' 22.12: PSNR 平? 怪物/霧自動被殺?
wait_gpu; echo "[queue] vpc-criterion (v=多視角貢獻, bottom5%) start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sb_b12_cap1m_vpc
conda run -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY_DIR/block_12.ply --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 --model.density.init_args.cap_max 1000000 \
  --model.density.init_args.vpc_prune_frac 0.05 \
  --model.density.init_args.vpc_interval 1500 \
  -n mcmc_sb_b12_cap1m_vpc > logs/mcmc_sb_b12_cap1m_vpc.log 2>&1
rc=$?
psnr=$(grep -oE 'val/psnr: [0-9.]+' outputs/mcmc_sb_b12_cap1m_vpc/blocks/block_12/results.txt 2>/dev/null | head -1)
nvpc=$(grep -c "value-per-cost" logs/mcmc_sb_b12_cap1m_vpc.log 2>/dev/null)
echo "[queue] vpc-criterion done rc=$rc $psnr (vpc事件 $nvpc 次; vs A' 22.12) $(date +%F_%T)" >> $PROG

# --- 3. 壓縮排程: densify 30k 早停 (vs A' 標準 42k停/22.12) ---
wait_gpu; echo "[queue] compress-sched (densify停30k) start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sb_b12_cap1m_densify30k
conda run -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY_DIR/block_12.ply --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 --model.density.init_args.cap_max 1000000 \
  --model.density.init_args.densify_until_iter 30000 \
  -n mcmc_sb_b12_cap1m_densify30k > logs/mcmc_sb_b12_cap1m_densify30k.log 2>&1
report mcmc_sb_b12_cap1m_densify30k 12 "vs A' 42k停 22.12,測早停省時無損?"

# --- 4. b8 count ablation + ★閉環: cap 從公式表自動讀 (非手設) ---
# 手設低 cap 1M 當對照; 公式 cap 從 block_caps.csv 讀 (b8=2.6M), 配動態 K → 真正的
# 「cap 自動算 + K 自動調」閉環. b8 是乖塊(τ=8.7 無怪物), 動態 K 風險低.
AUTOCAP=$(awk -F, '$1==8 {print $3}' 紀錄/block_caps.csv | tail -1)
echo "[queue] b8 公式 cap 讀到 = $AUTOCAP" >> $PROG
# 4a. 手設低 cap 對照
wait_gpu; echo "[queue] b8-manual-1m start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sb_b8_manual1m
conda run -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY_DIR/block_8.ply --data.parser.block_id 8 \
  --model.density.init_args.cap_max 1000000 \
  -n mcmc_sb_b8_manual1m > logs/mcmc_sb_b8_manual1m.log 2>&1
report mcmc_sb_b8_manual1m 8 "b8 手設 cap 1M 對照"
# 4b. ★閉環: 公式 cap (讀 CSV) + 動態 K
wait_gpu; echo "[queue] b8-AUTOcap-$AUTOCAP +dynK start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sb_b8_autocap
conda run -n gspl python -u main.py fit --config $SB \
  --model.initialize_from $PLY_DIR/block_8.ply --data.parser.block_id 8 \
  --model.density.init_args.cap_max "$AUTOCAP" \
  --model.dynamic_strips true --model.strip_max 8 \
  -n mcmc_sb_b8_autocap > logs/mcmc_sb_b8_autocap.log 2>&1
report mcmc_sb_b8_autocap 8 "★b8 公式cap=$AUTOCAP+動態K (閉環), vs 手設1M"

# --- 5. SH3@1.5M count arm (b12, SH3 count 斜率) ---
wait_gpu; echo "[queue] sh3-b12-1.5M start $(date +%F_%T)" >> $PROG
rm -rf outputs/mcmc_sh3_b12_cap1p5m
conda run -n gspl python -u main.py fit --config $SH3 \
  --model.initialize_from $PLY_DIR/block_12.ply --data.parser.block_id 12 \
  --model.density.init_args.screen_size_prune_px 300 --model.density.init_args.cap_max 1500000 \
  -n mcmc_sh3_b12_cap1p5m > logs/mcmc_sh3_b12_cap1p5m.log 2>&1
report mcmc_sh3_b12_cap1p5m 12 "SH3@1.5M vs SH3@1.0M 22.45"
echo "[queue] ALL DONE $(date +%F_%T)" >> $PROG
