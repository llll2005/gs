#!/bin/bash
# ★ 2026-11-09 牆面／樓頂清晰度（tools/probe/wall_blur_1108.py）的 lab 版：現行配方全場景 −0.33 dB 是牆面與樓頂紋理被抹掉（b1 −0.88、b4 −0.68），
#   拆解臂要直接量「細節能量比」。第一個模型自動放 lab 的 CityGS（../cityGS_origin，像素分類用）。
# 用法（佇列）：[solo] bash scripts/lab/wall_blur_lab.sh <塊> <標籤> 名稱=跑次 ...（跑次帶臂名）；輸出 logs/wall_blur_<標籤>.log
cd "$(dirname "$0")/../.." || exit 1
B=${1:?塊}; TAG=${2:?標籤}; shift 2
CGK=$(ls ../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/blocks/block_$B/checkpoints/*step=60000.ckpt 2>/dev/null | head -1)
[ -n "$CGK" ] || { echo "⛔ 找不到 CityGS b$B 參考 ckpt"; exit 2; }
M=("CityGS=$CGK")
for x in "$@"; do
  c=$(ls outputs/lab/${x#*=}/blocks/block_$B/checkpoints/*step=30000.ckpt 2>/dev/null | head -1)
  if [ -n "$c" ]; then M+=("${x%%=*}=$c"); else echo "⚠ 缺 $x"; fi
done
mkdir -p logs
conda run -n gspl --no-capture-output python -u tools/probe/wall_blur_1108.py --block $B --out outputs/lab/wall_blur_$TAG --models "${M[@]}" 2>&1 \
  | grep -v 'Warn\|warn\|pkg_res\|declare_namespace' | tee logs/wall_blur_$TAG.log
