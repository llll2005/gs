#!/bin/bash
# ★★★★★ 碎片是相機數造成的嗎 —— 同一個 ckpt、同一個上限，只掃相機數（純診斷，不訓練）
#
# 起因（2026-09-13）：speed3 在 5.66 GiB 上限下 OOM 了五次，而使用者問「為什麼碎片這麼嚴重，
# 近期跑次好像都沒這樣」。量台帳後發現：**峰值 gap 沒有變嚴重**（舊 12.3% -> 新 14.2%，n=12/5），
# OOM 也不是新的（09-07/08/12 已四次）。真正的對照訊號是這一組：
#   lab  speed3/b6   N=2.34M  保留 4.60G / 實佔 4.49G  gap 0.11G   548 台相機
#   本機 speed3/b13  N=2.58M  保留 5.22G / 實佔 4.48G  gap 0.74G   667 台相機
# **實佔幾乎相同而保留差 0.62G** => 相機數是候選來源。
# 機制假說：trim pass 每輪走全部相機，而每台的 binning 緩衝大小隨視角變動
#   => 相機愈多，進出配置器的「不同尺寸大塊」愈多 = 碎片的典型來源。
#
# 這支腳本把相機數當唯一變數（同 ckpt => N 固定），並同時對照 max_split_size_mb 的效果。
# ⚠ 一定要套 CITYGS_VRAM_CAP_GB：lab 是 24 GiB，沒有上限就永遠有餘裕、retries 恆為 0。
#   _common.sh 已經幫我們鎖 5.66 並開 max_split_size_mb:128。
# ⚠ vram_gap.py 用 glob 找 ckpt，沒有 --block 參數 => 靠「只有 b6 有 step=29999」來唯一指定。
#   若之後 b13 也存了 29999，這支要改成指定路徑。
source "$(dirname "$0")/_common.sh"
RUN=${1:-lab/speed3}; STEP=${2:-29999}
CAMS=${CAMS:-"64 137 274 548"}
for AC in "max_split_size_mb:128" ""; do
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "  PYTORCH_CUDA_ALLOC_CONF=[${AC:-未設}]   上限=${CITYGS_VRAM_CAP_GB:-未設} GiB"
  echo "════════════════════════════════════════════════════════════════"
  for NC in $CAMS; do
    echo "──────── trim-cams=$NC ────────"
    if [ -n "$AC" ]; then
      conda run -n gspl env PYTORCH_CUDA_ALLOC_CONF="$AC" \
        python tools/vram_gap.py --run "$RUN" --step "$STEP" --trim-cams "$NC" --frag 4 \
        2>&1 | grep -vE "pkg_resources|declare|down sample|loading|appearance|dataparser|found "
    else
      conda run -n gspl env -u PYTORCH_CUDA_ALLOC_CONF \
        python tools/vram_gap.py --run "$RUN" --step "$STEP" --trim-cams "$NC" --frag 4 \
        2>&1 | grep -vE "pkg_resources|declare|down sample|loading|appearance|dataparser|found "
    fi
  done
done
echo
echo "★ 判讀：retries 隨相機數上升 => 相機數確實在製造碎片；"
echo "  兩個 alloc 設定的差 => max_split_size_mb:128 到底買到多少（記憶說 +0.6GB，從沒對照過）"
