#!/bin/bash
# ★★★★★★ 失敗區在「零機制」下也失敗嗎 —— 1500 步三臂的逐 tile 判定（使用者 2026-09-12）
#
# normal_b12（vanilla 2DGS、SfM-init、零機制）已完賽 1500 步。PSNR 21.50 比四臂都高，
# 但 **PSNR 答不了這個問題** —— 失敗區只佔畫面一部分，全幅 PSNR 會被正常區稀釋
# （eval_protocol 記過：逐塊評官方 test 卡在 ~15dB 與好壞無關，同一個稀釋效應）。
# 要答的是「糊掉 tile 的位置與比例」，所以需要存出 val 影像再逐 tile 比。
#
# 三臂都存圖（各 36 張，同一組 val 相機）：
#   normal_b12           vanilla 2DGS + SfM-init      零機制
#   probe_armB_sfm       我方配方 + SfM-init          有機制、同起點
#   probe_armA_baseline  我方配方 + depth-init        有機制、原起點
# 然後跑 tools/blur_persistence.py 取三者的糊掉遮罩交集。
#   交集 ~100% => 失敗區與「有沒有我方機制」無關 => 病灶在輸入，整個殼層線要換方向
#   交集低     => 是某個機制造成的 => 逐一加回去二分定位
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
RC=0
for R in normal_b12 probe_armB_sfm probe_armA_baseline; do
  C=$(ls outputs/$R/blocks/block_12/lightning_logs/version_*/config.yaml 2>/dev/null | tail -1)
  if [ -z "$C" ]; then echo "⚠ 找不到 $R 的 config，跳過"; continue; fi
  echo "===== 存 val 影像：$R ====="
  conda run -n gspl --no-capture-output python -u main.py test --config "$C" --save_val || RC=$?
done
[ "$RC" -ne 0 ] && echo "⚠ 至少一個 test 失敗 rc=$RC（下面的分析可能不完整）"
echo "===== 逐 tile 糊掉遮罩交集 ====="
conda run -n gspl python tools/blur_persistence.py normal_b12 probe_armB_sfm probe_armA_baseline --blk 12 2>&1 | grep -vE "pkg_resources|declare"
exit "$RC"
