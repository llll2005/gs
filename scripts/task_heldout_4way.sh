#!/bin/bash
# ★★★★★★★ 最高優先：val⊂train 的排序 vs **真 held-out** 的排序（使用者 2026-09-12 目視發現）
#
# 使用者在 web viewer 看 `normal_b12`（vanilla 2DGS、零機制、**1500 步**）：
#   「長得比我們跑完 60000 步還要好！正常區顏色重建成功且完整 只差在細節」
# 而幾何稽核完全同向：
#   sfminit2 z0.33/懸空37.5% ／ normal z0.37/41.4% ／ speed3 z0.59/54.4% ／ agd2 z0.60/54.9%
# 但 val⊂train PSNR 的排序**幾乎完全相反**（speed3 26.70 > agd2 26.56 > sfminit2 > normal 21.50）。
#
# ⚠ 專案所有分數都是 `split_mode: reconstruction`（val ⊂ train），**從未量過泛化**
#   （記憶 northstar_merge_eval）。使用者看的自由視角正是沒量過的那一半。
# ⇒ 這支腳本補上那一半：官方 741 幀測試集，**姿態確定不在訓練集裡**
#   （最近的訓練姿態約 18.3 場景單位、0.00% 落在 1e-3 內）。
#
# 讀法（工具 docstring 自述的限制，不可忽略）：
#   · 單塊模型在測試視角會看到塊邊界外的內容，那些內容根本不存在 => 這是**下界**
#   · 可以用來**互相比較我方跑次**，不可直接對published 數字
#   · offset 由工具自己校準（名稱 0 起算 vs 1 起算差一幀＝曾經假造出 8 dB 的發現）；
#     若各 offset 的差距都在噪音內，就是沒建立配對，數字不可信 —— 要看它印的 margin
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
for R in normal_b12 sfminit2_b12 speed3_b12 agd2_b12; do
  CK=$(ls -t outputs/$R/blocks/block_12/checkpoints/*.ckpt 2>/dev/null | head -1)
  if [ -z "$CK" ]; then echo "⚠ $R 沒有 ckpt，跳過"; continue; fi
  echo "================ $R  ($(basename "$CK")) ================"
  conda run -n gspl python tools/eval_official_test.py --ckpt "$CK" --block 12 2>&1 \
    | grep -vE "pkg_resources|declare"
done
