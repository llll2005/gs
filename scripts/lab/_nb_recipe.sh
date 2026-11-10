#!/bin/bash
# 新基準的旗標陣列 NB（11-10：現行配方＋位置 lr 回 x1＋拿掉視線一致性 v4 連同「不准往上」）＝ 11-09 lr 倍率掃描的 g44s30ksfm10lrx100nov4_。
# task_tune25.sh source 進來；後面的覆寫會蓋掉前面的值。
source "$(dirname "${BASH_SOURCE[0]}")/_cur_recipe.sh"
NB=("${CUR[@]}"
    --model.gaussian.init_args.optimization.means_lr_init 6.4e-05 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 6.4e-07
    --model.density.init_args.sfm_vis false --model.density.init_args.sfm_vis_noup false)
NB_RUN=g44s30ksfm10lrx100nov4_best1maskdistrfed10b4    # 新基準 30k（b1／b4／b3 已有；換塊 +0.092／+0.026／+0.034，純訓練 12.5／13.5／12.0 分）
