#!/bin/bash
# 從 57 臂萃取的四個修法（使用者 11-09「你的想法都不錯，都排」），每臂＝現行配方＋一項改動，b1／b4／b3 [solo]
#   基準＝11-09 現行配方全場景裡的同塊模型（g44s30ksfm10nuPF2DB2_，不重跑）；另列 A＋dist4、lr x1（b1／b4＝拆解 looLR、b3＝lr 掃描 lrx100）。
#   評估＝換塊評測＋全面向（F1、地面／地下、立面、朝向、海面、燈車、時間）；時間只在 [solo] 跑次之間比。
# 用法（佇列）：[solo] bash scripts/lab/task_extract.sh <塊> <ng|lowres|bhnn|vgu>
#               [solo] bash scripts/lab/task_extract.sh eval <塊>
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
source scripts/lab/_cur_recipe.sh                 # CUR＝現行配方的旗標
SUF=best1maskdistrfed10b4
CUR_RUN=g44s30ksfm10nuPF2DB2_$SUF
A_RUN=g44s30ksfm10fssh16di4_$SUF

if [ "$1" = eval ]; then
  B=${2:?塊}
  case $B in 1|4) LR1=g44s30ksfm10looLR_$SUF ;; *) LR1=g44s30ksfm10lrx100_$SUF ;; esac
  R=(g44s30ksfm10ng2_$SUF g44s30ksfm10lowres_$SUF g44s30ksfm10bhnn_$SUF g44s30ksfm10vgu_$SUF)
  bash scripts/lab/task_full44.sh swap "$B" $A_RUN $CUR_RUN $LR1 "${R[@]}"
  EVAL_ONLY=time,f1,ground,facade,orient,sea,smallobj bash scripts/lab/eval_arms_lab.sh "$B" extract_b${B}_1109 \
    現行配方=$CUR_RUN A+dist4=$A_RUN lrx1=$LR1 法向x2=${R[0]} 前段低解析度=${R[1]} BH間距倍數=${R[2]} 熵閘門且地下=${R[3]}
  exit $?
fi

B=$1; ARM=${2:?臂}
case "$ARM" in
  # a 位置 lr 回 x1，只把法向那一半的更新放大 2 倍（GaussianSplatting means_normal_gain；切向＝紋理不再被抹）
  ng)     FAM=g44s30ksfm10ng2_
          XA=(--model.gaussian.init_args.optimization.means_lr_init 6.4e-5 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 6.4e-7
              --model.means_normal_gain 2.0) ;;
  # b 前段低解析度（< 7.5k 1/2、< 15k 3/4，同 36 的 C 臂）＋px_floor（C 失敗的原因＝點精靈 67.9%，當時還沒有 px_floor）
  lowres) FAM=g44s30ksfm10lowres_
          XA=(--model.train_res_schedule "7500:0.5,15000:0.75" --model.metric.init_args.lambda_dist_lowres 0) ;;
  # c BH（地下票落後上限 10x 容差、只套離地夠高的顆粒）改成 16 倍 SfM 最近鄰間距（不帶場景單位；局部地面格 6 倍間距）
  bhnn)   FAM=g44s30ksfm10bhnn_
          XA=(--model.density.init_args.sfm_vis_behind_max 10 --model.density.init_args.sfm_vis_behind_max_above_nn 16) ;;
  # d 熵閘門 0.002 只對 v4 也判地下的顆粒（兩個弱訊號取交集）
  vgu)    FAM=g44s30ksfm10vgu_
          XA=(--model.metric.init_args.vis_gate_tau 0.002 --model.renderer.init_args.vis_snapshot true --model.metric.init_args.vis_gate_need_under true) ;;
  *) echo "⛔ 臂只有 ng／lowres／bhnn／vgu"; exit 2 ;;
esac
CAP=$(python3 -c "import json;print(int(round(1e6 * json.load(open('configs/cap_alloc_sfm_g44.json'))['weights']['$B'])))")
DUP=${CITYGS_DUP44:-2}
CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=30000 CITYGS_CAP=$CAP CITYGS_DUP44=$DUP CITYGS_FAMILY=$FAM \
  bash scripts/lab/task_cmp.sh "$B" $SUF "${CUR[@]}" "${XA[@]}"
