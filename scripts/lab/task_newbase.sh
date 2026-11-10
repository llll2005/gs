#!/bin/bash
# 新基準上的單變數臂（使用者 11-10「v4 不能用、lr x1 可以」「lr x0.5、鄰域平面約束每 16 步都可以排」）
#   新基準＝現行配方＋位置 lr 回 x1＋拿掉視線一致性 v4（連不准往上；它要 v4 的投票）＝ 11-09 lr 倍率掃描的 g44s30ksfm10lrx100nov4_
#   （三塊畫質追平 A＋dist4、F1／地下層／時間都較好）。每臂＝新基準＋一項，b1／b4／b3 [solo]；基準跑次已存在，不重跑。
#   評估＝換塊評測＋全面向（F1、地面／地下、立面、朝向、海面、燈車、時間）；時間只在 [solo] 跑次之間比。
# 用法（佇列）：[solo] bash scripts/lab/task_newbase.sh <塊> <lr050|np16|lowres>
#               [solo] bash scripts/lab/task_newbase.sh eval <塊>
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
source scripts/lab/_cur_recipe.sh                 # CUR＝現行配方的旗標
SUF=best1maskdistrfed10b4
NB_RUN=g44s30ksfm10lrx100nov4_$SUF
CUR_RUN=g44s30ksfm10nuPF2DB2_$SUF
A_RUN=g44s30ksfm10fssh16di4_$SUF
NB=("${CUR[@]}"
    --model.gaussian.init_args.optimization.means_lr_init 6.4e-05 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 6.4e-07
    --model.density.init_args.sfm_vis false --model.density.init_args.sfm_vis_noup false)

if [ "$1" = eval ]; then
  B=${2:?塊}
  R=(g44s30ksfm10nblr050_$SUF g44s30ksfm10nbnp16_$SUF)
  [ -d outputs/lab/g44s30ksfm10nblowres_$SUF/blocks/block_$B ] && R+=(g44s30ksfm10nblowres_$SUF)
  N=(位置lrx0.5 鄰域平面每16步 前段低解析度)
  bash scripts/lab/task_full44.sh swap "$B" $A_RUN $CUR_RUN $NB_RUN "${R[@]}"
  ARGS=(新基準=$NB_RUN A+dist4=$A_RUN 現行配方=$CUR_RUN)
  for i in "${!R[@]}"; do ARGS+=("${N[$i]}=${R[$i]}"); done
  EVAL_ONLY=time,f1,ground,facade,orient,sea,smallobj bash scripts/lab/eval_arms_lab.sh "$B" newbase_b${B}_1110 "${ARGS[@]}"
  exit $?
fi

B=$1; ARM=${2:?臂}
case "$ARM" in
  # 位置 lr 再減半（3.2e-5）：11-09 掃描（有 v4）lr 越低地下層越少、立面越多；這裡看沒有 v4 時是否也成立
  lr050)  FAM=g44s30ksfm10nblr050_
          XA=(--model.gaussian.init_args.optimization.means_lr_init 3.2e-05 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 3.2e-07) ;;
  # 鄰域平面約束每 16 步（同 11-08 lab b3 的 nuPF2DNP16 臂：+0.75% 時間、地下 −0.19、朝向 +7～8 點）
  np16)   FAM=g44s30ksfm10nbnp16_
          XA=(--model.metric.init_args.surf_reg 0.01 --model.metric.init_args.surf_orient 1.0 --model.metric.init_args.surf_interval 16
              --model.metric.init_args.surf_every 400) ;;
  # 前段低解析度（< 7.5k 1/2、< 15k 3/4）＋px_floor（11-09 萃取臂在現行配方上快 12～15%；尚未排，等使用者說排）
  lowres) FAM=g44s30ksfm10nblowres_
          XA=(--model.train_res_schedule "7500:0.5,15000:0.75" --model.metric.init_args.lambda_dist_lowres 0) ;;
  *) echo "⛔ 臂只有 lr050／np16／lowres"; exit 2 ;;
esac
CAP=$(python3 -c "import json;print(int(round(1e6 * json.load(open('configs/cap_alloc_sfm_g44.json'))['weights']['$B'])))")
DUP=${CITYGS_DUP44:-2}
CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=30000 CITYGS_CAP=$CAP CITYGS_DUP44=$DUP CITYGS_FAMILY=$FAM \
  bash scripts/lab/task_cmp.sh "$B" $SUF "${NB[@]}" "${XA[@]}"
