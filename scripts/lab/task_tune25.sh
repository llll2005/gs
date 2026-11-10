#!/bin/bash
# 自主調參（使用者 11-10「開始自主實驗，定時抓 lab 看進度；用 25k 左右，自己調整比例跟一些參數，找到最佳解後排一次 4x4」）
#   底＝新基準（現行配方＋位置 lr x1＋拿掉 v4；_nb_recipe.sh），全長預設 25k。臂＝用 + 串起來的元件（例：g40+lowres、coarse+g40）。
#   對照＝新基準 30k（不重跑）與 A＋dist4；時間只在 [solo] 跑次之間比（coarse 起點另加 coarse 本身攤到每塊約 1 分）。
#   元件：
#     s<NN>    全長 NN k 步（預設 s25）
#     g<NN>    增生期＝全長的 NN%（預設 g50＝現行比例）
#     lowres   前段低解析度：< 全長 1/4 用 1/2、< 1/2 用 3/4，低解析度步不算 dist（11-09 在現行配方上快 12～15%）
#     coarse   起點＝輕量 coarse 裁塊（outputs/lab/coarse_g44r48sh0_crop；task_coarse.sh）
#     np16     鄰域平面約束每 16 步
#     lr050    位置 lr 再減半（3.2e-5）
#     mvu      v4 判準搬進合併事件（＋BH 用 SfM 間距倍數）
#     mvugrow  同 mvu 但收割期不投票不刪（只在增生期搬）
#     mvugnd   同 mvu 但只動貼地／地面以下的顆粒（離地高的不動）
#   合併窗口跟著全長縮放（3k ～ 全長 x 0.9；現行配方寫死 27k＝30k 的 90%）；二元熵切雙向（全長／3）與位置 lr 排程由 task_cmp 跟著全長縮放。
# 用法：[solo] bash scripts/lab/task_tune25.sh <塊> <臂>
#       [solo] bash scripts/lab/task_tune25.sh eval <塊> <標籤> <臂>...   （只比跑完的臂；對照新基準 30k、A＋dist4）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
source scripts/lab/_nb_recipe.sh
SUF=best1maskdistrfed10b4
A_RUN=g44s30ksfm10fssh16di4_$SUF

# 臂字串 => 全域 S（步數）、G（增生 %）、XT（其餘元件）、FAM（家族）、LBL（評估用名稱）
parse() {
  S=25; G=50; XT=()
  local t
  for t in ${1//+/ }; do
    if [[ $t =~ ^s([0-9]+)g([0-9]+)$ ]]; then S=${BASH_REMATCH[1]}; G=${BASH_REMATCH[2]}       # 3x3 的格子寫法 s20g40
    elif [[ $t =~ ^s([0-9]+)$ ]]; then S=${BASH_REMATCH[1]}
    elif [[ $t =~ ^g([0-9]+)$ ]]; then G=${BASH_REMATCH[1]}
    else
      case "$t" in
        lowres|coarse|np16|lr050|mvu|mvugrow|mvugnd) XT+=("$t") ;;
        *) echo "⛔ 不認得的元件 $t（s<NN>／g<NN>／s<NN>g<NN>／lowres／coarse／np16／lr050／mvu／mvugrow／mvugnd）"; return 2 ;;
      esac
    fi
  done
  FAM="g44tn_s${S}g${G}$(IFS=; echo "${XT[*]:-}")_"
  LBL="${S}k增生${G}"
  for t in "${XT[@]:-}"; do
    case "$t" in lowres) LBL+="低解析度" ;; coarse) LBL+="coarse起點" ;; np16) LBL+="鄰域平面" ;; lr050) LBL+="lr減半" ;; mvu) LBL+="合併地下" ;;
      mvugrow) LBL+="合併地下只增生期" ;; mvugnd) LBL+="合併地下只貼地" ;; esac
  done
}

if [ "$1" = eval ]; then
  B=${2:?塊}; TAG=${3:?標籤}; shift 3
  # TUNE_BASE／TUNE_BASE_LBL＝評估的基準跑次與名稱（預設新基準 30k；11-10 3x3 改建在新基準＋鄰域平面上 => 基準換成 30k 的鄰域平面臂）
  BASE=${TUNE_BASE:-$NB_RUN}; BLBL=${TUNE_BASE_LBL:-新基準30k}
  R=(); ARGS=("$BLBL=$BASE"); SW=($A_RUN $BASE)
  [ "$BASE" = "$NB_RUN" ] || { ARGS+=(新基準30k=$NB_RUN); SW+=($NB_RUN); }
  ARGS+=(A+dist4=$A_RUN)
  # TUNE_EXTRA="名稱=跑次 ..."：另外帶幾個現成跑次一起比（例：合併地下 v1＝g44s30ksfm10nbmvu_best1maskdistrfed10b4）
  for kv in ${TUNE_EXTRA:-}; do
    [ -d "outputs/lab/${kv#*=}/blocks/block_$B" ] && { ARGS+=("$kv"); SW+=("${kv#*=}"); } || echo "⚠ 缺 $kv"
  done
  for a in "$@"; do
    parse "$a" || exit 2
    r=$FAM$SUF
    if [ -d outputs/lab/$r/blocks/block_$B ]; then R+=($r); ARGS+=("$LBL=$r"); else echo "⚠ 缺 $a（$r）"; fi
  done
  bash scripts/lab/task_full44.sh swap "$B" "${SW[@]}" "${R[@]}"
  EVAL_ONLY=time,f1,ground,facade,orient,sea,smallobj bash scripts/lab/eval_arms_lab.sh "$B" tune_${TAG}_b${B} "${ARGS[@]}"
  exit $?
fi

B=$1; ARM=${2:?臂}
parse "$ARM" || exit 2
ST=$((S * 1000))
XA=(--model.density.init_args.densify_until_iter $((ST * G / 100))
    --model.density.init_args.merge_until_iter $((ST * 9 / 10)))
ENVX=()
for t in "${XT[@]:-}"; do
  case "$t" in
    lowres) XA+=(--model.train_res_schedule "$((ST / 4)):0.5,$((ST / 2)):0.75" --model.metric.init_args.lambda_dist_lowres 0) ;;
    coarse) CK=outputs/lab/coarse_g44r48sh0_crop/blocks/block_$B/checkpoints/coarse_crop.ckpt
            [ -f "$CK" ] || { echo "⛔ 缺 $CK（task_coarse.sh crop）"; exit 2; }
            XA+=(--model.initialize_from "$CK" --model.overwrite_config false); ENVX+=(CITYGS_KEEP_CFG_RENDERER=1) ;;
    np16)   XA+=(--model.metric.init_args.surf_reg 0.01 --model.metric.init_args.surf_orient 1.0 --model.metric.init_args.surf_interval 16
                 --model.metric.init_args.surf_every 400) ;;
    lr050)  XA+=(--model.gaussian.init_args.optimization.means_lr_init 3.2e-05 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 3.2e-07) ;;
    mvu|mvugrow|mvugnd)
            XA+=(--model.density.init_args.merge_vis_under true
                 --model.density.init_args.sfm_vis_behind_max 10 --model.density.init_args.sfm_vis_behind_max_above_nn 16)
            [ "$t" = mvugrow ] && XA+=(--model.density.init_args.merge_vis_under_harvest false)
            [ "$t" = mvugnd ] && XA+=(--model.density.init_args.merge_vis_under_ground true) ;;
  esac
done
CAP=$(python3 -c "import json;print(int(round(1e6 * json.load(open('configs/cap_alloc_sfm_g44.json'))['weights']['$B'])))")
echo "臂 $ARM => $LBL：全長 $ST、增生到 $((ST * G / 100))、合併到 $((ST * 9 / 10))、cap $CAP、家族 $FAM"
env "${ENVX[@]}" CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=$ST CITYGS_CAP=$CAP CITYGS_DUP44=${CITYGS_DUP44:-2} CITYGS_FAMILY=$FAM \
  bash scripts/lab/task_cmp.sh "$B" $SUF "${NB[@]}" "${XA[@]}"
