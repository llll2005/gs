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
#     tri      深度換成 SfM 三角形校正的單目（tools/make_tri_depths.py；只留可靠三角形內的像素）——深度支撐閘門、補洞放點跟著用新深度
#     trihf    tri＋補洞放點改成「渲染比校正深度深 2% 就補」（原本 10%）、每次最多 0.5% 顆（原本 0.2%）
#     trist    tri＋單側不能看穿只留「比校正深度深 2% 就罰」那一項，每 8 步（與法向同一步，權重 0.5）
#     trigate  tri＋深度支撐閘門收緊：不要 SfM 豁免、前方 3%、從未貼面 1.5%（原本 5%／5%）
#     tribh    tri＋只擋後面的增生限制：多數有效視角裡落在校正深度表面後 1%～5% 的顆粒，MCMC 幾乎不抽它當母點（不關熵、不刪不搬；
#              SfM 豁免不套用；後 5% 以上＝隔著樓看過去，不算）。規格 紀錄/規格_組合臂_表面起點增生限制深度上限_1111.md §2
#     hfd      tri＋需求地圖補貨：補洞事件依落後權重 W(p)（這個像素有多少顏色靠表面後面的顆粒撐）抽像素、新顆粒放在校正深度表面上
#              （＋原本的 α<0.5 洞；可單獨開、不加 loss）。規格 紀錄/規格2_表面補貨與落後貢獻計價_1111.md §2
#     tricap<W>[m<N>][g] tri＋逐貢獻的單側深度上限（光柵器 CAP_OFFSET；每一份混合貢獻落在校正深度後面超過 margin 就依混合權重罰）、每 8 步
#              W＝權重（小數點寫 p：tricap0p5）、m<N>＝margin N%（預設 1）、g＝只在增生期（到 densify_until 為止）
#     dense    起點換成稠密初始化（tools/make_tri_dense_init.py：SfM＋補點之外，把可靠像素沿校正深度反投影、多視角一致的體素質心補滿到 0.9 cap）
#     sni      初始化的面元朝向用 SfM 近鄰平面（sfm_normal_init；11-08 在 30k 效果被洗掉，短排程重測）
#     trimask  塊外遮罩換成用校正深度判的那一份（block_masks44_tri；tools/block_mask44.py --depth_dir estimated_depths_tri --depth_raw）
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
        lowres|coarse|np16|lr050|mvu|mvugrow|mvugnd|tri|trihf|trist|trigate|tribh|hfd|dense|sni|trimask) XT+=("$t") ;;
        tricap[0-9]*) [[ $t =~ ^tricap[0-9]+(p[0-9]+)?(m[0-9]+)?g?$ ]] || { echo "⛔ tricap 寫法：tricap<W>[m<N>][g]（例 tricap2、tricap0p5m3、tricap2g）"; return 2; }; XT+=("$t") ;;
        *) echo "⛔ 不認得的元件 $t（s<NN>／g<NN>／s<NN>g<NN>／lowres／coarse／np16／lr050／mvu／mvugrow／mvugnd／tri／trihf／trist／trigate／tribh／hfd／tricap<W>／dense／sni／trimask）"; return 2 ;;
      esac
    fi
  done
  FAM="g44tn_s${S}g${G}$(IFS=; echo "${XT[*]:-}")_"
  LBL="${S}k增生${G}"
  for t in "${XT[@]:-}"; do
    case "$t" in lowres) LBL+="低解析度" ;; coarse) LBL+="coarse起點" ;; np16) LBL+="鄰域平面" ;; lr050) LBL+="lr減半" ;; mvu) LBL+="合併地下" ;;
      mvugrow) LBL+="合併地下只增生期" ;; mvugnd) LBL+="合併地下只貼地" ;;
      tri) LBL+="校正深度" ;; trihf) LBL+="校正深度補洞2%" ;; trist) LBL+="校正深度不能看穿2%" ;; trigate) LBL+="校正深度閘門收緊" ;; tribh) LBL+="只擋後面增生" ;; hfd) LBL+="需求地圖補貨" ;;
      tricap*) LBL+="逐貢獻深度上限${t#tricap}" ;;
      dense) LBL+="稠密初始化" ;; sni) LBL+="SfM法向起點" ;; trimask) LBL+="校正深度遮罩" ;; esac
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
    tri|trihf|trist|trigate|tribh|hfd|tricap*)
            _TD=data/matrix_city/aerial/train/block_all/estimated_depths_tri
            [ -d "$_TD" ] && [ -d "${_TD}_rel" ] || { echo "⛔ 缺 $_TD（先跑 tools/make_tri_depths.py）"; exit 2; }
            XA+=(--data.parser.depth_dir estimated_depths_tri --data.parser.depth_scale_name estimated_depth_scales_tri
                 --data.parser.depth_rel_dir estimated_depths_tri_rel)
            [ "$t" = trihf ] && XA+=(--model.density.init_args.hole_fill_margin 0.02 --model.density.init_args.hole_fill_frac 0.005)
            [ "$t" = trist ] && XA+=(--model.metric.init_args.depth_see_through_reg 0.5 --model.metric.init_args.depth_see_through_margin 0.02
                                     --model.metric.init_args.depth_see_through_hole 0 --model.metric.init_args.depth_see_through_interval 8)
            if [[ $t =~ ^tricap([0-9]+)(p([0-9]+))?(m([0-9]+))?(g)?$ ]]; then
              _cw=${BASH_REMATCH[1]}; [ -n "${BASH_REMATCH[3]}" ] && _cw="$_cw.${BASH_REMATCH[3]}"
              _cm=$(printf '0.%02d' "${BASH_REMATCH[5]:-1}")
              XA+=(--model.metric.init_args.depth_cap_reg "$_cw" --model.metric.init_args.depth_cap_margin "$_cm" --model.metric.init_args.depth_cap_interval 8)
              [ -n "${BASH_REMATCH[6]}" ] && XA+=(--model.metric.init_args.depth_cap_until_iter $((ST * G / 100)))
            fi
            [ "$t" = hfd ] && XA+=(--model.density.init_args.hole_fill_demand true)
            [ "$t" = tribh ] && XA+=(--model.metric.init_args.depth_support_behind true --model.metric.init_args.depth_support_behind_margin 0.01
                                     --model.metric.init_args.depth_support_behind_max 0.05)
            [ "$t" = trigate ] && XA+=(--model.metric.init_args.depth_support_sfm_radius_nn 0 --model.metric.init_args.depth_support_sfm_radius 0
                                       --model.metric.init_args.depth_support_delta_front 0.03 --model.metric.init_args.depth_support_delta_back 0.015) ;;
    dense)  [ -f "data/matrix_city/aerial/train/block_all/sfmfill_tri44_crop/dup1/block_$B.ply" ] || { echo "⛔ 缺稠密初始化（tools/make_tri_dense_init.py --blocks $B）"; exit 2; }
            ENVX+=(CITYGS_CROP_INIT=sfmfill_tri44_crop); CITYGS_DUP44=1 ;;
    sni)    XA+=(--model.density.init_args.sfm_normal_init true) ;;
    trimask) [ -f "data/matrix_city/aerial/train/block_all/block_masks44_tri/block_$B/stats.json" ] || { echo "⛔ 缺校正深度遮罩 block_masks44_tri/block_$B"; exit 2; }
            ENVX+=(CITYGS_MASK_DIR=block_masks44_tri) ;;
  esac
done
CAP=$(python3 -c "import json;print(int(round(1e6 * json.load(open('configs/cap_alloc_sfm_g44.json'))['weights']['$B'])))")
echo "臂 $ARM => $LBL：全長 $ST、增生到 $((ST * G / 100))、合併到 $((ST * 9 / 10))、cap $CAP、家族 $FAM"
env "${ENVX[@]}" CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=$ST CITYGS_CAP=$CAP CITYGS_DUP44=${CITYGS_DUP44:-2} CITYGS_FAMILY=$FAM \
  bash scripts/lab/task_cmp.sh "$B" $SUF "${NB[@]}" "${XA[@]}"
