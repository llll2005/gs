#!/bin/bash
# 輕量 coarse 前訓練＋短排程細修（使用者 11-09「作一個一樣輕量的 coarse（基於我們當前的 dup 點雲）作為前訓練，排幾個訓練看步數組合、階段拆分」）
#   第一段 coarse：全域、1/4 訓練解析度（down_sample 4.8 => 400x225）、SH0、cap 2M、30k（增生到 15k），A＋dist4 的核心（無塊外遮罩）
#     ＋px_floor（低解析度會讓面元一出生就縮成點精靈：36 的前段低解析度 C 臂點精靈 67.9%；下限＝coarse 的 1 px＝細修的 4 px）。
#     門檻：峰值 VRAM（台帳）<= 6 GB、時間（佔全場景預算 3.16 h 的比例）、海面與幾何看第二段的細修結果。
#   第二段 細修：coarse 裁到各塊盒外擴 15%、最大的面元面內一拆四到 0.85 cap（tools/coarse_crop.py），用**精確的 A＋dist4**
#     （11-08 g44s30ksfm10fssh16di4_ 的旗標；11-07 之後的預設 lr x2、px_floor、刪死顆粒都釘回當時的值）＝唯一變數是起點與排程；
#     對照＝11-08 A＋dist4 SfM 分配全場景裡的同塊模型（不用重跑）。
#   排程：S30＝30k、增生到 15k（只換起點）；S20＝20k、增生到 5k；S15＝15k、增生到 3k。
# 用法（佇列，依序）：
#   [cpu]  bash scripts/lab/task_coarse.sh init                  全域起點 coarse_init/dup1_g44_1800k.ply
#   [solo] bash scripts/lab/task_coarse.sh coarse                訓練 coarse（outputs/lab/coarse_g44r48sh0_best1）
#   [cpu]  bash scripts/lab/task_coarse.sh crop                  裁到 16 塊（outputs/lab/coarse_g44r48sh0_crop）
#   [solo] bash scripts/lab/task_coarse.sh fine <塊> <S30|S20|S15>
#   [solo] bash scripts/lab/task_coarse.sh eval <塊>             換塊評測＋全面向（對 A＋dist4、現行配方）
# ★ 11-16 新底 5 版（細節見 init5／coarse5／crop5 的註解）：
#   [cpu]  bash scripts/lab/task_coarse.sh init5                 全域起點 coarse_init/tri_g44_1800k.ply（16 塊稠密起點拼起來）
#   [solo] bash scripts/lab/task_coarse.sh coarse5               10k、增生 70%、新底 5 配方的全域部分、1/4 解析度、SH0、cap 2M
#   [cpu]  bash scripts/lab/task_coarse.sh crop5                 裁到 16 塊（outputs/lab/coarse5_s10g70r48sh0_crop）
#   細修：[solo] CITYGS_DEMO=1 bash scripts/lab/task_tune25.sh <塊> s7g70+np16+tricap0p5+cmflat+trimask+geo8+log0+capfc+coarse5
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
MODE=$1
PY="conda run -n gspl --no-capture-output python -u"
CO_N=${CO_INIT_N:-1800000}
CO_CAP=${CO_CAP:-2000000}
CO_STEPS=${CO_STEPS:-30000}
CO_RUN=${CO_RUN:-coarse_g44r48sh0}                  # outputs/lab/<CO_RUN>_best1（task_cmp 的跑次名＝家族＋臂）
CO_INIT=coarse_init/dup1_g44_$((CO_N / 1000))k.ply
CROP=${CO_RUN}_crop
A_RUN=g44s30ksfm10fssh16di4_best1maskdistrfed10b4    # 11-08 A＋dist4 SfM 分配（對照）
CUR_RUN=g44s30ksfm10nuPF2DB2_best1maskdistrfed10b4   # 11-09 現行配方（參考）
# A＋dist4 的旗標（11-08 那次只給了前 5 個；後 4 個是 11-07 之後改了預設、要釘回當時值的）
A_FLAGS=(--model.renderer.init_args.record_reduce true --model.renderer.init_args.absgrad_gate true --model.metric.init_args.fused_ssim true
  --model.gaussian.init_args.optimization.optimizer.init_args.sh_rest_interval 16 --model.metric.init_args.lambda_dist_interval 4
  --model.gaussian.init_args.optimization.means_lr_init 6.4e-5 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 6.4e-7
  --model.density.init_args.px_floor 0 --model.density.init_args.harvest_prune_dead false)

case "$MODE" in
  # 全域起點：各塊 sfmfill_sweep44/dup1 只留自己盒內拼起來、抽到 CO_INIT_N 顆（CPU）
  init)
    $PY tools/coarse_crop.py init --dup dup1 --n "$CO_N" 2>&1 | grep -v -E 'Warn|pkg_res|declare' ;;
  # 訓練 coarse（全域、1/4 解析度、SH0；不跑圖表資料包）
  coarse)
    [ -f "data/matrix_city/aerial/train/block_all/$CO_INIT" ] || { echo "⛔ 缺 $CO_INIT（先跑 init）"; exit 2; }
    NCAM=1 STEPS=$CO_STEPS CITYGS_LEAN=1 CITYGS_CAP=$CO_CAP CITYGS_INIT_PLY=$CO_INIT CITYGS_POSTDATA=0 CITYGS_FAMILY=${CO_RUN}_ \
      bash scripts/lab/task_cmp.sh null best1 \
      --model.gaussian.init_args.optimization.optimizer.class_path FusedAdam \
      --model.renderer.init_args.record_reduce true --model.renderer.init_args.absgrad_gate true --model.metric.init_args.fused_ssim true \
      --model.metric.init_args.lambda_dist_interval 4 --model.metric.init_args.lambda_dist 100 \
      --model.renderer.init_args.trim_running_vc true --model.renderer.init_args.trim_running_warp true --model.renderer.init_args.bwd_zero_grad_skip true \
      --model.metric.init_args.opacity_entropy_side down --model.metric.init_args.opacity_entropy_both_from_iter $((CO_STEPS / 3)) \
      --model.metric.init_args.opacity_entropy_both_reg 0.008 \
      --model.gaussian.init_args.sh_degree 0 --data.parser.down_sample_factor 4.8 \
      --model.gaussian.init_args.optimization.means_lr_init 6.4e-5 --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.lr_final 6.4e-7 \
      --model.density.init_args.px_floor 1.0 --model.density.init_args.harvest_prune_dead false
    rc=$?
    echo "── coarse 的純訓練時間（train_status val step0 -> 最後一次 val）與峰值 VRAM（台帳）"
    python3 ../labrun/time_1024.py "${CO_RUN}_best1" 2>/dev/null || echo "（time_1024.py 不在 ../labrun）"
    grep -E "${CO_RUN}_best1" -A40 logs/quad_progress.log | grep -E "DONE|DIED" | tail -1
    exit $rc ;;
  # 裁到各塊（CPU）
  crop)
    # 全域跑次（block_id null）的 ckpt 直接在 <run>/checkpoints/（11-10 第一次重測在這裡找錯目錄）
    CK=$(ls -t outputs/lab/${CO_RUN}_best1/checkpoints/*.ckpt outputs/lab/${CO_RUN}_best1/blocks/block_null/checkpoints/*.ckpt 2>/dev/null | grep -v xyz_rgb | head -1)
    [ -n "$CK" ] || { echo "⛔ 找不到 coarse 的 ckpt（outputs/lab/${CO_RUN}_best1/checkpoints）"; exit 2; }
    echo "coarse ckpt：$CK"
    set -o pipefail
    $PY tools/coarse_crop.py crop --ckpt "$CK" --out "lab/$CROP" 2>&1 | grep -v -E 'Warn|pkg_res|declare' ;;
  # 從 coarse 起點細修（精確 A＋dist4；只換起點與排程）
  fine)
    B=${2:?塊}; S=${3:?排程 S30／S20／S15}
    case $S in
      # 30k、增生到 15k（排程同 A＋dist4，只換起點）
      S30) ST=30000; G=15000 ;;
      # 20k、增生到 5k
      S20) ST=20000; G=5000 ;;
      # 15k、增生到 3k
      S15) ST=15000; G=3000 ;;
      *) echo "⛔ 排程只有 S30／S20／S15"; exit 2 ;;
    esac
    CK=outputs/lab/$CROP/blocks/block_$B/checkpoints/coarse_crop.ckpt
    [ -f "$CK" ] || { echo "⛔ 缺 $CK（先跑 crop）"; exit 2; }
    CAP=$(python3 -c "import json;print(int(round(1e6 * json.load(open('configs/cap_alloc_sfm_g44.json'))['weights']['$B'])))")
    CITYGS_KEEP_CFG_RENDERER=1 CITYGS_GRID=44 CITYGS_LEAN=1 STEPS=$ST CITYGS_CAP=$CAP CITYGS_DUP44=2 \
      CITYGS_FAMILY=g44s$((ST / 1000))kcoarseG$((G / 1000))_ \
      bash scripts/lab/task_cmp.sh "$B" best1maskdistrfed10b4 "${A_FLAGS[@]}" \
      --model.initialize_from "$CK" --model.overwrite_config false --model.density.init_args.densify_until_iter "$G" ;;
  # 換塊評測＋全面向（對照 A＋dist4、現行配方；沒跑完的臂 eval_arms_lab 會印 ⚠ 缺）
  eval)
    B=${2:?塊}
    R30=g44s30kcoarseG15_best1maskdistrfed10b4; R20=g44s20kcoarseG5_best1maskdistrfed10b4; R15=g44s15kcoarseG3_best1maskdistrfed10b4
    bash scripts/lab/task_full44.sh swap "$B" $A_RUN $CUR_RUN $R30 $R20 $R15
    EVAL_ONLY=time,f1,ground,facade,orient,sea,smallobj bash scripts/lab/eval_arms_lab.sh "$B" coarse_b${B}_1109 \
      A+dist4=$A_RUN 現行配方=$CUR_RUN coarse30k=$R30 coarse20kG5=$R20 coarse15kG3=$R15 ;;
  # ★ 11-16 新底 5 版 coarse（使用者「當前最佳解但用 10k 步的 coarse 跟 7k 步的 4x4；coarse 也用我們的最佳解，參數依研究與 coarse 的目的調」）
  #   coarse 的目的＝便宜地先拿到全域一致的低頻幾何（每塊共用、看過全部 5,621 張 => 塔頂與接縫），細節交給細修 => 設計：
  #   - 配方＝新底 5 能用在全域的部分：NB（現行配方＋位置 lr x1＋拿掉 v4）＋校正深度＋逐貢獻深度上限 0.5（只罰躺平先驗、只補前面）
  #     ＋鄰域平面每 16 步＋dist 每 8 步＋不記 log；起點＝稠密起點（各塊 sfmfill_tri44_crop 只留自己盒內拼成全域、抽 1.8M＝0.9 x cap）
  #   - 不帶：塊外遮罩（全域沒有塊外）、前段低解析度（coarse 本身就是 1/4 解析度）
  #   - 1/4 訓練解析度（400x225；預縮影像 input_r4.8，與現縮同一算法）、SH0、cap 2M（11-09 輕量 coarse：峰值 2.76 GB、結束 1.80M 顆）
  #   - 10k、增生 70%（新底 5 的比例；11-09 的 20k／增生 25% 細修掉分是增生期太短）、合併到 90%；px_floor 1 px（＝細修的 4 px；11-09 理由）
  #   跑次 outputs/lab/<CO5_RUN>_best1；之後 crop5 裁成 16 塊（0.85 x 該塊 SfM 分配 cap、最大面元面內一拆四），細修用 task_tune25.sh 的 coarse5 元件
  init5)
    for b in $(seq 0 15); do [ -f "data/matrix_city/aerial/train/block_all/sfmfill_tri44_crop/dup1/block_$b.ply" ] || { echo "⛔ 缺稠密起點 block_$b（tools/make_tri_dense_init.py --blocks $b）"; exit 2; }; done
    $PY tools/coarse_crop.py init --src sfmfill_tri44_crop --dup dup1 --tag tri --n "${CO5_INIT_N:-1800000}" 2>&1 | grep -v -E 'Warn|pkg_res|declare' ;;
  coarse5)
    source scripts/lab/_nb_recipe.sh
    ST=${CO5_STEPS:-10000}; G5=${CO5_G:-70}; CO5_RUN=${CO5_RUN:-coarse5_s10g70r48sh0}
    CO5_INIT=coarse_init/tri_g44_$(( ${CO5_INIT_N:-1800000} / 1000 ))k.ply
    [ -f "data/matrix_city/aerial/train/block_all/$CO5_INIT" ] || [ "${CITYGS_DRY:-0}" = 1 ] || { echo "⛔ 缺 $CO5_INIT（先跑 init5）"; exit 2; }
    IMG=(); if [ -f data/matrix_city/aerial/train/block_all/input_r4.8/.complete ]; then IMG=(--data.parser.image_dir input_r4.8)
    else echo "⚠ 缺 input_r4.8/.complete（tools/make_resized_images.py --factor 4.8）=> 影像照舊現縮"; fi
    CITYGS_SFM_FAST=1 NCAM=1 STEPS=$ST CITYGS_LEAN=1 CITYGS_CAP=${CO5_CAP:-2000000} CITYGS_INIT_PLY=$CO5_INIT CITYGS_POSTDATA=0 \
      CITYGS_FAMILY=${CO5_RUN}_ bash scripts/lab/task_cmp.sh null best1 "${NB[@]}" \
      --model.gaussian.init_args.optimization.optimizer.class_path FusedAdam \
      --model.metric.init_args.lambda_dist 100 --model.renderer.init_args.trim_running_vc true \
      --model.metric.init_args.opacity_entropy_side down --model.metric.init_args.opacity_entropy_both_from_iter $((ST / 3)) \
      --model.metric.init_args.opacity_entropy_both_reg 0.008 \
      --model.density.init_args.densify_until_iter $((ST * G5 / 100)) --model.density.init_args.merge_until_iter $((ST * 9 / 10)) \
      --data.parser.depth_dir estimated_depths_tri --data.parser.depth_scale_name estimated_depth_scales_tri \
      --data.parser.depth_rel_dir estimated_depths_tri_rel \
      --model.metric.init_args.depth_cap_reg 0.5 --model.metric.init_args.depth_cap_margin 0.01 --model.metric.init_args.depth_cap_interval 8 \
      --model.metric.init_args.depth_cap_mask_dir data/matrix_city/aerial/train/block_all/cap_masks_flat \
      --model.metric.init_args.depth_cap_grad_mode 3 \
      --model.metric.init_args.surf_reg 0.01 --model.metric.init_args.surf_orient 1.0 --model.metric.init_args.surf_interval 16 \
      --model.metric.init_args.surf_every 400 \
      --model.metric.init_args.lambda_dist_interval 8 --model.train_log_interval -50 \
      --model.gaussian.init_args.sh_degree 0 --data.parser.down_sample_factor 4.8 "${IMG[@]}" \
      --model.density.init_args.px_floor 1.0 \
      --trainer.limit_val_batches 0 --trainer.num_sanity_val_steps 0
    exit $? ;;
  crop5)
    CO5_RUN=${CO5_RUN:-coarse5_s10g70r48sh0}
    CK=$(ls -t outputs/lab/${CO5_RUN}_best1/checkpoints/*.ckpt outputs/lab/${CO5_RUN}_best1/blocks/block_null/checkpoints/*.ckpt 2>/dev/null | grep -v xyz_rgb | head -1)
    [ -n "$CK" ] || { echo "⛔ 找不到 coarse5 的 ckpt（outputs/lab/${CO5_RUN}_best1）"; exit 2; }
    echo "coarse5 ckpt：$CK"
    set -o pipefail
    t0=$(date +%s)
    $PY tools/coarse_crop.py crop --ckpt "$CK" --out "lab/${CO5_RUN}_crop" 2>&1 | grep -v -E 'Warn|pkg_res|declare'; rc=$?
    echo "[crop5] 裁塊 $(( $(date +%s) - t0 )) s（全場景一次；時間總表要攤進 coarse 起點的成本）"; exit $rc ;;
  *) echo "⛔ 模式只有 init／coarse／crop／fine／eval／init5／coarse5／crop5"; exit 2 ;;
esac
