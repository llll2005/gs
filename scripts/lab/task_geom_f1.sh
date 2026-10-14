#!/bin/bash
# 幾何評測 A（使用者 10-14）：我方合併模型 vs 官方 CityGSV2 的表面 F1（MatrixCity 空拍），完全照官方流程：
#   GT＝官方 Google Drive 的 geometry_gt.zip（MC_Aerial/Block_all_ds.ply＋.json crop volume；doc/data_preparation.md）
#   抽網格＝有界 TSDF，voxel 0.01／sdf_trunc 0.04／depth_trunc 5.0、SH0 顏色、保留最大 50 個連通塊（scripts/citygs/run_citygs_mc_aerial.sh）
#     我方 tools/mesh_tsdf_stream.py：同算法，但逐張融合（官方一次存全部視角約 129 GB RAM，lab 只有 62 GB）
#   F1＝tools/eval_tnt/run.py --scene Block_all_ds（官方 README：MatrixCity Aerial P 0.432／R 0.790／F1 0.559）
#   兩個模型用**同一組相機**（我方合併 ckpt 的 parser：全場景、reconstruction、1.2 倍降採樣）。
# 用法（佇列）：
#   [solo] bash scripts/lab/task_geom_f1.sh prep
#   [solo] bash scripts/lab/task_geom_f1.sh mesh ours|official
#   [solo] bash scripts/lab/task_geom_f1.sh f1 ours|official
#   ⚠ official 的合併 ckpt 有 41.5 GB（含優化器狀態），torch.load 要整個讀進 RAM => 一定要 [solo]（不與訓練同時跑）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
# 無頭容器缺 libX11／libGL：裝在本資料夾底下的前綴（conda create -p，不動系統、不動 gspl 環境）
export LD_LIBRARY_PATH=/workspace/data/hdd/11213/o3dlibs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY="conda run -n gspl --no-capture-output python -u"
GT=data/geometry_gt/MC_Aerial
OUT=outputs/lab/geom_f1
OURS=outputs/lab/full44_best/checkpoints/merged.ckpt
OFF=../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/checkpoints/epoch=6-step=30000.ckpt
ck_of () { case "$1" in ours) echo "$OURS" ;; official) echo "$OFF" ;; *) echo "" ;; esac; }
MODE=$1; NAME=${2:-}
case "$MODE" in
  prep)
    [ -f "$GT/Block_all_ds.ply" ] && { echo "✅ GT 已在 $GT"; exit 0; }
    Z=data/geometry_gt_dl/geometry_gt.zip
    [ -f "$Z" ] || { echo "⛔ 缺 $Z（gdown 18L9AEJS2SNva7JgL2-DmqhoNtfDPSSY5）"; exit 2; }
    unzip -q -o "$Z" 'geometry_gt/MC_Aerial/*' -d data/ || exit 3
    ls -la "$GT" ;;
  mesh)
    C=$(ck_of "$NAME"); [ -n "$C" ] && [ -f "$C" ] || { echo "⛔ 未知或缺 ckpt：$NAME（$C）"; exit 2; }
    mkdir -p "$OUT/$NAME"
    $PY tools/mesh_tsdf_stream.py --ckpt "$C" --cameras_ckpt "$OURS" --out "$OUT/$NAME" \
      --voxel_size 0.01 --sdf_trunc 0.04 --depth_trunc 5.0 --num_cluster 50 2>&1 | grep -v -i warn | tee "$OUT/$NAME/mesh.log"
    exit "${PIPESTATUS[0]}" ;;
  f1)
    [ -f "$OUT/$NAME/fuse_post.ply" ] || { echo "⛔ 先跑 mesh $NAME"; exit 2; }
    [ -f "$GT/Block_all_ds.ply" ] || { echo "⛔ 先跑 prep"; exit 2; }
    $PY tools/eval_tnt/run.py --scene Block_all_ds --dataset-dir "$GT" --ply-path "$OUT/$NAME/fuse_post.ply" 2>&1 \
      | grep -v -i warn | tee "$OUT/$NAME/f1.log"
    grep -E 'precision|recall|f-score' "$OUT/$NAME/f1.log" ;;
  *) echo "⛔ 模式只能是 prep／mesh／f1"; exit 2 ;;
esac
