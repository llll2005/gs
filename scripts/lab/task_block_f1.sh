#!/bin/bash
# 區塊級幾何 F1（使用者 10-14；tools/block_f1.py；全場景 F1 我方 0.356 vs 官方 0.597 => 實驗分析/09 §6）
#   verify            現成全場景網格（我方、官方）不裁區塊、不配準 => 與 eval_tnt（有 ICP 配準）的 0.356／0.597 對照
#   fullcrop <塊>      我方／官方全場景網格裁到該塊 => 官方的逐塊 F1（兩者都用全部相機融合）
#   arms <塊> <跑次...> 各跑次（outputs/lab/<跑次>）的該塊換進 full44 合併模型，只用該塊相機抽網格 => 單變數臂的幾何分數
# 用法（佇列）：[solo] bash scripts/lab/task_block_f1.sh verify ；[solo] ... fullcrop 6 ；[solo] ... arms 6 g44_best0 g44_best1
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; "") bash "$(dirname "$0")/../_help.sh" "$0"; exit 2 ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
export LD_LIBRARY_PATH=/workspace/data/hdd/11213/o3dlibs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY="conda run -n gspl --no-capture-output python -u"
OUT=outputs/lab/geom_f1/blocks; mkdir -p "$OUT"
MODE=$1; shift
case "$MODE" in
  verify)
    for m in ours official; do
      $PY tools/block_f1.py --block -1 --mesh "outputs/lab/geom_f1/$m/fuse_post.ply" --out "$OUT/verify_$m" 2>&1 | grep -v -i -E 'warn|declare' | tee -a "$OUT/verify.log"
    done ;;
  fullcrop)
    B=${1:?塊}
    for m in ours official; do
      $PY tools/block_f1.py --block "$B" --mesh "outputs/lab/geom_f1/$m/fuse_post.ply" --out "$OUT/fullcrop_$m" 2>&1 | grep -v -i -E 'warn|declare' | tee -a "$OUT/fullcrop_b$B.log"
    done ;;
  arms)
    B=${1:?塊}; shift
    C=(); N=()
    for r in "$@"; do
      c=$(ls outputs/lab/$r/blocks/block_$B/checkpoints/*step=60000.ckpt 2>/dev/null | grep -v _inbox | head -1)
      [ -n "$c" ] || { echo "⚠ 略過 $r：沒有 block $B 的 60k ckpt"; continue; }
      C+=("$c"); N+=("$r")
    done
    [ ${#C[@]} -gt 0 ] || { echo "⛔ 沒有候選"; exit 2; }
    $PY tools/block_f1.py --block "$B" --cand "${C[@]}" --names "${N[@]}" --out "$OUT/arms_b$B" 2>&1 \
      | grep -v -i -E 'warn|declare|找不到深度圖|random points|cluster|num vertices' | tee -a "$OUT/arms_b$B.log" ;;
  *) echo "⛔ 模式只能是 verify／fullcrop／arms"; exit 2 ;;
esac
