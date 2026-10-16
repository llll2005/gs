#!/bin/bash
# ★ 2026-10-16 配對貢獻 bit（使用者「試試」）的 lab 驗證：forward（lean 訓練）記每個 (tile,顆粒) 配對有沒有被任一像素混合，
#   backward（renderCUDA_opt）只處理被混合過的。本機（4050、full44 b10 60k、6 台相機）：渲染逐位元相同、backward −6.8%；
#   梯度與舊 backward 不同，原因是上游光柵器的近平面判斷 forward／backward 不一致（forward 用交點深度、backward 濾波分支用 Tw.z）；
#   把 backward 改成與 forward 一致後，開／不開 bit 的梯度回到噪音底（rotations 1.6e-6 vs 噪音 1.5e-6）=> bit 本身精確等價。
#   這裡在 lab 重做：① g44_best1 b6 @29,999 六台相機的 dump／compare（噪音底、bit 開關）② 真實迴圈計時 best1 vs best1＋bit（增生期／收割期）
# 用法（佇列）：[solo] bash scripts/lab/task_pairbit_check.sh
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac
set -u
cd "$(dirname "$0")/../.." || exit 1
source scripts/lab/_common.sh >/dev/null
L=logs/pairbit_check_$(date +%m%d_%H%M).log
CK=$(ls outputs/${RUN_PREFIX}g44_best1/blocks/block_6/checkpoints/*step=29999.ckpt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "⛔ 沒有 g44_best1 b6 @29,999"; exit 2; }
O=logs/pairbit_dump; mkdir -p "$O"
D=(conda run -n gspl --no-capture-output python tools/check_lean_render.py dump --ckpt "$CK" --lean 1 --record-reduce 1 --repeat 20 --ncam 6 --bwd-sat 1 --bwd-red 1)
X=(conda run -n gspl --no-capture-output python tools/check_lean_render.py compare)
{
  conda run -n gspl python -c "import diff_trim_surfel_rasterization as m;F=m.GaussianRasterizationSettings._fields;assert 'bwd_pair_bit' in F;print('✅ 光柵器有 bwd_pair_bit')" || exit 3
  for x in sr1 sr2; do "${D[@]}" --out "$O/$x.pt" 2>&1 | grep "計時"; done
  "${D[@]}" --out "$O/pb.pt" --pair-bit 1 2>&1 | grep "計時"
  "${X[@]}" "$O/sr1.pt" "$O/sr2.pt" --label "噪音底：飽和跳過＋先加總 跑兩次" 2>&1 | grep -v -i warn | tail -8
  "${X[@]}" "$O/sr1.pt" "$O/pb.pt" --label "配對 bit 開 vs 關（梯度差＝上游近平面不一致；渲染應逐位元相同）" 2>&1 | grep -v -i warn | tail -8
  rm -f "$O"/*.pt
} 2>&1 | grep -v -E 'pkg_resources|declare_namespace' | tee "$L"
CITYGS_TIME_ARMS="best1 best1pb" bash scripts/lab/task_best1_time.sh loop 6 2>&1 | tee -a "$L"
echo "log：$L"
