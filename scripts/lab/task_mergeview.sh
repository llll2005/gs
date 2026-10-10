#!/bin/bash
# 2026-10-10 使用者問：v/c 類判準（vpctilek 等）val 好、塊內 held-out 差，是不是剪掉的是塊外內容、合併後會改善？
#   tools/merge_view_heldout.py：用 merge 的同一套規則切出塊內顆粒，把 held-out 誤差拆成塊內／塊外區域。只渲染 => 可平行。
# 用法：bash scripts/lab/task_mergeview.sh
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
L=logs/mergeview; mkdir -p "$L"
ck () { ls outputs/lab/$1/blocks/block_$2/checkpoints/*step=60000.ckpt 2>/dev/null | head -1; }
run () {   # run <輸出名> <塊> <跑次A> <跑次B> [<跑次C>]
  local o="$L/$1.txt" b=$2; shift 2
  [ -s "$o" ] && { echo "已有 $o"; return; }
  local cks=() ns=()
  for r in "$@"; do c=$(ck "$r" "$b"); [ -n "$c" ] || { echo "⛔ 缺 $r b$b"; return; }; cks+=("$c"); ns+=("$r"); done
  conda run -n gspl --no-capture-output python tools/merge_view_heldout.py --ckpt "${cks[@]}" --names "${ns[@]}" 2>&1 \
    | grep -vE "pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample" | tee "$o.tmp" \
    && grep -q "塊內區域" "$o.tmp" && mv "$o.tmp" "$o"
}
run b6_best_bestvt 6 cs60_best cs60_bestvt
run b6_conic_vpctilek 6 cs60_conic cs60_vpctilek cs60_oent
run b13_conic_vpctilek 13 cs60_conic cs60_vpctilek cs60_oent
