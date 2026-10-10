#!/bin/bash
# 2026-10-10 使用者問：v/c 類判準（vpctilek 等）val 好、塊內 held-out 差，是不是剪掉的是塊外內容、合併後會改善？
#   tools/merge_view_heldout.py：用 merge 的同一套規則切出塊內顆粒，把 held-out 誤差拆成塊內／塊外區域。只渲染 => 可平行。
# 2026-10-10（使用者：被淘汰的方案會不會只是塊外表現差、被量法錯殺）：screen 模式 => 所有新年代被淘汰／降級的方案、與已採用的方案，
#   對各自的基準在 val（val⊂train，也是整張畫面）與 held-out 兩種視角上拆塊內／塊外。這是**篩選**（近似：看不到合併後鄰塊遮擋；
#   塊內像素只佔畫面約 11~15%）；翻盤的候選再用 tools/swap_block_eval.py 在 4x4 上定案。輸出 logs/mergeview/<視角>_<組>.txt＋摘要 screen_summary.txt。
# 用法：bash scripts/lab/task_mergeview.sh            （10-10 原本的三組，held-out）
#       bash scripts/lab/task_mergeview.sh screen     （篩選：val＋held-out 全部組）
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
L=logs/mergeview; mkdir -p "$L"
ck () { ls outputs/lab/$1/blocks/block_$2/checkpoints/*.ckpt 2>/dev/null | grep -E 'step=[0-9]+[.]ckpt$' | awk -F"step=" '{split($2,a,"."); print a[1]"\t"$0}' | sort -n | tail -1 | cut -f2; }   # 該塊最後一步（22k 家族不是 60000）
run () {   # run <輸出名> <塊> <跑次A> <跑次B> [<跑次C>]
  local o="$L/${VPFX:-}$1.txt" b=$2; shift 2
  [ -s "$o" ] && { echo "已有 $o"; return; }
  local cks=() ns=()
  for r in "$@"; do c=$(ck "$r" "$b"); [ -n "$c" ] || { echo "⛔ 缺 $r b$b"; return; }; cks+=("$c"); ns+=("$r"); done
  conda run -n gspl --no-capture-output python tools/merge_view_heldout.py --ckpt "${cks[@]}" --names "${ns[@]}" --views "${VIEWS:-heldout}" 2>&1 \
    | grep -vE "pkg_resources|declare_namespace|caching images|appearance group|loading colmap|down sample" | tee "$o.tmp" \
    && grep -q "塊內區域" "$o.tmp" && mv "$o.tmp" "$o"
}
if [ "${1:-}" = screen ]; then
  for VIEWS in val heldout; do
    VPFX=${VIEWS}_
    for b in 6 13; do
      run 60k_conic_b$b $b cs60_conic cs60_conicvpc cs60_vpctilek cs60_oent cs60_cag0 cs60_coreg0 cs60_sfmdup4   # v/c 類、oent、absgrad 關；已採用的 oreg0／dup4（反向檢查）
      run 60k_speed3_b$b $b speed3 speed3_trimvpc elong_prune elong_relocate speed3_trimvpc_elong              # conic 關時代：v/c、elongation 硬剪／搬移
      run 22k_base_b$b $b cs_base cs_trimvpc cs_tilek                                                         # 22k：v/c、逐 tile 前 K 名
    done
    run 22k_conic_b6 6 cs_conic cs_conicvpc
    run 22k_cost_b6 6 cs_base cs_costdir cs_costdir_cal cs_costtaming                                         # 成本感知取樣
    run 22k_budget_b6 6 cs_base cs_cb25 cs_cb25cost cs_cb50 cs_cb50cost cs_cb25x cs_cb50x                     # 預算閘門（N 不同）
    run 22k_base_b12 12 cs_base cs_trimvpc cs_cb25 cs_cb25cost
  done
  { echo "差 = 方案 − 該組第一個（基準）；「只用塊內顆粒」≈ 合併後；val⊂train"
    for f in "$L"/val_*.txt "$L"/heldout_*.txt; do echo "=== $(basename "$f" .txt)"; grep -E "塊內區域像素佔畫面|−" "$f"; done; } | tee "$L/screen_summary.txt"
  exit 0
fi
run b6_best_bestvt 6 cs60_best cs60_bestvt
run b6_conic_vpctilek 6 cs60_conic cs60_vpctilek cs60_oent
run b13_conic_vpctilek 13 cs60_conic cs60_vpctilek cs60_oent
