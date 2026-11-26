#!/usr/bin/env bash
# ★ 2026-11-26 不展開 SH 的渲染驗證（使用者「試試看不展開」）：
#   低維 SH 的 ckpt 照舊載入時會展開回 15 維（每顆 232 B）；CITYGS_SH_KEEP_LOWDIM=1 讓顯卡上只放 K 維係數（8 維 136 B／顆），
#   渲染時由光柵器用基底展開（＝訓練時每一步走的 kernel）。這支對一個已經跑過 `task_fullscene.sh test` 的全場景跑次：
#     1) 同一支 tools/eval_official_test.py 在 741 幀上重評一次（分數、渲染速度、顯存峰值；存圖到 test_official_lowdim/）
#     2) tools/compare_render_dirs.py 跟原本展開版存的 test_official/ 逐像素比對（看起來一不一樣）
#   用法：bash scripts/lab/task_sh_lowdim_render.sh <跑次>     例：fs1122_base10_f32
set -uo pipefail
cd "$(dirname "$0")/../.."
if [ "${1:-}" = "--help" ] || [ -z "${1:-}" ]; then sed -n 2,8p "$0"; exit 0; fi
R=$1; OUT=outputs/lab/$R
PY="conda run -n gspl --no-capture-output python -u"
CK=$(ls "$OUT/checkpoints/"*.ckpt 2>/dev/null | tail -1)
[ -n "$CK" ] || { echo "⛔ $R 沒有合併 ckpt"; exit 2; }
[ -d "$OUT/test_official" ] || { echo "⛔ $R 沒有 test_official/（先跑 task_fullscene.sh test）"; exit 2; }
L=logs/sh_lowdim_render_${R}_$(date +%m%d_%H%M).log
{ echo "════ $R 不展開 SH：$CK ════"
  CITYGS_SH_KEEP_LOWDIM=1 $PY tools/eval_official_test.py --ckpt "$CK" --save_dir "$OUT/test_official_lowdim" || exit 2
  echo "════ 對照：原本展開版（同一支評測，task_fullscene.sh test）════"
  T=$(ls -t logs/fullscene_test_${R}_*.log 2>/dev/null | head -1)
  [ -n "$T" ] && grep -E "^PSNR|^SSIM|^LPIPS|^渲染" "$T"
  echo "════ 逐像素比對：test_official（展開）vs test_official_lowdim（不展開）════"
  $PY tools/compare_render_dirs.py "$OUT/test_official" "$OUT/test_official_lowdim" || exit 2
  # 比完就刪（741 張 3200x900 PNG 約 3 GB；lab 硬碟曾寫滿）；KEEP=1 保留
  [ "${KEEP:-0}" = "1" ] || { rm -rf "$OUT/test_official_lowdim"; echo "（已刪 test_official_lowdim/；KEEP=1 可保留）"; }
} 2>&1 | grep -vE 'pkg_resources|declare_namespace' | tee "$L"
exit "${PIPESTATUS[0]}"
