#!/bin/bash
# 評分工具落差（2026-10-09）：同一個官方 release 合併模型，官方 `main.py test` 25.79 vs 我方 eval_official_test.py 27.26。
#   已知（CPU，logs/evalgap/official_release_brightness.tsv）：官方存檔 741 張裡 8~13% 渲染變暗或全黑（0184 全黑），
#   排除後官方逐張平均 27.18 ≈ 我方工具 => 落差來自那批暗幀。本任務定位暗幀從哪來：
#     1 我方工具逐張（我方程式碼＋gspl）               => logs/evalgap/ours_release.csv，與官方 CSV 逐幀對照
#     2 官方程式碼＋gspl_official（官方光柵器），我方相機建法 => 暗幀重現 => 官方 renderer／光柵器層；不重現 => 官方 test 迴圈
#     3 官方程式碼＋gspl（我方光柵器）                  => 2 若重現，再看換光柵器會不會消失（介面不合會失敗，不影響 1、2）
# 用法：[solo] bash scripts/lab/task_evalgap.sh
case "${1:-}" in -h|--help) exec bash "$(dirname "$0")/../_help.sh" "$0" ;; esac   # 說明全部從本檔讀出（scripts/_help.sh）
set -u
cd "$(dirname "$0")/../.." || exit 1
GS=$(pwd); ORIG=$GS/../cityGS_origin
CK=$ORIG/outputs/citygsv2_mc_aerial_sh2_trim/checkpoints/epoch=6-step=30000.ckpt
TEST=$GS/data/matrix_city/aerial/test/block_all_test_official2
OCSV=$ORIG/outputs/citygsv2_mc_aerial_sh2_trim/metrics/test-step=60000.csv
L=$GS/logs/evalgap; mkdir -p "$L"
[ -f "$CK" ] || { echo "⛔ 找不到官方合併 ckpt $CK"; exit 2; }
F='pkg_resources|declare_namespace|\.\.\.[0-9]+/[0-9]+ +PSNR'

echo "════ 1 我方工具逐張（release 合併模型）"
[ -s "$L/ours_release.csv" ] || conda run -n gspl --no-capture-output python tools/eval_official_test.py --ckpt "$CK" \
    --test_dir "$TEST" --per_image_csv "$L/ours_release.csv" 2>&1 | grep -vE "$F" | grep -E "^(PSNR|SSIM|LPIPS)|逐張|⛔"

echo "════ 1b 逐幀對照（官方 CSV vs 我方）"
conda run -n gspl --no-capture-output python - "$OCSV" "$L/ours_release.csv" "$L/official_release_brightness.tsv" <<'PY' 2>&1 | grep -v pkg_resources
import csv, sys, numpy as np
off = {r["name"]: float(r["psnr"]) for r in csv.DictReader(open(sys.argv[1])) if r["name"] not in ("", "MEAN")}
our = {r["name"]: r for r in csv.DictReader(open(sys.argv[2]))}
brt = {r["name"]: float(r["bright_ratio"]) for r in csv.DictReader(open(sys.argv[3]), delimiter="\t")}
n = sorted(set(off) & set(our)); d = np.array([float(our[k]["psnr"]) - off[k] for k in n]); b = np.array([brt[k] for k in n])
print(f"共同幀 {len(n)}：官方 {np.mean([off[k] for k in n]):.3f}  我方 {np.mean([float(our[k]['psnr']) for k in n]):.3f}  差 {d.mean():+.3f}")
for lo, hi in [(0, .5), (.5, .85), (.85, .95), (.95, 9)]:
    m = (b >= lo) & (b < hi)
    if m.any():
        print(f"  官方亮度比 [{lo},{hi}) {m.sum():4d} 幀：我方−官方 平均 {d[m].mean():+6.3f}  中位 {np.median(d[m]):+6.3f}")
print(f"  |差| < 0.1 dB 的幀 {int((abs(d) < 0.1).sum())}/{len(n)}；我方亮度比 < 0.85 的幀 {sum(float(our[k]['bright_ratio']) < .85 for k in n)}")
PY

NAMES=$(python3 - "$L/official_release_brightness.tsv" <<'PY'
import csv, sys
r = sorted(csv.DictReader(open(sys.argv[1]), delimiter="\t"), key=lambda x: float(x["bright_ratio"]))
pick = [x["name"] for x in r[:16]] + [x["name"] for x in r if abs(float(x["bright_ratio"]) - 1) < 0.02][:6]
print(" ".join(pick))
PY
)
echo "════ 2 官方程式碼＋gspl_official：$NAMES"
( cd "$ORIG" && conda run -n gspl_official --no-capture-output python "$GS/tools/evalgap_render_official.py" --repo . \
    --ckpt "$CK" --test_dir "$TEST" --names $NAMES --out "$L/official_env_direct.tsv" ) 2>&1 | grep -vE "$F"
echo "════ 3 官方程式碼＋gspl（我方光柵器）"
( cd "$ORIG" && conda run -n gspl --no-capture-output python "$GS/tools/evalgap_render_official.py" --repo . \
    --ckpt "$CK" --test_dir "$TEST" --names $NAMES --out "$L/official_code_our_env.tsv" ) 2>&1 | grep -vE "$F" | tail -30 \
  || echo "（3 失敗：介面不合，不影響 1、2）"
