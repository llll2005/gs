#!/bin/bash
# ★★★★★★ 殼層：depth-init 1500 步，**參數與四臂測試完全相同**，唯一變數＝正確的 init PLY
#
# 對照組＝`probe_armA_baseline`（同一個 config、同一個路徑，但當時底下是 2026-05-29 的**汙染**
# PLY，每顆點用鄰幀的深度圖擺位置）。2026-09-12 已就地換成正確版（md5 64a9873b…）。
# ⇒ 這是「汙染 init vs 正確 init」在**殼層指標**上的單變數對照。
#
# 為什麼值得跑，即使 fixdepth_b12 的 60k 分數是平的：
#   fixdepth_b12 量的是**光度分數**（PSNR -0.2sd 平），而幾何上殼只薄 2.6%。
#   但那是 60k 的終點 —— 殼是在 **step 1 的起始 trim** 就定型的（80% 偏差第一步就存在），
#   而四臂探針正是為了看那一刻。1500 步的幾何差異可能比 60k 的分數差異清楚。
# ⚠ 判準看**幾何**不是分數：z 中位、離表面倍數、起始 trim 砍掉多少、真表面覆蓋率。
#   1500 步的 PSNR 本來就很低（arm A 是 20.11），不要拿它當品質判準。
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/probe_armA_fixinit
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/probe_shell_1500.yaml \
  --model.initialize_from data/matrix_city/aerial/train/block_all/depth_init/block_12.ply \
  --data.parser.block_id 12 \
  -n probe_armA_fixinit
echo "===== 幾何對照（正確 init vs 汙染 init）====="
for R in probe_armA_fixinit probe_armA_baseline; do
  echo "--- $R ---"
  conda run -n gspl python tools/audit_geometry.py "$R" --block 12 2>&1 | tail -14
done
echo "===== 真表面覆蓋率（只算 SfM 盒內，避免被場外幕布騙）====="
conda run -n gspl python tools/novel_view_coverage.py \
  --arms probe_armA_fixinit probe_armA_baseline probe_armB_sfm --step 1499 --max-cam 12 2>&1 \
  | grep -vE "pkg_resources|declare|appearance|dataparser|down sample|loading|found " | tail -12
