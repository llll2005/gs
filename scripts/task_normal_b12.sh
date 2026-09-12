#!/bin/bash
# ★★★★★★ 最原始基準線：vanilla 2DGS，1500 步（使用者 2026-09-12 要求）
#
# 要回答的問題：**失敗區在「完全沒有我方任何機制」的情況下也失敗嗎？**
#   也失敗 => 那不是我們疊加出來的，是 2DGS/資料本身的性質 => 整個殼層調查要換方向
#   不失敗 => 是我們某個機制造成的 => 逐一加回去就能二分定位
#
# configs/normal.yaml 只設了 9 個「不給就跑不起來」的值，其餘全吃預設：
#   density -> VanillaDensityController（論文原版梯度式 ADC，**沒有** MCMC/cap/absgrad/noise/
#              screen prune/value-per-cost/harvest-dust）
#   metric  -> VanillaMetrics（純 L1+SSIM，**沒有** normal/dist/opacity_reg/scale_reg/depth loss）
#   init    -> null ⇒ SfM 稀疏點（depth-init 是我方機制，不在基準線裡）
#   trim    -> 週期性與起始 trim 都關掉
#   解析度  -> 預設 1（全解析度 1920x1080，比我方的 1.2 更重）
#
# ⚠ 判準看**幾何**：z 中位（SfM 地表 0.26）、懸空%、懸空且不透明%。
#   1500 步的 PSNR 沒有意義（四臂當時是 19.5~20.8）。
# ⚠ 唯一殘差：真正的 vanilla 渲染器需要 `diff_surfel_rasterization`（本機沒裝），
#   所以用 trim 版但把 trim 全關。EXACT_SUPPORT/ABSGRAD 是編譯期寫死的，
#   兩者都已驗證位元級無損 => 不影響渲染與梯度。詳見 configs/normal.yaml 的註解。
set -u
cd "$(dirname "$0")/.." || exit 1
[ -n "${CITYGS_ALLOC_CONF:-}" ] && export PYTORCH_CUDA_ALLOC_CONF="$CITYGS_ALLOC_CONF"
rm -rf outputs/normal_b12
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/normal.yaml \
  --data.parser.block_id 12 \
  -n normal_b12
RC=$?
# ⚠ 台帳的 rc 是**腳本最後一個指令**的 rc。下面還有稽核迴圈，若不在這裡擋住，
#   訓練崩掉也會顯示「✔ DONE (rc=0)」—— 2026-09-12 第一次跑就這樣：RAM 被吃爆、
#   一張 ckpt 都沒存，台帳卻是 DONE。
if [ "$RC" -ne 0 ]; then
  echo "❌ 訓練失敗 rc=$RC —— 不做稽核，直接以此 rc 結束"
  exit "$RC"
fi
echo "===== 幾何：vanilla 基準線 vs 我方各臂 ====="
for R in normal_b12 probe_armA_fixinit probe_armA_baseline probe_armB_sfm; do
  echo "--- $R ---"
  conda run -n gspl python tools/audit_geometry.py "$R" --block 12 2>&1 \
    | grep -vE "pkg_resources|declare" | tail -6
done
