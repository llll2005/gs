#!/bin/bash
# ★★★★ AbsGS 訊號的前提驗證（重編光柵器 + 約 10 分鐘 GPU）
#
# 問題（使用者提問帶出來的）：MCMC 沒有梯度式 densify，能不能造一個訊號來用 AbsGS？
#
# 查碼查出來的實情（比原本以為的複雜，見 研究總覽 §11.22）：
#   * AbsGS（arXiv 2404.10484 §3.2）證明有符號的 screen-space 位置梯度會因為 splat 中心
#     兩側的像素推向相反方向而抵消 => 過度重建的大高斯永遠達不到分裂門檻。
#   * ⛔ **2DGS 的 viewspace 梯度不是 3DGS 的對應物**：`dL_dmean2D` 只在
#     `rho3d > rho2d`（次像素 fallback）分支被寫（backward.cu 全檔只有兩處），
#     解析良好的粒子恆為 0 => 天真移植 3DGS 梯度式 densify 只會看到次像素 splat。
#   * 2DGS 真正的對應物是 `dL_ds`＝「loss 想把 ray-splat 交點移到 splat 上的別處」，
#     但它立刻被轉成 transMat 梯度、與旋轉/尺度混在一起 => 逐顆訊號無處可取。
#
# 已實作（純新增，零刪除，已用 `git diff | awk '/^-[^-]/'` 稽核）：
#   auxiliary.h  `#define ABSGRAD 1`
#   backward.cu  在 ray-splat 分支 atomicAdd(|dL_ds.x|+|dL_ds.y|) 進 `dL_dmean2D.z`
#                （該分量全檔從未被讀，梯度鏈只用 .x/.y => **渲染與梯度逐位元不變**）
#   density controller  `absgrad_report`（預設 0，只讀不改行為）
#
# ⚠ 本任務要回答的**不是**「碰撞存不存在」，而是決策相關的那個問題：
#   **這個訊號排出來的名次，跟 MCMC 現在用的 `probs = opacity` 有沒有實質差別？**
#   top-10% 若幾乎重疊、rho 接近 1 => 換訊號不會換掉增生位置 => 整條線收掉，不必再花 GPU。
#   （這是 diagnosis-first。今天已經因為 intervention-first 浪費了一整輪。）
#
# 判準：
#   rho 低、top10% 重疊低、且 rho(|g|, 螢幕半徑) 明顯為正
#       => 訊號有實質差異且確實抓大粒子 => 值得做成 MCMC 的取樣權重
#   rho 高或 top10% 重疊高  => 與 opacity 同義 => 收掉
#   全為 0                  => .so 沒真的換掉（本腳本會先擋這個）
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
SUB=submodules/diff-surfel-rasterization-trim-pp

SO_BEFORE=$(find "$CONDA_PREFIX/lib" -name "*diff_trim_surfel*" -o -name "*_C*.so" 2>/dev/null | head -1)
STAMP_BEFORE=$(python - <<'PY'
import importlib.util, os
s = importlib.util.find_spec("diff_trim_surfel_rasterization")
p = os.path.dirname(s.origin) if s and s.origin else ""
c = [os.path.join(p, f) for f in os.listdir(p)] if p and os.path.isdir(p) else []
so = [f for f in c if f.endswith(".so")]
print(int(os.path.getmtime(so[0])) if so else 0)
PY
)
echo "重編前 .so mtime = $STAMP_BEFORE"

# 陷阱 1：distutils 只看 .cu 的 mtime，不看 header => 改了 auxiliary.h 不重編 build 會被跳過
rm -rf "$SUB/build"
# 陷阱 2：shell 匯出 CUDA_PATH=/opt/cuda (13.3)，torch 會優先用它而拒編；必須強制指回 env 內的 11.8
# 陷阱 3：pip 對同版本的本地路徑會靜默跳過 => 必須 --force-reinstall
conda run -n gspl env CUDA_HOME="$CONDA_PREFIX" CUDA_PATH="$CONDA_PREFIX" \
  pip install --no-build-isolation --no-deps --force-reinstall "$SUB" || { echo "!! 編譯失敗"; exit 1; }

STAMP_AFTER=$(python - <<'PY'
import importlib.util, os
s = importlib.util.find_spec("diff_trim_surfel_rasterization")
p = os.path.dirname(s.origin) if s and s.origin else ""
c = [os.path.join(p, f) for f in os.listdir(p)] if p and os.path.isdir(p) else []
so = [f for f in c if f.endswith(".so")]
print(int(os.path.getmtime(so[0])) if so else 0)
PY
)
echo "重編後 .so mtime = $STAMP_AFTER"
if [ "$STAMP_AFTER" = "$STAMP_BEFORE" ]; then
  echo "!! .so mtime 沒變 => 沒有真的重編（這是本專案踩過的第 3 個陷阱）。中止。"
  exit 1
fi

rm -rf outputs/absgrad_probe
conda run -n gspl --no-capture-output python -u main.py fit \
  --config configs/proxy/probe_ctrl.yaml \
  --trainer.max_steps 600 \
  --model.gaussian.init_args.optimization.means_lr_scheduler.init_args.max_steps 600 \
  --model.density.init_args.absgrad_report 100 \
  -n absgrad_probe 2>&1 | tee logs/absgrad_probe.txt | grep -E "absgrad|Trimming done|DIED|Error"

echo
echo "================ 開獎 ================"
grep -a "\[absgrad\]" logs/absgrad_probe.txt || echo "(無輸出 —— 檢查 ABSGRAD 是否為 1、.so 是否真的換掉)"
echo
echo "rho 高 / top10% 重疊高 => 與 opacity 同義，收掉。"
echo "rho 低 且 rho(|g|,螢幕半徑) 明顯為正 => 訊號有差異且確實抓大粒子，值得做成取樣權重。"
