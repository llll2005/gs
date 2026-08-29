#!/bin/bash
# ★★★ 診斷：trim 每次到底剪掉幾 %？（約 35 分鐘）
#
# 起因（2026-08-25）：s=0.2 的 proxy（densify 30 / trim 100）族群在 step 3,000 見頂 872k 後
# **下跌 20.3%**、成長率翻負，永遠到不了 2.6M cap => 跑在錯的 regime，該跑次作廢。
# 而全長跑次的成長率與破平衡公式吻合（實測 0.0001126 vs 理論 0.0001146/step）。
#
# 假說：`sep_depth_trim_2dgs_renderer.py` 的
#     tile = torch.quantile(contribution, prune_ratio);  prune_mask = (contribution <= tile)
# 用 `<=`，所以只要**超過 prune_ratio 比例的粒子 contribution 剛好是 0**，
# 這一刀就把它們全部剪掉，遠超過名目 10%。零貢獻的來源是 `EXACT_SUPPORT`
# （opacity <= 1/255 不進 binning）。MCMC relocation 的子代 opacity 很低，需要若干步才爬過去；
# 壓縮把「出生到首次 trim」從約 250 步壓到約 50 步 => 來不及 => 被當零貢獻剪掉。
#
# ⚠ 這是 diagnosis-first：**先量，不要先改**。上一輪我直接排了 14 小時的介入實驗，
#   驗了 config 正確／事件次數守恆／能解析，卻沒驗「族群軌跡在壓縮後成不成立」——
#   而那是整個設計的核心假設。這支腳本就是補那一步。
#
# 兩臂都是**截斷真實跑次**（刻意不動 means_lr_scheduler.max_steps），
# 所以前幾百步與原跑次逐位元相同，量到的就是原跑次的行為：
#   probe_diag  = px_noprior 的組態（densify 30 / trim 100），800 步 => 7 個 trim 事件
#   probe_ctrl  = noprior_b12 的組態原封不動（densify 150 / trim 500），2000 步 => 3 個 trim 事件
#
# 判準（看 `零貢獻=` 與 `剪=`）：
#   diag 的剪% >> 10% 且零貢獻% > 10%，而 ctrl 的剪% ≈ 10% 且零貢獻% < 10%
#       => 假說成立。修法方向：拉長「出生到首次 trim」的步數，或把 `<=` 改 `<`
#          （⚠ 後者會改變所有既有跑次的行為，不可輕易動）
#   兩臂的剪% 都 ≈ 10%
#       => 假說否證，族群衰減另有原因，回頭查 densify 端（add_new_gs 實際加了幾 %）
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

for TAG in diag ctrl; do
  NAME=probe_$TAG
  CFG=configs/proxy/probe_$TAG.yaml
  [ -f "$CFG" ] || { echo "!! 缺 $CFG，中止"; exit 1; }
  [ -d "outputs/$NAME" ] && { echo "== $NAME 殘骸移除"; rm -rf "outputs/$NAME"; }
  echo "===== $NAME ====="
  conda run -n gspl --no-capture-output python -u main.py fit --config "$CFG" -n "$NAME" \
    2>&1 | tee "logs/probe_$TAG.txt" | grep -E "Trimming done|step=" | tail -40
done

echo
echo "================ 開獎 ================"
for TAG in diag ctrl; do
  echo "--- probe_$TAG ---"
  grep "Trimming done" "logs/probe_$TAG.txt" 2>/dev/null || echo "  (無 trim 事件)"
done
echo
echo "對照組 ctrl 的剪% 應該 ≈ 10%。若 diag 明顯更高，且零貢獻% > 10%，假說成立。"
