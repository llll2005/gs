#!/bin/bash
# ★★★★★ proxy 驗證：壓縮 5 倍的跑次能不能排出跟 60k 一樣的名次？
#
# 動機（2026-08-25）：單次跑次 9.6 小時，佇列積了 70 小時，實驗做不完。
#
# 為什麼相信有機會（tensorboard 回測，零 GPU 成本，tools/proxy_horizon.py）：
#   每個跑次的 val 曲線都在 `densify_until` 那一刻跳 +1 dB，之前幾乎是平的
#   => 品質是「族群蓋好之後的收割」產生的，不是均勻累積的
#   => 整個排程等比例縮短，形狀會保留；而**單純截斷**會停在跳點之前，量到的不是配方。
#   實測佐證：截斷到 17,039 步（28%）時，同排程組內前三名已與最終一致（5 個裡對 4 個）。
#   等比例縮放包含完整收割段，理應比截斷更好。
#
# 設計（tools/make_proxy_config.py，s = 0.2）：
#   單一旋鈕。所有帶步數的參數同乘 s，包含 **interval**（縮 interval 才保得住事件次數：
#   族群軌跡是事件空間的 x1.05 / x0.9，事件次數一樣則軌跡一樣，破平衡關係也不變）。
#   實測事件次數 273 -> 273 / 193 -> 193，撞 cap 的位置同樣落在 densify 期的 73%。
#   ⚠ 解析度與 cap **刻意不動** —— 降解析度會直接刪掉高頻內容，而我方病灶就是細節與尾巴，
#     那會系統性低估「修細節」類的機制；降 cap 會離開 cap-bound regime（顆數槓桿在 2.34M
#     已飽和，1.3M 會回到顆數還有用的區間），兩者都會造成誤判。
#
# ⚠ 已知且**故意不補償**的兩個失真（補償要多加旋鈕，失敗時就無法歸因）：
#   1. LR 積分變 0.2 倍 => proxy 的絕對分數一定較低。**proxy 是篩子不是成品。**
#   2. MCMC noise 的總擴散量變 0.2 倍。
#   這兩個正是本實驗要回答的對象。
#
# 五臂＝直接縮放**各跑次自己留下的 resolved config**（不是從 task 腳本重建），
# 所以 proxy 臂之間的差異與全長臂之間的差異逐鍵相同。已複核：
#   px_noprior_b12       (參考臂)                    全長 26.050
#   px_sched30_b12       densify_until 8400->6000    全長 26.377   +0.327  排程
#   px_sh3_nonormal_b12  depth_loss 0->0.5           全長 25.828   -0.222  幾何先驗
#   px_dssim05_b12       + lambda_dssim 0.2->0.5     全長 25.519   -0.531  loss 權重
#   px_dssim08_b12       depth 0.5 + dssim 0.8       全長 25.025   -1.025  loss 權重
# 效應量 +0.327 / -0.222 / -0.531 / -1.025，其中兩個落在 0.2~0.35 的難帶
# ——那才是日常真正要分辨的量級（噪音底 0.0325）。
#
# 判準（`python tools/proxy_verdict.py`）：
#   五臂名次全對 且 0.222 那對分得開   => proxy 成立，之後所有篩選用它（9.6h -> 1.9h）
#   名次大致對但小效應分不開           => 只能篩大效應，小效應仍需全長
#   名次亂                             => 時間壓縮破壞排名，回頭考慮別的省時法（降解析度等）
#
# ⚠ 判準一律多指標（PSNR/SSIM/LPIPS/紋理比/最差10%），不可只看 PSNR。
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

ARMS="noprior_b12 sched30_b12 sh3_nonormal_b12 dssim05_b12 dssim08_b12"
BLK=12
MAXSTEP=12000

for ARM in $ARMS; do
  NAME=px_$ARM
  CFG=configs/proxy/px_$ARM.yaml
  D=outputs/$NAME/blocks/block_$BLK

  if [ ! -f "$CFG" ]; then
    echo "!! 缺 $CFG —— 先跑 tools/make_proxy_config.py。中止。"; exit 1
  fi

  # 可重入：已經跑到底的臂直接跳過（不刪、不重跑）
  DONE=$(find "$D/checkpoints" -name '*.ckpt' 2>/dev/null \
         | sed -n 's/.*step=\([0-9]*\)\.ckpt/\1/p' | sort -n | tail -1)
  if [ -n "$DONE" ] && [ "$DONE" -ge $((MAXSTEP - 100)) ]; then
    echo "== $NAME 已完成 (step=$DONE)，跳過"
    continue
  fi
  # 有殘骸但沒跑完 -> 只刪自己這一臂的目錄（名字與 -n 同源，不用變數拼接別的路徑）
  if [ -d "outputs/$NAME" ]; then
    echo "== $NAME 有未完成的殘骸 (step=${DONE:-none})，移除後重跑"
    rm -rf "outputs/$NAME"
  fi

  echo "===== $NAME  ($MAXSTEP 步，預估 ~1.9h) ====="
  conda run -n gspl --no-capture-output python -u main.py fit --config "$CFG" -n "$NAME" || {
    echo "!! $NAME 失敗，繼續下一臂（缺臂的結果由 proxy_verdict.py 標示）"; continue; }
  bash scripts/task_test.sh "$NAME" "$BLK" || true
done

echo "===== 全部完成，開獎 ====="
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/proxy_verdict.py || true
