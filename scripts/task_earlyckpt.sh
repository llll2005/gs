#!/bin/bash
# ★ 尾巴視角「有沒有曾經是好的」？—— 最便宜的診斷（~3 分鐘 GPU）
#
# 研究總覽 §12.29：修好最差 25% 的視角值 +1.551 dB（是 sched30 的 4.7 倍、超過到 27.23 的缺口），
# 但六個視角層級的假設全被否證（覆蓋度/約束數/相機高度距離傾角/內容複雜度/GT 對應）。
# 關鍵矛盾：**訓練損失本來就對誤差最大的視角給最大梯度，它卻修不好** => 不是分配問題。
#
# 這支在 sched30_b12 的**早期 ckpt** 上跑 test，看尾巴視角是「從頭就爛」還是「曾經好過後來壞掉」：
#   從頭就爛 -> 初始化局部極小（3DGS 論文：init 不良的 floater cannot be removed by optimization）
#              => 下一步是換 init（SfM-init 零成本，`--model.initialize_from null`）
#   曾經好過 -> 是 densify/trim 把它弄壞的 => 下一步是保護那些區域不被剪
set -u
cd "$(dirname "$0")/.." || exit 1
D=outputs/sched30_b12/blocks/block_12
CFG=$(ls -t $D/lightning_logs/version_*/config.yaml | head -1)
for S in 14999 29999 41999; do
  CK=$(ls $D/checkpoints/*step=$S.ckpt 2>/dev/null | head -1)
  [ -z "$CK" ] && { echo "缺 step=$S"; continue; }
  echo "=== step $S ==="
  conda run -n gspl --no-capture-output python -u main.py test --config "$CFG" \
    --ckpt_path "$CK" --save_val 2>&1 | tail -3
  mkdir -p $D/test_early/step$S
  find $D/test -name "*.png" -newer "$CK" -exec mv {} $D/test_early/step$S/ \; 2>/dev/null
done
echo "早期 test 圖在 $D/test_early/"
