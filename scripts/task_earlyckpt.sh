#!/bin/bash
# ★ 尾巴視角「有沒有曾經是好的」？—— 便宜診斷（~10 分鐘 GPU）
#
# 研究總覽 §12.29：修好最差 25% 的視角值 +1.551 dB（sched30 的 4.7 倍、超過到 27.23 的缺口），
# 但六個視角層級的假設全被否證。關鍵矛盾：**訓練損失本來就對誤差最大的視角給最大梯度，
# 它卻修不好** => 不是分配問題。這支看尾巴視角是「從頭就爛」還是「曾經好過後來壞掉」：
#   從頭就爛 -> 初始化局部極小 => 下一步換 init（SfM-init 零成本 `--model.initialize_from null`）
#   曾經好過 -> densify/trim 弄壞的 => 下一步保護那些區域不被剪
#
# ⛔ 2026-08-21 重寫：第一版用 `find $D/test -newer $CK -exec mv` 搬圖，**把原本的 60k test 圖
#    一起搬走並同名覆蓋，毀掉 36 張**（60k 圖比 ckpt 新，條件命中）。
#    現在改成**按 main.py 自己產生的子目錄名**搬（`epoch=*-step=<S>`），完全不碰其他目錄。
set -u
cd "$(dirname "$0")/.." || exit 1
NAME=${1:-sched30_b12}; BLK=${2:-12}
D=outputs/$NAME/blocks/block_$BLK
CFG=$(ls -t $D/lightning_logs/version_*/config.yaml | head -1)
[ -z "$CFG" ] && { echo "no config for $NAME"; exit 1; }
cp -f $D/results.txt $D/results.txt.bak 2>/dev/null
mkdir -p $D/test_early
for S in 14999 29999 41999 60000; do
  CK=$(ls $D/checkpoints/*step=$S.ckpt 2>/dev/null | head -1)
  [ -z "$CK" ] && { echo "缺 step=$S，跳過"; continue; }
  SUB=$(ls -d $D/test/*step=$S 2>/dev/null | head -1)
  [ -n "$SUB" ] && [ "$(ls -A "$SUB" 2>/dev/null | wc -l)" -gt 0 ] && { echo "step=$S 已有圖，跳過"; continue; }
  echo "=== step $S ==="
  conda run -n gspl --no-capture-output python -u main.py test --config "$CFG" --ckpt_path "$CK" --save_val 2>&1 | tail -2
done
mv -f $D/results.txt.bak $D/results.txt 2>/dev/null
echo "各 step 的 test 圖："; for d in $D/test/*/; do echo "  $(basename $d): $(ls $d/*.png 2>/dev/null | wc -l) 張"; done
