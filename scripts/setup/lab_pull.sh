#!/usr/bin/env bash
# lab 同步：本機 push -> lab git pull（2026-09-20 起改用這條，取代逐檔 put）
#
# 為什麼要有這支：lab 以前是用 `lab.py put` 逐檔覆蓋的，久了工作樹與 origin/main 差了
# 573 個檔、57 個未提交改動，HEAD 停在三週前 —— 「lab 跑的到底是哪一版」變成不可回答。
#
# 用法（JTOK 只能用環境變數帶，絕不寫進任何檔案）：
#     JTOK=xxx bash scripts/setup/lab_pull.sh
#     JTOK=xxx bash scripts/setup/lab_pull.sh --no-build    # 確定不用重編時
#
# 它會做的事：
#   1. 擋住「本機還有未提交/未 push 的改動」——否則 lab 拉到的不是你在測的東西
#   2. lab 端 fetch + reset --hard origin/main（lab 的本機狀態檔已在 .gitignore，不受影響）
#   3. 自動判斷光柵器原始碼或 glm 有沒有變，有才重編（重編要強制 CUDA_HOME，見下）
#   4. 回報兩邊的原始碼雜湊，對不上就非零結束
#
# ⚠ lab 重編必須強制 CUDA_HOME：容器預設指向系統 nvcc 12.9，而 gspl 的 torch 是 cu118，
#   torch 會直接丟 CUDA_MISMATCH 拒編。gspl 環境內自己就有 nvcc 11.8。
set -u
cd "$(dirname "$0")/../.." || exit 1
[ -n "${JTOK:-}" ] || { echo "⛔ 需要環境變數 JTOK"; exit 1; }
BUILD=1; [ "${1:-}" = "--no-build" ] && BUILD=0
RAST=submodules/diff-surfel-rasterization-trim-pp

# ── 1. 本機必須乾淨且已 push ────────────────────────────────────────────────
dirty=$(git status --porcelain | wc -l)
[ "$dirty" -eq 0 ] || { echo "⛔ 本機有 $dirty 個未提交改動 —— 先 commit，否則 lab 拉到的不是你在測的版本"; git status --short | head -10; exit 1; }
git fetch -q origin
L=$(git rev-parse HEAD); R=$(git rev-parse origin/main)
[ "$L" = "$R" ] || { echo "⛔ 本機 HEAD 與 origin/main 不同 —— 先 git push"; exit 1; }
echo "✅ 本機乾淨且已 push：$(git log --oneline -1)"

# ── 2~4. lab 端 ────────────────────────────────────────────────────────────
python scripts/setup/lab.py run "cd gs
OLD=\$(git rev-parse HEAD)
git fetch -q origin && git reset --hard -q origin/main || { echo '⛔ reset 失敗'; exit 1; }
NEW=\$(git rev-parse HEAD)
echo \"lab: \$(git log --oneline -1)\"
echo \"未提交改動：\$(git status --porcelain | wc -l)\"
if [ \"\$OLD\" = \"\$NEW\" ]; then echo '（lab 本來就是最新的）'; fi
NEEDBUILD=0
if [ \"\$OLD\" != \"\$NEW\" ] && ! git diff --quiet \$OLD \$NEW -- $RAST/cuda_rasterizer $RAST/third_party; then NEEDBUILD=1; fi
echo \"光柵器需要重編：\$NEEDBUILD（本次允許重編：$BUILD）\"
if [ \$NEEDBUILD = 1 ] && [ $BUILD = 1 ]; then
  G=/root/miniconda3/envs/gspl
  rm -rf $RAST/build
  conda run -n gspl env CUDA_HOME=\$G CUDA_PATH=\$G TORCH_CUDA_ARCH_LIST=8.6 \\
    pip install --no-build-isolation --no-deps --force-reinstall $RAST 2>&1 | tail -2
  conda run -n gspl python -c \"import diff_trim_surfel_rasterization as m,os,time;p=os.path.dirname(m.__file__);f=[x for x in os.listdir(p) if x.endswith('.so')][0];print('so:',f,time.strftime('%m-%d %H:%M',time.localtime(os.path.getmtime(os.path.join(p,f)))))\"
fi
echo 'HASH'
sha256sum $RAST/cuda_rasterizer/*.h $RAST/cuda_rasterizer/*.cu $RAST/third_party/glm/glm/glm.hpp | cut -c1-16,66-" > /tmp/.labpull.$$ 2>&1
sed -n '/^HASH$/,$p' /tmp/.labpull.$$ | tail -n +2 | tr -d '\r' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E '^[0-9a-f]{16} ' > /tmp/.labhash.$$
grep -vE '^\[\?2004|^\(base\)|^HASH$' /tmp/.labpull.$$ | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E 'lab:|未提交|重編|so:|⛔|本來就是' 

# ── 比對雜湊 ───────────────────────────────────────────────────────────────
bad=0
while read -r h f; do
  [ -f "$f" ] || { echo "  ⛔ 本機缺 $f"; bad=1; continue; }
  m=$(sha256sum "$f" | cut -c1-16)
  [ "$h" = "$m" ] || { echo "  ⛔ 不一致 $f  lab=$h 本機=$m"; bad=1; }
done < /tmp/.labhash.$$
n=$(wc -l < /tmp/.labhash.$$)
rm -f /tmp/.labpull.$$ /tmp/.labhash.$$
[ "$n" -gt 0 ] || { echo "⛔ 沒讀到 lab 的雜湊（上面的輸出可能有錯）"; exit 1; }
[ "$bad" = 0 ] && echo "✅ 光柵器原始碼與 glm 在兩台上逐位元相同（$n 個檔）" || exit 1
