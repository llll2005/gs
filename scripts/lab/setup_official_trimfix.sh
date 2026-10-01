#!/bin/bash
# 2026-10-01：第二條官方參考線＝「官方程式碼＋一行修正，讓 trim 真的執行」（使用者要求，先跑一塊）。
#
# 為什麼要改：官方 internal/gaussian_splatting.py 的 `_initialize_from_trained_model` 從 coarse ckpt 初始化時，
#   **一律**把 renderer 換成 ckpt 裡的那個（`self.renderer = renderer`）；coarse 的 renderer 設 diable_trimming: true
#   => 微調 config 寫的 trim 設定被蓋掉，官方 aerial 流程的 trim 從未執行（2026-09-17 查到）。
# 修正（唯一改動）：overwrite_config 為 False（官方微調的設定）時保留 config 的 renderer、只取 ckpt 的權重。
#   與我方 CITYGS_KEEP_CFG_RENDERER=1 的語意相同，但改在官方程式碼上、用官方套件跑。
# 做法：git worktree（與 cityGS_origin 共用 .git，cityGS_origin 本身一行不動）；驗證 diff 恰好只有這一處。
set -u
ORIG=/workspace/data/hdd/11213/cityGS_origin
FIX=/workspace/data/hdd/11213/cityGS_origin_trimfix
GS=/workspace/data/hdd/11213/gs
cd "$ORIG" || exit 1
if [ ! -d "$FIX" ]; then
  git worktree add --detach "$FIX" HEAD || exit 2
fi
cd "$FIX" || exit 1
F=internal/gaussian_splatting.py
python3 - "$F" <<'PY' || exit 3
import sys
p = sys.argv[1]; s = open(p).read()
old = "        # call for renderer\n        self.renderer = renderer\n"
new = ("        # call for renderer\n"
       "        # [2026-10-01 一行修正] 不覆寫 config 時保留 config 的 renderer（官方原碼一律換成 ckpt 的 =>\n"
       "        #   coarse 的 diable_trimming 蓋掉微調 config 的 trim 設定，trim 從未執行）\n"
       "        if self.hparams[\"overwrite_config\"]:\n"
       "            self.renderer = renderer\n")
if new in s:
    print("（修正已套用）")
else:
    assert s.count(old) == 1, f"找不到要改的那一段（出現 {s.count(old)} 次）"
    open(p, "w").write(s.replace(old, new)); print("✅ 已套用一行修正")
PY
[ -e data ] || ln -s "$GS/data" data
[ -e utils/Depth-Anything-V2 ] || ln -s "$ORIG/utils/Depth-Anything-V2" utils/Depth-Anything-V2
echo "── 與官方 HEAD 的差異（必須只有 $F 一個檔、只有這一段）──"
git diff --stat | cat
n=$(git diff --name-only | wc -l)
[ "$n" -eq 1 ] && [ "$(git diff --name-only)" = "$F" ] || { echo "⛔ 差異不只一個檔"; exit 4; }
git diff | cat
echo "✅ cityGS_origin_trimfix 就緒（$(git rev-parse --short HEAD) + 一行修正）"
