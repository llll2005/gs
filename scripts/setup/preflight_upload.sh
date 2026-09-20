#!/usr/bin/env bash
# 上傳到 GitHub **之前**跑這一支：檢查「clone 到新環境之後跑得起來嗎」。
#
# 用法:
#   bash scripts/setup/preflight_upload.sh                # 只report，不動任何東西
#   bash scripts/setup/preflight_upload.sh --fix-ignores  # 修白名單式 .gitignore（不 commit）
#   bash scripts/setup/preflight_upload.sh --fix-rasterizer  # 把光柵器原始碼收進本 repo
#   bash scripts/setup/preflight_upload.sh --fix          # 兩個都做
#
# 為什麼需要：本 repo 有兩類**靜默**的缺漏，`git status` 完全看不出來。
#   ① 上游用**白名單式 .gitignore**（`*` 然後一行一個 `!檔名`）
#      => 我們後來新增的每個檔都被 `git add -A` 靜默忽略，包含現行主線唯一在用的 config。
#      （tests/.gitignore 在 2026-08-06 已因同一個坑修過，其餘目錄還沒。）
#   ② 我方改過的光柵器是個 **gitlink**，.gitmodules 沒有它的條目，
#      它的 origin 指向上游（推不上去）且與我方歷史**沒有共同祖先**
#      => clone 之後那個目錄是空的，ABSGRAD / EXACT_SUPPORT 整組配方無法重現。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FIX_IGN=0; FIX_RAST=0
RAST=submodules/diff-surfel-rasterization-trim-pp
while [ $# -gt 0 ]; do
  case "$1" in
    --fix) FIX_IGN=1; FIX_RAST=1; shift;;
    --fix-ignores) FIX_IGN=1; shift;;
    --fix-rasterizer) FIX_RAST=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "不認得的參數: $1"; exit 2;;
  esac
done
say()  { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '  ✅ %s\n' "$*"; }
bad()  { printf '  ❌ %s\n' "$*"; PROBLEMS=$((PROBLEMS+1)); }
warn() { printf '  ⚠  %s\n' "$*"; }
PROBLEMS=0

# ── 1. 白名單式 .gitignore ────────────────────────────────────────────────────
say "1. 白名單式 .gitignore（第一行就是 \`*\` 的目錄）"
WL=$(find . -name .gitignore -not -path "./.git/*" -not -path "./submodules/*" 2>/dev/null \
     | while read -r g; do
         first=$(grep -vE '^\s*#|^\s*$' "$g" | head -1)
         [ "$first" = "*" ] && echo "$g"
       done)
if [ -z "$WL" ]; then ok "沒有"; else
  for g in $WL; do
    d=$(dirname "$g")
    n=$(git status --porcelain --ignored "$d" 2>/dev/null | grep -c "^!!")
    printf "  %-40s 被忽略 %s 項\n" "$g" "$n"
  done
fi

# ⚠ scripts/queue.txt 是**逐機器的可變狀態**不是原始碼，刻意不追蹤：
#   ① runner.sh:226 已容錯（檔案不存在就 idle）=> clone 下來不會壞
#   ② 追蹤它的話，一次 pull 就會覆蓋掉另一台正在跑的佇列
say "2. 被忽略、但看起來是「原始碼」的檔（clone 之後會直接消失）"
MISSING=$(git status --porcelain --ignored 2>/dev/null | sed -n 's/^!! //p' \
  | grep -vE '^(outputs|logs|data|參考論文|\.claude)/' \
  | grep -vxF 'scripts/queue.txt' \
  | grep -vE '__pycache__|\.pyc$|\.egg-info|\.ckpt$|\.ply$|\.pth$|\.npy$|\.pt$' \
  | grep -E '\.(py|yaml|yml|sh|md|h|cu|cpp|txt|csv|json)$' | sort)
if [ -z "$MISSING" ]; then ok "沒有"; else
  echo "$MISSING" | sed 's/^/  ❌ /'
  bad "共 $(echo "$MISSING" | wc -l) 個原始碼檔不會被上傳"
fi

say "3. 現行主線實際用到的檔，有沒有全部進 git"
NEED=$(grep -ho "configs/[A-Za-z0-9_./-]*\.ya\?ml" scripts/*.sh scripts/task_*.sh 2>/dev/null | sort -u)
NEED="$NEED $(grep -rhoE "internal/(models|renderers|density_controllers|metrics)/[a-z0-9_]+\.py" CLAUDE.md 紀錄/完整指令手冊.md 2>/dev/null | sort -u)"
BADN=0
for f in $NEED; do
  [ -f "$f" ] || continue
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { bad "$f （腳本/文件有引用，但沒進 git）"; BADN=$((BADN+1)); }
done
[ "$BADN" = 0 ] && ok "都在"

say "4. 我方改過的 Trim 光柵器"
if git ls-files -s "$RAST" 2>/dev/null | grep -q '^160000'; then
  bad "$RAST 是 gitlink（submodule 指標），但："
  grep -q "diff-surfel-rasterization-trim-pp" .gitmodules 2>/dev/null \
    && echo "     .gitmodules 有條目" || echo "     ❌ .gitmodules **沒有**它的條目 => clone 後是空目錄"
  if [ -d "$RAST/.git" ]; then
    HEADC=$(git -C "$RAST" rev-parse --short HEAD)
    ORIGIN=$(git -C "$RAST" remote get-url origin 2>/dev/null)
    echo "     本地 HEAD $HEADC，origin $ORIGIN"
    git -C "$RAST" merge-base HEAD origin/HEAD >/dev/null 2>&1 \
      && echo "     與 origin 有共同祖先" \
      || echo "     ❌ 與 origin **沒有共同祖先** => 那個 commit 不存在於任何公開遠端"
  fi
  echo "     => 修法：本腳本加 --fix-rasterizer（把 20 個原始檔收進本 repo；glm 仍走公開 submodule）"
elif [ -f "$RAST/cuda_rasterizer/auxiliary.h" ] && git ls-files --error-unmatch "$RAST/cuda_rasterizer/auxiliary.h" >/dev/null 2>&1; then
  ok "原始碼已收進本 repo（$(git ls-files "$RAST" | wc -l) 個檔）"
  grep -q "ABSGRAD 1" "$RAST/cuda_rasterizer/auxiliary.h" && ok "ABSGRAD=1 在追蹤中的那份裡"
else
  bad "$RAST 狀態不明"
fi

say "5. 體積與不該上傳的東西"
# ⚠ .gitignore **不會**讓已經被追蹤的檔消失 —— 規則加得比 git add 晚就沒用。
DIRTY=0
for d in outputs logs data 參考論文; do
  n=$(git ls-files "$d" 2>/dev/null | wc -l)
  [ "$n" = 0 ] && continue
  sz=$(git ls-files -z "$d" 2>/dev/null | xargs -0 du -ch 2>/dev/null | tail -1 | cut -f1)
  bad "$d/ 有 $n 個檔被追蹤（共 $sz）—— .gitignore 寫了也沒用，規則加得比 git add 晚"
  echo "     移出索引（保留磁碟上的檔）： git rm -r --cached $d"
  DIRTY=1
done
[ "$DIRTY" = 0 ] && ok "outputs/ logs/ data/ 參考論文/ 都沒被追蹤"
BIG=$(git ls-files -z 2>/dev/null | xargs -0 -I{} sh -c '[ -f "{}" ] && du -k "{}"' 2>/dev/null \
      | awk '$1>5120{printf "  %7.1f MB  %s\n",$1/1024,substr($0,index($0,$2))}' | sort -rn | head -8)
if [ -n "$BIG" ]; then
  warn "追蹤中 >5MB 的**檔案**（GitHub >50MB 會警告、>100MB 直接拒絕）："
  echo "$BIG"
else ok "沒有 >5MB 的追蹤檔"; fi
HITS=$(git ls-files -z | xargs -0 grep -lE "/home/[A-Za-z0-9_]+/" 2>/dev/null | head -5)
[ -n "$HITS" ] && { warn "含絕對路徑 /home/... 的追蹤檔（新環境會對不上，多半只是註解/範例）："; echo "$HITS" | sed 's/^/     /'; }

# ── 修 ────────────────────────────────────────────────────────────────────────
if [ "$FIX_IGN" = 1 ] && [ -n "$WL" ]; then
  say "修 A：把白名單式 .gitignore 改成「白名單 PATTERN」"
  for g in $WL; do
    d=$(dirname "$g")
    case "$d" in ./tests|./tests/dataset) echo "  跳過 $g（已修過）"; continue;; esac
    cp "$g" "$g.bak"
    {
      echo "# 上游把這裡設成白名單（\`*\` 然後一行一個 \`!檔名\`），所以**後來新增的每個檔都被靜默忽略**"
      echo "# —— 現行主線唯一在用的 config 就是這樣差點沒上傳。改成白名單 PATTERN。"
      echo "# \`*\` 仍然擋住 __pycache__、.pyc 與丟進來的資料/輸出。"
      echo "*"
      echo "!.gitignore"
      echo "!*.py"
      echo "!*.yaml"
      echo "!*.yml"
      echo "!*.md"
      grep -E '^!' "$g.bak" | grep -vE '^!(\.gitignore|\*\.(py|yaml|yml|md))$'
    } > "$g"
    echo "  ✏ $g （舊檔留在 $g.bak）"
  done
  echo "  接著： git add -A && git status --short | head -40   # 確認多出來的正是那些原始碼"
fi

if [ "$FIX_RAST" = 1 ]; then
  say "修 B：把光柵器原始碼收進本 repo"
  if [ ! -d "$RAST/.git" ]; then
    echo "  已經不是巢狀 repo，跳過"
  else
    mv "$RAST/.git" "submodules/.rasterizer_upstream_git" \
      && echo "  ✏ 巢狀 .git 移到 submodules/.rasterizer_upstream_git（上游歷史留在本機，沒刪）"
    git rm --cached "$RAST" >/dev/null 2>&1 && echo "  ✏ 從索引移除 gitlink"
    rm -rf "$RAST/build"      # ⚠ 不要刪 third_party/glm：它是隨 repo 附帶的（見下方註解）
    grep -v "diff-surfel-rasterization-trim-pp" .gitmodules 2>/dev/null > .gitmodules.new \
      && mv .gitmodules.new .gitmodules
    echo "  ✏ .gitmodules 移除光柵器相關條目（glm 已隨 repo 附帶，不再是 submodule）"
    git add -f "$RAST" >/dev/null 2>&1
    echo "  ✏ 已 git add -f $RAST（$(git diff --cached --name-only -- "$RAST" | wc -l) 個檔）"
    echo "  ✅ third_party/glm 隨 repo 附帶（glm 1.0.3，MIT，433 檔 3.2MB）=> clone 下來就能編"
    echo "     2026-09-20 改的：原本這裡把它刪掉並宣告成 submodule，結果 clone 後是空目錄、編譯 fail"
    echo "     （fatal error: glm/glm.hpp: No such file or directory）"
  fi
fi

say "結果"
[ "$PROBLEMS" = 0 ] && echo "  ✅ 沒有已知的阻斷問題，可以 push" \
  || echo "  ❌ $PROBLEMS 個問題 —— 現在 push 的話，clone 到新環境**跑不起來**"
echo "  push 之後在新機器上： bash scripts/setup/bootstrap.sh"
exit $([ "$PROBLEMS" = 0 ] && echo 0 || echo 1)
