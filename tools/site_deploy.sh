#!/usr/bin/env bash
# 展示網站部署到 GitHub Pages（2026-11-27）。
#   來源 outputs/site/：index.html 是 Claude Artifact 版（沒有 doctype／<head> 骨架，發佈時由 Artifact 包），加 v/ 影片、img/ 圖片。
#   這支補上骨架與分享預覽（OG 標籤、og.jpg）=> outputs/site_dist/，再推到 origin 的 gh-pages 分支。
#   gh-pages 每次都是一個孤兒 commit（-f 覆蓋），不累積影片歷史；main 完全不受影響。
#   第一次推完要在 GitHub：Settings → Pages → Build and deployment → Deploy from a branch → gh-pages / (root)。
#   用法：bash tools/site_deploy.sh        DRY=1 只建不推
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=outputs/site
DST=outputs/site_dist
URL=${SITE_URL:-https://llll2005.github.io/gs/}
PY=${PY:-/home/LnoArch/miniconda3/envs/gspl/bin/python}
[ -x "$PY" ] || PY=python3
[ -f "$SRC/index.html" ] || { echo "⛔ 找不到 $SRC/index.html"; exit 2; }

rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC/v" "$SRC/img" "$DST/"
"$PY" - "$SRC/index.html" "$DST" "$URL" <<'EOF'
import sys
from PIL import Image
src, dst, url = sys.argv[1:4]
s = open(src, encoding="utf-8").read()
i = s.index("</style>") + len("</style>")
desc = "一張 RTX 3090、1.33 小時重建整座 MatrixCity 空拍城市：飛行影片與測試視角對照（CityGaussianV2 對我們）。"
pre = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#3410c8">
<meta name="description" content="{desc}">
<meta property="og:type" content="website">
<meta property="og:title" content="單卡重建一座城市">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{url}img/og.jpg">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='32' r='30' fill='%233410c8'/%3E%3Ccircle cx='38' cy='26' r='14' fill='%235fdcff'/%3E%3C/svg%3E">
<style>:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}img{{max-width:100%}}</style>
"""
open(f"{dst}/index.html", "w", encoding="utf-8").write(pre + s[:i] + "\n</head>\n<body>\n" + s[i:] + "\n</body>\n</html>\n")
# 分享預覽圖 1200x630（LINE／Messenger 貼連結時顯示）
im = Image.open(f"{dst}/img/0179_ours.webp").convert("RGB")
w, h = im.size
ch = round(w * 630 / 1200)
im.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch)).resize((1200, 630), Image.LANCZOS).save(f"{dst}/img/og.jpg", quality=85)
print(f"建好 {dst}/index.html（{len(s)} 字元）＋ og.jpg")
EOF
touch "$DST/.nojekyll"
du -sh "$DST"
[ "${DRY:-0}" = "1" ] && { echo "DRY=1：只建不推"; exit 0; }

T=outputs/site_push_tmp
rm -rf "$T"; cp -r "$DST" "$T"
git -C "$T" init -q -b gh-pages
git -C "$T" add -A
git -C "$T" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" commit -q -m "展示網站 $(date '+%F %H:%M')"
git -C "$T" push -q -f "$(git remote get-url origin)" gh-pages:gh-pages
rm -rf "$T"
echo "✅ 已推到 gh-pages => $URL（GitHub 建置約 1～2 分鐘）"
