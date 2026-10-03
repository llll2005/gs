#!/usr/bin/env bash
# 印出任一支腳本的說明（2026-10-03）。由各腳本的 `-h／--help` 呼叫，也可以直接用：
#   bash scripts/_help.sh <腳本路徑>
# 內容**全部從該腳本本身讀出來**，不另外維護一份（另外維護的說明一定會過期 —— 舊版指令手冊就是這樣）：
#   ① 檔頭的標題行（第一行註解）
#   ② 檔頭的「用法」區塊：從 `# 用法` 那行開始，到第一個空註解行或非註解行為止
#   ③ 若有 `case "$ARM" in` / `case "$MODE" in`：逐項列出選項，附上緊鄰其上的第一行註解當說明
#   ④ 腳本認得的環境變數（`${CITYGS_...:-}` 之類有預設值的寫法）
set -u
f=${1:?用法: bash scripts/_help.sh <腳本路徑>}
[ -r "$f" ] || { echo "⛔ 讀不到 $f"; exit 2; }
name=${f#./}

title=$(awk 'NR>1 && /^#/ {sub(/^#[[:space:]]*/,""); if($0!=""){print; exit}} NR>1 && !/^#/ {exit}' "$f")
echo "$name —— ${title:-（檔頭沒有說明）}"
echo

# ② 用法區塊（可能有好幾行）
usage=$(awk '
  /^#[[:space:]]*(用法|Usage|usage)/ {on=1}
  on && /^#[[:space:]]*$/ {exit}
  on && !/^#/ {exit}
  on {sub(/^#[[:space:]]?/,""); print}' "$f")
if [ -n "$usage" ]; then
  echo "$usage"
  echo
fi

# ③ 選項（臂／模式）：只認兩格縮排的 case 標籤（巢狀 case 的縮排更深，不會被誤收）
#   臂：列出該臂實際加的覆寫旗標（從 case 本體抽 `--xxx 值`；變數照原樣顯示）——
#   不用上方註解當說明，因為註解區塊常常描述的是相鄰的另一臂（實測會配錯）。
if grep -q '^case "\$ARM" in' "$f"; then
  echo "臂（第二個參數；自動列自本檔的 case，右欄＝該臂加的覆寫）："
  awk '
    function flush() {
      if (lab == "") return
      gsub(/--model\./, "", fl); gsub(/init_args\./, "", fl); gsub(/[[:space:]]+/, " ", fl)
      if (fl == "" || fl == " ") fl = "（無，＝共同旗標）"
      printf "  %-24s %s\n", lab, fl; lab = ""; fl = ""
    }
    /^case "\$ARM" in/ {on=1; next}
    on && /^esac/ {flush(); exit}
    !on {next}
    /^[[:space:]]*#/ {next}
    depth == 0 && match($0, /^  [A-Za-z0-9_|]+\)/) {
      flush(); lab = substr($0, 3, RLENGTH-3); gsub(/\|/, " ", lab); body = substr($0, RLENGTH+1)
    }
    depth > 0 || !match($0, /^  [A-Za-z0-9_|]+\)/) { body = $0 }
    lab != "" {
      line = body
      while (match(line, /--[A-Za-z0-9_.]+[[:space:]]+(\$\(\([^)]*\)\)|[^[:space:])]+)/)) {
        fl = fl " " substr(line, RSTART, RLENGTH); line = substr(line, RSTART + RLENGTH)
      }
      if (match(body, /P="[^"]+"/)) fl = fl " init=" substr(body, RSTART+3, RLENGTH-4)
      n = gsub(/(^|[[:space:];(])case[[:space:]]/, "&", body); m = gsub(/(^|[[:space:];])esac([[:space:];]|$)/, "&", body)
      depth += n - m
    }' "$f"
  echo
fi
if grep -q '^case "\$MODE" in' "$f"; then
  modes=$(awk '/^case "\$MODE" in/ {on=1; next} on && /^esac/ {exit}
       on && match($0, /^  [A-Za-z0-9_|]+\)/) {l=substr($0,3,RLENGTH-3); gsub(/\|/," ",l); printf "  %s", l}' "$f")
  if [ -n "$modes" ]; then
    echo "模式（第一個參數；自動列自本檔的 case）："
    echo "$modes" | fold -s -w 100
    echo
  fi
fi

# ④ 環境變數（有預設值的 ${VAR:-...}／${VAR-...}／${VAR:?...}）
envs=$(grep -oE '\$\{(CITYGS_[A-Z0-9_]+|STEPS|NCAM|B0|RUN_PREFIX|RUNNER_[A-Z_]+|JTOK|LAB[A-Z]*|T)(:?-|:\?)' "$f" \
       | sed -E 's/^\$\{//; s/(:?-|:\?)$//' | sort -u | tr '\n' ' ')
[ -n "$envs" ] && { echo "認得的環境變數：$envs"; echo; }

case "$name" in
  *task_cmp*.sh|*task_speed3*.sh|*task_elong*.sh|*task_costbudget*.sh)
    echo "乾跑（只印出完整的訓練指令、不執行）：CITYGS_DRY=1 bash $name <塊> <臂>" ;;
esac
