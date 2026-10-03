#!/usr/bin/env bash
# 指令速查（2026-10-03）：寫指令前先看這裡。上半是常用流程（人工維護，只放「從哪裡查」），
# 下半是「支援 --help 的腳本」清單（自動掃描，不會過期）。各腳本的細節一律看它自己的 --help ——
# 那些說明是從腳本本身讀出來的（scripts/_help.sh），所以和實際行為一致。
# 用法：bash scripts/help.sh               速查＋腳本清單
#       bash scripts/help.sh <名稱片段>    直接看某支腳本的說明（例：bash scripts/help.sh task_cmp）
set -u
cd "$(dirname "$0")/.." || exit 1
if [ -n "${1:-}" ]; then
  case "$1" in -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac
  hits=$(ls scripts/*.sh scripts/lab/*.sh scripts/setup/*.sh 2>/dev/null | grep -F -- "$1")
  exact=$(printf '%s\n' $hits | awk -v n="${1%.sh}" '{b=$0; sub(/.*\//,"",b); sub(/\.sh$/,"",b)} b==n')
  [ -n "$exact" ] && hits=$exact
  n=$(printf '%s\n' "$hits" | grep -c .)
  if [ "$n" -eq 1 ]; then exec bash scripts/_help.sh "$hits"; fi
  py=$(ls tools/*.py scripts/setup/*.py 2>/dev/null | grep -F -- "$1")
  if [ "$n" -eq 0 ] && [ "$(printf '%s\n' "$py" | grep -c .)" -eq 1 ]; then
    echo "$py：";  python3 - "$py" <<'PY'
import ast, sys
d = ast.get_docstring(ast.parse(open(sys.argv[1], encoding="utf-8").read())) or "（沒有 docstring；試試 --help）"
print(d)
PY
    exit 0
  fi
  [ "$n" -eq 0 ] && { echo "⛔ 找不到含「$1」的腳本"; exit 2; }
  echo "符合「$1」的不只一支，請再具體一點："; printf '  %s\n' $hits; exit 2
fi

cat <<'TXT'
CityGaussian 指令速查（bash scripts/help.sh <名稱> 看單支；各腳本也都能直接加 --help）

紀錄       紀錄/README.md（入口）→ 紀錄/研究總覽.md §0 → 紀錄/實驗分析/NN_*.md＋同名圖；完整指令手冊在 紀錄/完整指令手冊.md

同步       git commit && git push，然後  JTOK=<token> bash scripts/setup/lab_pull.sh [--no-build]
           ⛔ token 只放在單一指令前面，不寫進任何檔案
lab 遠端   JTOK=<token> python scripts/setup/lab.py -h        子指令：run／q／bg／log／ps／ls／put／ckpts／get／pull／getfile
           JTOK=<token> python scripts/setup/lab.py q         看 lab 佇列（跑中的槽、待跑、最近完成）

佇列       bash scripts/runner.sh --help                      佇列行寫法（[cpu]／[solo]／標籤）、啟動／停止
           bash scripts/runner.sh status                      本機佇列現況

訓練（lab） bash scripts/lab/task_cmp.sh --help                對比骨架：臂清單（含每臂的覆寫旗標）與環境變數
           CITYGS_DRY=1 bash scripts/lab/task_cmp.sh <塊> <臂>   乾跑：只印完整訓練指令
           現行最佳：STEPS=60000 CITYGS_FAMILY=cs60_ bash scripts/lab/task_cmp.sh <塊> sfmdup4
官方線     bash scripts/lab/task_citygs_origin.sh --help      官方 CityGSV2 參考線（prep → coarse → 16 塊 → 合併 → held-out、量測）
舊結論重測 bash scripts/lab/task_recheck.sh --help            幾何／held-out／失敗區／空區雕刻／tau／coarse 閘門
速度驗證   bash scripts/lab/task_lean_check.sh --help ／ task_speed2_check.sh ／ task_tile_audit.sh

比較與量測
  python tools/run_status.py --runs <跑次...>                比分數前：確認跑完（results.txt 不寫步數）
  python tools/diff_resolved_config.py <A>[:塊] <B>[:塊]      比分數前：確認只差一個變數
  python tools/cmp_runs.py <基準> <臂...>                     最後 4 個 val 點平均的比較
  python tools/lab_cmp_report.py --blk <塊>                  lab 對比家族一覽（在 lab 上跑）
  bash scripts/task_load_compare.sh --help                   成本欄：同工具同相機量終點模型的 Load
  python tools/eval_official_test.py --ckpt <ckpt>           官方 741 幀 held-out
  python tools/plot_analysis.py                              重畫 紀錄/實驗分析/ 全部的圖
TXT

echo
echo "支援 --help 的腳本（自動掃描）："
for f in scripts/*.sh scripts/lab/*.sh scripts/setup/*.sh; do
  grep -q '_help.sh' "$f" 2>/dev/null || continue
  case "$f" in scripts/_help.sh|scripts/help.sh) continue ;; esac
  t=$(awk 'NR>1 && /^#/ {sub(/^#[[:space:]]*/,""); gsub(/[★⛔⚠]+[[:space:]]*/,""); if($0!=""){print; exit}} NR>1 && !/^#/ {exit}' "$f")
  printf '  %-40s %s\n' "$f" "$(printf '%s' "$t" | cut -c1-70)"
done
