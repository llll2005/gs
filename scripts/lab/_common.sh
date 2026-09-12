# lab 跑次的共同設定。source 它，不要直接執行。
#
# ★ 命名慣例（使用者 2026-09-13 要求）：**跑次名不帶塊編號**，因為
#   `outputs/<name>/blocks/block_N/` 的子資料夾本來就分開了。
#   例：`-n lab/speed3` + `--data.parser.block_id 6`
#       -> outputs/lab/speed3/blocks/block_6/
#   這樣同一個配方的多個塊會收在同一個 run 目錄下，合併/比較都方便。
# ★ `CITYGS_VRAM_CAP_GB=5.66`：3090 有 24GB，但本專案整篇的立論是**6GB 可行性**。
#   不鎖回信封的數字對命題沒有意義（記憶 lab_machine_workflow）。
#   同時這也是「3 個平行槽」能安全共存的前提（3 x 6.8GB < 24GB）。
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
# ⚠ 用 ${VAR-default}（**沒有冒號**）：這樣「明確設成空字串」會被尊重
#   => 本機（原生 6GB）可以用 `CITYGS_VRAM_CAP_GB= bash ...` 關掉上限，
#      而 lab 不帶這個變數時仍然自動鎖 5.66。
export CITYGS_VRAM_CAP_GB=${CITYGS_VRAM_CAP_GB-5.66}
[ -z "${CITYGS_VRAM_CAP_GB:-}" ] && unset CITYGS_VRAM_CAP_GB
echo "CITYGS_VRAM_CAP_GB=[${CITYGS_VRAM_CAP_GB:-未設 => 不限制（本機原生 6GB）}]"
CFG=configs/mcmc_2dgs_60k_sh3_aggr17_aerial.yaml
# ★ 跑次名的前綴。lab 用 `lab/` 讓產物收在 outputs/lab/ 底下（那裡不適用
#   「都是 6GB 跑的」鐵律）；**本機要設成空的** => outputs/<配方>/blocks/block_N/
#   用法：RUN_PREFIX= CITYGS_VRAM_CAP_GB= bash scripts/lab/task_speed3.sh 13
RUN_PREFIX=${RUN_PREFIX-lab/}
run_fit () {   # run_fit <run_name> <block_id> [額外參數...]
  local name="$1" blk="$2"; shift 2
  echo "=== $name / block $blk / cap ${CITYGS_VRAM_CAP_GB:-不限制} / $(date) ==="
  conda run -n gspl --no-capture-output python -u main.py fit \
    --config "$CFG" \
    --data.parser.block_id "$blk" \
    -n "$name" "$@"
  local rc=$?
  [ "$rc" -ne 0 ] && { echo "❌ $name block $blk 失敗 rc=$rc"; return "$rc"; }
  echo "=== $name / block $blk 完成 $(date) ==="
  conda run -n gspl python tools/audit_geometry.py "$name" --block "$blk" 2>&1 | tail -6
  return 0
}
