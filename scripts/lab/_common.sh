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
# ⚠⚠ **執行期**也要鎖 TORCH_CUDA_ARCH_LIST，不只建置期（2026-09-13 在 lab 實測）：
#   容器把它設成 `7.5 8.0 8.6 9.0 10.0 12.0+PTX`，torch 2.0.1 不認識 10.0/12.0；
#   而 **gsplat 在第一次 densify（step ~1,049）才 JIT 編譯 CUDA 後端** =>
#   前 1,048 步完全正常，然後 `ValueError: Unknown CUDA arch (10.0)` 當場 DIED。
#   用 `nvidia-smi --query-gpu=compute_cap` 直接問（回 `8.6`），不要巢狀 conda run。
if command -v nvidia-smi >/dev/null 2>&1; then
  _A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
  if [ -n "$_A" ]; then
    export TORCH_CUDA_ARCH_LIST="$_A"
  else
    echo "⚠ 問不到 compute_cap，沿用環境變數"
  fi
fi
# 守門：值裡若還含 torch 2.0.1 不支援的 arch，**立刻失敗**而不是跑到 step 1,049 才死
case "${TORCH_CUDA_ARCH_LIST:-}" in
  *10.0*|*11.0*|*12.0*)
    echo "⛔ TORCH_CUDA_ARCH_LIST=[$TORCH_CUDA_ARCH_LIST] 含 torch 2.0.1 不支援的 arch"
    echo "   => 會在第一次 densify（gsplat JIT）當場死，現在就停下來"
    exit 3 ;;
esac
echo "TORCH_CUDA_ARCH_LIST=[${TORCH_CUDA_ARCH_LIST:-未設}]"
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
