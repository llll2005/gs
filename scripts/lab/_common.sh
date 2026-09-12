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
#   「都是 6GB 跑的」鐵律）；本機必須是空的 => outputs/<配方>/blocks/block_N/
# ⚠⚠ 2026-09-13 踩到：預設寫死 `lab/`，而本機用
#   `CITYGS_VRAM_CAP_GB= bash scripts/lab/task_speed3.sh 13` 啟動時忘了帶 `RUN_PREFIX=`
#   => **本機跑次寫進了 outputs/lab/**。那裡是「outputs/ 都是這台 6GB 跑的」鐵律的
#   唯一例外區，來源一旦混進去就再也分不出是哪台機器跑的 ⇒ 分數作廢級的汙染。
#   ⇒ 改成**從路徑判機器**（lab 的 repo 在 /hdd/11213/gs），不再依賴呼叫端記得帶變數。
#   ⚠⚠ 判準第一版寫成 `case $(pwd) in /hdd/*)` —— **錯的，而且是反向汙染**：
#      `hdd/11213/gs` 是 **Jupyter contents API 的相對路徑**，lab 容器裡的 Jupyter
#      根目錄是 /workspace/data，所以 repo 的 pwd 其實是
#      `/workspace/data/hdd/11213/gs`，不會命中 `/hdd/*`
#      => lab 的跑次會被當成本機的，寫進 outputs/<配方>/，把 3090 的結果混進
#         「都是 6GB 跑的」命名空間。⇒ 以**標記檔**為主判準，路徑只當退路。
if [ -f .lab_machine ]; then
  _DEF_PREFIX=lab/
else
  case "$(pwd)" in
    */hdd/11213/*) _DEF_PREFIX=lab/ ;;
    *)             _DEF_PREFIX= ;;
  esac
fi
RUN_PREFIX=${RUN_PREFIX-$_DEF_PREFIX}
echo "RUN_PREFIX=[${RUN_PREFIX:-（空）=> 本機命名空間 outputs/<配方>/}]"
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
  # ⚠ 2026-09-13：`internal/cli.py:101` 斷言輸出目錄不存在，而前幾輪失敗的跑次
  #   留下了殘留（都有 step=499 的 ckpt）=> 重排同一個任務會
  #   `AssertionError: checkpoint or point cloud output already exists`。
  #   ⛔ 判準**不能**用「有沒有 ckpt」——失敗的殘留也有（499 步就存一次）。
  #   ⇒ **永不刪除，一律搬走**：舊的改名成 `<dir>.aborted_<時間>`，
  #     這樣失敗殘留與跑完的跑次都不會丟，而斷言也過得去。
  local out="outputs/$name/blocks/block_$blk"
  # ⚠⚠ 2026-09-13 本機雙開（第二次了）：使用者手動啟動了一支，runner 又排了同一支
  #   => 兩支都在 6GB 上長 N、**寫同一個輸出目錄**，其中一支在 step ~6,400 死掉，
  #      白燒 1h42m，而且早期 ckpt 是兩支同時寫的（有撕裂風險）。
  #   ⇒ 用 flock 從根上擋掉。鎖檔放 logs/locks/，**不能放輸出目錄裡**
  #      —— 下面的 move-aside 會把整個目錄搬走，鎖就跟著飛了。
  #   ★ fd 8 故意**讓子行程繼承**：今天死的是外層 bash 而 python 活著，
  #     繼承才能讓「孤兒訓練還在跑」也擋住重複啟動。要查是誰佔著：fuser -v <鎖檔>
  mkdir -p logs/locks
  local lk="logs/locks/$(printf '%s' "${name}_block_${blk}" | tr '/' '_').lock"
  exec 8>"$lk" || { echo "⛔ 開不了鎖檔 $lk"; return 9; }
  if ! flock -n 8; then
    echo "⛔ 已經有行程在跑 $name / block $blk（鎖 $lk）=> 放棄，不重複開"
    echo "   要查是誰：fuser -v $lk"
    return 9
  fi
  if [ -d "$out" ]; then
    local moved="${out}.aborted_$(date +%m%d_%H%M%S)"
    mv "$out" "$moved" || { echo "⛔ 搬不動 $out"; return 4; }
    echo "  舊輸出搬到 $moved（$(du -sh "$moved" 2>/dev/null | cut -f1)）"
  fi
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
