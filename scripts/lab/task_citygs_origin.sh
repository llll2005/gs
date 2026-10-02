#!/bin/bash
# ⛔⛔ 2026-09-24：本條線改跑 **gspl_official**（`scripts/lab/setup_official_env.sh` 照官方
#   doc/installation.md 三行 requirements 建的），prep／coarse／partition／16 塊／merge／test **全部重跑**。
#   先前的全部作廢（資源與「是否官方」都不乾淨）：
#     - 09-22 的 coarse 跑在 **gspl**（我方光柵器）：lab 的 HEAD 停在 44c1a21，從沒 pull 到換環境的 commit
#     - gspl_origin 只有光柵器是官方的；其餘套件來自我方 lock 檔，diff_gaussian_rasterization/simple_knn 編自我方 submodules/
#     - 分區是 09-18 用更早的 coarse 做的；深度圖/1600 影像是在 gspl 裡、用 gs 的 Depth-Anything-V2 做的
#     - 09-23~24 的 block 0~8 是手動啟動，沒有資源紀錄
#   允許的例外（使用者 2026-09-24）：我方的 log／計數器。=> tools/peakmem/sitecustomize.py 經 PYTHONPATH
#     外掛（官方原始碼一行不改）：行程峰值＋每 100 batch 的 N／it/s／VRAM（logs/citygs_official_train_*.tsv）
#     ＋每次加顆/剪枝的 churn（logs/citygs_official_churn_*.tsv）＋我方台帳 logs/quad_progress.log 的 START/DONE/DIED（名稱 OFF:…）。
#   `res`（離線 Load／逐步計時）是**我方量測工具**，跑在 gspl，量的是官方產出的 ckpt。
# ⚠⚠ 2026-09-22：本條線一律跑在 **gspl_off** 環境（可用 CITYGS_OFF_ENV 覆寫）。
#   為什麼：官方參考線的用途之一是**資源對照**，但 `diff_trim_surfel_rasterization` 是裝在
#   **共用的 conda 環境**裡，而那一份是**我方改過的**：
#     EXACT_SUPPORT=1（渲染逐位元相同，但 render VRAM -20.3%、tile 數大降 => 比真官方更省）
#     ABSGRAD=1（渲染/梯度逐位元相同，但 backward 多一次 atomicAdd => 略慢）
#     rects 緩衝 +8 B/顆、tiles 張量 +4 B/顆（即使沒開 return_tiles 也會配置）
#   => **品質那一欄是乾淨的**（以上前三項都逐位元相同），但**時間與 VRAM 不是官方的數字**。
#   而且這個污染從 EXACT_SUPPORT 啟用起就存在，不是 2026-09-22 才有的。
#   ⇒ `gspl_origin` 裝的是**官方 requirements 自己指定的那一份**：
#     `git+https://github.com/DekuLiuTesla/diff-surfel-rasterization.git@9eefc03…`
#     （DekuLiuTesla ＝ CityGaussian 作者；出處 cityGS_origin/requirements 第 8 行）。
#   ⚠ 使用者 2026-09-22 明確否決了「拿我方 fork 關旗標」的折衷方案 —— 那樣仍殘留
#     rects/tiles 的 +12 B/顆與 conic 分支的 codegen，**達不到「乾淨」的目的**。
# ★★★ 官方 CityGaussianV2 的**完整**流程，跑在**未修改的 cityGS_origin**（使用者 2026-09-17 決定）。
#
# 為什麼要這一條：使用者 09-15/09-16 手動跑的兩次雖然用的是官方 config（lab 上那份是乾淨的），
# 但 ① 跑在我方 repo（程式碼有大量改動）② `block_id: null` => **沒有分塊**，是一個全域模型
# ③ 依 `internal/gaussian_splatting.py` 的 renderer 覆蓋，**沒有 trim** => 那不是官方方法的數字。
#
# 唯一的硬障礙已解：官方 parser 只把目錄名加後綴（images -> images_1.2）而**不在載入時縮放**
# （官方 internal/dataset.py 只有一行 `# TODO: resize`），而 images_1.2 是指向全解析度 input 的
# 符號連結 => 1920 撞 1600 深度圖。`prep` 用**官方自己的** utils/image_downsample.py 生成真影像。
# ⚠ 我方跑次不受影響：我方 config 明確設 `image_dir: input`，get_image_dir 直接回傳它，不加後綴。
#
# ⚠⚠ 2026-09-17 第一次跑 coarse 失敗（250 秒）：`_depth_l1_loss` 裡 GT 逆深度寬 1920、渲染 1600。
#   量過：共用的 estimated_depths/*.npy 是 **(1080, 1920)**（我方流程沒加 -d），我方 metric 會把
#   預測縮到 GT 大小所以能跑，官方沒有這一步。而官方流程本來就是 `estimate_dataset_depths.py -d 1.2`。
#   => 替官方線建**獨立資料目錄**（影像/sparse 用連結，深度圖/尺度檔/分區是它自己的真檔案），
#      官方執行只多 `--data.path` 指過去；**共用的 block_all/ 完全不動**。測試集同樣處理
#      （官方 parser 有 `assert loaded_depth_count > 0`，測試集沒有深度圖會直接失敗）。
#
# ★ 不限制 VRAM（使用者 2026-09-17 明確要求）：官方原始碼本來就不認得 CITYGS_VRAM_CAP_GB，
#   這裡仍明確 unset；也不設 PYTORCH_CUDA_ALLOC_CONF（與官方一致）。
# ★ 資源紀錄（方便與我方比較、找缺口），**不改官方任何一行、不影響跑分**：
#   行程內峰值  tools/peakmem/sitecustomize.py 經 PYTHONPATH 載入，只在行程結束時讀
#               torch.cuda.max_memory_allocated / max_memory_reserved（= 我方台帳的「峰值實佔／保留」）
#   整卡最大    每 20 秒 nvidia-smi memory.used；⚠ 只有 [solo] 的模式（coarse/test）才有歸屬意義
#   其他        牆鐘秒數、結束碼、N、ckpt 大小 => logs/citygs_origin_resources.tsv
#
# 用法：task_citygs_origin.sh prep|coarse|partition|merge|test|res ／ task_citygs_origin.sh block <N>
set -u
OFFENV=${CITYGS_OFF_ENV:-gspl_official}
ORIG=/workspace/data/hdd/11213/cityGS_origin
GS=/workspace/data/hdd/11213/gs
NAME=citygsv2_mc_aerial_sh2_trim
COARSE=citygsv2_mc_aerial_coarse_sh2
TRAIN=$GS/data/matrix_city/aerial/train/block_all
TEST=$GS/data/matrix_city/aerial/test/block_all_test
OTRAIN=$GS/data/matrix_city/aerial/train/block_all_official       # 官方線專用（見表頭）
# ⛔ 2026-09-29：舊的 block_all_test_official 建在**我方自製**的 test input 上 —— 那份是從 10 個 tar 自己串接、
#   用本機一份不明來源的清單重編號的：741 張裡只有 657 張是官方 test 影格（另 84 張不在官方 transforms 裡），
#   而且順序對 sparse 相機是分段錯開（+1、+9、+26…）。官方 held-out 因此量到 15.62（渲染對錯的 GT）。
#   證據（不經模型）：sparse 相機 N 的姿態 = transforms_test.json 第 N 幀（位置殘差 1.6e-6、朝向 2.5e-7，741 對 741 唯一）；
#   另以官方模型的 741 張渲染佐證：舊 GT 平均 17.1 dB、官方對應平均 28.7 dB（160x90 探針）。
#   ⇒ 改用 prep_test_official **照官方 data_proc_mc.sh 從 tar 重建**的目錄（不經我方的 input/）
OTEST=$GS/data/matrix_city/aerial/test/block_all_test_official2
OTRAIN_REL=data/matrix_city/aerial/train/block_all_official        # 相對 cityGS_origin（data -> gs/data）
[ -d "$ORIG" ] || { echo "⛔ 找不到 $ORIG"; exit 2; }
MODE=${1:?用法: task_citygs_origin.sh prep|prep_test_official|coarse|partition|block <N>|blockfix <N>|blockpaper <N>|mergepaper|testpaper|merge|test|res|blockload|prune}
BLKARG=${2:-}

# ⚠ 執行期也要鎖 arch：容器設成含 10.0，torch 2.0.1 不認識，而 CUDA 後端第一次 densify 才 JIT 編譯
_A=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')
[ -n "$_A" ] && export TORCH_CUDA_ARCH_LIST="$_A"
echo "TORCH_CUDA_ARCH_LIST=[${TORCH_CUDA_ARCH_LIST:-未設}]"
unset CITYGS_VRAM_CAP_GB PYTORCH_CUDA_ALLOC_CONF
echo "VRAM 上限：不限制（官方原始碼不認得 CITYGS_VRAM_CAP_GB；配置器維持官方預設）"
LOG=$GS/logs/citygs_official_${MODE}${BLKARG:+_$BLKARG}.log
RES_TSV=$GS/logs/citygs_official_resources.tsv
TRAINLOG=$GS/logs/citygs_official_train_${MODE}${BLKARG:+_$BLKARG}.tsv
CHURN=$GS/logs/citygs_official_churn_${MODE}${BLKARG:+_$BLKARG}.tsv   # 每次加顆/剪枝一行（phase=clone/split/cull/trim）
PEAK_DIR=$GS/tools/peakmem
[ -f "$PEAK_DIR/sitecustomize.py" ] || { echo "⛔ 缺 $PEAK_DIR/sitecustomize.py（峰值紀錄）"; exit 2; }
[ -f "$RES_TSV" ] || printf 'mode\tblock\tstart\twall_s\trc\tpeak_alloc_MiB\tpeak_reserved_MiB\tgpu_used_max_MiB\tN\tckpt\text_max_MiB\tgate_wait_s\n' > "$RES_TSV"
# 2026-09-27：舊表頭補兩欄（ext_max_MiB＝同卡外部佔用估計峰值；gate_wait_s＝閘門等待秒數）
head -1 "$RES_TSV" | grep -q ext_max_MiB || sed -i '1s/$/\text_max_MiB\tgate_wait_s/' "$RES_TSV"

# ── GPU 閘門（2026-09-27，使用者要求）────────────────────────────────────────────
# 09-26 起同一張 3090 上出現我們容器外的佔用（約 10~13 GB，容器內看不到是誰）⇒ b10/b11 OOM、
#   b10~b12 同 N 下每步慢 2~3 倍（時間紀錄不可用）。規則（使用者原話的實作）：
#   ① 開始前卡是空的（整卡已用 < GATE_FREE_MIB）⇒ 直接開始（例如上一塊剛跑完、沒有別人）
#   ② 被佔用 ⇒ 等它空下來，然後**連續**空閒滿 GATE_HOLD_S（預設半小時；09-27 前為 1 小時）才開始；
#      計時中只要又被佔用就歸零重等
# 「空閒」只看 nvidia-smi 整卡已用量 —— 我們自己的其他訓練也會算進去（[solo] 下本來就不該有）。
# ⚠ nvidia-smi 讀不到（空值/非數字）一律當「被佔用」：偵測失敗不可當成空閒（見記憶 local_nvidia_driver_mismatch）。
GATE_FREE_MIB=${CITYGS_GATE_FREE_MIB:-1500}
GATE_HOLD_S=${CITYGS_GATE_HOLD_S:-1800}   # 2026-09-27 使用者：1 小時 -> 半小時
GATE_WAIT=0
_gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'; }
_is_free () { case "$1" in ''|*[!0-9]*) return 1 ;; esac; [ "$1" -lt "$GATE_FREE_MIB" ]; }   # 同一次讀數判斷與印出
gpu_gate () {
  local st="$GS/logs/citygs_official_status_${MODE}${BLKARG:+_$BLKARG}.txt" t0 since="" now u
  t0=$(date +%s)
  u=$(_gpu_used)
  if _is_free "$u"; then echo "[閘門] GPU 空閒（$u MiB）=> 直接開始"; return 0; fi
  echo "[閘門] ⏸ GPU 被佔用（${u:-讀不到} MiB）=> 等它空下來，再連續空閒 $((GATE_HOLD_S / 60)) 分鐘才開始 $(date '+%m-%d %H:%M')"
  while true; do
    now=$(date +%s); u=$(_gpu_used)
    if _is_free "$u"; then
      [ -z "$since" ] && { since=$now; echo "[閘門] $(date '+%m-%d %H:%M') 空了（${u} MiB），開始計時"; }
      if [ $((now - since)) -ge "$GATE_HOLD_S" ]; then
        GATE_WAIT=$((now - t0))
        echo "[閘門] $(date '+%m-%d %H:%M') 連續空閒滿 $((GATE_HOLD_S / 60)) 分鐘 => 開始（共等 $((GATE_WAIT / 60)) 分鐘）"
        return 0
      fi
      printf '# 官方參考線現況（%s%s）  更新 %s\n狀態     ⏸ 閘門計時中：已連續空閒 %d / %d 分鐘（整卡 %s MiB）\n已等待   %d 分鐘\n' \
        "$MODE" "${BLKARG:+ $BLKARG}" "$(date '+%m-%d %H:%M:%S')" $(((now - since) / 60)) $((GATE_HOLD_S / 60)) "$u" $(((now - t0) / 60)) > "$st"
    else
      [ -n "$since" ] && echo "[閘門] $(date '+%m-%d %H:%M') 計時中斷：又被佔用（${u:-讀不到} MiB），歸零"
      since=""
      printf '# 官方參考線現況（%s%s）  更新 %s\n狀態     ⏸ 閘門等待中：GPU 被佔用（整卡 %s MiB，門檻 < %s）\n已等待   %d 分鐘\n' \
        "$MODE" "${BLKARG:+ $BLKARG}" "$(date '+%m-%d %H:%M:%S')" "${u:-讀不到}" "$GATE_FREE_MIB" $(((now - t0) / 60)) > "$st"
    fi
    sleep 60
  done
}

cd "$ORIG" || exit 1
if [ "$MODE" != res ]; then
  OKF=$(conda run -n "$OFFENV" python -c "import sys;print(sys.prefix)" 2>/dev/null | tr -d '\r')/.citygs_official_env_ok
  [ -f "$OKF" ] || { echo "⛔ $OFFENV 還沒通過 setup_official_env.sh 的驗證（缺 $OKF）"; exit 2; }
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "⛔ cityGS_origin 的追蹤檔被改過"; git status --short; exit 2; }
  echo "環境：$OFFENV（$OKF）  cityGS_origin $(git rev-parse --short HEAD)（追蹤檔零改動）"
fi
[ -e data ] || ln -s "$GS/data" data
move_aside () { if [ -d "$1" ]; then mv "$1" "$1.aborted_$(date +%m%d_%H%M%S)" && echo "  舊輸出搬到 $1.aborted_*"; fi; return 0; }
coarse_ckpt () { find "outputs/$COARSE/checkpoints" -maxdepth 1 -name '*step=30000.ckpt' 2>/dev/null | head -1; }

run_measured () {   # run_measured <指令...>：資源寫進 RUN_* 變數；完整輸出在 $LOG
  local pk="$GS/logs/.peak_${MODE}${BLKARG:+_$BLKARG}_$$.tsv" t0 u bg r ext
  rm -f "$pk"
  gpu_gate
  RUN_START=$(date '+%m-%d %H:%M'); t0=$(date +%s); RUN_GMAX=0; RUN_EXT=0
  [ -f "$TRAINLOG" ] && mv "$TRAINLOG" "$TRAINLOG.old_$(date +%m%d_%H%M%S)"
  [ -f "$CHURN" ] && mv "$CHURN" "$CHURN.old_$(date +%m%d_%H%M%S)"
  ( export CITYGS_PEAK_OUT="$pk" CITYGS_TRAINLOG_OUT="$TRAINLOG" CITYGS_TRAINLOG_EVERY=100 \
      CITYGS_CHURN_OUT="$CHURN" CITYGS_LEDGER="$GS/logs/quad_progress.log" \
      CITYGS_STATUS_OUT="$GS/logs/citygs_official_status_${MODE}${BLKARG:+_$BLKARG}.txt" PYTHONPATH="$PEAK_DIR${PYTHONPATH:+:$PYTHONPATH}"; exec "$@" ) > "$LOG" 2>&1 &
  bg=$!
  while kill -0 "$bg" 2>/dev/null; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
    case "$u" in ''|*[!0-9]*) ;; *) [ "$u" -gt "$RUN_GMAX" ] && RUN_GMAX=$u
      # 外部佔用估計 = 整卡已用 − 本行程目前保留（train TSV 最後一行第 7 欄）− CUDA context 約 450 MiB
      #   （b0~b9 乾淨時實測差額恆為 ~418 MiB）；只在訓練開始後才有意義
      r=$(tail -1 "$TRAINLOG" 2>/dev/null | cut -f7)
      case "$r" in ''|*[!0-9]*) ;; *) ext=$((u - r - 450)); [ "$ext" -gt "$RUN_EXT" ] && RUN_EXT=$ext ;; esac ;;
    esac
    sleep 20
  done
  wait "$bg"; RUN_RC=$?
  RUN_WALL=$(( $(date +%s) - t0 ))
  tail -30 "$LOG"
  RUN_PA=-; RUN_PR=-
  if [ -s "$pk" ]; then
    RUN_PA=$(sort -t"$(printf '\t')" -k2,2n "$pk" | tail -1 | cut -f2)
    RUN_PR=$(sort -t"$(printf '\t')" -k3,3n "$pk" | tail -1 | cut -f3)
  else
    echo "⚠⚠ 沒拿到行程內峰值（行程非正常結束？）"
  fi
  echo "[資源] rc=$RUN_RC 牆鐘 ${RUN_WALL}s 配置峰值 ${RUN_PA} MiB 保留峰值 ${RUN_PR} MiB 整卡最大 ${RUN_GMAX} MiB 外部佔用估計峰值 ${RUN_EXT} MiB"
  [ "$RUN_EXT" -gt 2000 ] && echo "⚠⚠ 訓練期間同卡有外部佔用（估計峰值 ${RUN_EXT} MiB）=> 這一次的時間紀錄不可引用"
  return "$RUN_RC"
}

record () {   # record <block 或 -> <目錄>：N（PLY 標頭，沒有就讀 ckpt）與 ckpt 大小，寫進 TSV
  local blk=$1 dir=$2 n=- sz=- ck ply
  ck=$(find "$dir" -name '*.ckpt' 2>/dev/null | sed -E 's/^(.*step=([0-9]+)[^/]*\.ckpt)$/\2 \1/' | sort -n | tail -1 | sed -E 's/^[0-9]+ //')
  if [ -n "$ck" ]; then
    sz=$(du -h "$ck" | cut -f1)
    ply="${ck%.ckpt}-xyz_rgb.ply"
    if [ -f "$ply" ]; then
      n=$(head -c 4000 "$ply" | grep -a -m1 'element vertex' | awk '{print $3}')
    else
      n=$(conda run -n "$OFFENV" python -c "
import sys, torch
sd = torch.load(sys.argv[1], map_location='cpu')['state_dict']
k = [k for k, v in sd.items() if (k.endswith('means') or k.endswith('xyz')) and getattr(v, 'ndim', 0) == 2 and v.shape[-1] == 3]
print(sd[k[0]].shape[0] if k else -1)" "$ck" 2>/dev/null | tail -1)
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$MODE" "$blk" "$RUN_START" "$RUN_WALL" "$RUN_RC" "$RUN_PA" "$RUN_PR" "$RUN_GMAX" "$n" "$sz" "${RUN_EXT:--}" "${GATE_WAIT:-0}" >> "$RES_TSV"
  echo "[資源] 已記入 $RES_TSV：N=$n ckpt=$sz"
}

case "$MODE" in
  prep)
    gpu_gate   # 深度估計用 GPU
    # 2026-09-24：影像與深度圖**全部在 $OFFENV 裡重做**，放進官方線專用目錄（不再連到共用 block_all/images_1.2）。
    #   完成標記 .made_by_$OFFENV；沒有標記的舊產物一律搬走重做。
    # ── test 的檔名層（2026-09-24）──────────────────────────────────────────────
    #   官方 test 的 sparse 相機叫 0000.png..0740.png（MatrixCity／官方原本就從 0 起算），
    #   而我方 lab_fetch_test_images.sh 把串好的 741 幀存成 **1 起算** 的 input/0001.png..0741.png
    #   （md5 驗過內容與配對正確，只是名字整體 +1；記憶「相機 N <-> 影像 N+1」就是這件事）。
    #   官方工具照相機名找檔 => 找不到 0000.png.npy 而失敗。
    #   ⇒ 官方線專用的 $OTEST/images 改成**真目錄＋改名連結**：images/0000.png -> input/0001.png …
    #     把我方多加的那一號還原回官方檔名；原始 input/ 與我方流程都不動。
    #   逐幀驗證：sparse 名稱集合必須恰好等於 {input 編號 - 1}，否則中止。
    # ⛔ 2026-09-29 停用：這個「改名層」建在我方自製的 test input 上，對應是錯的（見 OTEST 定義處）。
    #   test 集一律由 prep_test_official 產生；prep 只處理 train。
    if false; then
      # ⚠ --no-capture-output 不可省：沒有它 conda run **不轉送 stdin** => python 讀到空腳本、rc=0 什麼都沒做
      #   （2026-09-24 就是這樣：改名層沒建，下面的 touch 穿過符號連結把標記寫進了共用 input/）
      conda run -n "$OFFENV" --no-capture-output python - "$TEST" "$OTEST" <<'PY' || exit $?
import os, sys
sys.path.insert(0, os.getcwd())
from internal.utils.colmap import read_images_binary
src, od = sys.argv[1], sys.argv[2]
names = sorted(v.name for v in read_images_binary(os.path.join(src, "sparse", "0", "images.bin")).values())
inp = sorted(os.listdir(os.path.join(src, "input")))
want = {f"{int(os.path.splitext(f)[0]) - 1:0{len(os.path.splitext(f)[0])}d}{os.path.splitext(f)[1]}": f for f in inp}
assert len(names) == len(inp) == len(want), (len(names), len(inp))
miss = [n for n in names if n not in want]
assert not miss, f"sparse 名稱對不上 input-1：{miss[:5]}"
dst = os.path.join(od, "images")
if os.path.islink(dst):
    os.remove(dst)
os.makedirs(dst, exist_ok=True)
for n in names:
    l = os.path.join(dst, n)
    if os.path.lexists(l):
        os.remove(l)
    os.symlink(os.path.join(src, "input", want[n]), l)
print(f"✅ test 改名層：{len(names)} 幀，{names[0]} -> input/{want[names[0]]} … {names[-1]} -> input/{want[names[-1]]}")
PY
      # 用舊（錯名）影像做的 1600 影像與深度圖一律搬走重做
      move_aside "$OTEST/images_1.2"
      move_aside "$OTEST/estimated_depths"
      [ -f "$OTEST/estimated_depth_scales.json" ] && mv "$OTEST/estimated_depth_scales.json" "$OTEST/estimated_depth_scales.json.aborted_$(date +%m%d_%H%M%S)"
      [ -d "$OTEST/images" ] && [ ! -L "$OTEST/images" ] || { echo "⛔ $OTEST/images 仍不是真目錄（改名層沒建成）"; exit 3; }
      [ -L "$OTEST/images/0000.png" ] || { echo "⛔ 改名層缺 0000.png"; exit 3; }
      touch "$OTEST/images/.renamed_to_sparse"
    fi
    for pair in "$TRAIN:$OTRAIN"; do   # 09-29：test 改由 prep_test_official
      src=${pair%%:*}; od=${pair##*:}; mkdir -p "$od"
      L="$od/images_1.2"; N_SRC=$(ls "$src/input" | wc -l)
      if [ -f "$L/.made_by_$OFFENV" ]; then echo "  已由 $OFFENV 產生，略過 $L"; continue; fi
      if [ -L "$L" ]; then rm -f "$L"; echo "  移除符號連結 $L"; else move_aside "$L"; fi
      echo "=== 縮放 $src/input -> $L（$N_SRC 張，官方 image_downsample.py）$(date) ==="
      conda run -n "$OFFENV" --no-capture-output python utils/image_downsample.py "$od/images" --dst "$L" --factor 1.2 || exit $?
      [ "$(ls "$L" | wc -l)" -ge "$N_SRC" ] || { echo "⛔ $L 張數不足"; exit 3; }
      touch "$L/.made_by_$OFFENV"
    done
    conda run -n "$OFFENV" python -c "
from PIL import Image
import os
for d in ['$OTRAIN']:
    p = os.path.join(d, 'images_1.2')
    fs = sorted(f for f in os.listdir(p) if not f.startswith('.'))
    w, h = Image.open(os.path.join(p, fs[0])).size
    print(f'{p}: {len(fs)} 張，第一張 {w}x{h}')
    assert (w, h) == (1600, 900), '⛔ 尺寸不是 1600x900'
print('✅ 影像完成')" || exit $?
    # Depth-Anything-V2：setup_official_env.sh 照 doc/data_preparation.md clone 的官方 repo（不再連到 gs）
    [ -d utils/Depth-Anything-V2/.git ] || { echo "⛔ utils/Depth-Anything-V2 不是官方 clone（先跑 setup_official_env.sh）"; exit 2; }
    for pair in "$TRAIN:$OTRAIN"; do   # 09-29：test 改由 prep_test_official
      src=${pair%%:*}; od=${pair##*:}
      mkdir -p "$od"
      [ -e "$od/sparse" ]     || ln -s "$src/sparse" "$od/sparse"
      [ -e "$od/images" ]     || ln -s "$src/input" "$od/images"
      n_img=$(ls "$src/input" | wc -l)
      if [ -f "$od/estimated_depths/.made_by_$OFFENV" ] && [ -f "$od/estimated_depth_scales.json" ]; then
        echo "  深度圖已由 $OFFENV 產生，略過 $od"; continue
      fi
      move_aside "$od/estimated_depths"
      [ -f "$od/estimated_depth_scales.json" ] && mv "$od/estimated_depth_scales.json" "$od/estimated_depth_scales.json.aborted_$(date +%m%d_%H%M%S)"
      echo "=== 官方深度工具（-d 1.2）：$od（$n_img 張）$(date) ==="
      conda run -n "$OFFENV" --no-capture-output python utils/estimate_dataset_depths.py "$od" -d 1.2 || exit $?
      [ "$(ls "$od/estimated_depths" | grep -c '\.npy$')" -ge "$n_img" ] || { echo "⛔ $od 深度圖張數不足"; exit 3; }
      touch "$od/estimated_depths/.made_by_$OFFENV"
    done
    conda run -n "$OFFENV" python -c "
import numpy as np, os, glob, json
for od in ['$OTRAIN']:
    fs = sorted(glob.glob(os.path.join(od, 'estimated_depths', '*.npy')))
    a = np.load(fs[0]); s = json.load(open(os.path.join(od, 'estimated_depth_scales.json')))
    print(f'{od}: 深度圖 {len(fs)} 張、形狀 {a.shape}、尺度 {len(s)} 筆')
    assert a.shape == (900, 1600), '⛔ 深度圖不是 1600x900'
print('✅ prep 完成（影像＋官方深度）')" ;;
  coarse)
    move_aside "outputs/$COARSE"
    echo "=== 官方 coarse（未修改原始碼＋原版 config，sh2 30k）$(date) ==="
    run_measured conda run -n "$OFFENV" --no-capture-output python -u main.py fit \
      --config configs/$COARSE.yaml -n $COARSE --data.train_max_num_images_to_cache 1024 \
      --data.num_workers 8 --data.path "$OTRAIN_REL"
    rc=$?; record - "outputs/$COARSE"; exit "$rc" ;;
  partition)
    CK=$(coarse_ckpt); [ -n "$CK" ] || { echo "⛔ 沒有 coarse ckpt"; exit 2; }
    # 官方 config 寫死 epoch=6-step=30000.ckpt；實際 epoch 可能不同 => 補一個同名連結，config 不動
    # 2026-09-24：舊分區（09-18，用更早的 coarse 做的）與舊的塊輸出一律搬走，從這個 coarse 重新分
    move_aside "$OTRAIN/partition"
    move_aside "outputs/$NAME"
    WANT="outputs/$COARSE/checkpoints/epoch=6-step=30000.ckpt"
    [ -e "$WANT" ] || { ln -s "$(basename "$CK")" "$WANT"; echo "  補連結 $WANT -> $(basename "$CK")"; }
    echo "=== 官方分區（4x4，content_threshold 0.05；與我方 5x5 不同目錄，不會互相覆寫）$(date) ==="
    conda run -n "$OFFENV" --no-capture-output python utils/partition_citygs.py \
      --config_path configs/$NAME.yaml --force 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    ls -d "$OTRAIN"/partition/partitions-dim_4_4_visibility_0.05 || rc=3
    exit "$rc" ;;
  block)
    [ -n "$BLKARG" ] || { echo "⛔ 用法: task_citygs_origin.sh block <N>"; exit 2; }
    [ -d "$OTRAIN/partition/partitions-dim_4_4_visibility_0.05" ] || { echo "⛔ 還沒分區（先跑 partition）"; exit 2; }
    move_aside "outputs/$NAME/blocks/block_$BLKARG"
    echo "=== 官方微調 block $BLKARG（60k，吃 coarse）$(date) ==="
    # ⚠⚠ 2026-09-18 踩過並已移除：我曾加 `--data.train_max_num_images_to_cache 512`（想省 RAM）。
    #   官方 4x4 每塊有 **538~1016 張**影像 => 快取比影像數小 ⇒ **每輪重讀**：
    #   block 0 的 log 裡 `caching images` 出現 **80 次**、block 1 出現 57 次，
    #   行程 CPU 101%（單核心讀檔滿載）而 GPU 抽樣 0% ⇒ 整個跑次卡在磁碟 I/O，不是 GPU 慢。
    #   ⇒ 改回官方預設（不傳這個參數 = -1 = 全快取）。官方 launcher 本來就不傳。
    #   （coarse 那一步的 1024 是**官方腳本自己寫的**，保留。）
    #   ⇒ 全快取每塊約 12~23 GB RAM ⇒ 官方塊必須 **[solo] 獨佔**（同時也避開 VRAM：每塊長到 10~12 GB）
    #   ⚠ `--data.num_workers 8`（官方 config 沒設 => 預設 **2**）：影像快取是 PIL 解碼，CPU 綁定。
    #     依據＝我方跑次在**同一顆碟**上用 8 條執行緒快取 548 張影像沒問題；lab 有 20 執行緒。
    #     ⚠ 資料在 **機械碟**（/dev/sda = ST2000DM008，rotational=1）=> 不再往上加，並行讀會造成尋道。
    #     純基礎建設，不影響演算法。
    run_measured conda run -n "$OFFENV" --no-capture-output python -u main.py fit \
      --config configs/$NAME.yaml --data.parser.block_id "$BLKARG" -n $NAME \
      --data.num_workers 8 --data.path "$OTRAIN_REL"
    rc=$?; record "$BLKARG" "outputs/$NAME/blocks/block_$BLKARG"; exit "$rc" ;;
  merge)
    # ⚠ 2026-09-29：官方 merge 只 assert「至少 1 塊」，而且掃 blocks/ 底下**每個目錄**（含 *.aborted_*）
    #   => 缺一塊時會**靜默用 15 塊合成**，test 分數照樣出來。這裡逐塊要求 block_0..15 都有 60000 步，
    #   且 aborted 目錄裡不得有 60000 步的 ckpt（否則同一塊會被合兩次）。
    # ⚠⚠ 2026-09-29 實測：沒有 ckpt 的 aborted 目錄也會讓官方 merge **當場崩潰**
    #   （search_load_file 找不到 ckpt 就 assert，不是略過）=> 合併前把它們搬出 blocks/（log 全保留）
    mkdir -p "outputs/$NAME/aborted_blocks"
    for d in "outputs/$NAME/blocks/"*.aborted_*; do
      [ -d "$d" ] || continue
      if ls "$d/checkpoints/"*step=60000.ckpt >/dev/null 2>&1; then continue; fi   # 留給下面的重複檢查報錯
      mv "$d" "outputs/$NAME/aborted_blocks/" && echo "  搬出 blocks/：$(basename "$d")"
    done
    NB=16; miss=""; dup=""
    for b in $(seq 0 $((NB - 1))); do
      ls "outputs/$NAME/blocks/block_$b/checkpoints/"*step=60000.ckpt >/dev/null 2>&1 || miss="$miss $b"
    done
    for d in "outputs/$NAME/blocks/"*.aborted_*; do
      [ -d "$d" ] && ls "$d/checkpoints/"*step=60000.ckpt >/dev/null 2>&1 && dup="$dup $(basename "$d")"
    done
    [ -z "$miss" ] || { echo "⛔ 這些塊沒有 60000 步 ckpt：$miss => 不合併（官方 merge 會靜默少合）"; exit 2; }
    [ -z "$dup" ] || { echo "⛔ aborted 目錄裡有 60000 步 ckpt（會被重複合併）：$dup"; exit 2; }
    echo "✅ 16 塊都有 60000 步 ckpt，aborted 目錄無完賽 ckpt"
    conda run -n "$OFFENV" --no-capture-output python utils/merge_citygs_ckpts.py "outputs/$NAME" 2>&1 | tee "$LOG"
    exit "${PIPESTATUS[0]}" ;;
  test)
    echo "=== 官方 held-out 評測（$TEST 全部）$(date) ==="
    # ⚠ 2026-09-29：官方腳本（scripts/citygs/run_citygs_mc_aerial.sh）寫 --config outputs/$COARSE/config.yaml，
    #   但官方 fit（lightning 2.3、save_config_kwargs overwrite）實際存在 lightning_logs/version_N/config.yaml
    #   => 照官方路徑找不到。沿用官方意圖（coarse 的 resolved config），只是換成它實際所在的位置。
    CFG=outputs/$COARSE/config.yaml
    [ -f "$CFG" ] || CFG=$(ls -d outputs/$COARSE/lightning_logs/version_*/config.yaml 2>/dev/null | sort -V | tail -1)
    [ -f "$CFG" ] || { echo "⛔ 找不到 coarse 的 config.yaml"; exit 2; }
    echo "test 用的 config：$CFG"
    run_measured conda run -n "$OFFENV" --no-capture-output python -u main.py test \
      --config "$CFG" -n $NAME \
      --data.path "$OTEST" \
      --data.parser.eval_image_select_mode ratio \
      --data.parser.eval_ratio 1.0 \
      --save_val --test_speed
    rc=$?; record - "outputs/$NAME/checkpoints"
    [ -f "outputs/$NAME/results.txt" ] && { echo "-- results.txt --"; cat "outputs/$NAME/results.txt"; }
    exit "$rc" ;;
  prep_test_official)
    # 2026-09-29：官方 held-out test 集，**逐步照官方** scripts/citygs/untar_matrixcity_test.sh + data_proc_mc.sh：
    #   ① 每個 block_k_test.tar 的 png 放到 test/block_k_test/input/（官方是 mv；這裡用連結指向已解開的 tar，內容相同）
    #   ② transforms.json = pose/block_all/transforms_test.json（官方原檔）
    #   ③ 官方 tools/transform_json2txt_mc_aerial.py：依 transforms 影格順序複製成 input/{idx:04d}.png
    #   ④ sparse 換成官方下載的 colmap_results/matrix_city_aerial/test/sparse
    #   ⑤ 官方 image_downsample（1.2）＋官方 estimate_dataset_depths（-d 1.2）
    #   我方 test/block_all_test/input 完全不參與。
    gpu_gate   # 深度估計用 GPU
    T=$GS/data/matrix_city/aerial/test
    STG=$T/_stage_official
    # 官方 data_proc_mc.sh 是把 colmap_results/.../test/sparse **mv** 到 block_all_test/sparse
    #   => lab 上 colmap_results/matrix_city_aerial/test 已是空的，官方那份就在 block_all_test/sparse。
    #   不信任位置：與 colmap_results.zip 裡的原檔逐檔比 md5，全部相同才用。
    CR=$GS/data/colmap_results/matrix_city_aerial/test/sparse
    if [ ! -d "$CR/0" ]; then
      CR=$T/block_all_test/sparse
      Z=$GS/data/colmap_results.zip
      [ -f "$Z" ] || { echo "⛔ 找不到 $Z，無法驗證 test sparse 是官方原檔"; exit 2; }
      for f in cameras.bin images.bin points3D.bin; do
        a1=$(unzip -p "$Z" "colmap_results/matrix_city_aerial/test/sparse/0/$f" 2>/dev/null | md5sum | cut -d' ' -f1)
        [ -n "$a1" ] && [ "$a1" != "d41d8cd98f00b204e9800998ecf8427e" ] || a1=$(unzip -p "$Z" "matrix_city_aerial/test/sparse/0/$f" 2>/dev/null | md5sum | cut -d' ' -f1)
        a2=$(md5sum < "$CR/0/$f" | cut -d' ' -f1)
        [ "$a1" = "$a2" ] || { echo "⛔ $CR/0/$f 與 zip 裡的官方原檔不同（$a1 vs $a2）"; exit 2; }
      done
      echo "  ④ test sparse：$CR（與 colmap_results.zip 原檔逐檔 md5 相同）"
    fi
    for k in $(seq 1 10); do
      src=$STG/x_block_$k/block_${k}_test; dst=$T/block_${k}_test/input
      [ -d "$src" ] || { echo "⛔ 缺解開的 tar：$src"; exit 2; }
      mkdir -p "$dst"
      for f in "$src"/*.png; do [ -e "$dst/$(basename "$f")" ] || ln -s "$f" "$dst/$(basename "$f")"; done
      echo "  ① block_${k}_test/input：$(ls "$dst" | wc -l) 張"
    done
    if [ ! -f "$OTEST/.built_by_official_steps" ]; then
      move_aside "$OTEST"
      mkdir -p "$OTEST"
      cp "$GS/data/matrix_city/aerial/pose/block_all/transforms_test.json" "$OTEST/transforms.json" || exit 3
      # ⚠ 2026-09-30：官方工具的 --intrinsic_path 預設是 **train** 的 transforms，而它的**影格清單也從這裡讀**；
      #   官方 data_proc_mc.sh 對 aerial test 沒傳這個參數 => 實際拿 train 的 5,621 幀去 test 目錄找檔，
      #   全部 cp 失敗、input/ 一張都沒有（09-30 第一次就是這樣，被下面的 md5 驗證擋下）。
      #   官方對 street 的 test 有傳 --intrinsic_path transforms_test.json => aerial 照同樣做法傳 test 自己的 transforms。
      #   結果＝input/{N} = transforms_test 第 N 幀的原始檔，與官方下載的 test sparse（相機 N = 第 N 幀，已驗證）一致。
      conda run -n "$OFFENV" --no-capture-output python tools/transform_json2txt_mc_aerial.py --source_path "$OTEST" \
        --intrinsic_path "$OTEST/transforms.json" || exit 3
      rm -rf "$OTEST/sparse"
      cp -r "$CR" "$OTEST/sparse" || exit 3
      ln -s input "$OTEST/images"          # 官方 parser 的 image_dir 預設 images（+ _1.2）；深度工具也讀 images
      # 驗證（不經模型）：input/{idx}.png 的 md5 必須等於 transforms 第 idx 幀在官方 tar 裡那個檔
      conda run -n "$OFFENV" --no-capture-output python - "$OTEST" "$STG/official_md5.txt" <<'PY' || exit 3
import hashlib, json, os, sys
od, mfile = sys.argv[1], sys.argv[2]
off = {"/".join(l.split(None, 1)[1].strip().split("/")[-2:]): l.split()[0] for l in open(mfile)}
fr = json.load(open(os.path.join(od, "transforms.json")))["frames"]
bad = 0
for i, f in enumerate(fr):
    key = "/".join(f["file_path"].split("/")[-2:])
    got = hashlib.md5(open(os.path.join(od, "input", f"{i:04d}.png"), "rb").read()).hexdigest()
    bad += got != off[key]
n = len([x for x in os.listdir(os.path.join(od, "input")) if x.endswith(".png")])
print(f"✅ input {n} 張；逐張 md5 對 transforms 影格的官方檔：不符 {bad}")
assert n == len(fr) and bad == 0
PY
      [ "$(md5sum < "$OTEST/sparse/0/images.bin")" = "$(md5sum < "$T/block_all_test/sparse/0/images.bin")" ] \
        && echo "  sparse/images.bin 與 block_all_test 那份相同（兩者都是官方下載）" || echo "  ⚠ sparse 與 block_all_test 那份不同（以官方 colmap_results 為準）"
      touch "$OTEST/.built_by_official_steps"
    fi
    L="$OTEST/images_1.2"
    if [ ! -f "$L/.made_by_$OFFENV" ]; then
      move_aside "$L"
      conda run -n "$OFFENV" --no-capture-output python utils/image_downsample.py "$OTEST/input" --dst "$L" --factor 1.2 || exit 3
      touch "$L/.made_by_$OFFENV"
    fi
    if [ ! -f "$OTEST/estimated_depths/.made_by_$OFFENV" ]; then
      move_aside "$OTEST/estimated_depths"
      conda run -n "$OFFENV" --no-capture-output python utils/estimate_dataset_depths.py "$OTEST" -d 1.2 || exit 3
      touch "$OTEST/estimated_depths/.made_by_$OFFENV"
    fi
    echo "✅ 官方 test 集：$OTEST（$(ls "$L" | wc -l) 張 1600 影像、$(ls "$OTEST/estimated_depths" | grep -c npy) 張深度）"
    exit 0 ;;
  blockfix)
    # 2026-10-01：第二條官方參考線（官方程式碼＋一行修正讓 trim 真的執行，見 setup_official_trimfix.sh），先跑一塊。
    #   其餘與 block 模式完全相同：官方 config、官方 coarse（cityGS_origin 那個，用絕對路徑）、官方分區、官方套件、外掛紀錄。
    [ -n "$BLKARG" ] || { echo "⛔ 用法: task_citygs_origin.sh blockfix <N>"; exit 2; }
    FIX=/workspace/data/hdd/11213/cityGS_origin_trimfix
    ( cd "$FIX" 2>/dev/null && [ "$(git diff --name-only)" = "internal/gaussian_splatting.py" ] ) \
      || { echo "⛔ $FIX 不存在或差異不是恰好那一個檔（先跑 scripts/lab/setup_official_trimfix.sh）"; exit 2; }
    CK=$(coarse_ckpt); [ -n "$CK" ] || { echo "⛔ 沒有官方 coarse ckpt"; exit 2; }
    CK=$ORIG/$CK
    NAMEF=citygsv2_mc_aerial_sh2_trimfix
    cd "$FIX" || exit 1
    move_aside "outputs/$NAMEF/blocks/block_$BLKARG"
    echo "=== 官方＋一行修正 微調 block $BLKARG（60k，吃官方 coarse $(basename "$CK")）$(date) ==="
    run_measured conda run -n "$OFFENV" --no-capture-output python -u main.py fit \
      --config configs/$NAME.yaml --data.parser.block_id "$BLKARG" -n $NAMEF \
      --model.initialize_from "$CK" --data.num_workers 8 --data.path "$OTRAIN_REL"
    rc=$?
    nt=$(grep -a -c 'Trimming' "$LOG"); echo "[檢查] log 裡 Trimming 出現 $nt 次（官方原版同塊 = 0）"
    [ "$nt" -ge 1 ] || { echo "⛔⛔ trim 仍然沒有執行 => 修正無效"; rc=5; }
    record "$BLKARG" "outputs/$NAMEF/blocks/block_$BLKARG"; exit "$rc" ;;
  blockpaper)
    # 2026-10-01：「照論文設定」的官方參考線（先兩塊驗證）：官方程式碼＋一行修正（trim 真的執行，同 blockfix）
    #   ＋ 論文 §5.1 寫的兩個值（發布的 aerial config 沒設、用的是預設）：
    #     ω（densify_grad_scaler）= 0.9（預設 0 => 梯度縮放不作用）、pruning ratio（trim 的 prune_ratio）= 0.025（預設 0.1）
    #   只用命令列覆寫，程式碼不另外改。
    [ -n "$BLKARG" ] || { echo "⛔ 用法: task_citygs_origin.sh blockpaper <N>"; exit 2; }
    FIX=/workspace/data/hdd/11213/cityGS_origin_trimfix
    ( cd "$FIX" 2>/dev/null && [ "$(git diff --name-only)" = "internal/gaussian_splatting.py" ] ) \
      || { echo "⛔ $FIX 不存在或差異不是恰好那一個檔（先跑 scripts/lab/setup_official_trimfix.sh）"; exit 2; }
    CK=$(coarse_ckpt); [ -n "$CK" ] || { echo "⛔ 沒有官方 coarse ckpt"; exit 2; }
    CK=$ORIG/$CK
    NAMEP=citygsv2_mc_aerial_sh2_paper
    cd "$FIX" || exit 1
    move_aside "outputs/$NAMEP/blocks/block_$BLKARG"
    echo "=== 照論文設定 微調 block $BLKARG（ω=0.9、prune_ratio=0.025、trim 執行；60k）$(date) ==="
    run_measured conda run -n "$OFFENV" --no-capture-output python -u main.py fit \
      --config configs/$NAME.yaml --data.parser.block_id "$BLKARG" -n $NAMEP \
      --model.initialize_from "$CK" --data.num_workers 8 --data.path "$OTRAIN_REL" \
      --model.density.init_args.densify_grad_scaler 0.9 --model.renderer.init_args.prune_ratio 0.025
    rc=$?
    nt=$(grep -a -c 'Trimming' "$LOG"); echo "[檢查] Trimming 出現 $nt 次"
    [ "$nt" -ge 1 ] || { echo "⛔⛔ trim 沒有執行"; rc=5; }
    record "$BLKARG" "outputs/$NAMEP/blocks/block_$BLKARG"; exit "$rc" ;;
  mergepaper|testpaper)
    # 2026-10-02：照論文設定（blockpaper）的 16 塊合併與官方 held-out test。程式碼＝trimfix worktree（一行修正）；
    #   merge／test 本身沒有改動，只是換目錄與跑次名。test 集＝prep_test_official 重建的官方 test。
    FIX=/workspace/data/hdd/11213/cityGS_origin_trimfix
    NAMEP=citygsv2_mc_aerial_sh2_paper
    CFGP=$ORIG/outputs/$COARSE/config.yaml
    [ -f "$CFGP" ] || CFGP=$(ls -d $ORIG/outputs/$COARSE/lightning_logs/version_*/config.yaml 2>/dev/null | sort -V | tail -1)
    cd "$FIX" || exit 1
    if [ "$MODE" = mergepaper ]; then
      mkdir -p "outputs/$NAMEP/aborted_blocks"
      for d in "outputs/$NAMEP/blocks/"*.aborted_*; do [ -d "$d" ] && mv "$d" "outputs/$NAMEP/aborted_blocks/"; done
      miss=""; for b in $(seq 0 15); do ls "outputs/$NAMEP/blocks/block_$b/checkpoints/"*step=60000.ckpt >/dev/null 2>&1 || miss="$miss $b"; done
      [ -z "$miss" ] || { echo "⛔ 這些塊沒有 60000 步 ckpt：$miss"; exit 2; }
      echo "✅ 16 塊都有 60000 步 ckpt"
      conda run -n "$OFFENV" --no-capture-output python utils/merge_citygs_ckpts.py "outputs/$NAMEP" 2>&1 | tee "$LOG"
      exit "${PIPESTATUS[0]}"
    fi
    gpu_gate
    echo "=== 照論文設定 官方 held-out test（$OTEST）$(date) ==="
    run_measured conda run -n "$OFFENV" --no-capture-output python -u main.py test \
      --config "$CFGP" -n $NAMEP --data.path "$OTEST" \
      --data.parser.eval_image_select_mode ratio --data.parser.eval_ratio 1.0 --save_val --test_speed
    rc=$?; record - "outputs/$NAMEP/checkpoints"
    [ -f "outputs/$NAMEP/results.txt" ] && { echo "-- results.txt --"; cat "outputs/$NAMEP/results.txt"; }
    exit "$rc" ;;
  blockload)
    # 2026-09-29：每塊在 30k（增生期結束附近）與 60k（終點）的離線 Load —— 我方跑次都有這一欄
    #   （同一支 cost_budget_calibrate、精確 Σtiles 與代理並列）。官方原始碼沒有訓練中 Load 打印，
    #   所以「訓練途中的負載」只能用這兩個 ckpt 近似。量測工具是我方的（使用者允許的例外），跑在 gspl。
    gpu_gate
    cd "$GS" || exit 1
    R="../cityGS_origin/outputs/$NAME"
    { bad=0
      for b in $(seq 0 15); do
        for st in 29999 60000; do
          CK=$(ls "$R/blocks/block_$b/checkpoints/"*step=$st.ckpt 2>/dev/null | head -1)
          [ -n "$CK" ] || { echo "⛔ block $b 沒有 step=$st 的 ckpt"; bad=1; continue; }
          echo "════════ 官方 block $b  step $st ════════"
          conda run -n gspl --no-capture-output python tools/cost_budget_calibrate.py --ckpt "$CK" --max-cam "${CITYGS_MAXCAM:-600}" || { echo "⛔ 失敗"; bad=1; }
        done
      done
      exit "$bad"; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images|WARNING depth scale' | tee "$LOG"
    exit "${PIPESTATUS[0]}" ;;
  prune)
    # 2026-09-29：官方合併模型依 opacity 剪 X% 後的 held-out 品質（評分走官方 main.py test，與 test 同一條路徑）
    gpu_gate
    CK=$(ls outputs/$NAME/checkpoints/*.ckpt 2>/dev/null | head -1)
    [ -n "$CK" ] || { echo "⛔ 沒有合併後的 ckpt（先 merge）"; exit 2; }
    CFG=outputs/$COARSE/config.yaml
    [ -f "$CFG" ] || CFG=$(ls -d outputs/$COARSE/lightning_logs/version_*/config.yaml 2>/dev/null | sort -V | tail -1)
    PT=$GS/logs/citygs_official_prune.tsv
    [ -f "$PT" ] || printf 'ratio\tN\tpsnr\tssim\tlpips\trc\n' > "$PT"
    # ⚠ PYTHONPATH=$ORIG 不可省：unpickle 官方 ckpt 要 import 官方的 internal（09-29 第一次因此失敗）
    { PYTHONPATH="$ORIG" conda run -n "$OFFENV" --no-capture-output python "$GS/tools/official_prune_ckpt.py" "$CK" "${NAME}_prune" 10 25 50 75 || exit 3
      for r in 10 25 50 75; do
        P=${NAME}_prune_p$r
        echo "════════ 官方 test：剪 ${r}% ($P) ════════"
        conda run -n "$OFFENV" --no-capture-output python -u main.py test --config "$CFG" -n "$P" \
          --data.path "$OTEST" --data.parser.eval_image_select_mode ratio --data.parser.eval_ratio 1.0
        rc=$?
        n=$(grep -a -m1 "剪 ${r}" "$LOG.tmp" 2>/dev/null | grep -o '剩 [0-9,]*' | tr -dc 0-9)
        m=$(tr -d ' ,' < "outputs/$P/results.txt" 2>/dev/null | awk -F: '/test\/psnr/{p=$2}/test\/ssim/{s=$2}/test\/lpips/{l=$2}END{print p"\t"s"\t"l}')
        printf '%s\t%s\t%s\t%s\n' "$r" "${n:--}" "${m:--\t-\t-}" "$rc" >> "$PT"
        # 評分完就刪剪枝 ckpt（每個 1~3 GiB；要重算可重跑本模式），保留 results.txt 與 metrics/
        rm -f "outputs/$P/checkpoints/"*.ckpt
      done; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$LOG" "$LOG.tmp"
    rc=${PIPESTATUS[0]}
    rm -f "$LOG.tmp"
    echo "-- 剪枝曲線 --"; cat "$PT"
    exit "$rc" ;;
  res)
    gpu_gate   # 逐步計時要獨佔整卡
    # 工具在 gs；用 ../ 讓 glob 指到 origin 的 outputs（ckpt 裡的 data 路徑相對 gs，剛好同一份資料）
    cd "$GS" || exit 1
    # 離線 Load：我方量測工具＋我方光柵器（gspl），當作**共同的量尺**——同一支工具、同一顆光柵器量雙方的模型。
    #   ⚠ 這不是官方光柵器實際的 binning 量：我方光柵器有 EXACT_SUPPORT（o<=1/255 不進 binning），
    #     而官方光柵器不暴露 tiles，無從直接量。報告時要標明「共同量尺」。
    OFFENV=gspl
    R="../cityGS_origin/outputs/$NAME"
    CK=$(find "$R/checkpoints" -maxdepth 1 -name '*.ckpt' 2>/dev/null | tail -1)
    [ -n "$CK" ] || { echo "⛔ 沒有合併後的 ckpt（先 merge）"; exit 2; }
    { bad=0
      echo "════ 官方參考線：訓練期資源表 ════"; cat "$RES_TSV"
      echo "════ 合併模型離線量測：$CK（$(du -h "$CK" | cut -f1)）════"
      conda run -n "$OFFENV" --no-capture-output python tools/cost_budget_calibrate.py --ckpt "$CK" --max-cam "${CITYGS_MAXCAM:-600}" || { echo "⛔ 離線 Load 失敗"; bad=1; }
      # 逐**塊**量（訓練是逐塊做的）。⚠ 2026-09-29 改：必須用**官方的程式碼與光柵器**量
      #   （gspl_official、CITYGS_CODE_ROOT=cityGS_origin）；先前在 gspl 裡量到的是我方改過的光柵器，
      #   還多了一項官方根本沒有的 MCMC noise => 那份時間組成作廢。
      for b in $(seq 0 15); do
        echo "── 逐步計時：官方 block $b（官方程式碼＋官方光柵器）──"
        ( cd "$ORIG" && CITYGS_CODE_ROOT="$ORIG" conda run -n gspl_official --no-capture-output \
            python "$GS/tools/step_breakdown.py" --run "$NAME/blocks/block_$b" --repeat 20 ) || { echo "⛔ 逐步計時失敗（block $b）"; bad=1; }
      done
      exit "$bad"; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$LOG"
    exit "${PIPESTATUS[0]}" ;;
  *) echo "⛔ 不認得的模式：$MODE"; exit 2 ;;
esac
