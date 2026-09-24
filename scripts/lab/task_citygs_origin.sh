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
OTEST=$GS/data/matrix_city/aerial/test/block_all_test_official
OTRAIN_REL=data/matrix_city/aerial/train/block_all_official        # 相對 cityGS_origin（data -> gs/data）
[ -d "$ORIG" ] || { echo "⛔ 找不到 $ORIG"; exit 2; }
MODE=${1:?用法: task_citygs_origin.sh prep|coarse|partition|block <N>|merge|test|res}
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
[ -f "$RES_TSV" ] || printf 'mode\tblock\tstart\twall_s\trc\tpeak_alloc_MiB\tpeak_reserved_MiB\tgpu_used_max_MiB\tN\tckpt\n' > "$RES_TSV"

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
  local pk="$GS/logs/.peak_${MODE}${BLKARG:+_$BLKARG}_$$.tsv" t0 u bg
  rm -f "$pk"
  RUN_START=$(date '+%m-%d %H:%M'); t0=$(date +%s); RUN_GMAX=0
  [ -f "$TRAINLOG" ] && mv "$TRAINLOG" "$TRAINLOG.old_$(date +%m%d_%H%M%S)"
  [ -f "$CHURN" ] && mv "$CHURN" "$CHURN.old_$(date +%m%d_%H%M%S)"
  ( export CITYGS_PEAK_OUT="$pk" CITYGS_TRAINLOG_OUT="$TRAINLOG" CITYGS_TRAINLOG_EVERY=100 \
      CITYGS_CHURN_OUT="$CHURN" CITYGS_LEDGER="$GS/logs/quad_progress.log" PYTHONPATH="$PEAK_DIR${PYTHONPATH:+:$PYTHONPATH}"; exec "$@" ) > "$LOG" 2>&1 &
  bg=$!
  while kill -0 "$bg" 2>/dev/null; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
    case "$u" in ''|*[!0-9]*) ;; *) [ "$u" -gt "$RUN_GMAX" ] && RUN_GMAX=$u ;; esac
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
  echo "[資源] rc=$RUN_RC 牆鐘 ${RUN_WALL}s 配置峰值 ${RUN_PA} MiB 保留峰值 ${RUN_PR} MiB 整卡最大 ${RUN_GMAX} MiB"
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$MODE" "$blk" "$RUN_START" "$RUN_WALL" "$RUN_RC" "$RUN_PA" "$RUN_PR" "$RUN_GMAX" "$n" "$sz" >> "$RES_TSV"
  echo "[資源] 已記入 $RES_TSV：N=$n ckpt=$sz"
}

case "$MODE" in
  prep)
    # 2026-09-24：影像與深度圖**全部在 $OFFENV 裡重做**，放進官方線專用目錄（不再連到共用 block_all/images_1.2）。
    #   完成標記 .made_by_$OFFENV；沒有標記的舊產物一律搬走重做。
    for pair in "$TRAIN:$OTRAIN" "$TEST:$OTEST"; do
      src=${pair%%:*}; od=${pair##*:}; mkdir -p "$od"
      L="$od/images_1.2"; N_SRC=$(ls "$src/input" | wc -l)
      if [ -f "$L/.made_by_$OFFENV" ]; then echo "  已由 $OFFENV 產生，略過 $L"; continue; fi
      if [ -L "$L" ]; then rm -f "$L"; echo "  移除符號連結 $L"; else move_aside "$L"; fi
      echo "=== 縮放 $src/input -> $L（$N_SRC 張，官方 image_downsample.py）$(date) ==="
      conda run -n "$OFFENV" --no-capture-output python utils/image_downsample.py "$src/input" --dst "$L" --factor 1.2 || exit $?
      [ "$(ls "$L" | wc -l)" -ge "$N_SRC" ] || { echo "⛔ $L 張數不足"; exit 3; }
      touch "$L/.made_by_$OFFENV"
    done
    conda run -n "$OFFENV" python -c "
from PIL import Image
import os
for d in ['$OTRAIN', '$OTEST']:
    p = os.path.join(d, 'images_1.2')
    fs = sorted(f for f in os.listdir(p) if not f.startswith('.'))
    w, h = Image.open(os.path.join(p, fs[0])).size
    print(f'{p}: {len(fs)} 張，第一張 {w}x{h}')
    assert (w, h) == (1600, 900), '⛔ 尺寸不是 1600x900'
print('✅ 影像完成')" || exit $?
    # Depth-Anything-V2：setup_official_env.sh 照 doc/data_preparation.md clone 的官方 repo（不再連到 gs）
    [ -d utils/Depth-Anything-V2/.git ] || { echo "⛔ utils/Depth-Anything-V2 不是官方 clone（先跑 setup_official_env.sh）"; exit 2; }
    for pair in "$TRAIN:$OTRAIN" "$TEST:$OTEST"; do
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
for od in ['$OTRAIN', '$OTEST']:
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
    n=$(find "outputs/$NAME/blocks" -maxdepth 3 -name '*step=60000.ckpt' 2>/dev/null | wc -l)
    echo "完賽的塊：$n"; [ "$n" -ge 1 ] || { echo "⛔ 沒有任何完賽的塊"; exit 2; }
    conda run -n "$OFFENV" --no-capture-output python utils/merge_citygs_ckpts.py "outputs/$NAME" 2>&1 | tee "$LOG"
    exit "${PIPESTATUS[0]}" ;;
  test)
    echo "=== 官方 held-out 評測（$TEST 全部）$(date) ==="
    run_measured conda run -n "$OFFENV" --no-capture-output python -u main.py test \
      --config outputs/$COARSE/config.yaml -n $NAME \
      --data.path "$OTEST" \
      --data.parser.eval_image_select_mode ratio \
      --data.parser.eval_ratio 1.0 \
      --save_val --test_speed
    rc=$?; record - "outputs/$NAME/checkpoints"
    [ -f "outputs/$NAME/results.txt" ] && { echo "-- results.txt --"; cat "outputs/$NAME/results.txt"; }
    exit "$rc" ;;
  res)
    # 工具在 gs；用 ../ 讓 glob 指到 origin 的 outputs（ckpt 裡的 data 路徑相對 gs，剛好同一份資料）
    cd "$GS" || exit 1
    OFFENV=gspl   # 量測工具是我方的（使用者允許的例外），量的是官方 ckpt
    R="../cityGS_origin/outputs/$NAME"
    CK=$(find "$R/checkpoints" -maxdepth 1 -name '*.ckpt' 2>/dev/null | tail -1)
    [ -n "$CK" ] || { echo "⛔ 沒有合併後的 ckpt（先 merge）"; exit 2; }
    { bad=0
      echo "════ 官方參考線：訓練期資源表 ════"; cat "$RES_TSV"
      echo "════ 合併模型離線量測：$CK（$(du -h "$CK" | cut -f1)）════"
      conda run -n "$OFFENV" --no-capture-output python tools/cost_budget_calibrate.py --ckpt "$CK" --max-cam "${CITYGS_MAXCAM:-600}" || { echo "⛔ 離線 Load 失敗"; bad=1; }
      conda run -n "$OFFENV" --no-capture-output python tools/step_breakdown.py --run "$R/checkpoints" --repeat 20 || { echo "⛔ 逐步計時失敗"; bad=1; }
      exit "$bad"; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$LOG"
    exit "${PIPESTATUS[0]}" ;;
  *) echo "⛔ 不認得的模式：$MODE"; exit 2 ;;
esac
