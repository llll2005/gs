#!/bin/bash
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
LOG=$GS/logs/citygs_origin_${MODE}${BLKARG:+_$BLKARG}.log
RES_TSV=$GS/logs/citygs_origin_resources.tsv
PEAK_DIR=$GS/tools/peakmem
[ -f "$PEAK_DIR/sitecustomize.py" ] || { echo "⛔ 缺 $PEAK_DIR/sitecustomize.py（峰值紀錄）"; exit 2; }
[ -f "$RES_TSV" ] || printf 'mode\tblock\tstart\twall_s\trc\tpeak_alloc_MiB\tpeak_reserved_MiB\tgpu_used_max_MiB\tN\tckpt\n' > "$RES_TSV"

cd "$ORIG" || exit 1
[ -e data ] || ln -s "$GS/data" data
move_aside () { if [ -d "$1" ]; then mv "$1" "$1.aborted_$(date +%m%d_%H%M%S)" && echo "  舊輸出搬到 $1.aborted_*"; fi; return 0; }
coarse_ckpt () { find "outputs/$COARSE/checkpoints" -maxdepth 1 -name '*step=30000.ckpt' 2>/dev/null | head -1; }

run_measured () {   # run_measured <指令...>：資源寫進 RUN_* 變數；完整輸出在 $LOG
  local pk="$GS/logs/.peak_${MODE}${BLKARG:+_$BLKARG}_$$.tsv" t0 u bg
  rm -f "$pk"
  RUN_START=$(date '+%m-%d %H:%M'); t0=$(date +%s); RUN_GMAX=0
  ( export CITYGS_PEAK_OUT="$pk" PYTHONPATH="$PEAK_DIR${PYTHONPATH:+:$PYTHONPATH}"; exec "$@" ) > "$LOG" 2>&1 &
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
      n=$(conda run -n gspl python -c "
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
    for d in "$TRAIN" "$TEST"; do
      L="$d/images_1.2"; N_SRC=$(ls "$d/input" | wc -l)
      if [ -L "$L" ]; then rm -f "$L"; echo "  移除符號連結 $L"; fi
      if [ -d "$L" ] && [ "$(ls "$L" | wc -l)" -ge "$N_SRC" ]; then echo "  已存在且張數足夠，略過 $L"; continue; fi
      echo "=== 縮放 $d/input -> $L（$N_SRC 張）$(date) ==="
      conda run -n gspl --no-capture-output python utils/image_downsample.py "$d/input" --dst "$L" --factor 1.2 || exit $?
    done
    conda run -n gspl python -c "
from PIL import Image
import os
for d in ['$TRAIN', '$TEST']:
    p = os.path.join(d, 'images_1.2')
    fs = sorted(os.listdir(p))
    w, h = Image.open(os.path.join(p, fs[0])).size
    print(f'{p}: {len(fs)} 張，第一張 {w}x{h}')
    assert (w, h) == (1600, 900), '⛔ 尺寸不是 1600x900'
print('✅ 影像完成')" || exit $?
    # Depth-Anything-V2（第三方模型程式碼＋權重，不是 CityGS 程式碼）：官方工具預設找 utils/Depth-Anything-V2
    [ -e utils/Depth-Anything-V2 ] || ln -s "$GS/utils/Depth-Anything-V2" utils/Depth-Anything-V2
    for pair in "$TRAIN:$OTRAIN" "$TEST:$OTEST"; do
      src=${pair%%:*}; od=${pair##*:}
      mkdir -p "$od"
      [ -e "$od/sparse" ]     || ln -s "$src/sparse" "$od/sparse"
      [ -e "$od/images" ]     || ln -s "$src/input" "$od/images"
      [ -e "$od/images_1.2" ] || ln -s "$src/images_1.2" "$od/images_1.2"
      n_img=$(ls "$src/input" | wc -l); n_dep=$(ls "$od/estimated_depths" 2>/dev/null | grep -c '\.npy$')
      if [ "$n_dep" -ge "$n_img" ] && [ -f "$od/estimated_depth_scales.json" ]; then
        echo "  已有深度圖 $n_dep 張＋尺度檔，略過 $od"; continue
      fi
      echo "=== 官方深度工具（-d 1.2）：$od（$n_img 張）$(date) ==="
      conda run -n gspl --no-capture-output python utils/estimate_dataset_depths.py "$od" -d 1.2 || exit $?
    done
    conda run -n gspl python -c "
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
    run_measured conda run -n gspl --no-capture-output python -u main.py fit \
      --config configs/$COARSE.yaml -n $COARSE --data.train_max_num_images_to_cache 1024 \
      --data.num_workers 8 --data.path "$OTRAIN_REL"
    rc=$?; record - "outputs/$COARSE"; exit "$rc" ;;
  partition)
    CK=$(coarse_ckpt); [ -n "$CK" ] || { echo "⛔ 沒有 coarse ckpt"; exit 2; }
    # 官方 config 寫死 epoch=6-step=30000.ckpt；實際 epoch 可能不同 => 補一個同名連結，config 不動
    WANT="outputs/$COARSE/checkpoints/epoch=6-step=30000.ckpt"
    [ -e "$WANT" ] || { ln -s "$(basename "$CK")" "$WANT"; echo "  補連結 $WANT -> $(basename "$CK")"; }
    echo "=== 官方分區（4x4，content_threshold 0.05；與我方 5x5 不同目錄，不會互相覆寫）$(date) ==="
    conda run -n gspl --no-capture-output python utils/partition_citygs.py \
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
    run_measured conda run -n gspl --no-capture-output python -u main.py fit \
      --config configs/$NAME.yaml --data.parser.block_id "$BLKARG" -n $NAME \
      --data.num_workers 8 --data.path "$OTRAIN_REL"
    rc=$?; record "$BLKARG" "outputs/$NAME/blocks/block_$BLKARG"; exit "$rc" ;;
  merge)
    n=$(find "outputs/$NAME/blocks" -maxdepth 3 -name '*step=60000.ckpt' 2>/dev/null | wc -l)
    echo "完賽的塊：$n"; [ "$n" -ge 1 ] || { echo "⛔ 沒有任何完賽的塊"; exit 2; }
    conda run -n gspl --no-capture-output python utils/merge_citygs_ckpts.py "outputs/$NAME" 2>&1 | tee "$LOG"
    exit "${PIPESTATUS[0]}" ;;
  test)
    echo "=== 官方 held-out 評測（$TEST 全部）$(date) ==="
    run_measured conda run -n gspl --no-capture-output python -u main.py test \
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
    R="../cityGS_origin/outputs/$NAME"
    CK=$(find "$R/checkpoints" -maxdepth 1 -name '*.ckpt' 2>/dev/null | tail -1)
    [ -n "$CK" ] || { echo "⛔ 沒有合併後的 ckpt（先 merge）"; exit 2; }
    { bad=0
      echo "════ 官方參考線：訓練期資源表 ════"; cat "$RES_TSV"
      echo "════ 合併模型離線量測：$CK（$(du -h "$CK" | cut -f1)）════"
      conda run -n gspl --no-capture-output python tools/cost_budget_calibrate.py --ckpt "$CK" --max-cam "${CITYGS_MAXCAM:-600}" || { echo "⛔ 離線 Load 失敗"; bad=1; }
      conda run -n gspl --no-capture-output python tools/step_breakdown.py --run "$R/checkpoints" --repeat 20 || { echo "⛔ 逐步計時失敗"; bad=1; }
      exit "$bad"; } 2>&1 | grep -vE 'pkg_resources|declare_namespace|caching images' | tee "$LOG"
    exit "${PIPESTATUS[0]}" ;;
  *) echo "⛔ 不認得的模式：$MODE"; exit 2 ;;
esac
