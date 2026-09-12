#!/usr/bin/env bash
# 在 lab 上取得 MatrixCity aerial + 預算 COLMAP，並跑到「可以開始訓練」為止。
#
# ⚠ 為什麼在 lab 下載而不是從本機推：我方上行 14 Mbps，lab 下載 71 MB/s（差 45 倍）。
#   27.9 GB 的 tar 在 lab 約 7 分鐘，從本機推要 4.5 小時。
# ⚠ 只在 $ROOT 底下做事，不動系統套件（使用者要求）。
#
# 官方權威（已核對）：
#   HuggingFace  BoDai/MatrixCity :: small_city/aerial/{pose,train,test}
#     block_1.tar 3,988,720,128 ... block_9.tar 7,933,188,608（與本機那份大小完全一致）
#   預算 COLMAP  CityGaussian 的 Google Drive（官方 doc 建議直接用，不要自己跑 5000+ 張）
#   ★ MatrixCity README 自述 pose 資料夾版本「rotation matrix needs to be multiplied by 100」
#     —— 與我方 2026-09-12 反推出的 R_w2c = diag(1,-1,-1) @ (100*R_json)^T 一致。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
D=data/matrix_city/aerial
GD_ID=1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7     # 預算 COLMAP

step() { echo; echo "=========== $* ==========="; date; }

step "0 前置檢查"
# ⚠ 2026-09-12：容器的 `python` 是 **3.14**（base 也是），`pip --user huggingface_hub`
#   會裝進 3.14 而 huggingface_hub 在那上面壞掉
#   （Error importing ...: 'typing.Union' object has no attribute '__module__'）
#   => 下載**完全不用 python**，直接 wget HF 的 resolve URL；
#      python 步驟一律走 `conda run -n gspl`（3.9）。
PY() { conda run -n gspl --no-capture-output python "$@"; }
command -v wget >/dev/null || { echo "⛔ 沒有 wget"; exit 2; }
conda run -n gspl python -c "import numpy" 2>/dev/null \
  && echo "  ✅ gspl 可用" \
  || echo "  ⚠ gspl 還沒建好 —— 下載與解 tar 不需要它，第 5 步之後才需要"

step "1 下載 MatrixCity small_city/aerial（tar 共 27.9 GB）"
HF=https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/small_city/aerial
mkdir -p "$D/train" "$D/test" "$D/pose/block_all"
# pose（小檔）
for f in transforms_train.json transforms_test.json; do
  t="$D/pose/block_all/$f"
  [ -s "$t" ] || wget -q --show-progress -O "$t" "$HF/pose/block_all/$f" || echo "  ⚠ $f 抓不到"
done
# 各 block 的 transforms.json（小檔）+ tar（大檔）
for n in $(seq 1 10); do
  mkdir -p "$D/train/block_$n"
  t="$D/train/block_$n/transforms.json"
  [ -s "$t" ] || wget -q -O "$t" "$HF/train/block_$n/transforms.json" || echo "  ⚠ block_$n transforms.json 抓不到"
  tr="$D/train/block_$n.tar"
  if [ -s "$tr" ]; then echo "  block_$n.tar 已存在 $(du -h "$tr" | cut -f1)"; else
    echo "  抓 block_$n.tar …"
    wget -q --show-progress -c -O "$tr" "$HF/train/block_$n.tar" || echo "  ⚠ block_$n.tar 失敗"
  fi
done
# test 集
for f in $(seq 1 3); do :; done
wget -q -O "$D/test/transforms.json" "$HF/test/transforms.json" 2>/dev/null || true
for nm in small_city_aerial_test big_city_aerial_test; do
  [ -s "$D/test/$nm.tar" ] || wget -q --show-progress -c -O "$D/test/$nm.tar" "$HF/test/$nm.tar" 2>/dev/null || rm -f "$D/test/$nm.tar"
done
echo "  train tar：$(ls -la $D/train/*.tar 2>/dev/null | wc -l) 個，共 $(du -shc $D/train/*.tar 2>/dev/null | tail -1 | cut -f1)"
echo "  各 block transforms.json："
for n in $(seq 1 10); do
  f="$D/train/block_$n/transforms.json"
  if [ -s "$f" ]; then echo "    block_$n: $(grep -o '"frame_index"' "$f" | wc -l) frames"; else echo "    block_$n: ✗ 缺"; fi
done

step "2 版面（第 1 步已直接下載到位，這裡只列出來確認）"
ls "$D/train" | head -25

step "3 解 tar（官方 untar_matrixcity_train.sh 的 aerial 部分）"
( cd "$D/train" && for n in $(seq 1 10); do
    [ -d "block_$n/input" ] && [ "$(ls block_$n/input 2>/dev/null | wc -l)" -gt 0 ] \
      && { echo "  block_$n 已解 ($(ls block_$n/input | wc -l) 張)"; continue; }
    mkdir -p "block_$n/input" && tar -xf "block_$n.tar" -C . \
      && mv block_$n/*.png "block_$n/input/" 2>/dev/null
    echo "  block_$n -> $(ls block_$n/input 2>/dev/null | wc -l) 張"
  done )

step "4 預算 COLMAP（Google Drive）"
if [ -d data/colmap_results/matrix_city_aerial/train/sparse ]; then
  echo "  已存在，跳過"
else
  gdown "$GD_ID" -O data/colmap_results.zip || \
    echo "⛔ gdown 失敗（Drive 可能要人工確認）=> 手動下載後放到 data/colmap_results.zip"
  [ -f data/colmap_results.zip ] && unzip -q -o data/colmap_results.zip -d data/ && echo "  解開完成"
fi
ls data/colmap_results 2>/dev/null

step "5 官方資料處理（只跑 aerial：蒐集影像 + 用預算 COLMAP 覆蓋 sparse）"
mkdir -p "$D/train/block_all/input" "$D/test/block_all_test/input"
cp "$D/pose/block_all/transforms_train.json" "$D/train/block_all/transforms.json"
cp "$D/pose/block_all/transforms_test.json"  "$D/test/block_all_test/transforms.json"
PY tools/transform_json2txt_mc_aerial.py --source_path "$D/train/block_all"
PY tools/transform_json2txt_mc_aerial.py --source_path "$D/test/block_all_test"
rm -rf "$D/train/block_all/sparse" "$D/test/block_all_test/sparse"
mv data/colmap_results/matrix_city_aerial/train/sparse "$D/train/block_all/"
mv data/colmap_results/matrix_city_aerial/test/sparse  "$D/test/block_all_test/"
echo "  input/ $(ls "$D/train/block_all/input" | wc -l) 張  前三：$(ls "$D/train/block_all/input" | head -3 | tr '\n' ' ')"

step "6 ★ 驗證配對（沒過就不要訓練）"
PY tools/verify_pairing_geometric.py --data "$D/train/block_all" || \
  { echo "⛔⛔ 配對驗證沒過 —— 停在這裡，不要開始訓練"; exit 9; }

step "7 partition（5x5，我方流程）"
PY utils/partition_from_colmap.py "$D/train/block_all" --block_dim 5 5 --content_threshold 0.08 --force

step "8 Depth-Anything-V2（使用者要求先裝好）"
if [ ! -d utils/Depth-Anything-V2 ]; then
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2
fi
mkdir -p utils/Depth-Anything-V2/checkpoints
CK=utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
[ -f "$CK" ] || wget -q --show-progress -O "$CK" \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
ls -la "$CK"
PY -c "
import sys; sys.path.insert(0,'utils/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
import torch
m=DepthAnythingV2(encoder='vitl', features=256, out_channels=[256,512,1024,1024])
sd=torch.load('$CK', map_location='cpu'); m.load_state_dict(sd)
m=m.cuda().eval()
import numpy as np
with torch.no_grad():
    d=m.infer_image(np.zeros((518,518,3), np.uint8))
print('✅ Depth-Anything-V2 可用，輸出', d.shape, d.dtype)
"

step "完成"
echo "下一步（訓練）："
echo "  python utils/estimate_dataset_depths.py $D/train/block_all --image_dir input   # 要開深度損失才需要"
echo "  bash scripts/task_speed3.sh                                                  # 我方配方"
