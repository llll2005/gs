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

step "0 前置：huggingface_hub / gdown"
python -m pip install -q --user huggingface_hub gdown 2>&1 | tail -2
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=0

step "1 下載 MatrixCity small_city/aerial（27.9 GB + test）"
mkdir -p data/mc_hf
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("BoDai/MatrixCity", repo_type="dataset",
                      allow_patterns=["small_city/aerial/**"],
                      local_dir="data/mc_hf", max_workers=8)
print("下載到", p)
PY
du -sh data/mc_hf 2>/dev/null

step "2 排成流程要的版面"
mkdir -p "$D/train" "$D/test" "$D/pose"
S=data/mc_hf/small_city/aerial
cp -r "$S/pose/." "$D/pose/" 2>/dev/null
for n in $(seq 1 10); do
  mkdir -p "$D/train/block_$n"
  [ -f "$S/train/block_$n.tar" ] && ln -sf "$(realpath "$S/train/block_$n.tar")" "$D/train/block_$n.tar"
  [ -f "$S/train/block_$n/transforms.json" ] && cp "$S/train/block_$n/transforms.json" "$D/train/block_$n/"
done
cp -r "$S/test/." "$D/test/" 2>/dev/null
ls "$D/train" | head -25
echo "各 block 的 transforms.json："
for n in $(seq 1 10); do
  f="$D/train/block_$n/transforms.json"
  [ -f "$f" ] && echo "  block_$n: $(python -c "import json;print(len(json.load(open('$f'))['frames']))" 2>/dev/null) frames" \
              || echo "  block_$n: ✗ 缺"
done

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
  gdown --fuzzy "https://drive.google.com/uc?id=$GD_ID" -O data/colmap_results.zip || \
    echo "⛔ gdown 失敗（Drive 可能要人工確認）=> 手動下載後放到 data/colmap_results.zip"
  [ -f data/colmap_results.zip ] && unzip -q -o data/colmap_results.zip -d data/ && echo "  解開完成"
fi
ls data/colmap_results 2>/dev/null

step "5 官方資料處理（只跑 aerial：蒐集影像 + 用預算 COLMAP 覆蓋 sparse）"
mkdir -p "$D/train/block_all/input" "$D/test/block_all_test/input"
cp "$D/pose/block_all/transforms_train.json" "$D/train/block_all/transforms.json"
cp "$D/pose/block_all/transforms_test.json"  "$D/test/block_all_test/transforms.json"
python tools/transform_json2txt_mc_aerial.py --source_path "$D/train/block_all"
python tools/transform_json2txt_mc_aerial.py --source_path "$D/test/block_all_test"
rm -rf "$D/train/block_all/sparse" "$D/test/block_all_test/sparse"
mv data/colmap_results/matrix_city_aerial/train/sparse "$D/train/block_all/"
mv data/colmap_results/matrix_city_aerial/test/sparse  "$D/test/block_all_test/"
echo "  input/ $(ls "$D/train/block_all/input" | wc -l) 張  前三：$(ls "$D/train/block_all/input" | head -3 | tr '\n' ' ')"

step "6 ★ 驗證配對（沒過就不要訓練）"
python tools/verify_pairing_geometric.py --data "$D/train/block_all" || \
  { echo "⛔⛔ 配對驗證沒過 —— 停在這裡，不要開始訓練"; exit 9; }

step "7 partition（5x5，我方流程）"
python utils/partition_from_colmap.py "$D/train/block_all" --block_dim 5 5 --content_threshold 0.08 --force

step "8 Depth-Anything-V2（使用者要求先裝好）"
if [ ! -d utils/Depth-Anything-V2 ]; then
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2
fi
mkdir -p utils/Depth-Anything-V2/checkpoints
CK=utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
[ -f "$CK" ] || wget -q --show-progress -O "$CK" \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
ls -la "$CK"
python -c "
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
echo "  python utils/estimate_dataset_depths.py $D/train/block_all --encoder vitl   # 要開深度損失才需要"
echo "  bash scripts/task_speed3.sh                                                  # 我方配方"
