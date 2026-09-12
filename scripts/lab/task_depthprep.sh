#!/bin/bash
# Depth-Anything V2 安裝 + 5,621 張深度圖 + depth-init PLY（25 塊）。
# 這是 `task_initcmp.sh <blk> depth` 的前提，也是唯一需要深度的地方
# （現行配方 depth_loss_weight=0 => 訓練本身不用深度）。
# ⚠ 很貴：5,621 張推論 + 25 塊體素化。放在一個槽裡跑。
source "$(dirname "$0")/_common.sh"
D=data/matrix_city/aerial/train/block_all
if [ ! -d utils/Depth-Anything-V2 ]; then
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2 || exit 1
fi
mkdir -p utils/Depth-Anything-V2/checkpoints
CK=utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
[ -s "$CK" ] || wget -q --show-progress -O "$CK" \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" || exit 1
ls -la "$CK"
# 本機是用 `depth_anything_v2` 符號連結指到外部 clone，**連結不會跟著 git 過來**
[ -e depth_anything_v2 ] || ln -s utils/Depth-Anything-V2/depth_anything_v2 depth_anything_v2
echo "=== 功能驗證 ==="
conda run -n gspl python -c "
import sys, numpy as np, torch
sys.path.insert(0,'utils/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
m=DepthAnythingV2(encoder='vitl', features=256, out_channels=[256,512,1024,1024])
m.load_state_dict(torch.load('$CK', map_location='cpu')); m=m.cuda().eval()
with torch.no_grad(): d=m.infer_image(np.zeros((518,518,3), np.uint8))
print('✅ Depth-Anything-V2 可用，輸出', d.shape)
" || exit 1
echo "=== 5,621 張深度圖（最貴的一步）==="
conda run -n gspl --no-capture-output python utils/estimate_dataset_depths.py "$D" || exit 1
echo "=== depth-init PLY（25 塊）==="
conda run -n gspl --no-capture-output python utils/depth_init_blocks.py "$D" \
  --block_dim 5 5 --voxel_min 0.03 --voxel_max 0.7 --chunk_size 75 || exit 1
ls "$D/depth_init" | head -5; ls "$D/depth_init" | wc -l
