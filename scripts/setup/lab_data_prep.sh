#!/usr/bin/env bash
# 在 **lab 主機**上重建 MatrixCity aerial 資料集（2026-09-12）
#
# 為什麼要重建：本機資料的影像↔相機配對壞掉 —— 6 台抽驗有 4 台的指派檔案分數不比
# 隨機檔案好，偏移量隨區段變化（+1/+2/+12/+42/+210）。根因是 `input/` 那批 6 位數檔案
# 不是官方 4 位數集合的改名，而是用別的順序重新蒐集的（官方是依 transforms_train.json
# 的 frames 順序編 4 位數，見 scripts/citygs/README.md）。
#
# ⚠ 在 lab 上跑而不是本機：我方上行 14 Mbps，lab 下載 71 MB/s（差 45 倍）=> 讓它自己拉。
# ⚠ 只在 $ROOT 底下做事，不動系統（使用者明確要求）。
set -u
ROOT="${ROOT:-$HOME}"
cd "$ROOT" || exit 1
D=data/matrix_city/aerial
echo "=== 0 環境檢查 ==="
df -h . | tail -1
python -c "import sys; print('python', sys.version.split()[0])" 2>/dev/null || echo "⚠ 沒有 python"
for t in curl tar unzip; do command -v $t >/dev/null || echo "⚠ 缺 $t"; done
command -v gdown >/dev/null || echo "ℹ 沒有 gdown（Google Drive 大檔要它）：pip install --user gdown"

cat <<'NOTE'

=== 1 需要手動放進來的兩包（腳本不自動下載，因為都要登入/確認）===
  ① MatrixCity small_city aerial     https://github.com/city-super/MatrixCity
     需要：aerial/pose/block_all/transforms_train.json + transforms_test.json
           aerial/train/block_1..10.tar   aerial/test/*.tar
     放到：data/matrix_city/aerial/{pose,train,test}/
  ② 預算 COLMAP（官方建議直接用）
     https://drive.google.com/file/d/1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7/view
     備用 https://pan.baidu.com/s/1zX34zftxj07dCM1x5bzmbA?pwd=1t6r
     解到：data/colmap_results/
     （gdown 'https://drive.google.com/uc?id=1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7'）

NOTE

echo "=== 2 現況 ==="
for p in "$D/pose/block_all/transforms_train.json" "$D/train/block_1.tar" \
         "data/colmap_results/matrix_city_aerial/train/sparse"; do
  [ -e "$p" ] && echo "  ✓ $p" || echo "  ✗ $p （還沒放）"
done
[ -e "$D/pose/block_all/transforms_train.json" ] || { echo "=> 缺①，停在這裡"; exit 2; }
[ -e "data/colmap_results/matrix_city_aerial/train/sparse" ] || { echo "=> 缺②，停在這裡"; exit 2; }

echo "=== 3 解開 tar（官方 untar_matrixcity_train.sh 的 aerial 部分）==="
( cd "$D/train" && for n in $(seq 1 10); do
    [ -d "block_$n/input" ] && { echo "  block_$n 已解，跳過"; continue; }
    mkdir -p "block_$n/input" && tar -xf "block_$n.tar" && mv block_$n/*.png "block_$n/input/" \
      && echo "  ✓ block_$n $(ls "block_$n/input" | wc -l) 張"
  done )

echo "=== 4 官方資料處理（蒐集影像 + 用預算 COLMAP 覆蓋 sparse）==="
echo "  只跑 aerial 那半（官方腳本含 street，我方不用）"
mkdir -p "$D/train/block_all/input" "$D/test/block_all_test/input"
cp "$D/pose/block_all/transforms_train.json" "$D/train/block_all/transforms.json"
cp "$D/pose/block_all/transforms_test.json"  "$D/test/block_all_test/transforms.json"
python tools/transform_json2txt_mc_aerial.py --source_path "$D/train/block_all" || exit 3
python tools/transform_json2txt_mc_aerial.py --source_path "$D/test/block_all_test" || exit 3
rm -rf "$D/train/block_all/sparse" "$D/test/block_all_test/sparse"
mv data/colmap_results/matrix_city_aerial/train/sparse "$D/train/block_all/"
mv data/colmap_results/matrix_city_aerial/test/sparse  "$D/test/block_all_test/"
echo "  input/ $(ls "$D/train/block_all/input" | wc -l) 張   前三個：$(ls "$D/train/block_all/input" | head -3 | tr '\n' ' ')"

echo "=== 5 ★ 驗證配對（沒過就不要開始訓練）==="
python tools/verify_image_pairing.py --data "$D/train/block_all" --from-name 2900 --to-name 3000 --step 10
echo
echo "判準：每一台都要 rank 1/5621 且 margin 明顯。"
echo "     現行壞資料是 6 台裡 4 台配錯（0242.png 排名 2101/5621）。"
echo
echo "=== 6 過了之後才做（不在本腳本內）==="
cat <<'NEXT'
  python utils/partition_from_colmap.py data/matrix_city/aerial/train/block_all --block_dim 5 5 --content_threshold 0.08 --force
  # 深度（現行配方 depth_loss_weight=0，只有要開深度損失才需要）
  git clone https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2
  wget -O utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
  python utils/estimate_dataset_depths.py data/matrix_city/aerial/train/block_all
  python utils/depth_init_blocks.py data/matrix_city/aerial/train/block_all --block_dim 5 5 --voxel_min 0.03 --voxel_max 0.7 --chunk_size 75
NEXT
