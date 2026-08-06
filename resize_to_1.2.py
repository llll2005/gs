import os
import numpy as np
import torch
import torch.nn.functional as F
import concurrent.futures
from tqdm import tqdm

base_dir = "/home/LnoArch/Projects/專題/CityGaussian/data/matrix_city/aerial/train/block_all"
src_dir = os.path.join(base_dir, "estimated_depths")
dst_dir = os.path.join(base_dir, "estimated_depths_1.2")
factor = 1.2

os.makedirs(dst_dir, exist_ok=True)
npy_files = [f for f in os.listdir(src_dir) if f.endswith('.npy')]

def process_depth(filename):
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    # 讀取 NumPy 陣列，形狀應為 (1080, 1920)
    depth = np.load(src_path)

    # 轉換為 PyTorch 張量，並增加 Batch 與 Channel 維度 (1, 1, 1080, 1920)
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)

    # 計算 1.2 倍縮小後的新維度
    new_h = int(depth.shape[0] / factor)
    new_w = int(depth.shape[1] / factor)

    # 執行矩陣雙線性插值
    resized_tensor = F.interpolate(depth_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # 壓縮回 2D 陣列並存檔
    resized_depth = resized_tensor.squeeze().numpy()
    np.save(dst_path, resized_depth)

def main():
    if not npy_files:
        print(f"在 {src_dir} 找不到任何 .npy 檔案！")
        return

    print(f"🚀 開始縮放深度圖矩陣 (共 {len(npy_files)} 個檔案)...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_depth, npy_files), total=len(npy_files)))

if __name__ == "__main__":
    main()