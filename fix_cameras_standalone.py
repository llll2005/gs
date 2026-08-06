import struct
import os

cam_path = '/home/LnoArch/Projects/專題/CityGS-X/data/matrix_city/small_city/aerial/train/block_all_3x/sparse/0/cameras.bin'
backup_path = cam_path + '.bak'

# 備份原始檔案
if not os.path.exists(backup_path):
    os.system(f'cp {cam_path} {backup_path}')

# 解析 COLMAP cameras.bin 結構
with open(backup_path, "rb") as fid:
    num_cameras = struct.unpack("<Q", fid.read(8))[0]
    cameras = []
    for _ in range(num_cameras):
        camera_id = struct.unpack("<I", fid.read(4))[0]
        model_id = struct.unpack("<I", fid.read(4))[0]
        width = struct.unpack("<Q", fid.read(8))[0]
        height = struct.unpack("<Q", fid.read(8))[0]
        
        # 匹配 COLMAP 的相機模型參數數量
        model_dict = {0:3, 1:4, 2:4, 3:5, 4:8, 5:8, 6:9, 7:12, 8:12, 9:6, 10:7, 11:10, 12:14}
        num_params = model_dict[model_id]
        params = list(struct.unpack("<" + "d" * num_params, fid.read(8 * num_params)))
        cameras.append((camera_id, model_id, width, height, params))

# 寫入修改後的解析度與焦距
count = 0
with open(cam_path, "wb") as fid:
    fid.write(struct.pack("<Q", num_cameras))
    for cam in cameras:
        c_id, m_id, w, h, p = cam
        if w != 640:
            scale_x = 640.0 / w
            scale_y = 360.0 / h
            new_w, new_h = 640, 360
            
            # 根據相機模型縮放焦距與光心
            if m_id in [0, 2, 9]: # SIMPLE 類型 (f, cx, cy)
                p[0] *= scale_x
                p[1] *= scale_x
                p[2] *= scale_y
            elif m_id in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]: # 標準類型 (fx, fy, cx, cy)
                p[0] *= scale_x
                p[1] *= scale_y
                p[2] *= scale_x
                p[3] *= scale_y
            count += 1
        else:
            new_w, new_h = w, h

        fid.write(struct.pack("<I", c_id))
        fid.write(struct.pack("<I", m_id))
        fid.write(struct.pack("<Q", new_w))
        fid.write(struct.pack("<Q", new_h))
        fid.write(struct.pack("<" + "d" * len(p), *p))

print(f"✅ 成功將 {count} 個相機的二進位內參物理鎖死為 640x360！")
