import struct
import os

cam_path = '/home/LnoArch/Projects/專題/CityGS-X/data/matrix_city/small_city/aerial/train/block_all_3x/sparse/0/cameras.bin'
backup_path = cam_path + '.bak'

# 1. 從我們上一輪的備份恢復最原始的 1600x900 相機
os.system(f'cp {backup_path} {cam_path}')

# 2. 解析
with open(cam_path, "rb") as fid:
    num_cameras = struct.unpack("<Q", fid.read(8))[0]
    cameras = []
    for _ in range(num_cameras):
        c_id = struct.unpack("<I", fid.read(4))[0]
        m_id = struct.unpack("<I", fid.read(4))[0]
        w = struct.unpack("<Q", fid.read(8))[0]
        h = struct.unpack("<Q", fid.read(8))[0]
        
        model_dict = {0:3, 1:4, 2:4, 3:5, 4:8, 5:8, 6:9, 7:12, 8:12, 9:6, 10:7, 11:10, 12:14}
        num_params = model_dict[m_id]
        p = list(struct.unpack("<" + "d" * num_params, fid.read(8 * num_params)))
        cameras.append((c_id, m_id, w, h, p))

# 3. 寫入預先放大 1.2 倍的數值 (768x432)
count = 0
with open(cam_path, "wb") as fid:
    fid.write(struct.pack("<Q", num_cameras))
    for cam in cameras:
        c_id, m_id, w, h, p = cam
        # 這裡的 w 是原始的 1600，我們將目標設為 768 (即 640 * 1.2)
        scale_x = 768.0 / w
        scale_y = 432.0 / h
        
        if m_id in [0, 2, 9]:
            p[0] *= scale_x
            p[1] *= scale_x
            p[2] *= scale_y
        elif m_id in [1, 3, 4, 5, 6, 7, 8, 10, 11, 12]:
            p[0] *= scale_x
            p[1] *= scale_y
            p[2] *= scale_x
            p[3] *= scale_y
        count += 1

        fid.write(struct.pack("<I", c_id))
        fid.write(struct.pack("<I", m_id))
        fid.write(struct.pack("<Q", 768))
        fid.write(struct.pack("<Q", 432))
        fid.write(struct.pack("<" + "d" * len(p), *p))

print(f"✅ 成功將 {count} 個相機預先放大至 768x432 (完美對抗框架的 1.2x 隱藏除法機制)！")
