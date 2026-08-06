import sys
sys.path.append('.')
from utils.colmap_read_model import read_model, write_model

sparse_dir = '/home/LnoArch/Projects/專題/CityGS-X/data/matrix_city/small_city/aerial/train/block_all_3x/sparse/0'
cameras, images, points3d = read_model(sparse_dir)

count = 0
for cam_id, cam in cameras.items():
    if cam.width != 640:
        scale_x = 640.0 / cam.width
        scale_y = 360.0 / cam.height
        new_params = list(cam.params)
        
        # 依照相機模型自動按比例縮放焦距與光心
        if cam.model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE']:
            new_params[0] *= scale_x # fx
            new_params[1] *= scale_y # fy
            new_params[2] *= scale_x # cx
            new_params[3] *= scale_y # cy
        elif cam.model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL']:
            new_params[0] *= scale_x # f
            new_params[1] *= scale_x # cx
            new_params[2] *= scale_y # cy
            
        cameras[cam_id] = cam._replace(width=640, height=360, params=new_params)
        count += 1

write_model(cameras, images, points3d, sparse_dir, '.bin')
print(f"✅ 成功將 {count} 個 COLMAP 相機內部參數與畫布大小鎖死為 640x360！")
