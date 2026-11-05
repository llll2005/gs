"""11-02 一次性：從真深度量海平面高度（cwd＝gs；CPU；只讀）。"""
import os, sys, json, glob, re
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np, cv2
sys.path.insert(0, ".")
from internal.utils.colmap import read_cameras_binary, read_images_binary, qvec2rotmat
D = "data/matrix_city/aerial/train/block_all"; GT = "../mc_depth_f32/aerial/train"; K = 1.000061e-4
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin"); imgs = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}
FMAP = json.load(open("data/matrix_city/aerial/global_frame_map_correct.json"))
SC = json.load(open(f"{D}/estimated_depth_scales.json"))
def world(name):
    m = FMAP[f"{int(name[:-4]) + 1:06d}.png"]
    p = glob.glob(f"{GT}/block_{m['block']}/**/{m['frame_index']:04d}.exr", recursive=True)[0]
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED); d = (d[..., 0] if d.ndim == 3 else d).astype(np.float64)
    im = imgs[name]; c = cams[im.camera_id]; fx, fy, cx, cy = c.params[:4]; R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec)
    H, W = d.shape; uu, vv = np.meshgrid(np.arange(0, W, 2), np.arange(0, H, 2)); z = d[::2, ::2] * K
    Xw = (np.stack([(uu + .5 - cx) / fx * z, (vv + .5 - cy) / fy * z, z], -1) - t) @ R
    raw = np.load(f"{D}/estimated_depths/{name}.npy").astype(np.float64); raw = raw[..., 0] if raw.ndim == 3 else raw
    inv = raw[::2, ::2] * SC[name]["scale"] + SC[name]["offset"] if name in SC else np.full(z.shape, np.nan)
    return Xw, z, inv
for name, rows, cols in (("4482.png", slice(0, 150), slice(0, 480)), ("5364.png", slice(0, 60), slice(380, 480)), ("2024.png", slice(0, 40), slice(0, 100))):
    Xw, z, inv = world(name); zz = Xw[rows, cols, 2].ravel(); e = (z * inv)[rows, cols].ravel() - 1
    print(f"{name} 區域 z 百分位 5/25/50/75/95：{np.round(np.percentile(zz, [5, 25, 50, 75, 95]), 4)}｜單目逆深度<=0：{np.mean(inv[rows, cols] <= 1e-6):.1%}｜e 中位 {np.nanmedian(e):+.3f}、e<−5% {np.nanmean(e < -0.05):.1%}")
    Z = Xw[..., 2].ravel(); h, b = np.histogram(Z, bins=np.arange(-0.10, 0.101, 0.005))
    print("  整張 z 直方圖（−0.10～0.10，0.005 格）：" + " ".join(f"{b[i]:+.3f}:{h[i]}" for i in range(len(h)) if h[i] > 0.002 * len(Z)))
