import add_pypath
import os
import argparse
import numpy as np
import cv2
import json
from joblib import delayed, Parallel
from internal.utils.colmap import read_model, qvec2rotmat

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir")
parser.add_argument("--depth_dir", type=str, default=None)
parser.add_argument("--output", "-o", type=str, default=None)
parser.add_argument("--point-max-error", type=float, default=1.5)
args = parser.parse_args()

if args.depth_dir is None:
    args.depth_dir = os.path.join(args.dataset_dir, "estimated_depths")
if args.output is None:
    args.output = os.path.join(args.dataset_dir, "estimated_depth_scales.json")

sparse_model_dir = os.path.join(args.dataset_dir, "sparse")
if os.path.exists(os.path.join(sparse_model_dir, "images.bin")) is False:
    sparse_model_dir = os.path.join(sparse_model_dir, "0")

cameras, images, points3d = read_model(sparse_model_dir)

# copied from https://github.com/graphdeco-inria/hierarchical-3d-gaussians/blob/main/preprocess/make_depth_scale.py

pts_indices = np.array([points3d[key].id for key in points3d])
pts_xyzs = np.array([points3d[key].xyz for key in points3d])
pts_errors = np.array([points3d[key].error for key in points3d])
points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
points3d_error_ordered = np.zeros([pts_indices.max() + 1, ])
points3d_ordered[pts_indices] = pts_xyzs
points3d_error_ordered[pts_indices] = pts_errors


_POSITIONAL_DEPTH = None


def _npy_for(img_name, images, depth_dir):
    """COLMAP 名 -> 深度 .npy，用**位置**對應，不是補零。

    ⛔ 2026-08-23 修正（與 dataparser 2026-08-12 `faeb4d4`、`depth_init_blocks.py` 2026-08-22
    完全相同的 off-by-one，這是第三處）：舊版 `base_name.zfill(6)` 把相機 `1729.png` 對到
    `001729.png.npy`，但 COLMAP 相機名 **0 起算**（0000.png~5620.png）而深度檔 **1 起算**
    （000001~005621）=> 正確是 `001730.png.npy`。
    ⇒ 舊版**用鄰幀的深度圖去擬合每張影像的 scale/offset**，而這個 JSON 同時餵給
      depth-init 與訓練時的深度監督。
    """
    global _POSITIONAL_DEPTH
    p1 = os.path.join(depth_dir, f"{img_name}.npy")
    if os.path.exists(p1):          # 同名直接對上（四位數檔名時代）
        return p1
    if _POSITIONAL_DEPTH is None:
        files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
        order = sorted(images, key=lambda k: images[k].name)
        if len(files) != len(order):
            raise SystemExit(
                f"深度圖 {len(files)} 張 != COLMAP 相機 {len(order)} 台，無法用位置對應。"
                " 位置對應是唯一正確的方式，數量不符必須先修資料。")
        _POSITIONAL_DEPTH = {images[k].name: files[i] for i, k in enumerate(order)}
    fn = _POSITIONAL_DEPTH.get(img_name)
    if fn is None:
        return None
    p2 = os.path.join(depth_dir, fn)
    return p2 if os.path.exists(p2) else None


def get_scales(key, cameras, images, points3d_ordered, points3d_error_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images[key].point3D_ids

    # filter out invalid 3D points
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    # get valid 3D point indices and 2D point xy
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    # reduce outliers
    pts_errors = points3d_error_ordered[pts_idx]
    valid_errors = pts_errors < args.point_max_error
    pts_idx = pts_idx[valid_errors]
    valid_xys = valid_xys[valid_errors]

    if len(pts_idx) > 0:
        # get 3D point xyz
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    # transform from world to camera
    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2]
    # invmonodepthmap = np.load(os.path.join(args.depth_dir, "{}.npy".format(image_meta.name)))  # already normalized
    # invmonodepthmap = np.load(os.path.join(args.depth_dir, "{}.npy".format(image_meta.name)))
    npy_path = _npy_for(image_meta.name, images, args.depth_dir)
    if npy_path is None:
        print(f"Warning: Missing depth map for {image_meta.name}, skipping.")
        return None # 找不到就回傳 None
    invmonodepthmap = np.load(npy_path)
    if invmonodepthmap is None:
        return None

    # if invmonodepthmap.ndim != 2:
    #     invmonodepthmap = invmonodepthmap[..., 0]

    # invmonodepthmap = invmonodepthmap.astype(np.float32)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    # xys inside image
    maps = (valid_xys * s).astype(np.float32)
    valid = (
            (maps[..., 0] >= 0) *
            (maps[..., 1] >= 0) *
            (maps[..., 0] < cam_intrinsic.width * s) *
            (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))

    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        # depth values from colmap
        invcolmapdepth = invcolmapdepth[valid]
        # get depth values of these 2D points from the depth map
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]

        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name, "scale": float(scale), "offset": float(offset)}


# depth_param_list = [get_scales(key, cameras, images, points3d_ordered, points3d_error_ordered, args) for key in images]
depth_param_list = Parallel(n_jobs=-1, backend="threading")(
    delayed(get_scales)(key, cameras, images, points3d_ordered, points3d_error_ordered, args) for key in images
)

depth_params = {
    depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
    for depth_param in depth_param_list if depth_param is not None
}

with open(args.output, "w") as f:
    json.dump(depth_params, f, indent=4, ensure_ascii=False)

print("Saved to `{}`".format(args.output))
