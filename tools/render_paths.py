"""一次載入模型，依序渲染 tools/video_paths.py 產生的多條相機路徑（GPU）。2026-11-25 新增。

python tools/render_paths.py --ckpt <模型 ckpt> --cam_from <提供相機內參的 ckpt> \
    --paths configs/video_paths_matrixcity_aerial.npz [--only capture street] \
    --out_dir outputs/lab/videos_1125 --prefix ours
=> <out_dir>/<prefix>_<路線>.mp4（30 fps）

相機內參／解析度＝--cam_from 那個跑次的第一台訓練相機（同 render_traj.py 的做法，解析度取偶數）。
兩個模型用同一個 --cam_from 和同一個路徑檔 => 每一幀視角完全相同，可以並排對照。
--cam_from 預設＝--ckpt；CityGS 的 ckpt 很大（約 43 GB），請指定我方的 merged.ckpt 當 --cam_from，免得讀兩次。
"""
import argparse
import copy
import gc
import os
import sys
import time

import imageio
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.gaussian_model_loader import GaussianModelLoader


def _resolve(p):
    return p if os.path.isfile(p) else GaussianModelLoader.search_load_file(p)


def template_camera(cam_from):
    src = torch.load(_resolve(cam_from), map_location="cpu")
    parser = src["datamodule_hyper_parameters"]["parser"]
    path = src["datamodule_hyper_parameters"]["path"]
    del src
    gc.collect()
    out = parser.instantiate(path=path, output_path=os.getcwd(), global_rank=0).get_outputs()
    cam = copy.deepcopy(out.train_set.cameras[0]).to_device("cuda")
    cam.height = int(cam.height / 2) * 2
    cam.width = int(cam.width / 2) * 2
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cam_from", default=None)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    data = np.load(a.paths)
    names = [n for n in data.files if not a.only or n in a.only]
    tmpl = template_camera(a.cam_from or a.ckpt)
    print(f"相機內參來自 {a.cam_from or a.ckpt}：{tmpl.width}x{tmpl.height}，fov_x {float(tmpl.fov_x):.4f}")

    ckpt = torch.load(_resolve(a.ckpt), map_location="cpu")
    bkgd = torch.tensor(ckpt["hyper_parameters"]["background_color"], device="cuda")
    model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device="cuda")
    model.freeze()
    model.pre_activate_all_properties()
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ckpt, stage="validate", device="cuda")
    del ckpt
    gc.collect()
    print(f"模型 {a.ckpt}：{model.get_xyz.shape[0]} 顆高斯")

    os.makedirs(a.out_dir, exist_ok=True)
    H = torch.tensor(tmpl.height, device="cuda")
    W = torch.tensor(tmpl.width, device="cuda")
    for name in names:
        M = data[name].astype(np.float64)
        out = os.path.join(a.out_dir, f"{a.prefix}_{name}.mp4")
        t0 = time.time()
        video = imageio.get_writer(out, fps=a.fps)
        with torch.no_grad():
            for c2w in tqdm(M, desc=name):
                cam = copy.copy(tmpl)
                cam.world_to_camera = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
                cam.full_projection = cam.world_to_camera.unsqueeze(0).bmm(cam.projection.unsqueeze(0)).squeeze(0)
                cam.camera_center = cam.world_to_camera.inverse()[3, :3]
                cam.height, cam.width = H, W
                img = renderer(cam, model, bkgd)["render"]
                video.append_data((img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        video.close()
        print(f"[{name}] {len(M)} 幀 {time.time() - t0:.0f} 秒 -> {out}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
