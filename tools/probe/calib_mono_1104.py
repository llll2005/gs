"""11-04 使用者「都排」：單目深度「看穿」判準校準（只渲染）。
補洞加嚴 A／B 在第 6k～12k 步每個視角有 35～48% 的像素被判「渲染深度比單目深 1～2% 以上」。
問題：那是我們的地面真的在地下，還是單目本身有系統偏差？=> 拿 CityGS（地面在正確高度）當校準：
  在 b3 的訓練視角（均勻抽 n 台），三個模型都實心（alpha>0.95）且單目有效的像素上，算
    比例(渲染深度 > 單目 x (1+m))、比例(渲染深度 < 單目 x (1-m))，m = 1%／2%／5%／10%；以及 渲染/單目 的中位比值。
  單目＝dataparser 同一套：np.load(.npy) * scale + offset 是「逆深度」，z_mono = 1 / inv（同 _hole_fill／hole_candidates）。
用法（cwd＝repo 根）：python <本檔> --block 3 --n 20 --models 標籤=ckpt ...
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, ".")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_images_binary, read_cameras_binary
from internal.cameras.cameras import Cameras
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--depth_dir", default="estimated_depths_lab")
ap.add_argument("--scale_json", default="estimated_depth_scales_lab.json")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda")
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu")
names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
names = [names[i] for i in np.linspace(0, len(names) - 1, a.n).round().astype(int)]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
sc = json.load(open(f"{D}/{a.scale_json}"))


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def camera(nm):
    im = ims[nm]; c = cams[im.camera_id]
    W, H = round(c.width / 1.2), round(c.height / 1.2); s = W / c.width
    fx, fy = c.params[0] * s, (c.params[1] if c.model != "SIMPLE_PINHOLE" else c.params[0]) * s
    cx, cy = (c.params[2] * s, c.params[3] * s) if c.model != "SIMPLE_PINHOLE" else (c.params[1] * s, c.params[2] * s)
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(qmat(im.qvec)), T=t(im.tvec), fx=t([fx])[0], fy=t([fy])[0], cx=t([cx])[0], cy=t([cy])[0],
                   width=torch.tensor([W], dtype=torch.int16), height=torch.tensor([H], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1),
                   distortion_params=None, camera_type=torch.zeros(1, dtype=torch.int8))[0], (W, H)


def mono_z(nm, W, H):
    inv = np.load(f"{D}/{a.depth_dir}/{nm}.npy") * sc[nm]["scale"] + sc[nm]["offset"]
    inv = torch.nn.functional.interpolate(torch.tensor(inv, dtype=torch.float32)[None, None], size=(H, W), mode="nearest")[0, 0]
    return torch.where(inv > 1e-6, 1.0 / inv.clamp_min(1e-6), torch.full_like(inv, float("nan")))


def inbox(means):
    coords = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
    boxes = coords.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.0)
    x = means.double() @ P["extra_data"]["rotation_transform"][:3, :3].double().T
    return CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=x[:, :2])[B]


renderer0 = None; Z, A = {}, {}
for spec in a.models:
    lab, ck = spec.split("=", 1)
    model, rnd, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
    pr = {k: v.detach().float() for k, v in model.properties.items()}
    if renderer0 is None:
        renderer0 = rnd
        for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
            setattr(renderer0, f, False)
    m = inbox(pr["means"]); model.properties = {k: v[m] for k, v in pr.items()}; model.to(dev); del pr
    for nm in names:
        cam, _ = camera(nm)
        with torch.no_grad():
            o = renderer0(cam.to_device(dev), model, bg_color=torch.zeros(3, device=dev))
        Z[(lab, nm)] = o["surf_depth"][0].float().cpu(); A[(lab, nm)] = o["rend_alpha"][0].float().cpu()
    del model; torch.cuda.empty_cache()
    print(f"[{lab}] 渲染 {len(names)} 台訓練視角", flush=True)

labs = [s.split("=", 1)[0] for s in a.models]
print(f"\nb{B}｜{len(names)} 台訓練視角｜像素＝三個模型都 alpha>0.95 且單目有效｜z＝渲染 surf_depth、zm＝單目（逆深度換算）")
print(f"{'模型':12}{'像素':>11}{'z/zm 中位':>10}{'p10':>7}{'p90':>7}" + "".join(f"{'深>' + str(m) + '%':>8}{'淺>' + str(m) + '%':>8}" for m in (1, 2, 5, 10)))
for lab in labs:
    rs = []
    for nm in names:
        W, H = Z[(lab, nm)].shape[1], Z[(lab, nm)].shape[0]
        zm = mono_z(nm, W, H)
        ok = torch.isfinite(zm)
        for l2 in labs:
            ok &= A[(l2, nm)] > 0.95
        rs.append((Z[(lab, nm)][ok] / zm[ok]).numpy())
    r = np.concatenate(rs)
    cols = "".join(f"{100 * np.mean(r > 1 + m / 100):>7.1f}%{100 * np.mean(r < 1 - m / 100):>7.1f}%" for m in (1, 2, 5, 10))
    print(f"{lab:12}{len(r):>11,}{np.median(r):>10.4f}{np.percentile(r, 10):>7.3f}{np.percentile(r, 90):>7.3f}{cols}", flush=True)
print("DONE calib_mono", flush=True)
