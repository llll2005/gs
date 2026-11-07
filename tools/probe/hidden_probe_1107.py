"""11-07 使用者 viewer：「看了建築內部，還是有高斯在建築中間，沒有達成建築中空」「CityGS 地底塊比我們少」。
量「被藏起來的顆粒」：每顆中心投到本塊訓練相機（均勻抽 --ncam 台），在該像素渲染 alpha > 0.95 且顆粒深度 > 表面深度 x (1+margin) 記「被擋」，
其餘在畫面內的記「看得到」。全部被擋＝畫面內 >= 3 台且 0 台看得到（＝遮擋回收 occl_recycle 的判準，只是改成看全部相機）。
分區（同 sprite_bands：2 m 格 SfM 中位 < 3 m＝地面格，其餘有 SfM 的格＝建築格；高度＝離地面格中位）：
  地下（地面格、< −0.5 m）、地面（地面格 ±0.5 m）、建築範圍內（建築格、離地 > 3 m）、其他。只看本塊盒內、o > 0.5 的顆粒。GPU；只讀。
用法：python tools/probe/hidden_probe_1107.py --block 3 --models 標籤=ckpt ...
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.ndimage import maximum_filter, uniform_filter
sys.path.insert(0, ".")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_images_binary, read_cameras_binary, read_points3D_binary
from internal.cameras.cameras import Cameras
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--ncam", type=int, default=150)
ap.add_argument("--margin", type=float, default=0.03)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda")
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.); bb1 = pc.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.05)
Rt = P["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double() @ Rt.T)[:, :2])[B].numpy()

# 地面格／建築格（同 sprite_bands_1107）
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
Pp = np.stack([q.xyz for q in pts.values()]); ok = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = Pp[ok & inb(Pp, bb1)]; G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u] = S[order, 2][f + c // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03); bldg = np.isfinite(med) & ~ground
gz = np.where(ground, med, np.nan); num = uniform_filter(np.nan_to_num(gz), size=9); den = uniform_filter(np.isfinite(gz).astype(float), size=9)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nanmedian(gz)))


def regions(x):
    cc = np.clip(np.floor(x[:, :2] / G).astype(int) - o0, 0, np.array(shp) - 1)
    h = x[:, 2] - gfill[cc[:, 0], cc[:, 1]]
    g = ground[cc[:, 0], cc[:, 1]]; b = bldg[cc[:, 0], cc[:, 1]]
    R = {"地下（地面格 < −0.5 m）": g & (h < -0.005), "地面（±0.5 m）": g & (np.abs(h) <= 0.005), "建築範圍內（離地 > 3 m）": b & (h > 0.03)}
    R["其他"] = ~(R["地下（地面格 < −0.5 m）"] | R["地面（±0.5 m）"] | R["建築範圍內（離地 > 3 m）"])
    return R


names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}; cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names = [n for n in names if n in ims]
names = [names[i] for i in np.linspace(0, len(names) - 1, min(a.ncam, len(names))).round().astype(int)]


def mk(nm):
    im = ims[nm]; cc = cams[im.camera_id]; W, H = round(cc.width / 1.2), round(cc.height / 1.2); s = W / cc.width
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(im.qvec2rotmat()), T=t(im.tvec), fx=t([cc.params[0] * s])[0], fy=t([cc.params[1] * s])[0], cx=t([cc.params[2] * s])[0],
                   cy=t([cc.params[3] * s])[0], width=torch.tensor([W], dtype=torch.int16), height=torch.tensor([H], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1), distortion_params=None,
                   camera_type=torch.zeros(1, dtype=torch.int8))[0]


CAMS = [mk(n) for n in names]
print(f"b{B}｜訓練相機 {len(CAMS)} 台（均勻抽）｜被擋＝該像素 alpha > 0.95 且顆粒深度 > 表面 x {1 + a.margin:g}｜全部被擋＝畫面內 >= 3 台且 0 台看得到｜只看本塊盒內 o > 0.5", flush=True)
rnd0 = None
rows = []
for spec in a.models:
    lab, pat = spec.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    model, rnd, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
    pr = {k: v.detach().float() for k, v in model.properties.items()}
    if rnd0 is None:
        rnd0 = rnd
        for fl in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
            if hasattr(rnd0, fl):
                setattr(rnd0, fl, False)
    k = torch.from_numpy(inb(pr["means"].numpy(), bb0)); model.properties = {kk: v[k] for kk, v in pr.items()}; model.to(dev); del pr
    X = model.get_xyz.detach().float(); N = X.shape[0]
    n_in = torch.zeros(N, device=dev); n_occ = torch.zeros(N, device=dev)
    Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    for cam in CAMS:
        cd = cam.to_device(dev)
        with torch.no_grad():
            o = rnd0(cd, model, bg_color=torch.zeros(3, device=dev))
            al = o["rend_alpha"].reshape(o["rend_alpha"].shape[-2:]); zs = o["surf_depth"].reshape(al.shape)
            H, W = al.shape
            p = Xh @ cd.world_to_camera.float(); zc = p[:, 2]
            uu = (float(cd.fx) * p[:, 0] / zc.clamp_min(1e-6) + float(cd.cx)).floor().long(); vv = (float(cd.fy) * p[:, 1] / zc.clamp_min(1e-6) + float(cd.cy)).floor().long()
            inn = (zc > 1e-3) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            uu, vv = uu.clamp(0, W - 1), vv.clamp(0, H - 1)
            occ = inn & (al[vv, uu] > 0.95) & (zs[vv, uu] > 0) & (zc > zs[vv, uu] * (1 + a.margin))
            n_in += inn.float(); n_occ += occ.float()
    op = (torch.sigmoid(model.gaussians["opacities"].detach()).reshape(-1) > 0.5).cpu().numpy()
    n_in = n_in.cpu().numpy(); n_vis = n_in - n_occ.cpu().numpy()
    hid = (n_in >= 3) & (n_vis == 0); most = (n_in >= 3) & (n_vis <= 0.05 * n_in)
    R = regions(X.cpu().numpy().astype(np.float64))
    print(f"\n[{lab}] 盒內 {N:,} 顆、不透明 {int(op.sum()):,}｜全部被擋的不透明 {int((hid & op).sum()):,}（{100 * (hid & op).sum() / max(op.sum(), 1):.1f}%）"
          f"、95% 以上視角被擋 {int((most & op).sum()):,}（{100 * (most & op).sum() / max(op.sum(), 1):.1f}%）", flush=True)
    for rn, m in R.items():
        mo = m & op
        print(f"   {rn:22} 不透明 {int(mo.sum()):>9,}｜全部被擋 {int((mo & hid).sum()):>8,}（{100 * (mo & hid).sum() / max(mo.sum(), 1):5.1f}%）"
              f"｜95% 以上被擋 {int((mo & most).sum()):>8,}（{100 * (mo & most).sum() / max(mo.sum(), 1):5.1f}%）", flush=True)
    del model; torch.cuda.empty_cache()
print("DONE")
