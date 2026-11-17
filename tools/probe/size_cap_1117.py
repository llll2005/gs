"""11-17 Fable 第四輪 Q2：面元大小上限（3σ 長軸 <= 最近可見相機 K px）設下去會怎樣。只讀、CPU。
每顆不透明度 > 0.02 的面元（本塊盒內）：
  長軸像素＝3 x 長軸 ÷（最近可見訓練相機的 1 px 長度＝最小深度 / fx_train；同 px_floor 的定義，不做遮擋）
  分類（GT 不用，只用 SfM 與面元自己的朝向）：海面（所在 4 m 格的可靠 SfM 中位在海平面 ±1 m，或 3x3 鄰域沒有可靠 SfM）、
        陸地躺平（|n_z| > 0.8）、立面（|n_z| < 0.3）、其他
  對每個 K：超過 K px 的顆數、它們佔該類 Σo·面積 的比例；把它們改用上限大小鋪滿需要的顆數（每軸 ceil(長度 / 上限)，相乘）
用法（lab）：python tools/probe/size_cap_1117.py --block 3 --models 新底6=<glob> ...
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.ndimage import maximum_filter
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--K", type=int, nargs="+", default=[16, 32, 64, 128]); ap.add_argument("--down", type=float, default=1.2)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); E = np.array([q.error for q in pts.values()]); T = np.array([len(q.image_ids) for q in pts.values()]); del pts
S = P[(T >= 3) & (E < 1.5)]; del P
zg = S[:, 2]; e = np.arange(-0.10, 0.0501, 0.001); hh, _ = np.histogram(zg, bins=e); hs = np.convolve(hh, np.ones(5) / 5, mode="same")
pk = [i for i in range(2, len(hs) - 2) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.02 * hs.max()]
ZS = e[pk[0]] + 0.0005
cell = 0.04; o0 = np.floor(S[:, :2].min(0) / cell).astype(int) - 2; ci = np.floor(S[:, :2] / cell).astype(int) - o0; shp = tuple(ci.max(0) + 3)
cid = ci[:, 0] * shp[1] + ci[:, 1]; od = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[od], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u] = S[od, 2][f + c // 2]; med = med.reshape(shp)
SEA = (np.isfinite(med) & (np.abs(med - ZS) < 0.01)) | ~(maximum_filter(np.isfinite(med).astype(np.uint8), size=3) > 0)
def is_sea(x):
    q = np.floor(x[:, :2] / cell).astype(int) - o0; ok = (q[:, 0] >= 0) & (q[:, 0] < shp[0]) & (q[:, 1] >= 0) & (q[:, 1] < shp[1])
    r = np.ones(len(x), bool); r[ok] = SEA[q[ok, 0], q[ok, 1]]; return r
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin"); imgs = read_images_binary(f"{D}/sparse/0/images.bin")
by_name = {v.name: v for v in imgs.values()}
lst = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/%03d_%03d.txt" % (B % 4, B // 4)) if l.strip()]
CAM = []
for nm in lst:
    im = by_name.get(nm)
    if im is None:
        continue
    cm = cams[im.camera_id]; fx = cm.params[0]; fy = cm.params[1] if len(cm.params) > 3 else fx
    cx, cy = (cm.params[2], cm.params[3]) if len(cm.params) > 3 else (cm.params[1], cm.params[2])
    CAM.append((qvec2rotmat(im.qvec), np.asarray(im.tvec), fx, fy, cx, cy, cm.width, cm.height))
FXT = CAM[0][2] / a.down
def px_len(X):
    best = np.full(len(X), np.inf)
    for R, t, fx, fy, cx, cy, W, H in CAM:
        Xc = X @ R.T + t; z = Xc[:, 2]
        uu = fx * Xc[:, 0] / np.maximum(z, 1e-9) + cx; vv = fy * Xc[:, 1] / np.maximum(z, 1e-9) + cy
        ok = (z > 1e-6) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        best = np.where(ok & (z < best), z, best)
    return best / FXT
print(f"b{B}｜相機 {len(CAM)}｜海平面 {ZS * 100:.1f} m", flush=True)
for spec in a.models:
    lab, pat = spec.split("=", 1); fsel = sorted(x for x in glob.glob(pat) if "xyz" not in x)[-1]
    sd = torch.load(fsel, map_location="cpu", weights_only=False)["state_dict"]; g = "gaussian_model.gaussians."
    m = sd[g + "means"].double().numpy(); o = torch.sigmoid(sd[g + "opacities"].float()).reshape(-1).numpy()
    s = torch.exp(sd[g + "scales"].float()).numpy()[:, :2]; q = sd[g + "rotations"].float().numpy().astype(np.float64); del sd
    k = (o > 0.02) & inblk(m); m, o, s, q = m[k], o[k], s[k], q[k]
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T; nz = np.abs(1 - 2 * (x * x + y * y))
    pl = px_len(m); ok = np.isfinite(pl); m, o, s, nz, pl = m[ok], o[ok], s[ok], nz[ok], pl[ok]
    Lpx = 3 * s.max(1) / pl; area = o * 2 * np.pi * s[:, 0] * s[:, 1]
    sea = is_sea(m); cls = {"海面": sea, "陸地躺平": ~sea & (nz > 0.8), "立面": ~sea & (nz < 0.3), "其他": ~sea & (nz >= 0.3) & (nz <= 0.8)}
    print(f"[{lab}] {fsel}｜{len(m):,} 顆｜長軸像素中位 {np.median(Lpx):.1f}、p99 {np.percentile(Lpx, 99):.0f}、最大 {Lpx.max():.0f}", flush=True)
    for K in a.K:
        big = Lpx > K
        need = np.ceil(np.clip(3 * s[:, 0] / pl / K, 1, None)) * np.ceil(np.clip(3 * s[:, 1] / pl / K, 1, None))
        row = f"   K={K:>3} px：超過 {big.sum():,} 顆（{100 * big.mean():.2f}%）"
        for cn, cm in cls.items():
            bk = big & cm
            row += f"｜{cn} {bk.sum():,} 顆、佔該類面積 {100 * area[bk].sum() / max(area[cm].sum(), 1e-12):.0f}%、要補 {int((need[bk] - 1).sum()):,}"
        print(row, flush=True)
print("DONE size_cap")
