"""11-17 Fable 第三輪第 5 題：轉角糊掉是誰引進的。只讀、CPU。
GT（MatrixCity 幾何 GT 點雲，16 近鄰 PCA 法向）本塊盒內：立面＝|n_z| < 0.3；轉角＝立面點 1.5 m 內有另一個立面點、兩者法向 |cos| < 0.7（夾角 > 45°），
它的兩面牆法向 n1（自己）、n2（鄰居裡最不一樣的那個）。
每個模型：本塊盒內不透明面元（o > 0.5），離最近 GT 立面點 0.75 m 內的歸給那個點：
  轉角點的面元：貼 n1 或 n2（|cos| > 0.9）＝貼某一面牆；兩面都不貼＝「跨角／斜切」（Fable 的跨角計數）
  平牆點的面元：朝向對不上（|cos| <= 0.9）的比例（對照）
依高度帶報：每個轉角 GT 點分到幾顆面元、跨角比例、跨角面元長軸中位；平牆的對不上比例。
對照模型（現成 ckpt，不重訓）：有／拿掉鄰域平面、拿掉法向一致性、A＋dist4、CityGS。
用法（lab）：python tools/probe/corner_1117.py --block 1 --models CityGS=<glob> 底=<glob> 拿掉鄰域平面=<glob> ...
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--gt", default="data/geometry_gt/MC_Aerial/Block_all_ds.ply")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
ZG = float(np.percentile(P[(TRK >= 3) & (ERR < 1.5), 2], 5)); del P

with open(a.gt, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"end_header\n"):
        hdr += f.readline()
    props, nv = [], 0
    for ln in hdr.decode().splitlines():
        t = ln.split()
        if t[:2] == ["element", "vertex"]:
            nv = int(t[2])
        elif t and t[0] == "property":
            props.append((t[2], {"double": "<f8", "float": "<f4", "uchar": "u1"}[t[1]]))
    G = np.fromfile(f, dtype=np.dtype(props), count=nv)
gx = np.stack([G["x"], G["y"], G["z"]], 1); del G
gx = gx[inblk(gx)]
tree = cKDTree(gx)
_, nn = tree.query(gx, k=16)
Q = gx[nn] - gx[nn].mean(1, keepdims=True); _, _, evec = np.linalg.svd(Q, full_matrices=False); del Q, nn
gn = evec[:, 2, :]; del evec
fac = np.abs(gn[:, 2]) < 0.3
fx_, fn_ = gx[fac], gn[fac]
ft = cKDTree(fx_)
dd, jj = ft.query(fx_, k=16, distance_upper_bound=0.015)
okn = np.isfinite(dd); jj = np.where(okn, jj, 0)
cosn = np.where(okn, np.abs((fn_[:, None, :] * fn_[jj]).sum(-1)), 1.0)
kmin = cosn.argmin(1); corner = cosn[np.arange(len(fx_)), kmin] < 0.7
n2 = fn_[jj[np.arange(len(fx_)), kmin]]
h = (fx_[:, 2] - ZG) * M
print(f"b{B}｜GT 盒內 {len(gx):,} 點｜立面 {len(fx_):,}（轉角 {corner.sum():,}＝{100 * corner.mean():.1f}%）", flush=True)

BANDS = [(-1e9, 50), (50, 150), (150, 1e9)]
rows = []
for spec in a.models:
    labm, pat = spec.split("=", 1); fs = sorted(f for f in glob.glob(pat) if "xyz" not in f)
    if not fs:
        print(f"⚠ 缺 {labm}：{pat}"); continue
    sd = torch.load(fs[-1], map_location="cpu", weights_only=False)["state_dict"]; g = "gaussian_model.gaussians."
    m = sd[g + "means"].double().numpy(); o = torch.sigmoid(sd[g + "opacities"].float()).reshape(-1).numpy()
    s = torch.exp(sd[g + "scales"].float()).numpy()[:, :2]; q = sd[g + "rotations"].float().numpy().astype(np.float64); del sd
    k = (o > 0.5) & inblk(m); m, s, q = m[k], s[k], q[k]
    d, j = ft.query(m, distance_upper_bound=0.0075)
    k2 = np.isfinite(d); s, q, j = s[k2], q[k2], j[k2]
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    nrm = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)   # R[:, 2]＝2DGS 面元法向
    c1 = np.abs((nrm * fn_[j]).sum(1)); c2 = np.abs((nrm * n2[j]).sum(1))
    isc = corner[j]; diag = isc & (c1 <= 0.9) & (c2 <= 0.9)
    lmax = s.max(1) * M
    r = [labm]
    for lo, hi in BANDS:
        gb = (h >= lo) & (h < hi); hb = gb[j]
        ncorner = int((corner & gb).sum()); nflat = int((~corner & gb).sum())
        kc = hb & isc; kf = hb & ~isc
        r.append((kc.sum() / max(ncorner, 1), 100 * diag[kc].mean() if kc.any() else np.nan,
                  np.median(lmax[hb & diag]) if (hb & diag).any() else np.nan, 100 * (c1[kf] <= 0.9).mean() if kf.any() else np.nan,
                  kf.sum() / max(nflat, 1)))
    rows.append(r)
    print(f"[載入] {labm}：{fs[-1]}｜立面 0.75 m 內不透明 {len(j):,} 顆", flush=True)

print("\n=== 轉角：每個轉角 GT 點分到的面元｜其中跨角（兩面牆都不貼）％｜跨角面元長軸中位（m）｜平牆對照：每點面元、朝向對不上％ ===")
names = [f"{'<50' if lo < -1e8 else lo}~{'∞' if hi > 1e8 else hi} m" for lo, hi in BANDS]
print(f"{'模型':14}" + "".join(f"{n:>44}" for n in names))
for r in rows:
    print(f"{r[0]:14}" + "".join(f"{f'{c:.2f} 顆｜跨角 {d:4.0f}%｜{l:4.2f} m｜平牆 {pf:.2f} 顆 {fm:3.0f}%':>44}" for c, d, l, fm, pf in r[1:]))
print("DONE corner")
