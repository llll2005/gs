"""11-17 Fable 第四輪 Q5：內凹轉角的抹痕是不是稠密起點的「體素質心」造成的。只讀、CPU。
嫌疑：起點把同一個體素裡的點取平均；轉角的體素同時裝了兩面牆的點，平均點不在任何一面牆上。
  外凸角：平均點落在建築內部（兩面牆的後面），看不到；內凹角：平均點落在凹口的空氣裡（兩面牆的前面），看得到、顏色是兩面的平均。
做法：
  GT（本塊盒內，16 近鄰 PCA 法向）立面點；轉角＝1.5 m 內有另一個立面點、兩者法向夾角 > 45°（同 corner_1117）。
  朝外：法向的水平分量往外 2 m 處的 GT 頂高（3 m 格）低於該點高度＝空氣 => 那一側是外；否則翻轉（GT 只用來分類）。
  凸／凹：轉角點 x 的牆 1 法向 n1（朝外）、鄰居 y 的牆 2 法向 n2（朝外）；(y − x)·n1 > 0（另一面牆在這面牆的前方）＝內凹，否則外凸。
  起點／模型的點：最近的轉角 GT 點 1.5 m 內的，算到兩面牆的有號距離 d1＝(p − x)·n1、d2＝(p − y)·n2：
    兩面牆前方（d1 > 0.3 m 且 d2 > 0.3 m）＝凹口的空氣（內凹）／角外的空氣（外凸）；兩面牆後方（都 < −0.3 m）＝建築內部；其他＝貼在某面牆上。
  模型另算面元朝向：兩面牆都對不上（|cos| <= 0.9）＝斜跨。
用法（lab）：python tools/probe/concave_corner_1117.py --block 2 --init <PLY> --models 新底6=<ckpt glob> ...
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--init", nargs="*", default=[]); ap.add_argument("--models", nargs="*", default=[])
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
gall = np.stack([G["x"], G["y"], G["z"]], 1); del G
# 頂高格（3 m）：判空氣
CELL = 0.03; lo = gall[:, :2].min(0) - 0.1
ij = ((gall[:, :2] - lo) / CELL).astype(int); shp = ij.max(0) + 2
top = np.full(shp, -1e9); np.maximum.at(top, (ij[:, 0], ij[:, 1]), gall[:, 2])
def top_at(xy):
    q = ((xy - lo) / CELL).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < shp[0]) & (q[:, 1] >= 0) & (q[:, 1] < shp[1])
    r = np.full(len(xy), -1e9); r[ok] = top[q[ok, 0], q[ok, 1]]; return r
gx = gall[inblk(gall)]
_, nn = cKDTree(gx).query(gx, k=16)
Q = gx[nn] - gx[nn].mean(1, keepdims=True); _, _, ev = np.linalg.svd(Q, full_matrices=False); del Q, nn
gn = ev[:, 2, :]; fac = np.abs(gn[:, 2]) < 0.3
fx_, fn_ = gx[fac], gn[fac].copy()
# 朝外
nh = fn_[:, :2] / np.linalg.norm(fn_[:, :2], axis=1, keepdims=True).clip(1e-9)
air = top_at(fx_[:, :2] + 0.02 * nh) < fx_[:, 2] - 0.01
fn_ *= np.where(air, 1.0, -1.0)[:, None]
ft = cKDTree(fx_)
dd, jj = ft.query(fx_, k=16, distance_upper_bound=0.015)
okn = np.isfinite(dd); jj = np.where(okn, jj, 0)
cosn = np.where(okn, np.abs((fn_[:, None, :] * fn_[jj]).sum(-1)), 1.0)
kmin = cosn.argmin(1); corner = cosn[np.arange(len(fx_)), kmin] < 0.7
jn = jj[np.arange(len(fx_)), kmin]
concave = corner & (((fx_[jn] - fx_) * fn_).sum(1) > 0)
convex = corner & ~concave
h = (fx_[:, 2] - ZG) * M
ci = np.where(corner)[0]; ct = cKDTree(fx_[ci])
print(f"b{B}｜GT 立面 {len(fx_):,}｜轉角 {corner.sum():,}（內凹 {concave.sum():,}、外凸 {convex.sum():,}）｜低於 150 m 的轉角 {int((corner & (h < 150)).sum()):,}", flush=True)

def classify(X, nrm=None):
    d, k = ct.query(X, distance_upper_bound=0.015)
    ok = np.isfinite(d); k = np.where(ok, k, 0); xi = ci[k]; yi = jn[xi]
    d1 = ((X - fx_[xi]) * fn_[xi]).sum(1) * M; d2 = ((X - fx_[yi]) * fn_[yi]).sum(1) * M
    front = ok & (d1 > 0.3) & (d2 > 0.3); back = ok & (d1 < -0.3) & (d2 < -0.3)
    out = {}
    for nm_, cm in (("內凹", concave[xi]), ("外凸", convex[xi])):
        s = ok & cm & (h[xi] < 150)
        r = [int(s.sum()), 100 * (front & s).sum() / max(s.sum(), 1), 100 * (back & s).sum() / max(s.sum(), 1)]
        if nrm is not None:
            c1 = np.abs((nrm * fn_[xi]).sum(1)); c2 = np.abs((nrm * fn_[yi]).sum(1)); diag = (c1 <= 0.9) & (c2 <= 0.9)
            r += [100 * (diag & s).sum() / max(s.sum(), 1), 100 * (diag & front & s).sum() / max((front & s).sum(), 1)]
        out[nm_] = r
    return out

for spec in a.init:
    lab, f = spec.split("=", 1)
    from plyfile import PlyData
    v = PlyData.read(f)["vertex"].data; X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    r = classify(X[inblk(X)])
    print(f"[起點] {lab}：" + "｜".join(f"{k} 轉角 1.5 m 內 {v_[0]:,} 點、兩面牆前方 {v_[1]:.1f}%、兩面牆後方 {v_[2]:.1f}%" for k, v_ in r.items()), flush=True)
for spec in a.models:
    lab, pat = spec.split("=", 1); f = sorted(x for x in glob.glob(pat) if "xyz" not in x)[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]; g = "gaussian_model.gaussians."
    m = sd[g + "means"].double().numpy(); o = torch.sigmoid(sd[g + "opacities"].float()).reshape(-1).numpy(); q = sd[g + "rotations"].float().numpy().astype(np.float64); del sd
    k = (o > 0.5) & inblk(m); m, q = m[k], q[k]
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    nrm = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    r = classify(m, nrm)
    print(f"[模型] {lab}：" + "｜".join(f"{k} 轉角 1.5 m 內 {v_[0]:,} 顆、兩面牆前方 {v_[1]:.1f}%、後方 {v_[2]:.1f}%、斜跨 {v_[3]:.0f}%（前方的裡面斜跨 {v_[4]:.0f}%）" for k, v_ in r.items()), flush=True)
print("DONE concave_corner")
