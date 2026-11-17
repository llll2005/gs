"""11-17 Fable 第四輪 Q3 第 1 步：牆後那層是沒用的殘留，還是在畫東西（玻璃反射）？用 GT 當 oracle 刪掉它們，再量換塊畫質與低角度透空。
範圍與 tower_gt_1113.py 相同：SfM 頂高 > top 的高樓足跡（10 m 格膨脹一格）內、本塊盒內；GT 立面點（16 近鄰 PCA，|n_z| < 0.3）。
朝外方向：每棟、每 50 m 高度帶，用該帶 GT 立面點的水平重心定（同 tower_gt；凹口處可能判反）。
刪除：不透明（o > 0.5）、離同帶 GT 立面點 15 m 內、沿朝外法向的有號距離 < −behind（預設 1 m）的面元。
輸出：outputs/lab/<out_run>/blocks/block_<B>/checkpoints/<同檔名>（state_dict 裡第一維＝N 的張量都刪同一批；不帶 optimizer 狀態）。
GT 只拿來挑要刪的顆粒（這是 oracle 測試，不是機制）。
用法（lab）：python tools/probe/oracle_behind_1117.py --block 2 --src "<ckpt glob>" --out_run oracle_behind_b6
"""
import argparse, glob, os, sys
import numpy as np, torch
from scipy.ndimage import label, maximum_filter
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--src", required=True); ap.add_argument("--out_run", required=True)
ap.add_argument("--top", type=float, default=1.5); ap.add_argument("--behind", type=float, default=0.01)
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
rel = (TRK >= 3) & (ERR < 1.5); ZG = float(np.percentile(P[rel, 2], 5)); P = P[rel & inblk(P)]
CELL = 0.1
o0 = np.floor(P[:, :2].min(0) / CELL).astype(int); ci = np.floor(P[:, :2] / CELL).astype(int) - o0
shp = tuple(ci.max(0) + 1); top = np.full(shp, -1e9); np.maximum.at(top, (ci[:, 0], ci[:, 1]), P[:, 2])
lab_, n = label(top > 1.0); bid = lab_[ci[:, 0], ci[:, 1]]
foot = np.zeros(shp, np.int32)
for k in range(1, n + 1):
    s = bid == k
    if s.sum() >= 50 and np.percentile(P[s, 2], 99) > a.top:
        foot[maximum_filter(lab_ == k, size=3) & (foot == 0)] = k
def bld(m):
    c = np.floor(m[:, :2] / CELL).astype(int) - o0
    ok = (c[:, 0] >= 0) & (c[:, 0] < shp[0]) & (c[:, 1] >= 0) & (c[:, 1] < shp[1])
    r = np.zeros(len(m), np.int32); r[ok] = foot[c[ok, 0], c[ok, 1]]; return r
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
gb = bld(gx); keep = (gb > 0) & inblk(gx); gx, gb = gx[keep], gb[keep]
_, nn = cKDTree(gx).query(gx, k=16)
Q = gx[nn] - gx[nn].mean(1, keepdims=True); _, _, ev = np.linalg.svd(Q, full_matrices=False); del Q, nn
gn = ev[:, 2, :]; fac = np.abs(gn[:, 2]) < 0.3
gx, gn, gb = gx[fac], gn[fac], gb[fac]
gband = np.floor((gx[:, 2] - ZG) * M / 50).astype(int)
# 朝外：每棟每帶的水平重心
for (k, bnd) in set(zip(gb.tolist(), gband.tolist())):
    s = (gb == k) & (gband == bnd)
    sg = np.sign(((gx[s, :2] - gx[s, :2].mean(0)) * gn[s, :2]).sum(1)); gn[s] *= np.where(sg == 0, 1, sg)[:, None]
tree = cKDTree(gx)
f = sorted(x for x in glob.glob(a.src) if "xyz" not in x)[-1]
ck = torch.load(f, map_location="cpu", weights_only=False); sd = ck["state_dict"]; g = "gaussian_model.gaussians."
m = sd[g + "means"].double().numpy(); o = torch.sigmoid(sd[g + "opacities"].float()).reshape(-1).numpy(); N = len(m)
d, j = tree.query(m, distance_upper_bound=0.15)
near = np.isfinite(d) & (o > 0.5) & inblk(m)
jj = np.where(near, j, 0)
sdv = ((m - gx[jj]) * gn[jj]).sum(1)
same = near & (bld(m) == gb[jj])
dele = same & (sdv < -a.behind)
h = (m[:, 2] - ZG) * M
print(f"b{B}｜{f}｜N {N:,}｜高樓立面 15 m 內不透明 {same.sum():,}｜牆後 > {a.behind * M:.0f} m：刪 {dele.sum():,}（{100 * dele.sum() / max(same.sum(), 1):.1f}%）", flush=True)
for lo, hi in ((0, 150), (150, 250), (250, 350), (350, 500)):
    k = same & (h >= lo) & (h < hi)
    print(f"   {lo}～{hi} m：立面附近 {k.sum():,}、刪 {int((dele & k).sum()):,}（{100 * (dele & k).sum() / max(k.sum(), 1):.0f}%）", flush=True)
keepm = torch.from_numpy(~dele)
for key, v in list(sd.items()):
    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == N:
        sd[key] = v[keepm]
ck["optimizer_states"] = []
out = os.path.join("outputs/lab", a.out_run, "blocks", f"block_{B}", "checkpoints", os.path.basename(f))
os.makedirs(os.path.dirname(out), exist_ok=True); torch.save(ck, out)
print(f"=> {out}（N {int((~dele).sum()):,}）")
print("DONE oracle_behind")
