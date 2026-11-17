"""11-18 使用者「錨定層能不能用在高樓牆面（尤其高樓層）」的前提檢查（只讀、CPU；GT 只當量尺）。
F1 立面的多視角共識平面夠不夠準：make_plane_depth.py 每張影像的立面區（kind 1）平面 => 世界座標；區重心落在某棟高樓足跡內的，
   在該棟裡依法向方位（< --ang 度）與截距（< --ctol x 該區相機距離）貪婪分群，每群加權中位＋去離群重擬＝一面牆的共識平面。
   對 GT 立面點（16 近鄰 PCA、|n_z| < 0.3）：同棟、法向相容（夾角 < 15°）的共識平面裡最近的 |n·X − c|（m）；分高度帶報中位、p90、< 1／2 m 比例、
   沒有相容平面的比例（＝涵蓋缺口）。對照＝單張影像的區平面（每區重心 10 m 內、法向相容的 GT 點，同一個量）。
F2 分面：每棟的共識平面數、每面的區數、法向分佈（轉角／退縮會變成多面）。
F3 顆數：高度帶內 GT 立面面積（點數 x 間距²）÷（k px 的足跡長度）²，k＝2、4；足跡＝最近可見訓練相機的 1 px 長度（深度／fx，同 px_floor，不算遮擋）。
用法（lab）：python tools/probe/facade_plane_1118.py --block 2 [--block 3]
"""
import argparse, os, sys
import numpy as np, torch
from scipy.ndimage import label, maximum_filter
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary, read_cameras_binary
from internal.utils import colmap_fast
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--top", type=float, default=1.5)
ap.add_argument("--plane_dir", default="estimated_depths_plane"); ap.add_argument("--gt", default="data/geometry_gt/MC_Aerial/Block_all_ds.ply")
ap.add_argument("--ang", type=float, default=8.0); ap.add_argument("--ctol", type=float, default=0.02); ap.add_argument("--down", type=float, default=1.2)
ap.add_argument("--nsample", type=int, default=20000)
ap.add_argument("--ctol_px", type=float, default=0.0, help="> 0：截距容許值改成 該區相機距離／fx x 這麼多像素（像素足跡；取代 --ctol）")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()
# ── 棟（同 tower_gt_1113）──
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
P = P[(TRK >= 3) & (ERR < 1.5) & inblk(P)]
ZG = float(np.percentile(P[:, 2], 5))
CELL = 0.1
o0 = np.floor(P[:, :2].min(0) / CELL).astype(int); ci = np.floor(P[:, :2] / CELL).astype(int) - o0
shp = tuple(ci.max(0) + 1); top = np.full(shp, -1e9); np.maximum.at(top, (ci[:, 0], ci[:, 1]), P[:, 2])
lab_, n = label(top > 1.0); bid = lab_[ci[:, 0], ci[:, 1]]
tall = [k for k in range(1, n + 1) if (bid == k).sum() >= 50 and np.percentile(P[bid == k, 2], 99) > a.top]
foot = np.zeros(shp, np.int32)
for k in tall:
    foot[maximum_filter(lab_ == k, size=3) & (foot == 0)] = k
def bld(X):
    c = np.floor(X[:, :2] / CELL).astype(int) - o0
    ok = (c[:, 0] >= 0) & (c[:, 0] < shp[0]) & (c[:, 1] >= 0) & (c[:, 1] < shp[1])
    r = np.zeros(len(X), np.int32); r[ok] = foot[c[ok, 0], c[ok, 1]]; return r
# ── GT 立面 ──
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
nd, nn = cKDTree(gx).query(gx, k=16)
Q = gx[nn] - gx[nn].mean(1, keepdims=True); _, _, ev = np.linalg.svd(Q, full_matrices=False); del Q, nn
gn = ev[:, 2, :]; fac = np.abs(gn[:, 2]) < 0.3
gx, gn, gb = gx[fac], gn[fac], gb[fac]; sp = float(np.median(nd[:, 1]))
gh = (gx[:, 2] - ZG) * M
print(f"b{B}｜高樓 {len(tall)} 棟｜GT 立面點 {len(gx):,}（間距中位 {sp * M:.2f} m）", flush=True)
# ── 相機與區平面 ──
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin"); imgs = colmap_fast.read_images_binary_light(f"{D}/sparse/0/images.bin")
by_name = {v.name: v for v in imgs.values()}
names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/%03d_%03d.txt" % (B % 4, B // 4)) if l.strip() in by_name]
from internal.utils.colmap import qvec2rotmat
REG = []    # (棟, n_w, c_w, 權重, 相機距離, 重心世界點)
CAMS = []
for nm in names:
    im = by_name[nm]; cm = cams[im.camera_id]; R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec, float)
    fx, fy, cx, cy = cm.params[:4]; CAMS.append((R, t, fx, fy, cx, cy, cm.width, cm.height))
    f = os.path.join(D, a.plane_dir, nm + "_planes.npz")
    if not os.path.exists(f):
        continue
    z = np.load(f); s = z["kind"] == 1
    if not s.any():
        continue
    W, H = z["size"]; C = -R.T @ t
    for n_c, c, npx, cu, cv in zip(z["n"][s], z["c"][s], z["npix"][s], z["cu"][s], z["cv"][s]):
        n_w = n_c @ R; c_w = c - n_c @ t
        if n_w @ C - c_w < 0:
            n_w, c_w = -n_w, -c_w
        d = np.array([((cu + 0.5) * cm.width / W - cx) / fx, ((cv + 0.5) * cm.height / H - cy) / fy, 1.0])
        den = d @ n_c
        if abs(den) < 1e-9:
            continue
        zc = c / den
        if zc <= 0:
            continue
        Xw = (d * zc - t) @ R
        k = int(bld(Xw[None])[0])
        if k > 0 and inblk(Xw[None])[0]:
            REG.append((k, n_w, c_w, float(npx), float(np.linalg.norm(Xw - C)), Xw))
print(f"區平面：{len(names)} 張影像、落在高樓足跡內的立面區 {len(REG):,}", flush=True)
FXT = CAMS[0][2] / a.down
# ── 分群＋共識 ──
def wmed(x, w):
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])
PL = []     # (棟, n, c, 區數)
for k in tall:
    R_ = [r for r in REG if r[0] == k]
    R_.sort(key=lambda r: -r[3]); cl = []
    for r in R_:
        for g in cl:
            tol = a.ctol_px * r[4] / FXT if a.ctol_px > 0 else a.ctol * r[4]
            if np.degrees(np.arccos(np.clip(r[1] @ g["n"], -1, 1))) < a.ang and abs(r[2] - g["c"]) < tol:
                g["m"].append(r); break
        else:
            cl.append({"n": r[1], "c": r[2], "m": [r]})
    for g in cl:
        if len(g["m"]) < 3:
            continue
        Nw = np.stack([r[1] for r in g["m"]]); Cw = np.array([r[2] for r in g["m"]]); Wt = np.array([r[3] for r in g["m"]])
        n0 = np.array([wmed(Nw[:, i], Wt) for i in range(3)]); n0 /= np.linalg.norm(n0); c0 = wmed(Cw, Wt)
        ang = np.degrees(np.arccos(np.clip(Nw @ n0, -1, 1)))
        mA = wmed(np.abs(ang - wmed(ang, Wt)), Wt) + 1e-9; mC = wmed(np.abs(Cw - c0), Wt) + 1e-12
        inl = (ang <= wmed(ang, Wt) + 3 * 1.4826 * mA) & (np.abs(Cw - c0) <= 3 * 1.4826 * mC)
        n1 = (Nw[inl] * Wt[inl, None]).sum(0); n1 /= np.linalg.norm(n1); c1 = float((Cw[inl] * Wt[inl]).sum() / Wt[inl].sum())
        PL.append((k, n1, c1, int(inl.sum())))
print(f"F2｜共識平面 {len(PL)} 面（每面 >= 3 區）｜每棟：" + "、".join(f"棟{k} {sum(1 for q in PL if q[0] == k)} 面" for k in tall), flush=True)
# ── F1：GT 立面點到相容平面的距離 ──
def dist_to(planes, X, N, B_):
    out = np.full(len(X), np.inf)
    for (k, n_, c_, *_r) in planes:
        s = (B_ == k) & (np.abs(N @ n_) > np.cos(np.radians(15)))
        if s.any():
            out[s] = np.minimum(out[s], np.abs(X[s] @ n_ - c_))
    return out * M
dc = dist_to(PL, gx, gn, gb)
BANDS = ((0, 150), (150, 250), (250, 350), (350, 600))
def row(lab, d, sel):
    dd = d[sel]; fin = np.isfinite(dd)
    if not fin.any():
        return f"   {lab}：沒有相容平面"
    return (f"   {lab}：GT 立面點 {int(sel.sum()):,}｜沒有相容平面 {100 * (~fin).mean():.0f}%｜距離中位 {np.median(dd[fin]):.2f} m、p90 {np.percentile(dd[fin], 90):.2f} m｜"
            f"< 1 m {100 * (dd[fin] < 1).mean():.0f}%、< 2 m {100 * (dd[fin] < 2).mean():.0f}%")
print("F1｜共識平面（同棟、法向相容的最近一面）", flush=True)
for lo, hi in BANDS:
    print(row(f"{lo}～{hi} m", dc, (gh >= lo) & (gh < hi)), flush=True)
# 對照：單張區平面（每區重心 10 m 內、法向相容的 GT 點）
gt_tree = cKDTree(gx)
ds_, dsel = [], []
for (k, n_w, c_w, w, dep, Xw) in REG:
    idx = gt_tree.query_ball_point(Xw, 0.1)
    if not idx:
        continue
    idx = np.array(idx); idx = idx[(gb[idx] == k) & (np.abs(gn[idx] @ n_w) > np.cos(np.radians(15)))]
    if len(idx):
        ds_.append(np.median(np.abs(gx[idx] @ n_w - c_w)) * M); dsel.append((Xw[2] - ZG) * M)
ds_ = np.array(ds_); dsel = np.array(dsel)
for (k, n1, c1, ni) in PL:
    pass
print("F1 對照｜單張區平面（區重心 10 m 內的 GT 點，逐區中位）", flush=True)
for lo, hi in BANDS:
    s = (dsel >= lo) & (dsel < hi)
    if s.any():
        print(f"   {lo}～{hi} m：{int(s.sum()):,} 區｜逐區距離中位的中位 {np.median(ds_[s]):.2f} m、p90 {np.percentile(ds_[s], 90):.2f} m", flush=True)
# 共識平面用同樣的量（區重心 10 m 內）比
dcr = []
for (k, n_w, c_w, w, dep, Xw) in REG:
    cand = [q for q in PL if q[0] == k and abs(q[1] @ n_w) > np.cos(np.radians(15))]
    if not cand:
        continue
    q = min(cand, key=lambda q: abs(Xw @ q[1] - q[2]))
    idx = gt_tree.query_ball_point(Xw, 0.1)
    if not idx:
        continue
    idx = np.array(idx); idx = idx[(gb[idx] == k) & (np.abs(gn[idx] @ q[1]) > np.cos(np.radians(15)))]
    if len(idx):
        dcr.append(((Xw[2] - ZG) * M, np.median(np.abs(gx[idx] @ q[1] - q[2])) * M))
if dcr:
    dcr = np.array(dcr)
    print("F1 同量比｜共識平面（同一批區的位置）", flush=True)
    for lo, hi in BANDS:
        s = (dcr[:, 0] >= lo) & (dcr[:, 0] < hi)
        if s.any():
            print(f"   {lo}～{hi} m：{int(s.sum()):,} 區｜距離中位的中位 {np.median(dcr[s, 1]):.2f} m、p90 {np.percentile(dcr[s, 1], 90):.2f} m", flush=True)
# ── F3：顆數 ──
rng = np.random.default_rng(0)
sel = rng.choice(len(gx), size=min(a.nsample, len(gx)), replace=False)
X = gx[sel]; best = np.full(len(X), np.inf)
for R, t, fx, fy, cx, cy, Wc, Hc in CAMS:
    Xc = X @ R.T + t; z = Xc[:, 2]
    u = fx * Xc[:, 0] / np.maximum(z, 1e-9) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-9) + cy
    ok = (z > 1e-6) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
    best = np.where(ok & (z < best), z, best)
fxt = CAMS[0][2] / a.down
pl = best / fxt; ok = np.isfinite(pl)
scale = len(gx) / len(X)
print(f"F3｜每個 GT 立面點代表 {sp * M:.2f}² m²；最近可見相機的 1 px 長度中位 {np.median(pl[ok]) * M:.2f} m", flush=True)
for lo, hi in BANDS:
    s = ok & (gh[sel] >= lo) & (gh[sel] < hi)
    if s.any():
        area = s.sum() * scale * (sp * M) ** 2
        cnt = {kpx: int(scale * np.sum((sp / (kpx * pl[s])) ** 2)) for kpx in (2, 4)}
        print(f"   {lo}～{hi} m：立面面積約 {area:,.0f} m²｜1 px 長度中位 {np.median(pl[s]) * M:.2f} m｜鋪滿要 2 px：{cnt[2]:,} 顆、4 px：{cnt[4]:,} 顆", flush=True)
print("DONE facade_plane", flush=True)
