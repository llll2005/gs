"""11-17 Fable 第三輪第 6 題：使用者圈的三棟高樓（A 397 m／B 427 m／C 355 m，b2／b3）低樓層平視看到崩，是不是「被旁邊的塔擋住、看到的次數少」。只讀、CPU。
每個目標 GT 立面點：所屬塊的訓練相機裡，有幾台「真的看得到」它：
  投影在畫面內、在前方；相機在牆的外側（法向朝相機）；不被擋（GT 點雲 z-buffer，1/8 解析度、每台相機只投視錐內的點；深度 <= 該格最小深度 x (1 + tol)）。
目標：三棟的足跡（GT 頂高 > 100 m 的連通足跡、含圈心那棟，外擴 12 m）內 0～150 m 的立面點。
對照：同一塊、同高度帶（0～150 m）、周圍 100 m 內沒有 > 150 m 建築的立面點（空曠處的一般樓）。
用法（lab）：python tools/probe/lowfloor_vis_1117.py [--sub 4000000] [--tol 0.02]
"""
import argparse, sys
import numpy as np, torch
from scipy.ndimage import label, binary_dilation, maximum_filter
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--sub", type=int, default=4000000); ap.add_argument("--tol", type=float, default=0.02)
ap.add_argument("--down", type=int, default=8); ap.add_argument("--ntgt", type=int, default=6000)
ap.add_argument("--gt", default="data/geometry_gt/MC_Aerial/Block_all_ds.ply")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
Rp = Rt.numpy()
inblk = lambda x, B: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()
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
rng = np.random.default_rng(0)
occ = gx[rng.choice(len(gx), min(a.sub, len(gx)), replace=False)]       # 遮擋用（全場景抽樣）
U = gx @ Rp.T
# 頂高格（3 m）=> 高樓足跡
res = 0.03; lo = U[:, :2].min(0) - 0.1
ij = ((U[:, :2] - lo) / res).astype(int); shp = ij.max(0) + 2
top = np.full(shp, -1e9); np.maximum.at(top, (ij[:, 0], ij[:, 1]), gx[:, 2])
labT, _ = label(top > ZG + 1.0)
tall150 = maximum_filter(top > ZG + 1.5, size=int(round(1.0 / res)) * 2 + 1)    # 100 m 內有 > 150 m 的建築
TW = {"A 397 m（b2）": (2.48, -4.66, 2), "B 427 m（四塊交界）": (2.83, -2.65, 2), "C 355 m 深藍（b3）": (4.28, -3.67, 3)}
# 目標與對照只抽一部分（法向要 PCA）
_TREE = []
def pca_n(X):
    if not _TREE:
        _TREE.append(cKDTree(gx))
    _, nn = _TREE[0].query(X, k=16)
    Q = gx[nn] - gx[nn].mean(1, keepdims=True); _, _, ev = np.linalg.svd(Q, full_matrices=False); return ev[:, 2, :]
groups = {}
hgt = (gx[:, 2] - ZG) * M
low = (hgt > 2) & (hgt < 150)
for nm_, (u, v, B) in TW.items():
    ci = ((np.array([u, v]) - lo) / res).astype(int); i = labT[ci[0], ci[1]]
    if i == 0:
        win = labT[ci[0] - 3:ci[0] + 4, ci[1] - 3:ci[1] + 4]; i = np.bincount(win[win > 0]).argmax()
    fp = binary_dilation(labT == i, iterations=4)
    k = np.nonzero(fp[ij[:, 0], ij[:, 1]] & low & inblk(gx, B))[0]
    groups[nm_] = (B, k)
for B in (2, 3):
    k = np.nonzero(~tall150[ij[:, 0], ij[:, 1]] & low & inblk(gx, B))[0]
    groups[f"對照：b{B} 空曠處 0～150 m"] = (B, k)
sel = {}
for nm_, (B, k) in groups.items():
    k = k[rng.choice(len(k), min(a.ntgt * 4, len(k)), replace=False)] if len(k) else k
    n = pca_n(gx[k]); f = np.abs(n[:, 2]) < 0.3
    k, n = k[f][:a.ntgt], n[f][:a.ntgt]
    sel[nm_] = (B, gx[k], n, hgt[k])
    print(f"{nm_}：立面點 {len(k):,}", flush=True)

cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin"); imgs = read_images_binary(f"{D}/sparse/0/images.bin")
by_name = {v.name: v for v in imgs.values()}
cnt = {nm_: np.zeros(len(v[1]), int) for nm_, v in sel.items()}
inframe = {nm_: np.zeros(len(v[1]), int) for nm_, v in sel.items()}
for B in (2, 3):
    lst = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/%03d_%03d.txt" % (B % 4, B // 4)) if l.strip()]
    for ii, name in enumerate(lst):
        im = by_name.get(name)
        if im is None:
            continue
        cm = cams[im.camera_id]; R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec); C = -R.T @ t
        fx = cm.params[0]; fy = cm.params[1] if len(cm.params) > 3 else fx
        cx, cy = (cm.params[2], cm.params[3]) if len(cm.params) > 3 else (cm.params[1], cm.params[2])
        W, H = cm.width // a.down, cm.height // a.down
        Xc = occ @ R.T + t; z = Xc[:, 2]
        u = (fx * Xc[:, 0] / np.maximum(z, 1e-9) + cx) / a.down; v = (fy * Xc[:, 1] / np.maximum(z, 1e-9) + cy) / a.down
        ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        lin = v[ok].astype(int) * W + u[ok].astype(int)
        zb = np.full(W * H, np.inf); np.minimum.at(zb, lin, z[ok])
        for nm_, (Bt, X, n, hh) in sel.items():
            if Bt != B:
                continue
            Xc2 = X @ R.T + t; z2 = Xc2[:, 2]
            u2 = (fx * Xc2[:, 0] / np.maximum(z2, 1e-9) + cx) / a.down; v2 = (fy * Xc2[:, 1] / np.maximum(z2, 1e-9) + cy) / a.down
            ok2 = (z2 > 1e-6) & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
            facing = np.ones(len(X), bool)                  # PCA 法向沒有正反面 => 背面由 z-buffer（同棟較近的牆）擋掉
            vis = np.zeros(len(X), bool)
            lin2 = v2[ok2].astype(int) * W + u2[ok2].astype(int)
            vis[ok2] = z2[ok2] <= zb[lin2] * (1 + a.tol)
            inframe[nm_] += ok2 & facing; cnt[nm_] += vis
        if ii % 200 == 0:
            print(f"  b{B} 相機 {ii}/{len(lst)}", flush=True)

print("\n=== 低樓層（0～150 m）立面點：本塊訓練相機裡，畫面內幾台／真的看得到（沒被擋）幾台 ===")
print(f"{'組':26}{'點數':>8}{'畫面內 中位':>12}{'看得到 中位':>12}{'看得到 p10':>11}{'看得到 < 5 台':>13}{'0～50 m 看得到中位':>18}{'50～150 m':>11}")
for nm_, (B, X, n, hh) in sel.items():
    c = cnt[nm_]; f_ = inframe[nm_]
    if len(c) == 0:
        print(f"{nm_:26}{0:>8}"); continue
    lo_ = hh < 50
    print(f"{nm_:26}{len(c):>8,}{np.median(f_):>12.0f}{np.median(c):>12.0f}{np.percentile(c, 10):>11.0f}{100 * (c < 5).mean():>12.0f}%"
          f"{np.median(c[lo_]) if lo_.any() else float('nan'):>18.0f}{np.median(c[~lo_]) if (~lo_).any() else float('nan'):>11.0f}")
print("DONE lowfloor_vis")
