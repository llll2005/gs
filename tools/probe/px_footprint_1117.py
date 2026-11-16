"""11-17 Fable 第三輪「塔頂顆粒為什麼小、CityGS 為什麼修得正」的兩個離線檢查（使用者「直接做」）。只讀、CPU。
假說：面元要轉向，梯度來自它在畫面上跨好幾個像素；只佔一兩個像素的面元拿不到旋轉訊號。塔頂的 0.30 m 可能只是「大小跟著像素走」。
檢查一：各高度帶的「長軸 ÷ 最近可見訓練相機的 1 像素長度」（＝它在最近那台相機裡佔幾個像素，訓練解析度＝原圖 /1.2）。
        各高度差不多 => 大小跟著像素走；塔頂特別貼近 1 => 被縮到點精靈下限（px_floor 1 px）。
檢查二：高樓立面帶內，朝向對齊率（|n·n_GT| > 0.9）依「畫面上的大小」分組。大的對得齊、小的對不齊 => 解釋成立。
範圍與 tower_gt_1113.py 相同：SfM 頂高 > top 的高樓足跡（10 m 格膨脹一格）內、本塊盒內；GT 立面點（PCA 法向 |n_z| < 0.3）1 m 內的不透明面元（o > 0.5）。
「最近可見」＝投影落在畫面內、在相機前方的本塊訓練相機裡深度最小的那台（不做遮擋，與 px_floor 的定義一致）。
用法：python tools/probe/px_footprint_1117.py --block 2 --models CityGS=<ckpt glob> 新底6=<ckpt glob> ...
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.ndimage import label, maximum_filter
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=2)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--top", type=float, default=1.5, help="只看 SfM 頂高 > top 的高樓（1.5＝150 m）")
ap.add_argument("--gt", default="data/geometry_gt/MC_Aerial/Block_all_ds.ply")
ap.add_argument("--down", type=float, default=1.2, help="訓練解析度＝原圖 / down（down_sample_factor）")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()

pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
rel = (TRK >= 3) & (ERR < 1.5)
ZG = float(np.percentile(P[rel, 2], 5))
P = P[rel & inblk(P)]
CELL = 0.1
o0 = np.floor(P[:, :2].min(0) / CELL).astype(int); ci = np.floor(P[:, :2] / CELL).astype(int) - o0
shp = tuple(ci.max(0) + 1); top = np.full(shp, -1e9)
np.maximum.at(top, (ci[:, 0], ci[:, 1]), P[:, 2])
lab_, n = label(top > 1.0)
bid = lab_[ci[:, 0], ci[:, 1]]
anyfoot = np.zeros(shp, bool)
for k in range(1, n + 1):
    s = bid == k
    if s.sum() >= 50 and np.percentile(P[s, 2], 99) > a.top:
        anyfoot |= maximum_filter(lab_ == k, size=3)
def onfoot(m):
    c = np.floor(m[:, :2] / CELL).astype(int) - o0
    ok = (c[:, 0] >= 0) & (c[:, 0] < shp[0]) & (c[:, 1] >= 0) & (c[:, 1] < shp[1])
    r = np.zeros(len(m), bool); r[ok] = anyfoot[c[ok, 0], c[ok, 1]]; return r

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
gx = gx[onfoot(gx)]; gx = gx[inblk(gx)]
gtree = cKDTree(gx)
_, nn_j = gtree.query(gx, k=16)
Q = gx[nn_j] - gx[nn_j].mean(1, keepdims=True)
_, _, evec = np.linalg.svd(Q, full_matrices=False); del Q, nn_j
gn = evec[:, 2, :]
fac = np.abs(gn[:, 2]) < 0.3
fx_, fn_ = gx[fac], gn[fac]
ftree = cKDTree(fx_)
print(f"b{B}｜GT 高樓足跡內＋塊內 {len(gx):,} 點、立面 {fac.sum():,}｜地面參考 z {ZG:.4f}", flush=True)

# ── 本塊訓練相機 ──
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin"); imgs = read_images_binary(f"{D}/sparse/0/images.bin")
by_name = {v.name: v for v in imgs.values()}
lst = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/%03d_%03d.txt" % (B % 4, B // 4)) if l.strip()]
CR, CT, CK = [], [], []
for nm in lst:
    im = by_name.get(nm)
    if im is None:
        continue
    cm = cams[im.camera_id]
    fx = cm.params[0]; fy = cm.params[1] if len(cm.params) > 3 else fx
    cx, cy = (cm.params[2], cm.params[3]) if len(cm.params) > 3 else (cm.params[1], cm.params[2])
    CR.append(qvec2rotmat(im.qvec)); CT.append(np.asarray(im.tvec)); CK.append((fx / a.down, fx, fy, cx, cy, cm.width, cm.height))
CR = np.stack(CR); CT = np.stack(CT); CK = np.array(CK)
print(f"本塊訓練相機 {len(CR)} 台｜訓練解析度 fx {CK[0, 0]:.1f}（原圖 {CK[0, 1]:.1f} / {a.down}）", flush=True)

def near_px_len(X):
    """每點：本塊訓練相機裡，投影在畫面內、在前方的最小深度 / fx_train（＝最近那台相機 1 像素的長度，場景單位）。沒有 => nan。"""
    best = np.full(len(X), np.inf)
    for i in range(len(CR)):
        Xc = X @ CR[i].T + CT[i]; z = Xc[:, 2]
        _, fx0, fy0, cx, cy, W, H = CK[i]
        u = fx0 * Xc[:, 0] / np.maximum(z, 1e-9) + cx; v = fy0 * Xc[:, 1] / np.maximum(z, 1e-9) + cy
        ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        best = np.where(ok & (z < best), z, best)
    return np.where(np.isfinite(best), best / CK[0, 0], np.nan)

def qrot(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)

BANDS = [(0, 50), (50, 150), (150, 250), (250, 350), (350, 450)]
SIZE = [(0, 1.5), (1.5, 3), (3, 6), (6, 12), (12, 1e9)]
res = {}
for spec in a.models:
    labm, pat = spec.split("=", 1); fs = sorted(glob.glob(pat)); assert fs, pat
    sd = torch.load(fs[-1], map_location="cpu", weights_only=False)["state_dict"]; g = "gaussian_model.gaussians."
    m = sd[g + "means"].double().numpy(); o = torch.sigmoid(sd[g + "opacities"].float()).reshape(-1).numpy()
    s = torch.exp(sd[g + "scales"].float()).numpy()[:, :2]; r = sd[g + "rotations"].float().numpy(); del sd
    k = (o > 0.5) & onfoot(m); k &= inblk(m)
    m, s, r = m[k], s[k], r[k]
    d, j = ftree.query(m, distance_upper_bound=0.01)       # GT 立面 1 m 內
    k2 = np.isfinite(d); m, s, r, j = m[k2], s[k2], r[k2], j[k2]
    nrm = qrot(r.astype(np.float64))[:, :, 2]
    al = np.abs((nrm * fn_[j]).sum(1)) > 0.9
    h = (fx_[j, 2] - ZG) * M
    pl = near_px_len(m)
    lmax = s.max(1); npx = lmax / pl
    res[labm] = (h, lmax, pl, npx, al)
    print(f"[載入] {labm}：{fs[-1]}｜立面 1 m 內不透明 {len(m):,} 顆（沒有相機看得到 {np.isnan(pl).sum():,}）", flush=True)

print("\n=== 檢查一：各高度帶 長軸（m）｜最近相機 1 px 長度（m）｜長軸佔幾像素（中位、p10）｜< 1.5 px 的比例 ===")
print(f"{'模型':10}" + "".join(f"{f'{lo}~{hi} m':>36}" for lo, hi in BANDS))
for labm, (h, lmax, pl, npx, al) in res.items():
    row = f"{labm:10}"
    for lo, hi in BANDS:
        kk = (h >= lo) & (h < hi) & np.isfinite(npx)
        row += (f"{np.median(lmax[kk]) * M:5.2f}｜{np.median(pl[kk]) * M:5.3f}｜{np.median(npx[kk]):4.1f}／{np.percentile(npx[kk], 10):3.1f} px｜{100 * (npx[kk] < 1.5).mean():3.0f}%".rjust(36)
                if kk.sum() >= 30 else f"{'—':>36}")
    print(row)

print("\n=== 檢查二：朝向對齊率 依「長軸佔幾像素」分組（括號＝該組佔比）===")
for lo, hi in [(0, 150), (150, 250), (250, 450)]:
    print(f"-- 高度 {lo}~{hi} m")
    print(f"{'模型':10}" + "".join(f"{f'{a0}~{a1 if a1 < 1e8 else chr(8734)} px':>16}" for a0, a1 in SIZE))
    for labm, (h, lmax, pl, npx, al) in res.items():
        hb = (h >= lo) & (h < hi) & np.isfinite(npx); row = f"{labm:10}"
        for a0, a1 in SIZE:
            kk = hb & (npx >= a0) & (npx < a1)
            row += (f"{100 * al[kk].mean():5.0f}%（{100 * kk.sum() / max(hb.sum(), 1):3.0f}%）".rjust(16) if kk.sum() >= 30 else f"{'—':>16}")
        print(row)
print("DONE px_footprint")
