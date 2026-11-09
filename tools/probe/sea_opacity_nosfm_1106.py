"""11-06 使用者 viewer：海面上方的霧變稀疏但很高，「SfM 距離限制沒生效嗎」。量 b3 各模型海面上方（> 海平面 5 m）的顆粒，含半透明；
類別定義同 sfm_dist_premise_1105／sfm_vis_diag_1106（4 cm 格 SfM 中位≈海平面的格）。只讀、CPU。"""
import glob, sys
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
import os; D = "data/matrix_city/aerial/train/block_all"; B = int(os.environ.get("SEA_BLOCK", "3"))
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.); bb1 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(x).double() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
gd = (TRK >= 3) & (ERR < 1.5)
zg = P[gd, 2]; e = np.arange(-0.10, 0.0501, 0.001); h, _ = np.histogram(zg, bins=e); hs = np.convolve(h, np.ones(5) / 5, mode="same")
pk = [i for i in range(2, len(hs) - 2) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.02 * hs.max()]
ZS = e[pk[0]] + 0.0005
S = P[gd]; tree = cKDTree(S)   # 全場景可靠 SfM（塊外的海也要有定義）
def cellmed(cell):
    o0 = np.floor(S[:, :2].min(0) / cell).astype(int) - 2
    ci = np.floor(S[:, :2] / cell).astype(int) - o0; shp = tuple(ci.max(0) + 3)
    cid = ci[:, 0] * shp[1] + ci[:, 1]; o = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[o], return_index=True, return_counts=True)
    med = np.full(shp[0] * shp[1], np.nan); med[u] = S[o, 2][f + c // 2]
    return o0, shp, med.reshape(shp)
def look(x, cell, o0, shp, grid):
    ci = np.floor(x[:, :2] / cell).astype(int) - o0
    ok = (ci[:, 0] >= 0) & (ci[:, 0] < shp[0]) & (ci[:, 1] >= 0) & (ci[:, 1] < shp[1])
    r = np.full(len(x), np.nan); r[ok] = grid[ci[ok, 0], ci[ok, 1]]; return r
G4 = cellmed(0.04)
from scipy.ndimage import maximum_filter
_o0, _shp, _med = G4; NOSFM = (~(maximum_filter(np.isfinite(_med).astype(np.uint8), size=3) > 0)).astype(float)  # 3x3 鄰域都沒有可靠 SfM 的 4 m 格
o10 = np.floor(S[:, :2].min(0) / 0.1).astype(int) - 2; ci10 = np.floor(S[:, :2] / 0.1).astype(int) - o10
shp10 = tuple(ci10.max(0) + 3); top = np.full(shp10, -1e9); np.maximum.at(top, (ci10[:, 0], ci10[:, 1]), S[:, 2])
TALL = maximum_filter((top > 1.0).astype(np.uint8), size=3).astype(float)   # 高樓格（同 sfm_vis_diag／tower_probe）
print(f"b{B}｜海平面估 {ZS * 100:.1f} m｜海面上方＝4 cm 格（4 m）3x3 鄰域沒有可靠 SfM、或格內 SfM 中位在海平面 ±1 m、扣掉高樓格（SfM 頂高 > 100 m 的 10 m 格膨脹 1 格）、顆粒高於海平面 5 m 以上（單位 m；場景 1＝100 m）")
print(f"{'模型':14}{'o>0.5':>8}{'0.1~0.5':>9}{'0.02~0.1':>10}{'Σo·面積 m²':>12}{'高度中位':>9}{'高度p90':>9}{'離SfM中位':>10}{'離SfM>5m':>10}{'>20m':>8}{'>50m':>8}")
for spec in sys.argv[1:]:
    lab, pat = spec.split("=", 1); f = sorted(glob.glob(pat))[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    m = sd["gaussian_model.gaussians.means"].double().numpy(); o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy()
    sc = torch.exp(sd["gaussian_model.gaussians.scales"].float()).numpy(); del sd
    IN = inb(m, bb0)
    sm = look(m, 0.04, *G4)
    nos = look(m, 0.04, _o0, _shp, NOSFM) > 0
    tl = look(m, 0.1, o10, shp10, TALL) > 0
    sea0 = ~tl & ((np.isfinite(sm) & (np.abs(sm - ZS) < 0.01)) | nos) & (m[:, 2] > ZS + 0.05) & (o > 0.02)
    for where, sel in (("塊內", IN), ("塊外", ~IN)):
     sea = sea0 & sel; ms, os_, ss = m[sea], o[sea], sc[sea]
     hgt = (ms[:, 2] - ZS) * 100; d = tree.query(ms, k=1)[0] * 100
     area = (os_ * 2 * np.pi * ss[:, 0] * ss[:, 1]).sum() * 1e4
     nb = lambda a, b: int(((os_ > a) & (os_ <= b)).sum())
     q = lambda x, p_: np.percentile(x, p_) if len(x) else float("nan")
     print(f"{lab + where:14}{nb(0.5, 1.01):>8,}{nb(0.1, 0.5):>9,}{nb(0.02, 0.1):>10,}{area:>12,.0f}{q(hgt, 50):>9.1f}{q(hgt, 90):>9.1f}{q(d, 50):>10.1f}{100 * np.mean(d > 5) if len(d) else 0:>9.0f}%{int((hgt > 20).sum()):>8,}{int((hgt > 50).sum()):>8,}", flush=True)
