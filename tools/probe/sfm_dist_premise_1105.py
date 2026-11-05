"""11-05 使用者構想「鎖住 SfM 位置＋增生只准在 SfM 附近 r 內」的前提檢查（CPU；cwd＝repo 根；只讀；不用真值）。
問：離最近可靠 SfM 點的距離 d，能不能把「好的顆粒」和「浮塊／地下層」分開？r 取多少會餓死正確的表面？
類別（只用 SfM 定義）：
  地下     2 m 格裡有可靠 SfM、且 SfM 中位高度 < 3 m（地面格），顆粒比該格 SfM 中位低 1 m 以上
  海面上方 4 m 格的可靠 SfM 中位在 SfM 估的海平面（高度直方圖最低峰）±1 m（不靠真值的海面格，precision 92%），顆粒高出 5 m 以上
  高處     顆粒 z > 100 m
  其他     其餘
對每個模型（不透明 o>0.5）報：各類顆粒數、d < 1／2／5 m 的比例。CityGS 當「合理場景」的參考 => 它在 r 外的比例＝r 會餓死多少。
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--models", nargs="+", required=True)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
bb1 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(x).double() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
good = (TRK >= 3) & (ERR < 1.5)
# 海平面（不靠真值）：全場景可靠 SfM 高度直方圖的最低強峰
zg = P[good, 2]; e = np.arange(-0.10, 0.0501, 0.001); h, _ = np.histogram(zg, bins=e); hs = np.convolve(h, np.ones(5) / 5, mode="same")
pk = [i for i in range(2, len(hs) - 2) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.02 * hs.max()]
ZS = e[pk[0]] + 0.0005
S = P[good & inb(P, bb1)]                 # 塊盒外擴 5%（邊界附近也有 SfM 可依）
tree = cKDTree(S)
def cellmed(cell):
    o0 = np.floor(S[:, :2].min(0) / cell).astype(int) - 2
    ci = np.floor(S[:, :2] / cell).astype(int) - o0; shp = tuple(ci.max(0) + 3)
    cid = ci[:, 0] * shp[1] + ci[:, 1]; o = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[o], return_index=True, return_counts=True)
    med = np.full(shp[0] * shp[1], np.nan); med[u] = S[o, 2][f + c // 2]
    return o0, shp, med.reshape(shp)
g0, gshp, gmed = cellmed(0.02); s0, sshp, smed = cellmed(0.04)
def look(x, o0, shp, med):
    ci = np.floor(x[:, :2] / (0.02 if med is gmed else 0.04)).astype(int) - o0
    ok = (ci[:, 0] >= 0) & (ci[:, 0] < shp[0]) & (ci[:, 1] >= 0) & (ci[:, 1] < shp[1])
    r = np.full(len(x), np.nan); r[ok] = med[ci[ok, 0], ci[ok, 1]]; return r
print(f"b{B}｜可靠 SfM（塊盒外擴 5%）{len(S):,}｜SfM 估海平面 {ZS * M:+.1f} m（真值 −3.4 m）")
print(f"{'模型':14}{'類別':10}{'不透明顆粒':>11}{'d<1m':>8}{'d<2m':>8}{'d<5m':>8}{'d 中位 m':>10}")
for spec in a.models:
    lab, pat = spec.split("=", 1); f = sorted(glob.glob(pat))[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    m = sd["gaussian_model.gaussians.means"].double().numpy(); o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy(); del sd
    k = inb(m, bb0) & (o > 0.5); m = m[k]
    d, _ = tree.query(m, k=1, distance_upper_bound=0.2); d = np.minimum(d, 0.2) * M
    gm = look(m, g0, gshp, gmed); sm = look(m, s0, sshp, smed)
    under = np.isfinite(gm) & (gm < 0.03) & (m[:, 2] < gm - 0.01)
    sea = np.isfinite(sm) & (np.abs(sm - ZS) < 0.01) & (m[:, 2] > ZS + 0.05)
    high = m[:, 2] > 1.0
    cats = {"全部": np.ones(len(m), bool), "地下": under, "海面上方": sea & ~under, "高處>100m": high, "其他": ~under & ~sea & ~high}
    for c, s in cats.items():
        if s.sum() == 0:
            continue
        dd = d[s]
        print(f"{lab:14}{c:10}{int(s.sum()):>11,}{100 * np.mean(dd < 1):>7.1f}%{100 * np.mean(dd < 2):>7.1f}%{100 * np.mean(dd < 5):>7.1f}%{np.median(dd):>10.2f}", flush=True)
print("DONE sfm_dist_premise")
