"""★ 2026-11-20 遠海補值（使用者「排看看」）：合併後在海平面上把海面延伸到地平線，只往「海」的方向鋪。

為什麼：test 幀裡地平線那條遠海，86～91% 的點沒有任何訓練影像看得到 => 訓練不會在那裡長高斯 => 渲染是黑的（CityGS 是灰的）。
⚠ 那一帶沒有任何訓練監督，補進去的是「合理的猜測」（顏色抄最近的海面格），報告要註明；不改訓練、不加訓練時間。

做法（全部用相對尺度，不用場景公尺）：
  1. 讀全部塊的海面物件（sea_object_v9/block_*.npz）：海面格位置 tiles、顏色 colors、海平面 c、法向 n、格距 g。
  2. 已建模範圍：合併模型 o>0.5 的顆粒在 xy 上的佔用格（格寬 occ_k × g），往外膨脹 1 格；有海面格落入的佔用格標成「海」。
  3. 外圍環狀網格：以佔用範圍中心為圓心，角度步 dtheta、半徑每圈乘 (1+dtheta)（格子大小與距離成正比），從佔用範圍內緣一路到 r_max_k × 佔用範圍半徑。
  4. 每個外圍格：落在已建模範圍內就跳過；否則找最近的佔用格，是「海」才鋪，顏色＝最近海面格的顏色（只填 SH 0 階）。
  5. 新面元：位置在海平面、法向＝海面法向、尺度＝0.6 × 格寬、不透明度 0.99。
用法：python tools/add_far_sea.py --ckpt <merged.ckpt> --sea "data/.../sea_object_v9/block_*.npz" --out <new.ckpt> [--dtheta_deg 0.5] [--r_max_k 20]
"""
import argparse, glob, math, os, sys
import numpy as np
import torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C0 = 0.28209479177387814

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--sea", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dtheta_deg", type=float, default=0.5)
ap.add_argument("--r_max_k", type=float, default=20.0, help="鋪到 佔用範圍半徑 × 這個倍數")
ap.add_argument("--occ_k", type=float, default=4.0, help="佔用格寬＝海面格距 g × 這個倍數")
ap.add_argument("--opacity", type=float, default=0.99)
a = ap.parse_args()

# 1. 海面物件
fs = sorted(glob.glob(a.sea))
T, Ccol, G, cs, ns = [], [], [], [], []
for f in fs:
    z = np.load(f)
    T.append(z["tiles"]); Ccol.append(z["colors"]); G.append(float(z["g"])); cs.append(float(z["c"])); ns.append(z["n"])
T = np.concatenate(T); Ccol = np.concatenate(Ccol).astype(np.float32)
g = float(np.median(G)); c = float(np.median(cs)); n = np.mean(ns, 0); n /= np.linalg.norm(n)
if Ccol.max() > 1.5:
    Ccol = Ccol / 255.0
print(f"[海面物件] {len(fs)} 塊、海面格 {len(T):,}、格距 g {g:.4f}、海平面 c {c:.4f}、法向 {np.round(n, 3)}、顏色範圍 {Ccol.min():.2f}～{Ccol.max():.2f}", flush=True)

# 2. 合併模型的佔用格
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd = ck["state_dict"]; P = "gaussian_model.gaussians."
means = sd[P + "means"].numpy(); op = torch.sigmoid(sd[P + "opacities"]).numpy().reshape(-1)
N = len(means)
q = a.occ_k * g
xy = means[op > 0.5][:, :2]
lo = np.percentile(xy, 0.1, 0) - 2 * q; hi = np.percentile(xy, 99.9, 0) + 2 * q
nx, ny = (np.ceil((hi - lo) / q)).astype(int) + 1
occ = np.zeros((nx, ny), bool)
ij = np.clip(((xy - lo) / q).astype(int), 0, [nx - 1, ny - 1]); occ[ij[:, 0], ij[:, 1]] = True
occ = occ | np.roll(occ, 1, 0) | np.roll(occ, -1, 0) | np.roll(occ, 1, 1) | np.roll(occ, -1, 1)
sea = np.zeros_like(occ)
tij = np.clip(((T[:, :2] - lo) / q).astype(int), 0, [nx - 1, ny - 1]); sea[tij[:, 0], tij[:, 1]] = True
sea &= occ
oi, oj = np.nonzero(occ)
oc = np.stack([lo[0] + (oi + 0.5) * q, lo[1] + (oj + 0.5) * q], 1); osea = sea[oi, oj]
print(f"[佔用] 合併 N {N:,}、o>0.5 {len(xy):,}、佔用格寬 {q:.3f}、佔用格 {len(oc):,}（海 {osea.sum():,}）", flush=True)

# 3. 外圍環狀網格
ctr = oc.mean(0); R0 = float(np.percentile(np.linalg.norm(oc - ctr, axis=1), 99.5)); rmin = float(np.percentile(np.linalg.norm(oc - ctr, axis=1), 5))
dth = math.radians(a.dtheta_deg); th = np.arange(0, 2 * math.pi, dth)
rs = [max(rmin, q)]
while rs[-1] < a.r_max_k * R0:
    rs.append(rs[-1] * (1 + dth))
rs = np.array(rs)
RR, TH = np.meshgrid(rs, th, indexing="ij")
px = ctr[0] + RR * np.cos(TH); py = ctr[1] + RR * np.sin(TH); cell = (RR * dth).reshape(-1)
P2 = np.stack([px.reshape(-1), py.reshape(-1)], 1)
# 4. 在已建模範圍內的跳過；最近佔用格是海才鋪
gi = ((P2 - lo) / q).astype(int)
inside = (gi[:, 0] >= 0) & (gi[:, 0] < nx) & (gi[:, 1] >= 0) & (gi[:, 1] < ny)
inocc = np.zeros(len(P2), bool); inocc[inside] = occ[gi[inside, 0], gi[inside, 1]]
cand = ~inocc
_, k = cKDTree(oc).query(P2[cand])
keep = np.zeros(len(P2), bool); keep[np.nonzero(cand)[0][osea[k]]] = True
P2 = P2[keep]; cell = cell[keep]
_, kt = cKDTree(T[:, :2]).query(P2)
rgb = Ccol[kt]
print(f"[外圍] 圓心 {np.round(ctr, 2)}、佔用半徑 R0 {R0:.2f}、鋪到 {a.r_max_k} x R0＝{a.r_max_k * R0:.1f}、環 {len(rs)} 圈 x {len(th)} 扇＝{RR.size:,} 格 => 在範圍外 {cand.sum():,}、往海的方向 {len(P2):,} 顆", flush=True)

# 5. 新面元（海平面上；法向＝海面法向）
pz = (c - P2 @ n[:2]) / n[2]                      # 平面 n·x = c
new_means = np.concatenate([P2, pz[:, None]], 1).astype(np.float32)
z = np.array([0, 0, 1.0]); v = np.cross(z, n); s = np.linalg.norm(v); cth = float(z @ n)
if s < 1e-8:
    quat = np.array([1.0, 0, 0, 0])
else:
    ang = math.atan2(s, cth); v = v / s; quat = np.array([math.cos(ang / 2), *(v * math.sin(ang / 2))])
M = len(new_means)
add = {"means": torch.from_numpy(new_means),
       "shs_dc": torch.from_numpy(((rgb - 0.5) / C0).astype(np.float32))[:, None, :],
       "shs_rest": torch.zeros((M,) + tuple(sd[P + "shs_rest"].shape[1:])),
       "opacities": torch.full((M, 1), math.log(a.opacity / (1 - a.opacity))),
       "scales": torch.log(torch.from_numpy((0.6 * cell).astype(np.float32)))[:, None].repeat(1, sd[P + "scales"].shape[1]),
       "rotations": torch.from_numpy(np.tile(quat, (M, 1)).astype(np.float32))}
for kname, t in add.items():
    sd[P + kname] = torch.cat([sd[P + kname], t.to(sd[P + kname].dtype)], 0)
for kname in ("optimizer_states", "loops", "callbacks", "lr_schedulers"):
    ck.pop(kname, None)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
torch.save(ck, a.out)
print(f"=> {a.out}：N {N:,} + 遠海 {M:,} = {N + M:,}（格寬 {cell.min():.3f}～{cell.max():.2f}）", flush=True)
print("DONE add_far_sea")
