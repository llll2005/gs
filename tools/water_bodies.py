#!/usr/bin/env python
"""★ 11-18 海面 v6 ①：水體與自動水面高度（全場景層級；取代 v3 手填的 --sea_level）。CPU。
使用者 11-18：「海平面高度改自動算」「若能依某種屬性判定出它是同一個面（湖面、海面）且分離 => 各自處理」。

輸入：某一版各塊海面物件（預設 sea_object_v2：每塊用自己的共識平面投票＋長過的海面格；有海的塊才有 npz）。
做法：
  1 每塊的海面格格心 => 世界 xy，畫進全場景同一張格網（解析度＝各塊格邊長的最小值）；鄰塊的格網對不齊 => 先膨脹 1 格再取連通區（8 連通）
     => 每個連通區＝一個水體（互不相連的海、湖分開）。小於 --min_cells 的連通區丟掉（零星誤判）。
  2 每個水體的高度＝落在該水體格上、而且離「該塊共識平面」在 band x 格邊長內的可靠 SfM 點（track >= 3、重投影誤差 < 1.5 px）的 z 中位，
     先去掉離中位 > 3 x 1.4826 x MAD 的點再取一次中位（band 用格邊長的倍數＝像素準則，沒有公尺常數）。
     點數 < --min_pts => 退回成員塊共識平面在該水體格心的 z 中位（標記 fallback）。
  3 另外報：每個水體 SfM 水面的傾斜（最小平方平面）與殘差 MAD；不同水體高度差。
輸出：<dataset>/<out>.npz（labels 格網、原點、解析度、每個水體的高度／點數／fallback 旗標）、<out>.json（摘要）。
make_sea_object.py --water_bodies <out>.npz 用它：每塊的海面改成「本塊盒內佔最多格的那個水體」的水平面。
用法：python tools/water_bodies.py data/matrix_city/aerial/train/block_all [--src sea_object_v2] [--out water_bodies]
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dataset"); ap.add_argument("--src", default="sea_object_v2"); ap.add_argument("--out", default="water_bodies")
ap.add_argument("--band", type=float, default=2.0, help="SfM 水面點：離該塊共識平面 <= band x 格邊長")
ap.add_argument("--min_cells", type=int, default=200); ap.add_argument("--min_pts", type=int, default=50)
a = ap.parse_args()
from scipy import ndimage
os.environ.setdefault("CITYGS_SFM_FAST", "1")
from internal.utils import colmap_fast
D = a.dataset
fs = sorted(glob.glob(os.path.join(D, a.src, "block_*.npz")), key=lambda f: int(f.split("_")[-1][:-4]))
if not fs:
    raise SystemExit(f"⛔ {D}/{a.src} 沒有任何 block_*.npz")
OB = {}
for f in fs:
    b = int(f.split("_")[-1][:-4]); z = np.load(f)
    keys = z["keys"]; g = float(z["g"]); base = int(z["base"])
    i = np.round(keys / base).astype(np.int64); j = keys - i * base
    cen = z["origin"] + np.outer((i + 0.5) * g, z["e1"]) + np.outer((j + 0.5) * g, z["e2"])
    OB[b] = dict(n=z["n"], c=float(z["c"]), g=g, org=z["origin"], e1=z["e1"], e2=z["e2"], keys=np.sort(keys), base=base, cen=cen)
res = min(v["g"] for v in OB.values())
allc = np.concatenate([v["cen"][:, :2] for v in OB.values()])
x0, y0 = allc.min(0) - 4 * res
nx, ny = (np.ceil((allc.max(0) - (x0, y0)) / res) + 8).astype(int)
occ = np.zeros((nx, ny), bool)
for b, v in OB.items():
    # 每格用它自己的邊長畫（格邊長 g_b >= res => 格心附近 ceil(g_b/res) 格都塗上）
    r = int(np.ceil(v["g"] / res / 2))
    ix = np.floor((v["cen"][:, 0] - x0) / res).astype(int); iy = np.floor((v["cen"][:, 1] - y0) / res).astype(int)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            occ[np.clip(ix + dx, 0, nx - 1), np.clip(iy + dy, 0, ny - 1)] = True
lab, nb = ndimage.label(ndimage.binary_dilation(occ, iterations=1), structure=np.ones((3, 3)))
lab = np.where(occ, lab, 0)
cnt = np.bincount(lab.ravel(), minlength=nb + 1); cnt[0] = 0
keep = [k for k in range(1, nb + 1) if cnt[k] >= a.min_cells]
print(f"全場景格網 {nx}x{ny}（解析度 {res * 100:.2f} m）｜海面格塗色 {int(occ.sum()):,}｜連通區 {nb}，>= {a.min_cells} 格的 {len(keep)} 個：" +
      "、".join(f"#{k} {cnt[k]:,} 格" for k in keep), flush=True)
remap = np.zeros(nb + 1, np.int32)
for t, k in enumerate(keep, 1):
    remap[k] = t
lab = remap[lab]
d = colmap_fast.points3D_arrays(os.path.join(D, "sparse/0/points3D.bin"))
S = d["xyz"][(np.diff(d["off"]) >= 3) & (d["err"] < 1.5)].astype(np.float64)
sx = np.floor((S[:, 0] - x0) / res).astype(int); sy = np.floor((S[:, 1] - y0) / res).astype(int)
ins = (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
sl = np.zeros(len(S), np.int32); sl[ins] = lab[sx[ins], sy[ins]]
# 每個 SfM 點離「它所在那塊海面物件的共識平面」：用離它最近的塊（格心最近）
near_b = np.full(len(S), -1)
from scipy.spatial import cKDTree
bc = [(b, cKDTree(v["cen"][:, :2])) for b, v in OB.items()]
best = np.full(len(S), np.inf)
cand = np.nonzero(sl > 0)[0]
for b, tr in bc:
    dd, _ = tr.query(S[cand, :2]); m = dd < best[cand]; best[cand[m]] = dd[m]; near_b[cand[m]] = b
bodies = []
for t in range(1, len(keep) + 1):
    m = (sl == t) & (near_b >= 0)
    h = np.array([S[i] @ OB[near_b[i]]["n"] - OB[near_b[i]]["c"] for i in np.nonzero(m)[0]]) if m.any() else np.zeros(0)
    gg = np.array([OB[near_b[i]]["g"] for i in np.nonzero(m)[0]]) if m.any() else np.zeros(0)
    w = np.nonzero(m)[0][np.abs(h) <= a.band * gg] if m.any() else np.zeros(0, int)
    zc = S[w, 2] if len(w) else np.zeros(0)
    members = sorted({int(near_b[i]) for i in np.nonzero(m)[0]}) if m.any() else []
    if len(zc) >= a.min_pts:
        md = np.median(zc); mad = np.median(np.abs(zc - md)) * 1.4826 + 1e-12; zc2 = zc[np.abs(zc - md) <= 3 * mad]
        z_w = float(np.median(zc2)); fb = False
        A = np.c_[S[w][np.abs(zc - md) <= 3 * mad][:, :2], np.ones(len(zc2))]; coef, *_ = np.linalg.lstsq(A, zc2, rcond=None)
        tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1])))); rmad = float(np.median(np.abs(zc2 - A @ coef)))
    else:
        # 退回：成員塊共識平面在本水體格心的 z
        zz = []
        for b, v in OB.items():
            cx = np.floor((v["cen"][:, 0] - x0) / res).astype(int); cy = np.floor((v["cen"][:, 1] - y0) / res).astype(int)
            mm = lab[np.clip(cx, 0, nx - 1), np.clip(cy, 0, ny - 1)] == t
            zz.append(v["cen"][mm, 2])
        zz = np.concatenate(zz) if zz else np.zeros(1)
        z_w = float(np.median(zz)); fb = True; tilt = float("nan"); rmad = float("nan"); zc2 = zc
    bodies.append(dict(id=t, cells=int((lab == t).sum()), z=z_w, n_sfm=int(len(zc2)), fallback=fb, tilt_deg=tilt, resid_mad=rmad, blocks=members))
    print(f"水體 #{t}：{bodies[-1]['cells']:,} 格｜塊 {members}｜SfM 水面點 {len(zc2):,}｜高度 {z_w:.5f}（{z_w * 100:.1f} m）"
          f"{'（點太少，退回共識平面）' if fb else f'｜SfM 水面傾斜 {tilt:.2f}°、殘差 MAD {rmad * 100:.2f} m'}", flush=True)
OUT = os.path.join(D, a.out)
np.savez(OUT + ".npz", labels=lab, x0=x0, y0=y0, res=res, z=np.array([bb["z"] for bb in bodies]), n_sfm=np.array([bb["n_sfm"] for bb in bodies]),
         fallback=np.array([bb["fallback"] for bb in bodies]))
json.dump(dict(src=a.src, res=res, bodies=bodies), open(OUT + ".json", "w"), indent=1, ensure_ascii=False)
print(f"DONE water_bodies => {OUT}.npz", flush=True)
