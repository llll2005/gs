#!/usr/bin/env python
"""★ 11-18 海面格溢到陸地的檢查（使用者 viewer：海面深藍溢到岸邊路面）。只讀、CPU；GT 只拿來量。
每個海面物件（npz）的第 0 層海面格：格心 2 m 半徑內 GT 點的最高點，比水面高 > --land_m 的＝底下其實是陸地／結構（溢上岸）。
另報：水面附近（GT 最高點在水面 ±1 m）但不是海面格的「漏掉的海」——只算離海面格 <= 3 格的（交界），給 gap 規則對照。
用法：python tools/probe/sea_spill_1118.py --objs v6=<npz> v8=<npz> ...
"""
import argparse, os, sys
import numpy as np
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser()
ap.add_argument("--objs", nargs="+", required=True); ap.add_argument("--gt", default="data/geometry_gt/MC_Aerial/Block_all_ds.ply")
ap.add_argument("--land_m", type=float, default=1.5)
a = ap.parse_args()
with open(a.gt, "rb") as fh:
    hdr = b""
    while not hdr.endswith(b"end_header\n"):
        hdr += fh.readline()
    props, nv = [], 0
    for ln in hdr.decode().splitlines():
        t = ln.split()
        if t[:2] == ["element", "vertex"]:
            nv = int(t[2])
        elif t and t[0] == "property":
            props.append((t[2], {"double": "<f8", "float": "<f4", "uchar": "u1"}[t[1]]))
    G = np.fromfile(fh, dtype=np.dtype(props), count=nv)
g = np.stack([G["x"], G["y"], G["z"]], 1); del G
CELL = 0.01; lo = g[:, :2].min(0) - 0.05
ij = ((g[:, :2] - lo) / CELL).astype(int); shp = ij.max(0) + 2
top = np.full(shp, -1e9); np.maximum.at(top, (ij[:, 0], ij[:, 1]), g[:, 2])
def top_at(xy, r=2):
    q = ((xy - lo) / CELL).astype(int); best = np.full(len(xy), -1e9)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            qq = q + [di, dj]; ok = (qq[:, 0] >= 0) & (qq[:, 0] < shp[0]) & (qq[:, 1] >= 0) & (qq[:, 1] < shp[1])
            v = np.full(len(xy), -1e9); v[ok] = top[qq[ok, 0], qq[ok, 1]]; best = np.maximum(best, v)
    return best
for spec in a.objs:
    nm, f = spec.split("=", 1)
    z = np.load(f); gg = float(z["g"]); org = z["origin"]; e1 = z["e1"]; e2 = z["e2"]; c = float(z["c"]); keys = np.sort(z["keys"])
    i = np.round(keys / 1000003).astype(np.int64); j = keys - i * 1000003
    cen = org + np.outer((i + 0.5) * gg, e1) + np.outer((j + 0.5) * gg, e2)
    t = top_at(cen[:, :2]); has = t > -1e8
    land = has & (t - c > a.land_m / 100)
    # 漏掉的海：海面格 3 格內的格，GT 最高點在水面 ±1 m、但不是海面格
    ks = set(keys.tolist()); cand = set()
    for di in range(-3, 4):
        for dj in range(-3, 4):
            cand |= set(((i + di) * 1000003 + (j + dj)).tolist())
    cand = np.array(sorted(cand - ks), np.int64)
    ci = np.round(cand / 1000003).astype(np.int64); cj = cand - ci * 1000003
    cc = org + np.outer((ci + 0.5) * gg, e1) + np.outer((cj + 0.5) * gg, e2)
    tc = top_at(cc[:, :2]); water = (tc > -1e8) & (np.abs(tc - c) < 0.01)
    print(f"{nm}｜海面格 {len(keys):,}（格邊長 {gg * 100:.2f} m）｜底下其實是陸地／結構（GT 最高點高出水面 > {a.land_m:g} m）{int(land.sum()):,} 格（{100 * land.mean():.2f}%）"
          f"｜交界 3 格內漏掉的水面格 {int(water.sum()):,}（佔交界候選 {100 * water.mean():.1f}%）", flush=True)
print("DONE sea_spill", flush=True)
