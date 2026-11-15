"""11-16 H3（Fable 提案的前提檢查）：塔頂沒有 SfM、校正深度又不可靠時，單目深度推出的法向準不準。只讀、CPU；GT 深度只在 lab（../mc_depth_f32）。
每張抽到的訓練影像：GT 深度、單目深度（estimated_depths x 逐張 scale/offset＝逆深度）、校正深度（estimated_depths_tri＝逆深度）各自反投影成相機座標點，
用中央差分的叉積推法向；比單目／校正深度法向對 GT 法向的夾角。分組（GT 只用來分類）：
  高度帶（GT 世界點離地面參考，地面＝可靠 SfM z 的 5% 分位，只拿來分帶）：塔頂 > 250 m、較低 50～150 m
  表面類型：立面（GT 法向 |n_z| < 0.3）
  校正深度可靠度：可靠（斜率級 <= 1 且大小級 <= 3，訓練預設）／不可靠
深度不連續的像素（3x3 內 GT 深度相對變化 > 5%）不算。判讀：塔頂不可靠像素上，單目法向誤差中位 < 約 20 度 => 法向先驗 loss 值得試。
用法（lab，cwd＝repo 根）：python tools/probe/h3_mono_normal_1116.py --block 3 [--ncam 60]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
ap = argparse.ArgumentParser(); ap.add_argument("--block", type=int, required=True); ap.add_argument("--ncam", type=int, default=60)
ap.add_argument("--stride", type=int, default=2)
a, _rest = ap.parse_known_args(); sys.argv = sys.argv[:1]
import monodepth_sea_1102 as M            # 真深度配對、單位（K_UNIT＝場景單位／cm）、相機、單目 scale
import torch
D = M.D
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
bx, by = [int(v) for v in p["partition_coordinates"]["id"][a.block]]
lst = [n for n in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{bx:03d}_{by:03d}.txt").read().split() if n in M.BYNAME]
rng = np.random.default_rng(0); names = [lst[i] for i in sorted(rng.choice(len(lst), min(a.ncam, len(lst)), replace=False))]
ZG = float(np.percentile(M.PX[M.GOODP][:, 2], 5))


def resize(arr, shape):
    if arr.shape == shape:
        return arr
    import cv2
    return cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float64)


def normals(z, fx, fy, cx, cy, s):
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, s:-s] = P[:, 2 * s:] - P[:, :-2 * s]; dy[s:-s] = P[2 * s:] - P[:-2 * s]
    n = np.cross(dx, dy); n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)
    return P, n


acc = {}
for name in names:
    gt = M.load_gt(name)
    if gt is None:
        continue
    im = M.BYNAME[name]; W, H, fx, fy, cx, cy, R, t = M.cam_of(im)
    sc = M.SCALES.get(name)
    if sc is None:
        continue
    zg = gt * M.K_UNIT                                                     # 真深度（場景單位）
    sh = zg.shape
    kx, ky = sh[1] / W, sh[0] / H
    mono_inv = resize(np.load(f"{D}/estimated_depths/{name}.npy").squeeze(), sh) * sc["scale"] + sc["offset"]
    tri_inv = resize(np.load(f"{D}/estimated_depths_tri/{name}.npy").squeeze(), sh)
    rel = resize(np.load(f"{D}/estimated_depths_tri_rel/{name}.npy"), sh).astype(np.int64)
    zm = np.where(mono_inv > 1e-6, 1.0 / np.clip(mono_inv, 1e-6, None), np.nan)
    zt = np.where(tri_inv > 1e-6, 1.0 / np.clip(tri_inv, 1e-6, None), np.nan)
    s = a.stride
    Pg, ng = normals(zg, fx * kx, fy * ky, cx * kx, cy * ky, s)
    _, nm = normals(zm, fx * kx, fy * ky, cx * kx, cy * ky, s)
    _, nt = normals(zt, fx * kx, fy * ky, cx * kx, cy * ky, s)
    Pw = (Pg - t) @ R                                                      # 世界點（R^T (P - t)）
    nw = ng @ R                                                            # GT 法向轉世界（只拿來分立面／躺平）
    h = (Pw[..., 2] - ZG) * 100.0
    from scipy.ndimage import maximum_filter, minimum_filter
    smooth = (maximum_filter(zg, 3) / np.clip(minimum_filter(zg, 3), 1e-9, None)) < 1.05
    ok = np.isfinite(zg) & (zg > 0) & smooth
    ok[:2 * s] = ok[-2 * s:] = False; ok[:, :2 * s] = ok[:, -2 * s:] = False
    fac = np.abs(nw[..., 2]) < 0.3
    reliable = ((rel & 7) <= 1) & ((rel >> 3) <= 3) & np.isfinite(zt)
    ang = lambda n2: np.degrees(np.arccos(np.clip(np.abs((n2 * ng).sum(-1)), 0, 1)))
    am, at = ang(nm), ang(nt)
    for band, bm in (("塔頂 >250 m", h > 250), ("較低 50~150 m", (h > 50) & (h <= 150))):
        for rname, rm in (("校正深度可靠", reliable), ("校正深度不可靠", ~reliable)):
            m = ok & fac & bm & rm
            if m.sum() < 50:
                continue
            k = (band, rname); e = acc.setdefault(k, {"mono": [], "tri": [], "n": 0, "nt": 0})
            # 11-16 修：鄰點是 NaN（無效深度）時法向是 NaN => 只收有限值，另記校正深度有法向的比例
            xm = am[m]; e["mono"].append(xm[np.isfinite(xm)][::7]); e["n"] += int(m.sum())
            xt = at[m]; xt = xt[np.isfinite(xt)]; e["nt"] += int(xt.size)
            if xt.size >= 50:
                e["tri"].append(xt[::7])
print(f"b{a.block}：抽 {len(names)} 張訓練影像；立面像素（GT |n_z| < 0.3），深度不連續排除；夾角＝與 GT 法向（度）")
print(f"{'':30}{'像素':>10}{'單目 中位':>10}{'單目 <20°':>10}{'校正有法向':>10}{'校正 中位':>10}{'校正 <20°':>10}")
for k, e in sorted(acc.items()):
    mo = np.concatenate(e["mono"]) if e["mono"] else np.zeros(0); tr = np.concatenate(e["tri"]) if e["tri"] else np.zeros(0)
    f = lambda x: (f"{np.median(x):.0f}", f"{100 * np.mean(x < 20):.0f}%") if len(x) else ("—", "—")
    cov = f"{100 * e['nt'] / max(e['n'], 1):.0f}%"
    print(f"{k[0] + '｜' + k[1]:30}{e['n']:>10,}{f(mo)[0]:>10}{f(mo)[1]:>10}{cov:>10}{f(tr)[0]:>10}{f(tr)[1]:>10}")
