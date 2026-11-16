"""11-17 Fable 第三輪「把海面與塔頂當成平面物件」的前提檢查（使用者「直接做」）。只讀、CPU；GT 深度只在 lab（../mc_depth_f32，經 monodepth_sea_1102）。
做法（全部在影像上，門檻只用角度、相對深度比、影像面積比例、顏色差；沒有場景單位）：
立面（--mode facade）
  候選像素：單目法向（單目深度 x 逐張 scale/offset 反投影、中央差分）在世界座標不躺平 |n_z| < 0.3、不在單目深度斷層（3x3 最大/最小 < 1.05）。
  分區：依單目世界法向的方位角分 12 格（15°），每格各自取連通區；只留面積 >= 影像 0.2% 的區。
  擬合：區內「校正深度可靠」（斜率級 <= 1、大小級 <= 3）的像素反投影成相機座標點，最小平方平面、去掉殘差 > 3 MAD 再擬一次；
        平面法向要和區內單目法向中位一致（< 15°）、可靠像素 >= 200 才收。
  自我檢查：把可靠像素裡世界高度最高的 20% 留出來不參與擬合 => 「往上外推」的誤差（Fable：留一部分可靠像素驗外推）。
  外推：區內不可靠像素沿視線與平面求交；跟 GT 深度比相對誤差。依 GT 世界點的高度分帶（GT 只拿來評分）。
海面（--mode sea）
  候選像素：校正深度無效或在 SfM 凸包外（斜率級 4）、顏色均勻（5x5 內 RGB 標準差 < --col_std）。連通區面積 >= 影像 --min_area 才收（影像定義，不用高度）。
  擬合：區的外圈帶（膨脹 --ring 像素扣掉區本身）上「校正深度可靠」的像素 => 相機座標點 => 平面（同上的穩健擬合）。
  外推：區內每個像素求交；GT 評分只拿來分類：海面＝GT 世界 z 在海平面 ±1 m（Z_SEA，同 monodepth_sea_1102）；其餘是碼頭、船、樓等。
  報：區內海面像素的相對誤差；區內非海面像素「GT 比平面近 1% 以上」的比例（＝深度上限會誤罰的）；全部海面像素被區涵蓋的比例。
用法（lab，cwd＝repo 根）：python tools/probe/plane_extrap_1117.py --mode facade --block 2 [--ncam 60]
                          python tools/probe/plane_extrap_1117.py --mode sea --block 1 [--ncam 60]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["facade", "sea"], required=True); ap.add_argument("--block", type=int, required=True)
ap.add_argument("--ncam", type=int, default=60); ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--min_area", type=float, default=0.02, help="海面區最小面積（影像比例）")
ap.add_argument("--col_std", type=float, default=6.0, help="海面候選：5x5 內 RGB 標準差上限（0～255 色階）")
ap.add_argument("--ring", type=int, default=12, help="海面區外圈帶寬（GT 解析度像素）")
a, _rest = ap.parse_known_args(); sys.argv = sys.argv[:1]
import monodepth_sea_1102 as M
import torch, cv2
from scipy.ndimage import label, maximum_filter, minimum_filter, binary_dilation, uniform_filter
D = M.D
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
bx, by = [int(v) for v in p["partition_coordinates"]["id"][a.block]]
lst = [n for n in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{bx:03d}_{by:03d}.txt").read().split() if n in M.BYNAME]
rng = np.random.default_rng(0); names = [lst[i] for i in sorted(rng.choice(len(lst), min(a.ncam, len(lst)), replace=False))]
ZG = float(np.percentile(M.PX[M.GOODP][:, 2], 5))


def rs(arr, shape, interp=cv2.INTER_NEAREST):
    if arr.shape[:2] == shape:
        return arr
    return cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=interp).astype(np.float64)


def cam_pts(z, fx, fy, cx, cy):
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    return np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1), np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1)


def normals_from(P, s):
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, s:-s] = P[:, 2 * s:] - P[:, :-2 * s]; dy[s:-s] = P[2 * s:] - P[:-2 * s]
    n = np.cross(dx, dy); return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)


def fit_plane(X):
    """穩健平面：n·X = c（|n|=1）；回傳 (n, c, 殘差 MAD 相對中位深度) 或 None。"""
    if len(X) < 30:
        return None
    for _ in range(2):
        mu = X.mean(0); _, _, vt = np.linalg.svd(X - mu, full_matrices=False); n = vt[2]; c = n @ mu
        r = X @ n - c; mad = np.median(np.abs(r - np.median(r))) + 1e-12
        keep = np.abs(r - np.median(r)) < 3 * 1.4826 * mad
        if keep.sum() < 30:
            return None
        X = X[keep]
    mu = X.mean(0); _, _, vt = np.linalg.svd(X - mu, full_matrices=False); n = vt[2]; c = n @ mu
    return n, c, float(np.median(np.abs(X @ n - c)) / max(np.median(X[:, 2]), 1e-9))


def intersect(dirs, n, c):
    den = dirs @ n
    z = np.where(np.abs(den) > 1e-9, c / np.where(np.abs(den) > 1e-9, den, 1.0), np.nan)
    return np.where(z > 0, z, np.nan)


ERRB = [0.01, 0.03, 0.10]
acc = {}
def add(key, err):
    e = acc.setdefault(key, [])
    e.append(np.abs(err[np.isfinite(err)]))

stat = dict(img=0, regions=0, accepted=0, unrel_fac=0, unrel_fac_cov=0, sea_px=0, sea_cov=0, reg_px=0, reg_sea=0, reg_front=0)
for name in names:
    gt = M.load_gt(name)
    sc = M.SCALES.get(name)
    if gt is None or sc is None:
        continue
    im = M.BYNAME[name]; W, H, fx, fy, cx, cy, R, t = M.cam_of(im)
    zg = gt * M.K_UNIT; sh = zg.shape; kx, ky = sh[1] / W, sh[0] / H
    fxs, fys, cxs, cys = fx * kx, fy * ky, cx * kx, cy * ky
    tri_inv = rs(np.load(f"{D}/estimated_depths_tri/{name}.npy").squeeze(), sh)
    rel = rs(np.load(f"{D}/estimated_depths_tri_rel/{name}.npy"), sh).astype(np.int64)
    zt = np.where(tri_inv > 1e-6, 1.0 / np.clip(tri_inv, 1e-6, None), np.nan)
    reliable = ((rel & 7) <= 1) & ((rel >> 3) <= 3) & np.isfinite(zt)
    Pg, dirs = cam_pts(zg, fxs, fys, cxs, cys)
    Pw = (Pg - t) @ R                                         # GT 世界點（只拿來評分）
    hgt = (Pw[..., 2] - ZG) * 100.0
    okg = np.isfinite(zg) & (zg > 0)
    stat["img"] += 1
    if a.mode == "facade":
        mono_inv = rs(np.load(f"{D}/estimated_depths/{name}.npy").squeeze(), sh) * sc["scale"] + sc["offset"]
        zm = np.where(mono_inv > 1e-6, 1.0 / np.clip(mono_inv, 1e-6, None), np.nan)
        Pm, _ = cam_pts(np.nan_to_num(zm, nan=0.0), fxs, fys, cxs, cys)
        nmc = normals_from(Pm, a.stride); nmw = nmc @ R
        zz = np.nan_to_num(zm, nan=0.0)
        smooth = (maximum_filter(zz, 3) / np.clip(minimum_filter(zz, 3), 1e-9, None)) < 1.05
        cand = np.isfinite(zm) & smooth & (np.abs(nmw[..., 2]) < 0.3) & np.isfinite(nmw).all(-1)
        cand[:2 * a.stride] = cand[-2 * a.stride:] = False; cand[:, :2 * a.stride] = cand[:, -2 * a.stride:] = False
        az = np.degrees(np.arctan2(nmw[..., 1], nmw[..., 0])) % 180.0
        Pt, _ = cam_pts(np.nan_to_num(zt, nan=0.0), fxs, fys, cxs, cys)
        gtn = normals_from(Pg, a.stride) @ R
        gfac = okg & (np.abs(gtn[..., 2]) < 0.3)
        unrel_fac = gfac & ~reliable
        stat["unrel_fac"] += int(unrel_fac.sum())
        cov = np.zeros(sh, bool)
        for k in range(12):
            lab_, nl = label(cand & (az >= 15 * k) & (az < 15 * (k + 1)))
            if nl == 0:
                continue
            sizes = np.bincount(lab_.ravel())
            for r_ in np.nonzero(sizes >= 0.002 * lab_.size)[0]:
                if r_ == 0:
                    continue
                reg = lab_ == r_; stat["regions"] += 1
                rr = reg & reliable
                if rr.sum() < 200:
                    continue
                hw = (((Pt[rr] - t) @ R)[:, 2])                  # 可靠像素的世界高度（校正深度）
                cut = np.percentile(hw, 80)
                fitm = hw < cut
                f = fit_plane(Pt[rr][fitm])
                if f is None:
                    continue
                n_, c_, _ = f
                nr = nmc[reg]; nr = nr * np.sign(nr @ nr[len(nr) // 2])[:, None]   # 統一正反面再取中位
                nmed = np.median(nr, 0); nmed /= np.linalg.norm(nmed) + 1e-12
                if np.degrees(np.arccos(min(1.0, abs(float(n_ @ nmed))))) > 15:
                    continue
                stat["accepted"] += 1
                # 自我檢查：最高 20% 的可靠像素（沒參與擬合）
                ho = rr.copy(); ho[rr] = ~fitm
                zp = intersect(dirs[ho], n_, c_)
                add(("自我檢查：可靠像素最高 20%（往上外推）", "全部"), zp / zt[ho] - 1.0)
                tgt = reg & ~reliable & okg
                zp = intersect(dirs[tgt], n_, c_)
                err = zp / zg[tgt] - 1.0
                hh = hgt[tgt]; gf = gfac[tgt]
                for lo, hi in ((0, 150), (150, 250), (250, 1e9)):
                    kk = gf & (hh >= lo) & (hh < hi)
                    add(("不可靠立面像素：外推深度 vs GT", f"{lo}~{hi if hi < 1e8 else '∞'} m"), err[kk])
                cov |= tgt
        stat["unrel_fac_cov"] += int((cov & unrel_fac).sum())
    else:
        img = cv2.imread(f"{D}/input/{name}")
        if img is None:
            continue
        img = rs(img, sh, cv2.INTER_AREA)
        mu = uniform_filter(img, size=(5, 5, 1)); sd = np.sqrt(np.clip(uniform_filter(img ** 2, size=(5, 5, 1)) - mu ** 2, 0, None)).max(-1)
        nodepth = ~np.isfinite(zt) | ((rel & 7) == 4)
        cand = nodepth & (sd < a.col_std)
        sea = okg & (np.abs(Pw[..., 2] - M.Z_SEA) < 0.01)
        stat["sea_px"] += int(sea.sum())
        Pt, _ = cam_pts(np.nan_to_num(zt, nan=0.0), fxs, fys, cxs, cys)
        lab_, nl = label(cand)
        if nl == 0:
            continue
        sizes = np.bincount(lab_.ravel()); cov = np.zeros(sh, bool)
        for r_ in np.nonzero(sizes >= a.min_area * lab_.size)[0]:
            if r_ == 0:
                continue
            reg = lab_ == r_; stat["regions"] += 1
            ring = binary_dilation(reg, iterations=a.ring) & ~reg & reliable
            f = fit_plane(Pt[ring]) if ring.sum() >= 100 else None
            if f is None:
                continue
            n_, c_, _ = f; stat["accepted"] += 1
            tgt = reg & okg
            zp = intersect(dirs[tgt], n_, c_)
            err = zp / zg[tgt] - 1.0
            s_ = sea[tgt]
            add(("區內海面像素：平面深度 vs GT", "全部"), err[s_])
            front = (~s_) & np.isfinite(err) & (err > 0.01)     # 平面比 GT 遠 1% 以上 => GT 的東西在平面前方 => 上限會誤罰
            stat["reg_px"] += int(tgt.sum()); stat["reg_sea"] += int(s_.sum()); stat["reg_front"] += int(front.sum())
            add(("區內非海面像素（碼頭、船、樓）：平面深度 vs GT", "全部"), err[~s_])
            cov |= tgt
        stat["sea_cov"] += int((cov & sea).sum())

print(f"b{a.block}｜{a.mode}｜影像 {stat['img']}｜候選區 {stat['regions']}、擬合通過 {stat['accepted']}")
if a.mode == "facade":
    print(f"不可靠的 GT 立面像素 {stat['unrel_fac']:,}，被通過的平面涵蓋 {100 * stat['unrel_fac_cov'] / max(stat['unrel_fac'], 1):.0f}%")
else:
    print(f"GT 海面像素 {stat['sea_px']:,}，被區涵蓋 {100 * stat['sea_cov'] / max(stat['sea_px'], 1):.0f}%｜區內像素 {stat['reg_px']:,}："
          f"海面 {100 * stat['reg_sea'] / max(stat['reg_px'], 1):.0f}%、非海面且 GT 在平面前方 > 1%（上限會誤罰）{100 * stat['reg_front'] / max(stat['reg_px'], 1):.1f}%")
print(f"{'項目':44}{'分組':>12}{'像素數':>10}{'|誤差|中位':>11}" + "".join(f"{f'< {int(b * 100)}%':>8}" for b in ERRB))
for (k1, k2), es in acc.items():
    e = np.concatenate(es) if es else np.zeros(0)
    if len(e) < 30:
        print(f"{k1:44}{k2:>12}{len(e):>10}{'—':>11}"); continue
    print(f"{k1:44}{k2:>12}{len(e):>10,}{100 * np.median(e):>10.1f}%" + "".join(f"{100 * (e < b).mean():>7.0f}%" for b in ERRB))
print("DONE plane_extrap")
