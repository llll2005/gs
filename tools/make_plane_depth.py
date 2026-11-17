#!/usr/bin/env python
"""★ 2026-11-17 平面外推深度（Fable 第三輪「把海面與塔頂當成平面物件」；使用者「作」）。CPU；lab 上跑。
每張訓練影像（解析度同校正深度，960x540）產生一張「平面外推的逆深度」，只放在校正深度不可靠／沒有的像素上：
  立面（kind 1）：單目深度 x 逐張 scale/offset 反投影、差分外積得法向 => 世界法向不躺平（|n_z| < facade_nz）、不在單目深度斷層的像素，
        依世界法向方位角分 12 格（15°）各自取連通區（面積 >= 影像 facade_area）。區內校正深度可靠的像素反投影成相機座標點，穩健擬合平面；
        收下的條件：可靠像素 >= min_rel；平面法向和區內單目法向中位一致（< agree_deg）；**自我檢查**＝可靠像素裡世界高度最高的 20% 不參與擬合，
        用平面往上外推到它們，相對誤差中位 < self_tol（Fable：留一部分可靠像素驗外推）。通過後用全部可靠像素重擬，外推到區內不可靠的像素。
  海面（kind 2）：校正深度無效或在 SfM 凸包外（斜率級 4）、顏色均勻（5x5 內 RGB 標準差 < col_std）的連通區（面積 >= 影像 sea_area）。
        區的外圈帶（膨脹 ring 像素、扣掉區本身）上校正深度可靠的像素擬合平面（殘差 MAD 相對中位深度 < self_tol），外推到整個區。
        海面由影像定義，不用高度挑（Fable：湖面、大片空地也適用）。
  兩種都擋掠射：視線和平面法向夾角 > 80° 的像素不放（|cos| < 0.17）。
沒有場景單位：門檻只有角度、相對深度比、影像面積比例、顏色差（0～255 色階）、像素。
前提檢查（tools/probe/plane_extrap_1117.py，GT 深度）：立面塔頂（> 250 m）外推誤差中位 0.8～1.5%、自我檢查 0.2%；海面中位 0.6～1.8%、區內幾乎全是海。
輸出：<dataset>/<out>/<影像名>.npy（float16 逆深度 [540,960]，0＝沒有）、<影像名>_k.png（uint8：1 立面、2 海面）、stats_<blocks>.json
用法（lab，cwd＝repo 根）：python tools/make_plane_depth.py data/matrix_city/aerial/train/block_all --blocks 1 2 3 --nproc 12
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
A = None


def cam_pts(z, fx, fy, cx, cy):
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    return d * z[..., None], d


def normals_from(P, s):
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, s:-s] = P[:, 2 * s:] - P[:, :-2 * s]; dy[s:-s] = P[2 * s:] - P[:-2 * s]
    n = np.cross(dx, dy); return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)


def fit_plane(X):
    """穩健平面 n·X = c；回傳 (n, c, 殘差 MAD／中位深度) 或 None。"""
    if len(X) < 30:
        return None
    for _ in range(2):
        mu = X.mean(0); _, _, vt = np.linalg.svd(X - mu, full_matrices=False); n = vt[2]; c = n @ mu
        r = X @ n - c; med = np.median(r); mad = np.median(np.abs(r - med)) + 1e-12
        keep = np.abs(r - med) < 3 * 1.4826 * mad
        if keep.sum() < 30:
            return None
        X = X[keep]
    mu = X.mean(0); _, _, vt = np.linalg.svd(X - mu, full_matrices=False); n = vt[2]; c = n @ mu
    return n, c, float(np.median(np.abs(X @ n - c)) / max(float(np.median(X[:, 2])), 1e-12))


def intersect(dirs, n, c):
    den = dirs @ n
    dn = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    graze = np.abs(dn @ n) < 0.17                      # 視線與平面法向夾角 > 80°
    z = np.where(np.abs(den) > 1e-12, c / np.where(np.abs(den) > 1e-12, den, 1.0), np.nan)
    return np.where((z > 0) & ~graze, z, np.nan)


def one(job):
    name, cam, R = job
    import cv2
    from PIL import Image
    from scipy.ndimage import label, maximum_filter, minimum_filter, binary_dilation, uniform_filter
    a = A
    o_npy = os.path.join(a["out"], name + ".npy"); o_k = os.path.join(a["out"], name + "_k.png"); o_p = os.path.join(a["out"], name + "_planes.npz")
    have_maps = os.path.exists(o_npy) and os.path.exists(o_k)
    if have_maps and os.path.exists(o_p) and not a["force"]:
        return name, None
    regs = []                                        # ★ 11-17 規格 6 §1 P1：每個收下的區的平面參數（相機座標 n·X = c）
    D = a["D"]
    ft = os.path.join(D, "estimated_depths_tri", name + ".npy"); fr = os.path.join(D, "estimated_depths_tri_rel", name + ".npy")
    if not (os.path.exists(ft) and os.path.exists(fr)):
        return name, "缺校正深度"
    tri = np.load(ft).squeeze().astype(np.float64); rel = np.load(fr).astype(np.int64)
    H, W = tri.shape
    fx0, fy0, cx0, cy0, w0, h0 = cam
    fx, fy, cx, cy = fx0 * W / w0, fy0 * H / h0, cx0 * W / w0, cy0 * H / h0
    zt = np.where(tri > 1e-6, 1.0 / np.clip(tri, 1e-6, None), np.nan)
    reliable = ((rel & 7) <= a["slope_max"]) & ((rel >> 3) <= a["size_max"]) & np.isfinite(zt)
    Pt, dirs = cam_pts(np.nan_to_num(zt, nan=0.0), fx, fy, cx, cy)
    out_inv = np.zeros((H, W), np.float32); kind = np.zeros((H, W), np.uint8)
    st = dict(fac_reg=0, fac_ok=0, sea_reg=0, sea_ok=0)
    if a["facade"]:
        sc = a["scales"].get(name); fm = os.path.join(D, a["depth_dir"], name + ".npy")
        if sc is not None and os.path.exists(fm):
            mono = cv2.resize(np.load(fm).squeeze().astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) * sc["scale"] + sc["offset"]
            zm = np.where(mono > 1e-6, 1.0 / np.clip(mono, 1e-6, None), np.nan)
            Pm, _ = cam_pts(np.nan_to_num(zm, nan=0.0), fx, fy, cx, cy)
            s = a["stride"]
            nmc = normals_from(Pm, s); nmw = nmc @ R
            zz = np.nan_to_num(zm, nan=0.0)
            smooth = (maximum_filter(zz, 3) / np.clip(minimum_filter(zz, 3), 1e-9, None)) < 1 + a["disc"]
            cand = np.isfinite(zm) & smooth & (np.abs(nmw[..., 2]) < a["facade_nz"]) & np.isfinite(nmw).all(-1)
            cand[:2 * s] = cand[-2 * s:] = False; cand[:, :2 * s] = cand[:, -2 * s:] = False
            az = np.degrees(np.arctan2(nmw[..., 1], nmw[..., 0])) % 180.0
            for k in range(12):
                lab_, nl = label(cand & (az >= 15 * k) & (az < 15 * (k + 1)))
                if nl == 0:
                    continue
                sizes = np.bincount(lab_.ravel())
                for r_ in np.nonzero(sizes >= a["facade_area"] * lab_.size)[0]:
                    if r_ == 0:
                        continue
                    reg = lab_ == r_; st["fac_reg"] += 1
                    rr = reg & reliable
                    if rr.sum() < a["min_rel"]:
                        continue
                    Xr = Pt[rr]; zw = ((Xr - a["t"][name]) @ R)[:, 2]
                    cut = np.percentile(zw, 80); fitm = zw < cut
                    f = fit_plane(Xr[fitm])
                    if f is None:
                        continue
                    nr = nmc[reg]; nr = nr * np.sign(nr @ nr[len(nr) // 2])[:, None]
                    nmed = np.median(nr, 0); nmed /= np.linalg.norm(nmed) + 1e-12
                    if np.degrees(np.arccos(min(1.0, abs(float(f[0] @ nmed))))) > a["agree_deg"]:
                        continue
                    ho = ~fitm
                    if ho.sum() >= 20:
                        zp = intersect(dirs[rr][ho], f[0], f[1])
                        e = np.abs(zp / Xr[ho, 2] - 1.0)
                        if not np.isfinite(e).any() or np.nanmedian(e) > a["self_tol"]:
                            continue
                    f = fit_plane(Xr)
                    if f is None:
                        continue
                    tgt = reg & ~reliable
                    zp = intersect(dirs[tgt], f[0], f[1])
                    ok = np.isfinite(zp)
                    ii = np.nonzero(tgt)
                    out_inv[ii[0][ok], ii[1][ok]] = (1.0 / zp[ok]).astype(np.float32); kind[ii[0][ok], ii[1][ok]] = 1
                    st["fac_ok"] += 1
                    vv_, uu_ = np.nonzero(reg); regs.append((1, f[0], f[1], int(reg.sum()), f[2], float(uu_.mean()), float(vv_.mean())))
    if a["sea"]:
        img = cv2.imread(os.path.join(D, "input", name))
        if img is not None:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float64)
            mu = uniform_filter(img, size=(5, 5, 1)); sd = np.sqrt(np.clip(uniform_filter(img ** 2, size=(5, 5, 1)) - mu ** 2, 0, None)).max(-1)
            nodepth = ~np.isfinite(zt) | ((rel & 7) == 4)
            lab_, nl = label(nodepth & (sd < a["col_std"]) & (kind == 0))
            if nl:
                sizes = np.bincount(lab_.ravel())
                for r_ in np.nonzero(sizes >= a["sea_area"] * lab_.size)[0]:
                    if r_ == 0:
                        continue
                    reg = lab_ == r_; st["sea_reg"] += 1
                    ring = binary_dilation(reg, iterations=a["ring"]) & ~reg & reliable
                    if ring.sum() < 100:
                        continue
                    f = fit_plane(Pt[ring])
                    if f is None or f[2] > a["self_tol"]:
                        continue
                    zp = intersect(dirs[reg], f[0], f[1]); ok = np.isfinite(zp)
                    ii = np.nonzero(reg)
                    out_inv[ii[0][ok], ii[1][ok]] = (1.0 / zp[ok]).astype(np.float32); kind[ii[0][ok], ii[1][ok]] = 2
                    st["sea_ok"] += 1
                    regs.append((2, f[0], f[1], int(reg.sum()), f[2], float(ii[1].mean()), float(ii[0].mean())))
    if a["force"] or not have_maps:                  # 已有的深度圖不覆寫（訓練中的跑次可能正在讀）
        np.save(o_npy, out_inv.astype(np.float16))
        Image.fromarray(kind).save(o_k)
    np.savez(o_p, kind=np.array([r[0] for r in regs], np.int8), n=np.array([r[1] for r in regs], np.float64).reshape(-1, 3),
             c=np.array([r[2] for r in regs], np.float64), npix=np.array([r[3] for r in regs], np.int64), resid=np.array([r[4] for r in regs], np.float64),
             cu=np.array([r[5] for r in regs], np.float64), cv=np.array([r[6] for r in regs], np.float64), size=np.array([W, H]))
    st["fac_px"] = float((kind == 1).mean()); st["sea_px"] = float((kind == 2).mean())
    return name, st


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset"); ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
    ap.add_argument("--out", default="estimated_depths_plane")
    ap.add_argument("--kinds", default="facade,sea")
    ap.add_argument("--slope_max", type=int, default=1); ap.add_argument("--size_max", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1, help="單目法向的差分步長（960 寬；＝原圖 2 px）")
    ap.add_argument("--disc", type=float, default=0.05); ap.add_argument("--facade_nz", type=float, default=0.3)
    ap.add_argument("--facade_area", type=float, default=0.002); ap.add_argument("--min_rel", type=int, default=50)
    ap.add_argument("--agree_deg", type=float, default=15.0); ap.add_argument("--self_tol", type=float, default=0.01)
    ap.add_argument("--sea_area", type=float, default=0.02); ap.add_argument("--col_std", type=float, default=6.0)
    ap.add_argument("--ring", type=int, default=6)
    ap.add_argument("--depth_dir", default="estimated_depths", help="單目逆深度（本機舊命名 estimated_depths_lab）")
    ap.add_argument("--nproc", type=int, default=12); ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from internal.utils.colmap import read_cameras_binary, read_images_binary, qvec2rotmat
    D = a.dataset
    cams = read_cameras_binary(os.path.join(D, "sparse/0/cameras.bin")); imgs = read_images_binary(os.path.join(D, "sparse/0/images.bin"))
    by_name = {v.name: v for v in imgs.values()}
    names = []
    for b in a.blocks:
        names += [l.strip() for l in open(os.path.join(D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))) if l.strip()]
    names = sorted(set(n for n in names if n in by_name))
    if a.limit:
        names = names[:: max(1, len(names) // a.limit)][: a.limit]
    out = os.path.join(D, a.out); os.makedirs(out, exist_ok=True)
    jobs, tvec = [], {}
    for n in names:
        im = by_name[n]; cm = cams[im.camera_id]
        fx = cm.params[0]; fy = cm.params[1] if len(cm.params) > 3 else fx
        cx, cy = (cm.params[2], cm.params[3]) if len(cm.params) > 3 else (cm.params[1], cm.params[2])
        jobs.append((n, (fx, fy, cx, cy, cm.width, cm.height), qvec2rotmat(im.qvec)))
        tvec[n] = np.asarray(im.tvec, dtype=np.float64)
    ks = set(a.kinds.split(","))
    A = dict(D=D, out=out, scales=json.load(open(os.path.join(D, "estimated_depth_scales.json"))), depth_dir=a.depth_dir, force=a.force,
             facade="facade" in ks, sea="sea" in ks, slope_max=a.slope_max, size_max=a.size_max, stride=a.stride, disc=a.disc,
             facade_nz=a.facade_nz, facade_area=a.facade_area, min_rel=a.min_rel, agree_deg=a.agree_deg, self_tol=a.self_tol,
             sea_area=a.sea_area, col_std=a.col_std, ring=a.ring, t=tvec)
    t0 = time.time(); res = []; bad = 0
    with Pool(a.nproc) as pool:
        for i, (n, r) in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            if isinstance(r, str):
                bad += 1
            elif r is not None:
                res.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}（{time.time() - t0:.0f} s）", flush=True)
    agg = {k: float(np.sum([r[k] for r in res])) for k in ("fac_reg", "fac_ok", "sea_reg", "sea_ok")} if res else {}
    agg.update(fac_px=float(np.mean([r["fac_px"] for r in res])) if res else 0.0, sea_px=float(np.mean([r["sea_px"] for r in res])) if res else 0.0)
    st = dict(blocks=a.blocks, images=len(jobs), new=len(res), failed=bad, **agg, args=vars(a))
    json.dump(st, open(os.path.join(out, f"stats_{'_'.join(map(str, a.blocks))}.json"), "w"), indent=1, default=str)
    print(f"完成：{len(jobs)} 張（新產生 {len(res)}、失敗 {bad}）｜立面區 {agg.get('fac_ok', 0):.0f}/{agg.get('fac_reg', 0):.0f} 通過、"
          f"海面區 {agg.get('sea_ok', 0):.0f}/{agg.get('sea_reg', 0):.0f} 通過｜平均每張 立面平面像素 {100 * agg.get('fac_px', 0):.1f}%、"
          f"海面平面像素 {100 * agg.get('sea_px', 0):.1f}%｜{time.time() - t0:.0f} s => {out}", flush=True)


if __name__ == "__main__":
    main()
