"""★ 2026-11-11 SfM 三角形校正的單目深度（每張影像一張逆深度圖＋一張可靠度分級圖）。

做什麼：每張訓練影像，用「它自己觀測到的可靠 SfM 點」（images.bin 的 point3D_ids；track>=3 且重投影誤差<1.5）在影像上做 Delaunay 三角形，
        三角形內線性內插「SfM 逆深度 − 現行單目逆深度」的殘差，加回單目：
            inv(p) = 單目(p)·s + o ＋ 內插_三角形( 1/z_SfM − (單目·s + o) )(p)
        SfM 點上精確；平面的逆深度在影像座標上是仿射的 => 三角形內線性內插對平面（路面、屋頂、立面）是精確的（🟨 ACMP 的平面先驗，
        Xu & Tao AAAI 2020；我方改成「校正單目」而不是取代它，三角形內保留單目的起伏）。凸包外用最近 SfM 點的殘差。
為什麼：單目深度（每張一組 SfM 縮放）誤差中位 2.15%、判前方 15.7%，塔頂 18%／77%；校正後 0.15%／2.2%，塔頂 0.45%／16%；
        可靠三角形內（斜率 < 3，佔 70% 像素）0.05%／0.3%（lab 350 張對 MatrixCity 真深度，`tools/probe/tri_depth_1111.py`；真深度只診斷）。
        好幾條線的重開條件都是「比單目更準的稠密深度」（紀錄/實驗分析/31、33、34）。
可靠度分級（uint8，與深度圖同大小）＝斜率級 + 8 x 大小級：
        斜率級 0／1／2／3／4＝三角形的無因次斜率 < 1／< 3／< 8／>= 8／凸包外；斜率＝|∇ log z| x fx（表面對視線的 cot，沒有場景單位）
        大小級 0／1／2／3＝三角形最長邊 < 16／< 64／< 256／>= 256 px
輸出（相對 <dataset>）：<out>/<影像名>.npy（float32 逆深度，場景單位，<=0＝無效）、<out>_rel/<影像名>.npy（uint8）、<out_scales>.json（全部 scale 1、offset 0）
        大小＝相機大小 / --down（預設 2 => 960x540；使用端都按比例取樣）。
用法（lab，cwd＝repo 根目錄，CPU）：
  conda run -n gspl --no-capture-output python tools/make_tri_depths.py data/matrix_city/aerial/train/block_all --nproc 6
訓練時：--data.parser.depth_dir estimated_depths_tri --data.parser.depth_scale_name estimated_depth_scales_tri
        --data.parser.depth_rel_dir estimated_depths_tri_rel [--data.parser.depth_rel_slope_max 1 --data.parser.depth_rel_size_max 3]
"""
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import cv2
from multiprocessing import Pool
from scipy.spatial import Delaunay, cKDTree
sys.path.insert(0, ".")
cv2.setNumThreads(1)

SL_E = [1.0, 3.0, 8.0]
SZ_E = [16.0, 64.0, 256.0]


def tri_correct(uv, zc, f_mono_pts, mono_grid, W, H, fx, down: int):
    """uv [n,2]（相機像素座標）、zc [n]（相機 z）、f_mono_pts [n]（單目逆深度在 SfM 點上的值；None＝沒有單目）、
    mono_grid [h,w]（輸出解析度的單目逆深度；None＝沒有單目）。回傳 (inv [h,w] float32, cls [h,w] uint8)。"""
    h, w = H // down, W // down
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * down, (np.arange(h) + 0.5) * down)
    P = np.stack([uu.ravel(), vv.ravel()], 1)
    f_sfm = 1.0 / zc
    res = f_sfm - f_mono_pts if f_mono_pts is not None else f_sfm
    tri = Delaunay(uv)
    s = tri.find_simplex(P)
    ins = s >= 0
    sidx = np.where(ins, s, 0)
    T = tri.transform[sidx]
    b2 = np.einsum("nij,nj->ni", T[:, :2], P - T[:, 2])
    bc = np.concatenate([b2, 1.0 - b2.sum(1, keepdims=True)], 1)
    vid = tri.simplices[sidx]
    lin = (res[vid] * bc).sum(1)
    if (~ins).any():
        lin[~ins] = res[cKDTree(uv).query(P[~ins])[1]]
    inv = lin.reshape(h, w) + (mono_grid if mono_grid is not None else 0.0)
    sv = tri.simplices
    p0, p1, p2 = uv[sv[:, 0]], uv[sv[:, 1]], uv[sv[:, 2]]
    A = np.stack([p1 - p0, p2 - p0], 1)
    rhs = np.stack([f_sfm[sv[:, 1]] - f_sfm[sv[:, 0]], f_sfm[sv[:, 2]] - f_sfm[sv[:, 0]]], 1)
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    okd = np.abs(det) > 1e-9
    gx = np.where(okd, (rhs[:, 0] * A[:, 1, 1] - rhs[:, 1] * A[:, 0, 1]) / np.where(okd, det, 1), 0)
    gy = np.where(okd, (rhs[:, 1] * A[:, 0, 0] - rhs[:, 0] * A[:, 1, 0]) / np.where(okd, det, 1), 0)
    slope_t = np.where(okd, np.hypot(gx, gy) / f_sfm[sv].mean(1) * fx, np.inf)
    edge_t = np.maximum.reduce([np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p1, axis=1), np.linalg.norm(p0 - p2, axis=1)])
    sl = np.where(ins, np.searchsorted(SL_E, slope_t[sidx], side="right"), 4)
    sz = np.where(ins, np.searchsorted(SZ_E, edge_t[sidx], side="right"), 3)
    inv = np.where(np.isfinite(inv) & (inv > 0), inv, 0.0).astype(np.float32)
    cls = (sl + 8 * sz).astype(np.uint8).reshape(h, w)
    cls[inv <= 0] = 4 + 8 * 3
    return inv, cls


def sfm_obs(im):
    from internal.utils.colmap import qvec2rotmat
    R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec, dtype=np.float64)
    ok = im.point3D_ids >= 0
    ii = ID2I[im.point3D_ids[ok]]
    keep = ii >= 0
    ii = ii[keep]; uv = im.xys[ok][keep]
    zc = (PX[ii] @ R.T + t)[:, 2]
    return uv, zc, ii


def one(name):
    im = BYNAME[name]
    out = f"{A.dataset}/{A.out}/{name}.npy"; outr = f"{A.dataset}/{A.out}_rel/{name}.npy"
    if not A.force and os.path.exists(out) and os.path.exists(outr):
        return name, "已有", None
    cam = CAMS[im.camera_id]
    W, H = int(cam.width), int(cam.height); fx = float(cam.params[0])
    uv, zc, ii = sfm_obs(im)
    g = (zc > 0) & GOODP[ii] & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, zc = uv[g], zc[g]
    if len(zc):
        _, ix = np.unique(np.round(uv * 4).astype(np.int64), axis=0, return_index=True)
        uv, zc = uv[ix], zc[ix]
    h, w = H // A.down, W // A.down
    sc = SCALES.get(name)
    mono_ok = sc is not None and (A.lb * MED <= sc["scale"] <= A.ub * MED) and os.path.exists(f"{A.dataset}/{A.mono}/{name}.npy")
    raw = None
    if mono_ok:
        raw = np.load(f"{A.dataset}/{A.mono}/{name}.npy").astype(np.float64)
        raw = raw[..., 0] if raw.ndim == 3 else raw
        if raw.shape != (H, W):
            raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
        raw = raw * sc["scale"] + sc["offset"]
    if len(zc) < A.min_pts:                    # SfM 觀測太少：只能原樣給單目（全部標成凸包外＝不可靠）；連單目都沒有就全 0
        inv = np.zeros((h, w), np.float32) if raw is None else np.maximum(cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA), 0).astype(np.float32)
        cls = np.full((h, w), 4 + 8 * 3, np.uint8)
        why = "SfM 觀測太少" + ("、也沒有單目" if raw is None else "（原樣給單目）")
    else:
        ui = np.clip(np.floor(uv[:, 0]).astype(int), 0, W - 1); vi = np.clip(np.floor(uv[:, 1]).astype(int), 0, H - 1)
        if raw is None:                        # 沒有可用的單目（縮放超界／缺檔）：純 SfM 三角形，凸包外不可靠
            inv, cls = tri_correct(uv, zc, None, None, W, H, fx, A.down)
            why = "沒有單目（純 SfM 三角形）"
        else:
            inv, cls = tri_correct(uv, zc, raw[vi, ui], cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA), W, H, fx, A.down)
            why = ""
    np.save(out, inv); np.save(outr, cls)
    sl = cls & 7
    # 自我檢查：SfM 點所在的輸出像素，逆深度應該≈1/z（半解析度取樣 => 不會完全相等）
    dev = np.nan
    if len(zc) >= A.min_pts:
        uo = np.clip((uv[:, 0] / A.down).astype(int), 0, w - 1); vo = np.clip((uv[:, 1] / A.down).astype(int), 0, h - 1)
        v = inv[vo, uo]
        dev = float(np.median(np.abs(v[v > 0] * zc[v > 0] - 1))) if (v > 0).any() else np.nan
    return name, why, (len(zc), float((sl < 1).mean()), float((sl < 2).mean()), float((sl < 3).mean()), float((sl == 4).mean()), float((inv > 0).mean()), dev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--mono", default="estimated_depths"); ap.add_argument("--mono_scales", default="estimated_depth_scales")
    ap.add_argument("--out", default="estimated_depths_tri"); ap.add_argument("--out_scales", default="estimated_depth_scales_tri")
    ap.add_argument("--down", type=int, default=2); ap.add_argument("--nproc", type=int, default=6)
    ap.add_argument("--min_pts", type=int, default=50); ap.add_argument("--lb", type=float, default=0.2); ap.add_argument("--ub", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="只做前 N 張（冒煙測試）")
    ap.add_argument("--force", action="store_true")
    A = ap.parse_args()
    from internal.utils.colmap import read_cameras_binary, read_images_binary, read_points3D_binary
    t0 = time.time()
    CAMS = read_cameras_binary(f"{A.dataset}/sparse/0/cameras.bin")
    imgs = read_images_binary(f"{A.dataset}/sparse/0/images.bin")
    pts = read_points3D_binary(f"{A.dataset}/sparse/0/points3D.bin")
    pid = np.fromiter(pts.keys(), dtype=np.int64, count=len(pts))
    PX = np.stack([p.xyz for p in pts.values()]).astype(np.float64)
    GOODP = (np.fromiter((len(p.image_ids) for p in pts.values()), dtype=np.int64, count=len(pts)) >= 3) & \
            (np.fromiter((p.error for p in pts.values()), dtype=np.float64, count=len(pts)) < 1.5)
    del pts
    ID2I = np.full(pid.max() + 1, -1, dtype=np.int64); ID2I[pid] = np.arange(len(pid))
    BYNAME = {im.name: im for im in imgs.values()}
    SCALES = json.load(open(f"{A.dataset}/{A.mono_scales}.json"))
    MED = float(np.median([v["scale"] for v in SCALES.values()]))
    names = sorted(BYNAME)
    if A.limit:
        names = names[:: max(1, len(names) // A.limit)][: A.limit]
    os.makedirs(f"{A.dataset}/{A.out}", exist_ok=True); os.makedirs(f"{A.dataset}/{A.out}_rel", exist_ok=True)
    print(f"SfM 點 {len(PX):,}（可靠 {int(GOODP.sum()):,}）｜影像 {len(names):,}｜{time.time() - t0:.0f} s｜輸出 {A.dataset}/{A.out}（1/{A.down} 解析度）", flush=True)
    stats, whys, done = [], {}, 0
    with Pool(A.nproc) as pool:
        for name, why, st in pool.imap_unordered(one, names, chunksize=8):
            done += 1
            if why:
                whys[why] = whys.get(why, 0) + 1
            if st is not None:
                stats.append(st)
            if done % 500 == 0:
                print(f"  {done:,}/{len(names):,}｜{time.time() - t0:.0f} s", flush=True)
    json.dump({n: {"scale": 1.0, "offset": 0.0} for n in sorted(BYNAME)}, open(f"{A.dataset}/{A.out_scales}.json", "w"))
    print(f"完成 {done:,} 張｜{time.time() - t0:.0f} s｜備註 {whys}")
    if stats:
        S = np.array(stats, dtype=np.float64)
        print(f"每張可靠 SfM 觀測 中位 {np.median(S[:, 0]):,.0f}（p10 {np.percentile(S[:, 0], 10):,.0f}）")
        print(f"像素比例（逐張平均）：斜率<1 {100 * S[:, 1].mean():.1f}%、<3 {100 * S[:, 2].mean():.1f}%、<8 {100 * S[:, 3].mean():.1f}%、凸包外 {100 * S[:, 4].mean():.1f}%、有效 {100 * S[:, 5].mean():.1f}%")
        print(f"[自我檢查] SfM 點所在像素的 |逆深度 x z_SfM − 1| 中位（逐張中位再取中位）{100 * np.nanmedian(S[:, 6]):.3f}%（應 << 1%；半解析度取樣造成的殘差）")
    print("DONE make_tri_depths", flush=True)
