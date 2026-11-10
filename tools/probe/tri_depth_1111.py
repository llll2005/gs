"""11-11 自主研究：每張影像用「它自己看得到的 SfM 點」做 Delaunay 三角形，校正單目深度——到底有多準。
動機：好幾條線的重開條件都是「有比單目（約 ±10%）更準的稠密深度」（紀錄/實驗分析/31「不做」表：單目看穿判準、單目深度 loss、硬性貼齊；
33：塊外遮罩在高樓接縫誤判、深度支撐閘門把塔頂壓成煙；34：補洞放點沿視線放到單目深度）。
SfM 觀測（images.bin 的 point3D_ids）＝這張影像真的看得到的點，沒有遮擋歧義；平面的逆深度在影像座標上是仿射的
=> 三角形內線性內插 SfM 逆深度，對平面（路面、屋頂、立面）是精確的（ACMP 的平面先驗，Xu & Tao 2020）。
三種深度（都只用訓練輸入；MatrixCity 真深度只拿來量誤差、不進訓練）：
  ① 現行：單目＋每張一組 SfM 縮放（訓練用的那份）
  ⑤ 三角形校正單目：inv = ①＋三角形內插(SfM 逆深度 − ① 在 SfM 點的值)   （SfM 點上精確、三角形內保留單目的起伏）
  ⑥ 純 SfM 三角形：inv = 三角形內插(SfM 逆深度)                          （平面先驗；跨高低落差的三角形會錯）
e = z_真 × inv − 1。分組：三角形斜率（無因次：|∇ log z| × fx，＝表面對視線的 cot）、三角形最長邊（px）、高度帶、朝向、有無紋理、離 SfM 遠近。
用法（lab，cwd＝gs，CPU）：STRIDE=16 NPROC=6 conda run -n gspl --no-capture-output python tools/probe/tri_depth_1111.py
輸出：stdout 表格、outputs/lab/tri_depth_1111/{hist.npz, *.jpg}
"""
import os, sys, time
os.environ.setdefault("STRIDE", "16")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, "tools/probe"); sys.path.insert(0, "../labrun")
import numpy as np
import cv2
from multiprocessing import Pool
from scipy.spatial import Delaunay, cKDTree
from scipy.ndimage import uniform_filter
import monodepth_sea_1102 as M            # 共用：SfM、真深度配對、單位檢查（import 時會跑 md5 與單位檢查）

OUT = "outputs/lab/tri_depth_1111"
os.makedirs(OUT, exist_ok=True)
ST = 2
EB = np.linspace(-0.2, 0.2, 401)          # e 的直方圖（0.001 一格）；超出夾到邊
NB = len(EB) - 1
ALIGN = ["①現行單目", "⑤三角形校正單目", "⑥純SfM三角形"]
SL_E = [1.0, 3.0, 8.0]                    # 斜率分級（無因次）
SL_N = ["斜率<1", "1~3", "3~8", ">8", "凸包外"]
SZ_E = [16.0, 64.0, 256.0]                # 三角形最長邊（px）
SZ_N = ["邊<16px", "16~64", "64~256", ">256"]
BAND_N = M.BAND_N
ORI_N = M.ORI_N
TX_N = ["無紋理", "有紋理"]
NCODE = 5 * 3 * 5 * 4 * 2 * 2             # 高度帶, 朝向, 斜率級, 大小級, 紋理, 近 SfM


def code_of(band, ori, sl, sz, tx, near):
    return ((((band * 3 + ori) * 5 + sl) * 4 + sz) * 2 + tx) * 2 + near


def one(name, want_maps=False):
    im = M.BYNAME[name]
    sc = M.SCALES.get(name)
    if sc is None or not (M.LB * M.MED <= sc["scale"] <= M.UB * M.MED):
        return None, "訓練時沒載深度"
    d = M.load_gt(name)
    if d is None:
        return None, "缺真深度"
    raw = np.load(f"{M.D}/estimated_depths/{name}.npy").astype(np.float64)
    raw = raw[..., 0] if raw.ndim == 3 else raw
    W, H, fx, fy, cx, cy, R, t = M.cam_of(im)
    if d.shape != (H, W):
        return None, "真深度大小不合"
    if raw.shape != (H, W):
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
    uu, vv = np.meshgrid(np.arange(0, W, ST), np.arange(0, H, ST))
    dg = d[::ST, ::ST]; x = raw[::ST, ::ST]
    vg = np.isfinite(dg) & (dg > 0) & (dg < 1e8)
    zg = np.where(vg, dg, 1.0) * M.K_UNIT
    inv1 = x * sc["scale"] + sc["offset"]
    # 這張影像看得到的可靠 SfM 點
    uv, zc, ii = M.sfm_obs(im, R, t)
    g = (zc > 0) & M.GOODP[ii] & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, zc = uv[g], zc[g]
    _, ix = np.unique(np.round(uv * 4).astype(np.int64), axis=0, return_index=True)
    uv, zc = uv[ix], zc[ix]
    if len(zc) < 50:
        return None, "SfM 觀測 < 50"
    ui = np.clip(np.floor(uv[:, 0]).astype(int), 0, W - 1); vi = np.clip(np.floor(uv[:, 1]).astype(int), 0, H - 1)
    f_sfm = 1.0 / zc                                   # SfM 逆深度
    f_mono = raw[vi, ui] * sc["scale"] + sc["offset"]  # ① 在 SfM 點的值
    tri = Delaunay(uv)
    P = np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5], 1)
    s = tri.find_simplex(P)
    ins = s >= 0
    sidx = np.where(ins, s, 0)
    T = tri.transform[sidx]
    b2 = np.einsum("nij,nj->ni", T[:, :2], P - T[:, 2])
    bc = np.concatenate([b2, 1.0 - b2.sum(1, keepdims=True)], 1)
    vid = tri.simplices[sidx]
    lin_sfm = (f_sfm[vid] * bc).sum(1)
    lin_mono = (f_mono[vid] * bc).sum(1)
    # 凸包外：最近 SfM 點的殘差（⑤）；⑥ 沒有先驗
    if (~ins).any():
        nn = cKDTree(uv).query(P[~ins])[1]
        lin_sfm[~ins] = np.nan
        res_out = (f_sfm - f_mono)[nn]
    inv6 = lin_sfm.reshape(x.shape)
    inv5 = inv1 + np.where(ins, lin_sfm - lin_mono, 0.0).reshape(x.shape)
    if (~ins).any():
        tmp = inv5.ravel(); tmp[~ins] = inv1.ravel()[~ins] + res_out; inv5 = tmp.reshape(x.shape)
    # 每個三角形：逆深度梯度 => 無因次斜率；最長邊
    sv = tri.simplices
    p0, p1, p2 = uv[sv[:, 0]], uv[sv[:, 1]], uv[sv[:, 2]]
    A = np.stack([p1 - p0, p2 - p0], 1)                # (n,2,2)
    rhs = np.stack([f_sfm[sv[:, 1]] - f_sfm[sv[:, 0]], f_sfm[sv[:, 2]] - f_sfm[sv[:, 0]]], 1)
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    okd = np.abs(det) > 1e-9
    gx = np.where(okd, (rhs[:, 0] * A[:, 1, 1] - rhs[:, 1] * A[:, 0, 1]) / np.where(okd, det, 1), 0)
    gy = np.where(okd, (rhs[:, 1] * A[:, 0, 0] - rhs[:, 0] * A[:, 1, 0]) / np.where(okd, det, 1), 0)
    slope_t = np.hypot(gx, gy) / f_sfm[sv].mean(1) * fx
    edge_t = np.maximum.reduce([np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p1, axis=1), np.linalg.norm(p0 - p2, axis=1)])
    sl = np.where(ins, np.searchsorted(SL_E, slope_t[sidx], side="right"), 4).reshape(x.shape)
    sz = np.where(ins, np.searchsorted(SZ_E, edge_t[sidx], side="right"), 3).reshape(x.shape)
    # 分類：高度帶、朝向（真深度的法向）、紋理（影像局部標準差）、離 SfM 遠近
    Xc = np.stack([(uu + 0.5 - cx) / fx * zg, (vv + 0.5 - cy) / fy * zg, zg], -1)
    Xw = (Xc - t) @ R
    band = np.clip(np.searchsorted(M.BANDS, Xw[..., 2], side="right") - 1, 0, 4)
    du = np.zeros_like(Xw); dv_ = np.zeros_like(Xw)
    du[:, 1:-1] = Xw[:, 2:] - Xw[:, :-2]; dv_[1:-1] = Xw[2:] - Xw[:-2]
    nrm = np.cross(du, dv_); nl = np.linalg.norm(nrm, axis=-1)
    nz = np.abs(nrm[..., 2]) / np.maximum(nl, 1e-30)
    ori = np.where(nl <= 0, 1, np.where(nz > 0.8, 0, np.where(nz < 0.3, 2, 1)))
    gray = cv2.imread(f"{M.D}/input/{name}", cv2.IMREAD_GRAYSCALE)[::ST, ::ST].astype(np.float64) / 255.0
    mu = uniform_filter(gray, 9); sd = np.sqrt(np.maximum(uniform_filter(gray * gray, 9) - mu * mu, 0))
    tx = (sd >= 0.02).astype(np.int64)
    import torch
    near = M.SFMV.near(torch.from_numpy(Xw.reshape(-1, 3).astype(np.float32))).numpy().reshape(x.shape).astype(np.int64)
    code = code_of(band, ori, sl, sz, tx, near)
    hist = np.zeros((3, NCODE * NB), dtype=np.int64)
    es = []
    for j, inv in enumerate((inv1, inv5, inv6)):
        ok = vg & np.isfinite(inv) & (inv > 1e-6)
        e = np.clip(zg * np.where(ok, inv, 1.0) - 1.0, -0.19999, 0.19999)
        b = np.clip(np.searchsorted(EB, e[ok], side="right") - 1, 0, NB - 1)
        hist[j] += np.bincount(code[ok] * NB + b, minlength=NCODE * NB)
        es.append(np.where(ok, e, np.nan))
    row = {"name": name, "n_sfm": int(len(zc)), "n_tri": int(len(sv)), "tower_px": int((vg & (band >= 3)).sum()),
           "flat_ground_px": int((vg & (band == 0) & (tx == 0)).sum()), "in_hull": float(ins.mean())}
    if want_maps:
        return (hist, row, {"es": es, "sl": sl, "vg": vg, "tri": tri, "uv": uv}), ""
    return (hist, row), ""


def chunk(names):
    H = np.zeros((3, NCODE * NB), dtype=np.int64); rows = []; skip = {}
    for n in names:
        try:
            r, why = one(n)
        except Exception as ex:
            r, why = None, f"例外 {type(ex).__name__}: {ex}"
        if r is None:
            skip[why] = skip.get(why, 0) + 1; continue
        H += r[0]; rows.append(r[1])
    return H, rows, skip


if __name__ == "__main__":
    t1 = time.time()
    NP = int(os.environ.get("NPROC", "6"))
    names = M.NAMES
    parts = [names[i::2 * NP] for i in range(2 * NP)]
    with Pool(NP) as pool:
        res = pool.map(chunk, parts)
    Hs = sum(r[0] for r in res).reshape(3, NCODE, NB)
    rows = [x for r in res for x in r[1]]
    skip = {}
    for r in res:
        for k, v in r[2].items():
            skip[k] = skip.get(k, 0) + v
    np.savez_compressed(f"{OUT}/hist.npz", H=Hs, EB=EB)
    print(f"\n逐張完成：{len(rows):,} 張、略過 {skip}｜{time.time() - t1:.0f} s", flush=True)
    print(f"每張可靠 SfM 觀測 中位 {np.median([r['n_sfm'] for r in rows]):,.0f}（p10 {np.percentile([r['n_sfm'] for r in rows], 10):,.0f}）、"
          f"三角形 中位 {np.median([r['n_tri'] for r in rows]):,.0f}、像素在凸包內 中位 {100 * np.median([r['in_hull'] for r in rows]):.1f}%")

    c = np.arange(NCODE)
    c_near = c % 2; c_tx = (c // 2) % 2; c_sz = (c // 4) % 4; c_sl = (c // 16) % 5; c_ori = (c // 80) % 3; c_band = c // 240
    mids = 0.5 * (EB[:-1] + EB[1:])
    order = np.argsort(np.abs(mids)); am = np.abs(mids)[order]

    def stats(h):
        n = h.sum()
        if n == 0:
            return None
        cdf = np.cumsum(h[order]) / n
        med = am[min(np.searchsorted(cdf, 0.5), NB - 1)]
        return med, h[np.abs(mids) > 0.01].sum() / n, h[np.abs(mids) > 0.02].sum() / n, h[np.abs(mids) > 0.05].sum() / n, h[mids < -0.05].sum() / n, h[mids > 0.02].sum() / n

    def table(title, groups, aligns=(0, 1, 2)):
        print(f"\n=== {title} ===")
        print(f"{'':30}{'像素佔比':>7}  " + "  ".join(f"{ALIGN[j]}：中位|e|／>1%／>2%／>5%／判前方／比真的近>2%" for j in aligns))
        tot = Hs[0].sum()
        for nm, sel in groups:
            out = []
            for j in aligns:
                st_ = stats(Hs[j][sel].sum(0))
                cov = Hs[j][sel].sum() / max(Hs[0][sel].sum(), 1)
                out.append("—" if st_ is None else f"{100 * st_[0]:5.2f}%／{100 * st_[1]:4.1f}／{100 * st_[2]:4.1f}／{100 * st_[3]:4.1f}／{100 * st_[4]:4.1f}／{100 * st_[5]:4.1f}（有值 {100 * cov:3.0f}%）")
            print(f"{nm:30}{100 * Hs[0][sel].sum() / max(tot, 1):6.1f}%  " + "  ".join(out))

    print("\n欄位：e＝z_真×逆深度−1。中位|e|；|e|>1%／>2%／>5% 的像素比例；判前方＝e<−5%（真表面會被深度支撐閘門當浮塊）；"
          "比真的近>2%＝e>+2%（拿它判「看穿」會誤罰真表面）。地下層約在視線距離的 1.5～3% => 要抓它，|e| 得在 1% 上下。")
    ALLC = np.ones(NCODE, bool)
    table("全部", [("全部", ALLC)])
    table("依三角形斜率（無因次；凸包外＝沒有三角形）", [(SL_N[k], c_sl == k) for k in range(5)])
    table("依三角形最長邊（凸包內）", [(SZ_N[k], (c_sz == k) & (c_sl < 4)) for k in range(4)])
    table("斜率<3 × 最長邊", [(f"斜率<3、{SZ_N[k]}", (c_sz == k) & (c_sl < 2)) for k in range(4)])
    table("依高度（世界 z，場景單位）", [(BAND_N[b], c_band == b) for b in range(5)])
    table("依高度 × 斜率<3", [(f"{BAND_N[b]}、斜率<3", (c_band == b) & (c_sl < 2)) for b in range(5)])
    table("依朝向", [(ORI_N[o], c_ori == o) for o in range(3)])
    table("地面（z<0.05）水平 × 紋理", [(f"地面水平、{TX_N[k]}", (c_band == 0) & (c_ori == 0) & (c_tx == k)) for k in range(2)]
          + [(f"地面水平、{TX_N[k]}、斜率<3", (c_band == 0) & (c_ori == 0) & (c_tx == k) & (c_sl < 2)) for k in range(2)])
    table("立面 × 高度", [(f"立面、{BAND_N[b]}", (c_ori == 2) & (c_band == b)) for b in range(5)])
    table("離 SfM（0.03 單位體素）", [("近 SfM", c_near == 1), ("遠離 SfM", c_near == 0), ("遠離 SfM、斜率<3", (c_near == 0) & (c_sl < 2))])

    # 可靠規則的涵蓋率與準度
    print("\n=== 可靠規則（⑤／⑥ 只在「斜率 < S 的三角形」內當可靠）===")
    for S, kmax in (("1", 1), ("3", 2), ("8", 3)):
        sel = c_sl < kmax
        cov = Hs[0][sel].sum() / Hs[0].sum()
        for j in (1, 2):
            st_ = stats(Hs[j][sel].sum(0))
            print(f"斜率 < {S}：涵蓋 {100 * cov:5.1f}% 像素｜{ALIGN[j]} 中位|e| {100 * st_[0]:.2f}%、>1% {100 * st_[1]:.1f}%、>2% {100 * st_[2]:.1f}%、>5% {100 * st_[3]:.1f}%、判前方 {100 * st_[4]:.1f}%")

    # 示意圖：塔最多、無紋理地面最多、凸包內最少各一張
    pick = [max(rows, key=lambda r: r["tower_px"])["name"], max(rows, key=lambda r: r["flat_ground_px"])["name"], min(rows, key=lambda r: r["in_hull"])["name"]]

    def cmap_e(e, full=0.05):
        tt = np.clip(np.nan_to_num(e, nan=0.0) / full, -1, 1)
        w = 1 - np.abs(tt)
        img = np.stack([w + (tt < 0) * np.abs(tt), w, w + (tt > 0) * np.abs(tt)], -1)   # BGR：負（比真的遠）＝藍、正（比真的近）＝紅
        img[np.isnan(e)] = 0.3
        return (255 * np.clip(img, 0, 1)).astype(np.uint8)

    for n in dict.fromkeys(pick):
        r, _ = one(n, want_maps=True)
        if r is None:
            continue
        mp = r[2]
        rgb = cv2.imread(f"{M.D}/input/{n}")[::ST, ::ST]
        tri_img = rgb.copy()
        for sv_ in mp["tri"].simplices[:: max(1, len(mp["tri"].simplices) // 6000)]:
            cv2.polylines(tri_img, [np.round(mp["uv"][sv_] / ST).astype(np.int32)], True, (0, 255, 255), 1)
        slv = cv2.applyColorMap((63 * np.clip(mp["sl"], 0, 4)).astype(np.uint8), cv2.COLORMAP_JET)
        tiles = [rgb, tri_img, slv, cmap_e(mp["es"][0]), cmap_e(mp["es"][1]), cmap_e(mp["es"][2])]
        labs = ["RGB", "SfM triangles (subsampled)", "tri slope class (blue<1 .. red=outside hull)", "err mono current (+-5% full scale)", "err tri-corrected mono", "err pure SfM triangles"]
        tiles = [cv2.putText(tl.copy(), lb, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5) for tl, lb in zip(tiles, labs)]
        tiles = [cv2.putText(tl, lb, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2) for tl, lb in zip(tiles, labs)]
        big = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
        cv2.imwrite(f"{OUT}/{os.path.splitext(n)[0]}.jpg", cv2.resize(big, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\n示意圖：{OUT}/*.jpg（{list(dict.fromkeys(pick))}）")
    print("DONE tri_depth", flush=True)
