#!/usr/bin/env python
"""★ 11-18 內凹凹槽的破洞研究（使用者 viewer：b2 建築兩翼之間的凹槽一片破洞＋暗色煙；regvis／shn5／窄角度法向先驗都沒改善）。只讀、CPU。
舊量尺（concave_corner_1117）只看轉角線 1.5 m 內 => 量不到凹槽兩面牆離轉角幾公尺的破洞。這支改量「凹槽牆面」：
  GT（MatrixCity 幾何 GT 點雲，16 近鄰 PCA 法向、依頂高判朝外）本塊盒內立面點；內凹轉角＝同 concave_corner_1117。
  凹槽牆面＝離最近內凹轉角點 <= R（--R_m，預設 8 m）的立面點；平牆＝離任何轉角 > 15 m 的立面點（對照）。
  內凹轉角點依 xy（--cell_m 格）連通分群＝一條條凹槽；凹槽牆面點歸給最近的那群。
每個模型（o > 0.5 的面元）與起點：
  覆蓋＝1 m 內有面元／點；貼牆覆蓋＝1 m 內有法向對得上（|cos| > 0.9）的面元。
每個 GT 點的「看得到的訓練鏡頭數」＝本塊清單裡：在畫面內、而且鏡頭在牆的外側（(C − p)·n > 0）——不算遮擋（凹槽另一翼的遮擋另外由 GT 深度驗，見 --occl）。
  --occl：用 GT 深度圖（lab 的 monodepth_sea_1102.load_gt）驗遮擋，只算真的看得到的（慢；每群抽 --occl_n 點）。
輸出：凹槽牆面 vs 平牆的整體數字、最差的 --top 個凹槽（中心、高度範圍、點數、各模型覆蓋、起點覆蓋、看得到的鏡頭數），
  以及每個最差凹槽「最能看清楚它」的訓練影像名（在畫面內、正面、最近）＝給 render_views --train 用。
用法（lab）：python tools/probe/recess_1118.py --block 2 --models 名=<glob> ... [--init <ply>] [--occl]
"""
import argparse, glob, os, sys
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy import ndimage
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--models", nargs="*", default=[]); ap.add_argument("--init", nargs="*", default=[])
ap.add_argument("--gt", default="data/geometry_gt/MC_Aerial/Block_all_ds.ply")
ap.add_argument("--R_m", type=float, default=8.0); ap.add_argument("--cell_m", type=float, default=6.0); ap.add_argument("--top", type=int, default=12)
ap.add_argument("--occl", action="store_true"); ap.add_argument("--occl_n", type=int, default=60)
ap.add_argument("--write_clean_init", default="", help="★ 診斷用（用 GT，不是方法）：第一個 --init 拿掉凹槽空氣裡（兩面牆前方 > 0.5 m、離內凹轉角 <= R）的點，寫到這個 PLY（其他欄位照抄）")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
ZG = float(np.percentile(P[(TRK >= 3) & (ERR < 1.5), 2], 5)); del P
with open(a.gt, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"end_header\n"):
        hdr += f.readline()
    props, nv = [], 0
    for ln in hdr.decode().splitlines():
        t = ln.split()
        if t[:2] == ["element", "vertex"]:
            nv = int(t[2])
        elif t and t[0] == "property":
            props.append((t[2], {"double": "<f8", "float": "<f4", "uchar": "u1"}[t[1]]))
    G = np.fromfile(f, dtype=np.dtype(props), count=nv)
gall = np.stack([G["x"], G["y"], G["z"]], 1); del G
CELL = 0.03; lo = gall[:, :2].min(0) - 0.1
ij = ((gall[:, :2] - lo) / CELL).astype(int); shp = ij.max(0) + 2
top = np.full(shp, -1e9); np.maximum.at(top, (ij[:, 0], ij[:, 1]), gall[:, 2])
def top_at(xy):
    q = ((xy - lo) / CELL).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < shp[0]) & (q[:, 1] >= 0) & (q[:, 1] < shp[1])
    r = np.full(len(xy), -1e9); r[ok] = top[q[ok, 0], q[ok, 1]]; return r
gx = gall[inblk(gall)]
_, nn = cKDTree(gx).query(gx, k=16)
Q = gx[nn] - gx[nn].mean(1, keepdims=True); _, _, ev = np.linalg.svd(Q, full_matrices=False); del Q, nn
gn = ev[:, 2, :]; fac = np.abs(gn[:, 2]) < 0.3
fx_, fn_ = gx[fac], gn[fac].copy()
nh = fn_[:, :2] / np.linalg.norm(fn_[:, :2], axis=1, keepdims=True).clip(1e-9)
air = top_at(fx_[:, :2] + 0.02 * nh) < fx_[:, 2] - 0.01
fn_ *= np.where(air, 1.0, -1.0)[:, None]
ft = cKDTree(fx_)
dd, jj = ft.query(fx_, k=16, distance_upper_bound=0.015)
okn = np.isfinite(dd); jj = np.where(okn, jj, 0)
cosn = np.where(okn, np.abs((fn_[:, None, :] * fn_[jj]).sum(-1)), 1.0)
kmin = cosn.argmin(1); corner = cosn[np.arange(len(fx_)), kmin] < 0.7
jn = jj[np.arange(len(fx_)), kmin]
concave = corner & (((fx_[jn] - fx_) * fn_).sum(1) > 0)
h = (fx_[:, 2] - ZG) * M
ci = np.where(concave)[0]
if len(ci) == 0:
    raise SystemExit("沒有內凹轉角")
# 分群（xy 格連通）
cs = a.cell_m / M; cxy = np.floor(fx_[ci, :2] / cs).astype(int); o_ = cxy.min(0); g_ = np.zeros(cxy.max(0) - o_ + 1, bool); g_[tuple((cxy - o_).T)] = True
lab_, ncl = ndimage.label(g_, structure=np.ones((3, 3))); cl_of = lab_[tuple((cxy - o_).T)] - 1
ctree = cKDTree(fx_[ci])
dcor, kcor = ctree.query(fx_)
allc = np.where(corner)[0]; dany, _ = cKDTree(fx_[allc]).query(fx_)
recess = (dcor <= a.R_m / M) & ~concave
plain = dany > 15 / M
rec_cl = np.where(recess, cl_of[kcor], -1)
print(f"b{B}｜GT 立面 {len(fx_):,}｜內凹轉角點 {len(ci):,}、{ncl} 條凹槽（xy {a.cell_m:g} m 格連通）｜凹槽牆面（離內凹轉角 <= {a.R_m:g} m）{int(recess.sum()):,}｜平牆（離轉角 > 15 m）{int(plain.sum()):,}", flush=True)
# 訓練鏡頭（本塊清單）
imgs = read_images_binary(f"{D}/sparse/0/images.bin"); cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/%03d_%03d.txt" % (B % 4, B // 4)) if l.strip()]
byn = {v.name: v for v in imgs.values()}
CAM = []
for nm in names:
    if nm not in byn:
        continue
    im = byn[nm]; cm = cams[im.camera_id]; pr = cm.params
    fx, fy, cx, cy = (pr[0], pr[1], pr[2], pr[3]) if len(pr) > 3 else (pr[0], pr[0], pr[1], pr[2])
    R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec, float)
    CAM.append((nm, R, t, fx, fy, cx, cy, cm.width, cm.height, -R.T @ t))
def visible_cams(X, N):
    """X、N：點與朝外法向 => 每點看得到的鏡頭數（畫面內＋鏡頭在外側；不算遮擋）、每點最好的鏡頭（離最近、離畫面中心近）。"""
    cnt = np.zeros(len(X), int); best = np.full(len(X), -1); bscore = np.full(len(X), np.inf)
    for k_, (nm, R, t, fx, fy, cx, cy, W, H, C) in enumerate(CAM):
        Xc = X @ R.T + t; z = Xc[:, 2]
        u = fx * Xc[:, 0] / np.maximum(z, 1e-9) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-9) + cy
        ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (((C - X) * N).sum(1) > 0)
        cnt += ok
        sc = z * (1 + np.hypot((u - W / 2) / W, (v - H / 2) / H))
        m_ = ok & (sc < bscore); bscore[m_] = sc[m_]; best[m_] = k_
    return cnt, best
vis_r, best_r = visible_cams(fx_[recess], fn_[recess])
vis_p, _ = visible_cams(fx_[plain][::7], fn_[plain][::7])
print(f"看得到的訓練鏡頭數（畫面內＋正面，不算遮擋）：凹槽牆面 中位 {np.median(vis_r):.0f}（p10 {np.percentile(vis_r, 10):.0f}）｜平牆 中位 {np.median(vis_p):.0f}（p10 {np.percentile(vis_p, 10):.0f}）", flush=True)
def cover(X, nrm=None, r=1.0):
    t_ = cKDTree(X); res = {}
    for nm_, msk in (("凹槽牆面", recess), ("平牆", plain)):
        d_, k_ = t_.query(fx_[msk], distance_upper_bound=r / M)
        c_ = np.isfinite(d_)
        res[nm_] = [100 * c_.mean()]
        if nrm is not None:
            kk = np.where(c_, k_, 0); al = c_ & (np.abs((nrm[kk] * fn_[msk]).sum(1)) > 0.9)
            res[nm_].append(100 * al.mean())
        res[nm_].append(c_)
    return res
covs = {}
for spec in a.init:
    labn, f = spec.split("=", 1)
    from plyfile import PlyData
    v = PlyData.read(f)["vertex"].data; X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64); X = X[inblk(X)]
    r = cover(X); covs["起點:" + labn] = r["凹槽牆面"][-1]
    dc, kc = ctree.query(X); xi = ci[kc]; yi = jn[xi]
    d1 = ((X - fx_[xi]) * fn_[xi]).sum(1) * M; d2 = ((X - fx_[yi]) * fn_[yi]).sum(1) * M
    inair0 = (dc <= a.R_m / M) & (d1 > 0.5) & (d2 > 0.5); near0 = dc <= a.R_m / M
    if a.write_clean_init and spec == a.init[0]:
        from plyfile import PlyData, PlyElement
        pd_ = PlyData.read(f); vv_ = pd_["vertex"].data
        Xall = np.stack([vv_["x"], vv_["y"], vv_["z"]], 1).astype(np.float64)
        ib = inblk(Xall); drop = np.zeros(len(Xall), bool); drop[np.where(ib)[0][inair0]] = True
        os.makedirs(os.path.dirname(os.path.abspath(a.write_clean_init)), exist_ok=True)
        PlyData([PlyElement.describe(vv_[~drop], "vertex")], text=False).write(a.write_clean_init)
        print(f"[診斷起點] {a.write_clean_init}：拿掉凹槽空氣裡的 {int(drop.sum()):,} 點（{len(Xall):,} => {int((~drop).sum()):,}；用 GT，只供診斷）", flush=True)
    print(f"[起點] {labn}：1 m 內有點 凹槽牆面 {r['凹槽牆面'][0]:.1f}%｜平牆 {r['平牆'][0]:.1f}%｜凹槽附近（離內凹轉角 <= {a.R_m:g} m）的點 {int(near0.sum()):,}，"
          f"其中在凹槽空氣裡（兩面牆前方 > 0.5 m）{int(inair0.sum()):,}（{100 * inair0.sum() / max(near0.sum(), 1):.1f}%）", flush=True)
for spec in a.models:
    labn, pat = spec.split("=", 1); f = sorted(x for x in glob.glob(pat) if "xyz" not in x)[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]; g = "gaussian_model.gaussians."
    m = sd[g + "means"].double().numpy(); o = torch.sigmoid(sd[g + "opacities"].float()).reshape(-1).numpy(); q = sd[g + "rotations"].float().numpy().astype(np.float64); del sd
    k = (o > 0.5) & inblk(m); m, q = m[k], q[k]
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    nrm = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    r = cover(m, nrm); covs[labn] = r["凹槽牆面"][-1]
    # 凹槽空氣裡的顆粒（兩面牆前方）：最近內凹轉角點的兩面牆都在前方 > 0.5 m、離轉角 <= R
    dc, kc = ctree.query(m); xi = ci[kc]; yi = jn[xi]
    d1 = ((m - fx_[xi]) * fn_[xi]).sum(1) * M; d2 = ((m - fx_[yi]) * fn_[yi]).sum(1) * M
    inair = (dc <= a.R_m / M) & (d1 > 0.5) & (d2 > 0.5); nearm = dc <= a.R_m / M
    print(f"[模型] {labn}：1 m 內有面元 凹槽牆面 {r['凹槽牆面'][0]:.1f}%（貼牆 {r['凹槽牆面'][1]:.1f}%）｜平牆 {r['平牆'][0]:.1f}%（貼牆 {r['平牆'][1]:.1f}%）｜"
          f"凹槽空氣裡（兩面牆前方 > 0.5 m）的不透明面元 {int(inair.sum()):,}（凹槽附近的 {100 * inair.sum() / max(nearm.sum(), 1):.1f}%）", flush=True)
# 每條凹槽
ri = np.where(recess)[0]
rows = []
for c_ in range(ncl):
    s = rec_cl[ri] == c_
    if s.sum() < 80:
        continue
    X = fx_[ri[s]]
    row = dict(cl=c_, n=int(s.sum()), cen=X.mean(0), hlo=float(np.percentile(h[ri[s]], 5)), hhi=float(np.percentile(h[ri[s]], 95)),
               cams=float(np.median(vis_r[s])), cov={k_: 100 * v_[s].mean() for k_, v_ in covs.items()})
    bi = best_r[s]; bi = bi[bi >= 0]
    row["best"] = CAM[int(np.bincount(bi).argmax())][0] if len(bi) else "-"
    rows.append(row)
key0 = next((k_ for k_ in covs if not k_.startswith("起點:")), None)
rows.sort(key=lambda r: r["cov"].get(key0, 100))
print(f"\n最差的 {a.top} 條凹槽（依「{key0}」凹槽牆面覆蓋排序）：", flush=True)
for r in rows[: a.top]:
    print(f"  凹槽 #{r['cl']}｜中心 ({r['cen'][0]:.3f}, {r['cen'][1]:.3f}, {r['cen'][2]:.3f})｜高 {r['hlo']:.0f}～{r['hhi']:.0f} m｜GT 牆點 {r['n']:,}｜看得到的鏡頭中位 {r['cams']:.0f}｜"
          + "、".join(f"{k_} {v_:.0f}%" for k_, v_ in r["cov"].items()) + f"｜最好的訓練影像 {r['best']}", flush=True)
if a.occl and rows:
    sys.path.insert(0, "tools/probe"); _argv = sys.argv; sys.argv = sys.argv[:1]
    import monodepth_sea_1102 as MS
    sys.argv = _argv
    import cv2
    print("\n--occl：用 GT 深度驗遮擋（每條凹槽抽點）", flush=True)
    rng = np.random.default_rng(0); TRI = []
    for r in rows[: a.top]:
        s = np.where(rec_cl[ri] == r["cl"])[0]; s = rng.choice(s, min(a.occl_n, len(s)), replace=False); X = fx_[ri[s]]; N_ = fn_[ri[s]]
        seen = np.zeros(len(X), int)
        for (nm, R, t, fx, fy, cx, cy, W, H, C) in CAM:
            Xc = X @ R.T + t; z = Xc[:, 2]
            u = fx * Xc[:, 0] / np.maximum(z, 1e-9) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-9) + cy
            ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (((C - X) * N_).sum(1) > 0)
            if not ok.any():
                continue
            gt = MS.load_gt(nm)
            if gt is None:
                continue
            zg = gt * MS.K_UNIT; hh, ww = zg.shape
            ui = np.clip((u * ww / W).astype(int), 0, ww - 1); vi = np.clip((v * hh / H).astype(int), 0, hh - 1)
            vis_ = ok & (np.abs(zg[vi, ui] - z) <= 0.02 * z)
            seen += vis_
            ft_ = f"{D}/estimated_depths_tri/{nm}.npy"; fr_ = f"{D}/estimated_depths_tri_rel/{nm}.npy"
            if vis_.any() and os.path.exists(ft_):
                tri = np.load(ft_).squeeze(); h3, w3 = tri.shape
                u3 = np.clip((u * w3 / W).astype(int), 0, w3 - 1); v3 = np.clip((v * h3 / H).astype(int), 0, h3 - 1)
                ti = tri[v3, u3]; okt = vis_ & (ti > 1e-6)
                if os.path.exists(fr_):
                    rl = np.load(fr_); rel_ok = ((rl[v3, u3] & 7) <= 1) & ((rl[v3, u3] >> 3) <= 3)
                else:
                    rel_ok = np.ones(len(X), bool)
                zt = np.where(okt, 1.0 / np.maximum(ti, 1e-9), np.nan)
                TRI.append(np.c_[(zt[okt] / z[okt] - 1.0), rel_ok[okt].astype(float)])
        print(f"  凹槽 #{r['cl']}：抽 {len(X)} 點，真的看得到（GT 深度對得上）的訓練影像數 中位 {np.median(seen):.0f}、p10 {np.percentile(seen, 10):.0f}、0 張的點 {100 * (seen == 0).mean():.0f}%", flush=True)
        if TRI:
            T_ = np.concatenate(TRI); TRI.clear()
            for nm_, mm in (("全部", np.ones(len(T_), bool)), ("可靠", T_[:, 1] > 0)):
                e = T_[mm, 0]
                if len(e):
                    print(f"     校正深度 vs 真深度（{nm_}，{len(e):,} 個像素觀測）：相對誤差中位 {100 * np.median(e):+.1f}%｜偏淺（在牆前方）> 2% 佔 {100 * (e < -0.02).mean():.0f}%、偏深 > 2% 佔 {100 * (e > 0.02).mean():.0f}%", flush=True)
print("DONE recess", flush=True)
