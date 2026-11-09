"""11-09 使用者「細節被抹平感覺是投票的問題？投票只考慮相對位置、沒考慮附近可能不是平面 => 平面度、紋理量、深度拿來加權投票？」
在已訓練好的模型上用現行配方的參數重算 SfM 視線一致性 v4 的投票（同 sfm_vis_diag_1106 的深度格；GPU；cwd＝repo 根；只讀），
看「正確表面」（CityGS 與 A＋dist4 的不透明顆粒）被判成浮塊（空區票）／地下（被擋票）／沒證據（不准往上）的比例，
依三個無因次特徵分組——候選加權項能不能把真的細節和真的浮塊／地下層分開：
  平面度＝本模型塊內不透明顆粒 16 近鄰 PCA 的 λ_min/Σλ（大＝附近不平：樓頂設備、窗框、凹凸）
  顏色起伏＝16 近鄰 DC 顏色的標準差（3D 的紋理量）
  格內起伏＝有證據的相機上，該 8 px 格 SfM 最遠減最近深度 ÷ 容差（> 1＝格內本來就不是一個平面）
規則（現行配方 sfm_vis_cams 0、cell 8、px 4、free_frac 0.5、rep 0.1、behind_min 3、behind_frac 0.9；noup：視野 >= 3 台但有證據 0 台）。
用法：python tools/probe/vote_relief_1109.py --block 1 --cams <跑次>/blocks/block_1/cameras.json --models CityGS=ckpt 標籤=ckpt ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--cams", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--cell", type=int, default=8)
ap.add_argument("--px", type=float, default=4.0)
ap.add_argument("--free_min", type=int, default=2)
ap.add_argument("--free_frac", type=float, default=0.5)
ap.add_argument("--rep", type=float, default=0.1)
ap.add_argument("--behind_min", type=int, default=3)
ap.add_argument("--behind_frac", type=float, default=0.9)
ap.add_argument("--nmax", type=int, default=300000)
ap.add_argument("--wins", default="", help="例 1,3,5：鄰格否決——w×w 鄰域有 SfM 深度對得上這顆的格就不投浮塊／地下票（1＝現行）；給了就只印這張比較表")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = "cuda"
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb0, coordinates=(torch.from_numpy(x).double() @ Rt.T)[:, :2])[B].numpy()

# ---- 每台相機的 SfM 深度格（同 sfm_vis_diag_1106）----
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
good = {pid: q.xyz for pid, q in pts.items() if len(q.image_ids) >= 3 and q.error < 1.5}
ims = read_images_binary(f"{D}/sparse/0/images.bin"); cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names = {os.path.basename(c["img_name"]) for c in json.load(open(a.cams))}
sel = [im for im in ims.values() if os.path.basename(im.name) in names]
g = a.cell
Hc = (max(cams[im.camera_id].height for im in sel) + g - 1) // g; Wc = (max(cams[im.camera_id].width for im in sel) + g - 1) // g
K = len(sel)
dmin = np.full((K, Hc, Wc), np.inf, np.float32); dmax = np.full((K, Hc, Wc), -np.inf, np.float32)
R = np.zeros((K, 3, 3), np.float32); T = np.zeros((K, 3), np.float32); intr = np.zeros((K, 6), np.float32)
for k, im in enumerate(sel):
    cam = cams[im.camera_id]; pr = cam.params
    fx, fy, cx, cy = (pr[0], pr[0], pr[1], pr[2]) if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL") else (pr[0], pr[1], pr[2], pr[3])
    Rk = qvec2rotmat(im.qvec); R[k] = Rk; T[k] = im.tvec; intr[k] = (fx, fy, cx, cy, cam.width, cam.height)
    ids = [int(i) for i in im.point3D_ids if int(i) in good]
    if not ids:
        continue
    P = np.array([good[i] for i in ids]) @ Rk.T + im.tvec
    z = P[:, 2]; m = z > 1e-6
    u = fx * P[m, 0] / z[m] + cx; v = fy * P[m, 1] / z[m] + cy; zz = z[m]
    ok = (u >= 0) & (v >= 0) & (u < cam.width) & (v < cam.height)
    np.minimum.at(dmin[k], ((v[ok] // g).astype(int), (u[ok] // g).astype(int)), zz[ok])
    np.maximum.at(dmax[k], ((v[ok] // g).astype(int), (u[ok] // g).astype(int)), zz[ok])
emp = ~np.isfinite(dmin); dmin[emp] = np.nan; dmax[emp] = np.nan
t = lambda x: torch.from_numpy(x).to(dev)
dmin_t, dmax_t, R_t, T_t, I_t = t(dmin), t(dmax), t(R), t(T), t(intr)
print(f"b{B}｜{K} 台本塊訓練相機｜{g} px 格、有 SfM 的格 {100 * (~emp).mean():.1f}%｜容差 {a.px:g} px 足跡", flush=True)

# SfM 地面（描述用的分組，不進規則）：2 m 格 SfM 中位
S = np.stack([q for q in good.values()])
o0 = np.floor(S[:, :2].min(0) / 0.02).astype(int) - 2; ci = np.floor(S[:, :2] / 0.02).astype(int) - o0; shp = tuple(ci.max(0) + 3)
cid = ci[:, 0] * shp[1] + ci[:, 1]; oo = np.lexsort((S[:, 2], cid)); uu, ff, cc = np.unique(cid[oo], return_index=True, return_counts=True)
GZ = np.full(shp[0] * shp[1], np.nan); GZ[uu] = S[oo, 2][ff + cc // 2]; GZ = GZ.reshape(shp)
GROUND = np.nanpercentile(S[:, 2], 10)


@torch.no_grad()
def votes(X, win=1):
    """win > 1（★ 11-09 鄰格否決）：證據照舊只來自自己那格；但 win×win 鄰域裡只要有一格的 SfM 深度範圍［最近−容差, 最遠＋容差］
    包住這顆的深度（附近有它自己的表面）=> 這台相機的浮塊／地下票作廢。（11-09 先試的「鄰域取最近／最遠」會把別棟的點也算成證據，更糟。）"""
    DMIN, DMAX = dmin_t, dmax_t
    N = X.shape[0]; z0 = lambda: torch.zeros(N, dtype=torch.int32, device=dev)
    fr_n, be_n, cov, inv, rel1 = z0(), z0(), z0(), z0(), z0()
    rel_sum = torch.zeros(N, device=dev)
    for k in range(K):
        fx, fy, cx, cy, W, H = I_t[k]
        Xc = X @ R_t[k].T + T_t[k]; z = Xc[:, 2]; zs = z.clamp_min(1e-6)
        u = fx * Xc[:, 0] / zs + cx; v = fy * Xc[:, 1] / zs + cy
        ok = (z > 1e-6) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
        cu = (u / g).long().clamp(0, Wc - 1); cv = (v / g).long().clamp(0, Hc - 1)
        dm = DMIN[k][cv, cu]; dM = DMAX[k][cv, cu]
        has = ok & torch.isfinite(dm); tol = a.px * z / fx
        sup = torch.zeros_like(has)
        if win > 1:
            r_ = win // 2
            for dy in range(-r_, r_ + 1):
                for dx in range(-r_, r_ + 1):
                    if dx == 0 and dy == 0:
                        continue
                    cu2 = (cu + dx).clamp(0, Wc - 1); cv2 = (cv + dy).clamp(0, Hc - 1)
                    d1 = DMIN[k][cv2, cu2]; d2 = DMAX[k][cv2, cu2]
                    sup |= torch.isfinite(d1) & (z >= d1 - tol) & (z <= d2 + tol)
        fr_n += (has & (z < dm - tol) & ~sup).int(); be_n += (has & (z > dM + tol) & ~sup).int()
        cov += has.int(); inv += ok.int()
        rr = torch.where(has, (dM - dm) / tol.clamp_min(1e-9), torch.zeros_like(z))
        rel_sum += rr; rel1 += (has & (rr > 1)).int()
    return [x.cpu().numpy() for x in (fr_n, be_n, cov, inv, rel_sum, rel1)]


def nrm(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)


def pct(m):
    return f"{100 * m.mean():5.1f}" if m.size else "  —  "


edges = None
for spec in a.models:
    lab, f = spec.split("=", 1)
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    gg = lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
    m = gg("means").double().numpy(); o = torch.sigmoid(gg("opacities")).reshape(-1).numpy()
    n = nrm(gg("rotations").double().numpy()); dc = gg("shs_dc").reshape(len(m), 3).numpy()
    del sd
    keep = inb(m) & (o > 0.5)
    m, n, dc = m[keep], n[keep], dc[keep]
    tree = cKDTree(m)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(m), min(a.nmax, len(m)), replace=False))
    X = m[idx]
    _, nb = tree.query(X, k=16)
    Q = m[nb] - m[nb].mean(1, keepdims=True)
    ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", Q, Q) / 16)
    plan = ev[:, 0] / np.clip(ev.sum(1), 1e-18, None)
    col = (0.5 + 0.28209479 * dc[nb]).clip(0, 1).std(1).mean(1)
    ci2 = np.floor(X[:, :2] / 0.02).astype(int) - o0
    okc = (ci2[:, 0] >= 0) & (ci2[:, 0] < shp[0]) & (ci2[:, 1] >= 0) & (ci2[:, 1] < shp[1])
    gz = np.full(len(X), GROUND); gz[okc] = np.where(np.isfinite(GZ[ci2[okc, 0], ci2[okc, 1]]), GZ[ci2[okc, 0], ci2[okc, 1]], GROUND)
    hz = (X[:, 2] - gz) * 100
    nz = np.abs(n[idx, 2])
    if a.wins:
        cw = [("地面 ±3 m", np.abs(hz) < 3), ("地下 < −3 m", hz < -3), ("樓頂 高>3m", (hz > 3) & (nz > 0.85)),
              ("立面 高>3m", (hz > 3) & (nz < 0.3)), ("斜的 高>3m", (hz > 3) & (nz >= 0.3) & (nz <= 0.85)), ("地面以上合計", hz > 3)]
        print(f"\n== {lab}｜塊內不透明 {len(m):,}、抽 {len(X):,}｜各格：判浮塊／判地下／沒證據 %（顆數）")
        print("  鄰域  " + "".join(f"{nm}（{k.sum():,}）".ljust(30) for nm, k in cw))
        Xt = torch.from_numpy(X).float().to(dev)
        for w in [int(x) for x in a.wins.split(",")]:
            f_, b_, c_, i_, _, _ = votes(Xt, w)
            fr = (f_ >= a.free_min) & (f_ >= a.free_frac * c_)
            be = (c_ >= a.behind_min) & (b_ >= a.behind_frac * c_) & (c_ >= a.rep * i_)
            ne = (i_ >= 3) & (c_ == 0)
            print(f"  {w}x{w}   " + "".join(f"{pct(fr[k])}／{pct(be[k])}／{pct(ne[k])}".ljust(30) for nm, k in cw), flush=True)
        continue
    fr_n, be_n, cov, inv, rel_sum, rel1 = votes(torch.from_numpy(X).float().to(dev))
    rel = np.where(cov > 0, rel_sum / np.maximum(cov, 1), np.nan)
    relfrac = np.where(cov > 0, rel1 / np.maximum(cov, 1), np.nan)
    free = (fr_n >= a.free_min) & (fr_n >= a.free_frac * cov)
    behind = (cov >= a.behind_min) & (be_n >= a.behind_frac * cov) & (cov >= a.rep * inv)
    noev = (inv >= 3) & (cov == 0)
    anyf = free | behind
    cats = [("全部", np.ones(len(X), bool)), ("地面 ±3 m", np.abs(hz) < 3), ("地下 < −3 m", hz < -3),
            ("樓頂／水平面 高 > 3 m", (hz > 3) & (nz > 0.85)), ("立面 |n_z|<0.3 高 > 3 m", (hz > 3) & (nz < 0.3)), ("斜的 高 > 3 m", (hz > 3) & (nz >= 0.3) & (nz <= 0.85))]
    print(f"\n== {lab}｜塊內不透明 {len(m):,}、抽 {len(X):,}｜判浮塊 {pct(free)}%、判地下 {pct(behind)}%、沒證據 {pct(noev)}%")
    print("  類別                         顆數    浮塊%   地下%  沒證據%  平面度中位(‰)  顏色起伏中位  格內起伏中位(容差)")
    for nm, k in cats:
        if k.sum() < 50:
            continue
        print(f"  {nm:24} {k.sum():8,}  {pct(free[k])}  {pct(behind[k])}  {pct(noev[k])}   {1000 * np.median(plan[k]):10.2f}   {np.median(col[k]):10.3f}   {np.nanmedian(rel[k]):10.2f}")
    if edges is None:   # 分位邊界用第一個模型（CityGS）的分布 => 各模型同一把尺
        edges = {"平面度": np.quantile(plan, [0.25, 0.5, 0.75, 0.9]), "顏色起伏": np.quantile(col, [0.25, 0.5, 0.75, 0.9])}
    for fname, val in (("平面度", plan), ("顏色起伏", col)):
        e = edges[fname]; bins = np.digitize(val, e)
        print(f"  依{fname}（{lab if lab == a.models[0].split('=')[0] else '同第一個模型的分位邊界'}；0＝最平／最單色 25%，4＝最不平／最花 10%）：判浮塊／判地下／判掉合計 %")
        for hz_nm, hk in (("全部", np.ones(len(X), bool)), ("高 > 3 m", hz > 3)):
            print("    " + f"{hz_nm:8}" + "  ".join(f"[{b}] {pct(free[(bins == b) & hk])}／{pct(behind[(bins == b) & hk])}／{pct(anyf[(bins == b) & hk])}（{((bins == b) & hk).sum():,}）" for b in range(5)))
    rb = np.digitize(np.nan_to_num(rel, nan=-1), [0, 0.5, 1, 2, 4])
    print("  依格內起伏（容差的倍數；有證據才算）：判浮塊／判地下 %（顆數）")
    print("    " + "  ".join(f"{lbl} {pct(free[rb == b])}／{pct(behind[rb == b])}（{(rb == b).sum():,}）" for b, lbl in ((1, "<0.5"), (2, "0.5~1"), (3, "1~2"), (4, "2~4"), (5, ">4"))))
print("DONE")
