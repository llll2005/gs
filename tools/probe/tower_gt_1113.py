"""11-13 高樓 GT 量尺（使用者：「CityGS 好像也有一樣問題（塔上半部一條條直紋、塔尖散成煙），如果我們解了就是一個突破」）。
CPU；cwd＝repo 根；只讀。

為什麼要新量尺：33 的高樓工具（破洞、立面對齊、低角度透空）都拿 CityGS 的表面當參考 => CityGS 自己破掉的地方我們量不到。
這支改用 MatrixCity 幾何 GT 點雲（data/geometry_gt/MC_Aerial/Block_all_ds.ply，1,325 萬點、帶法向；區塊 F1 用的同一份），
只拿來量、不進訓練。座標與我們相同（GT 姿態 x1/100，block_f1.py 也不做配準）。

棟的分割照 tower_probe_1105／tower_interior_1108（可靠 SfM 10 m 格頂高 > 100 m 的連通區，棟號＝連通區編號，與 33 的表一致），
每棟足跡＝該棟 10 m 格膨脹 1 格。GT 依法向分立面（|n_z| < 0.3）與平面（|n_z| > 0.8）。
量（每棟 x 高度帶 x 模型；顆粒＝塊內 o > 0.5 的面元中心）：
  完整度   GT 立面點 r 內有不透明顆粒的比例（r＝1 m；另列 2 m）
  朝向     GT 立面點 1.5 m 內最近不透明顆粒的 |n·n_gt| > 0.9 的比例（只算 1.5 m 內有顆粒的點）
  離表面   足跡內該帶不透明顆粒離最近 GT 點 > 2 m 的比例（煙柱／浮在外面）
  拉長     GT 立面 1 m 內不透明顆粒：長短軸比中位數；長短軸比 > 3 且長軸近垂直（|a_z| > 0.9）的比例
  半透明   足跡內該帶 0.05 < o <= 0.5 的顆粒數
另列每棟：GT 頂高（最高點、99.9 分位）、SfM 頂高、各模型不透明顆粒頂高，以及頂部 30 m 的 GT 點數、在畫面內的訓練相機數
（不算遮擋）與該段在畫面上的寬度（像素，中位數）=> 回答「GT 有沒有塔尖、訓練影像看不看得到」。
用法：python tools/probe/tower_gt_1113.py --block 3 --models CityGS=<ckpt glob> 新底3=<ckpt glob> ...
"""
import argparse, glob, sys
import numpy as np, torch
from scipy.ndimage import label, maximum_filter
from scipy.spatial import cKDTree
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--top", type=float, default=1.5, help="逐棟細表只看頂高 > top 的高樓（1.5＝150 m）")
ap.add_argument("--gt", default="data/geometry_gt/MC_Aerial/Block_all_ds.ply")
ap.add_argument("--cam_sample", type=int, default=400, help="頂部相機統計抽的 GT 點數")
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inblk = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()

# ── 棟（同 tower_interior_1108）──
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); RGB = np.stack([q.rgb for q in pts.values()]).astype(float)
ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
good = (TRK >= 3) & (ERR < 1.5) & inblk(P)
P, RGB = P[good], RGB[good]
CELL = 0.1
o0 = np.floor(P[:, :2].min(0) / CELL).astype(int); ci = np.floor(P[:, :2] / CELL).astype(int) - o0
shp = tuple(ci.max(0) + 1); top = np.full(shp, -1e9)
np.maximum.at(top, (ci[:, 0], ci[:, 1]), P[:, 2])
lab_, n = label(top > 1.0)
bid = lab_[ci[:, 0], ci[:, 1]]
tall = []
for k in range(1, n + 1):
    s = bid == k
    if s.sum() < 50:
        continue
    zt = np.percentile(P[s, 2], 99)
    if zt > a.top:
        up = s & (P[:, 2] > 0.5 * zt); c = RGB[up].mean(0)
        tall.append((k, zt, (c[2] - c[0]) / max(c.mean(), 1), P[s, :2].mean(0), P[s, 2].max()))
foot10 = {t[0]: maximum_filter(lab_ == t[0], size=3) for t in tall}
def on(m, k):
    c = np.floor(m[:, :2] / CELL).astype(int) - o0
    ok = (c[:, 0] >= 0) & (c[:, 0] < shp[0]) & (c[:, 1] >= 0) & (c[:, 1] < shp[1])
    r = np.zeros(len(m), bool); r[ok] = foot10[k][c[ok, 0], c[ok, 1]]; return r

# ── GT 點雲（binary ply，直接讀）──
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
gx = np.stack([G["x"], G["y"], G["z"]], 1); gn = np.stack([G["nx"], G["ny"], G["nz"]], 1); del G
k_in = np.zeros(len(gx), bool)
anyfoot = np.zeros(shp, bool)
for t in tall:
    anyfoot |= foot10[t[0]]
c = np.floor(gx[:, :2] / CELL).astype(int) - o0
ok = (c[:, 0] >= 0) & (c[:, 0] < shp[0]) & (c[:, 1] >= 0) & (c[:, 1] < shp[1])
k_in[ok] = anyfoot[c[ok, 0], c[ok, 1]]
gx, gn = gx[k_in], gn[k_in]
k_b = inblk(gx); gx, gn = gx[k_b], gn[k_b]
print(f"[GT] ply 自帶法向長度中位 {np.median(np.linalg.norm(gn, axis=1)):.3f}（0＝沒有法向 => 用 16 近鄰 PCA 估）", flush=True)
gtree = cKDTree(gx)
nn_d, nn_j = gtree.query(gx, k=16)
Q = gx[nn_j] - gx[nn_j].mean(1, keepdims=True)
_, ev, evec = np.linalg.svd(Q, full_matrices=False)
gn = evec[:, 2, :]                                   # 最小奇異值方向＝法向
print(f"b{B}｜GT 點雲在高樓足跡內＋塊內 {len(gx):,} 點｜GT 最近鄰間距中位 {np.median(nn_d[:, 1]) * M:.2f} m｜"
      f"立面（|n_z|<0.3）{100 * (np.abs(gn[:, 2]) < 0.3).mean():.0f}%、平面（|n_z|>0.8）{100 * (np.abs(gn[:, 2]) > 0.8).mean():.0f}%", flush=True)
del Q, nn_j

# ── 模型 ──
def qrot(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
def load(spec):
    lab, pat = spec.split("=", 1); fs = sorted(glob.glob(pat)); assert fs, f"找不到 {pat}"
    f = fs[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    g = "gaussian_model.gaussians."
    m = sd[g + "means"].double().numpy()
    o = torch.sigmoid(sd[g + "opacities"].float()).reshape(-1).numpy()
    s = torch.exp(sd[g + "scales"].float()).numpy()
    r = sd[g + "rotations"].float().numpy(); del sd
    c = np.floor(m[:, :2] / CELL).astype(int) - o0
    ok = (c[:, 0] >= 0) & (c[:, 0] < shp[0]) & (c[:, 1] >= 0) & (c[:, 1] < shp[1])
    kk = np.zeros(len(m), bool); kk[ok] = anyfoot[c[ok, 0], c[ok, 1]]
    kk &= inblk(m)
    m, o, s, r = m[kk], o[kk], s[kk], r[kk]
    R = qrot(r.astype(np.float64))
    s2 = s[:, :2]
    ia = np.argmax(s2, 1)
    ax = R[np.arange(len(R)), :, ia]                 # 長軸方向（世界座標）
    nrm = R[:, :, 2]                                  # 2DGS 面元法向
    ratio = s2.max(1) / np.maximum(s2.min(1), 1e-12)
    print(f"[載入] {lab}：{f}｜足跡內＋塊內 {len(m):,} 顆（不透明 {int((o > 0.5).sum()):,}）｜不透明長軸 中位 {np.median(s2.max(1)[o > 0.5]) * M:.2f} m／"
          f"p99 {np.percentile(s2.max(1)[o > 0.5], 99) * M:.1f} m", flush=True)
    return dict(lab=lab, m=m, o=o, ax=ax, n=nrm, ratio=ratio, lmax=s2.max(1), t1=R[:, :, 0], t2=R[:, :, 1], s=s2)

def cover(md, X, op):
    """GT 點 X 有沒有被不透明面元蓋到：候選＝最近 48 個中心（15 m 內），條件＝離面元平面 < 1 m 且面內 (u/s1)^2+(v/s2)^2 < 9（3 sigma）。
    回傳 covered（bool）、蓋到它的面元裡離平面最近那顆的 |n·n_gt| 所需的索引（-1＝沒蓋到）。"""
    idx = np.where(op)[0]
    if len(idx) == 0:
        return np.zeros(len(X), bool), np.full(len(X), -1)
    tr = cKDTree(md["m"][idx])
    out_c = np.zeros(len(X), bool); out_j = np.full(len(X), -1)
    for s0 in range(0, len(X), 20000):
        Xs = X[s0:s0 + 20000]
        dd, jj = tr.query(Xs, k=48, distance_upper_bound=0.15)
        valid = np.isfinite(dd); jj = np.where(valid, jj, 0); J = idx[jj]
        d = Xs[:, None, :] - md["m"][J]
        u = (d * md["t1"][J]).sum(-1) / md["s"][J, 0]; v = (d * md["t2"][J]).sum(-1) / md["s"][J, 1]
        h = np.abs((d * md["n"][J]).sum(-1))
        ok = valid & (u * u + v * v < 9) & (h < 0.01)
        hh = np.where(ok, h, np.inf); best = hh.argmin(1)
        out_c[s0:s0 + 20000] = ok.any(1)
        out_j[s0:s0 + 20000] = np.where(ok.any(1), J[np.arange(len(Xs)), best], -1)
    return out_c, out_j
models = [load(sp) for sp in a.models]

# ── 相機（原解析度 COLMAP；不算遮擋）──
ims = read_images_binary(f"{D}/sparse/0/images.bin"); cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
bx, by = [int(v) for v in pc.id[B]]
blist = set(open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{bx:03d}_{by:03d}.txt").read().split())
CR, CT, CK, CB = [], [], [], []
for im in ims.values():
    cm = cams[im.camera_id]; fx = cm.params[0]; fy = cm.params[1] if len(cm.params) > 3 else fx
    cx, cy = (cm.params[2], cm.params[3]) if len(cm.params) > 3 else (cm.params[1], cm.params[2])
    CR.append(qvec2rotmat(im.qvec)); CT.append(im.tvec); CK.append((fx, fy, cx, cy, cm.width, cm.height)); CB.append(im.name in blist)
CR, CT, CK, CB = np.array(CR), np.array(CT), np.array(CK), np.array(CB)
print(f"[相機] 全部 {len(CR):,} 台、b{B} 清單 {int(CB.sum()):,} 台", flush=True)
def cam_stats(X, width):
    """X：GT 點（抽樣）；width：該段寬度（場景單位）。回傳 每點畫面內相機數中位（全部／本塊清單）、寬度像素中位、至少看到一點的相機數（全部／本塊）。"""
    Xc = np.einsum("cij,pj->cpi", CR, X) + CT[:, None, :]
    z = Xc[..., 2]; u = CK[:, None, 0] * Xc[..., 0] / np.maximum(z, 1e-9) + CK[:, None, 2]
    v = CK[:, None, 1] * Xc[..., 1] / np.maximum(z, 1e-9) + CK[:, None, 3]
    vis = (z > 0.01) & (u >= 0) & (u < CK[:, None, 4]) & (v >= 0) & (v < CK[:, None, 5])
    pw = CK[:, None, 0] * width / np.maximum(z, 1e-9)
    return (np.median(vis.sum(0)), np.median(vis[CB].sum(0)), (np.median(pw[vis]) if vis.any() else 0.0),
            int(vis.any(1).sum()), int(vis[CB].any(1).sum()))

BANDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5]
bl = lambda i: f"{BANDS[i] * M:.0f}~{BANDS[i + 1] * M:.0f}m"
glass = max(tall, key=lambda t: t[2])[0]
SUM = {}
for k, zt, blue, ctr, zmax_sfm in sorted([t for t in tall], key=lambda t: -t[1]):
    gsel = on(gx, k)
    X, N = gx[gsel], gn[gsel]
    if len(X) < 100:
        print(f"\n棟 {k}：GT 點太少（{len(X)}）略過"); continue
    fac = np.abs(N[:, 2]) < 0.3
    gtop = X[:, 2].max(); gtop99 = np.percentile(X[:, 2], 99.9)
    print(f"\n棟 {k}（SfM 頂高 {zt * M:.0f} m（最高點 {zmax_sfm * M:.0f}）、中心 {ctr[0]:.2f},{ctr[1]:.2f}{'，藍色玻璃塔' if k == glass else ''}）"
          f"｜GT {len(X):,} 點（立面 {int(fac.sum()):,}）｜GT 頂高 最高 {gtop * M:.0f} m／99.9% {gtop99 * M:.0f} m")
    # 頂部 30 m：GT 點數、相機、像素寬
    zt0 = gtop - 0.3
    tp = X[:, 2] >= zt0
    if tp.sum() >= 5:
        Xt = X[tp]; ext = Xt[:, :2].max(0) - Xt[:, :2].min(0); wid = float(np.sqrt(max(ext[0] * ext[1], 1e-6)))
        rs = np.random.default_rng(0).choice(len(Xt), min(a.cam_sample, len(Xt)), replace=False)
        nc, ncb, pxw, ncam, ncamb = cam_stats(Xt[rs], wid)
        print(f"  頂部 30 m（{zt0 * M:.0f}~{gtop * M:.0f} m）：GT {int(tp.sum()):,} 點、水平範圍約 {ext[0] * M:.0f} x {ext[1] * M:.0f} m｜"
              f"每點在畫面內的相機中位 全部 {nc:.0f}／本塊清單 {ncb:.0f} 台（至少一點在畫面內：{ncam}／{ncamb} 台；不算遮擋）｜該段寬約 {pxw:.0f} px（中位，原解析度）")
    tops = []
    for md in models:
        s = on(md["m"], k) & (md["o"] > 0.5)
        tops.append(f"{md['lab']} {md['m'][s, 2].max() * M:.0f}／99.9% {np.percentile(md['m'][s, 2], 99.9) * M:.0f}" if s.sum() > 10 else f"{md['lab']} —")
    print("  不透明顆粒頂高（m）：" + "｜".join(tops))
    hdr = f"  {'':14}" + "".join(f"{bl(i):>14}" for i in range(len(BANDS) - 1) if BANDS[i] < gtop)
    rows = {key: [] for key in ["GT 立面點", "面元覆蓋", "中心1m內", "朝向對齊", "離表面>2m", "長短軸比", "直的拉長", "半透明數"]}
    per = []
    for md in models:
        mo = on(md["m"], k)
        op = mo & (md["o"] > 0.5)
        tr = cKDTree(md["m"][op]) if op.sum() else None
        mm, mn, mr, ma = md["m"][op], md["n"][op], md["ratio"][op], md["ax"][op]
        dg, _ = gtree.query(mm, k=1) if len(mm) else (np.zeros(0), None)
        cov, cj = cover(md, X, (md["o"] > 0.5))   # 候選用全部不透明面元（足跡內），不只本棟
        r = {key: [] for key in rows}
        for i in range(len(BANDS) - 1):
            if BANDS[i] >= gtop:
                continue
            gb = fac & (X[:, 2] >= BANDS[i]) & (X[:, 2] < BANDS[i + 1])
            mb = (mm[:, 2] >= BANDS[i]) & (mm[:, 2] < BANDS[i + 1])
            ng = int(gb.sum())
            r["GT 立面點"].append(f"{ng:,}")
            if ng < 30 or tr is None:
                for key in list(rows)[1:]:
                    r[key].append("—")
                continue
            d1, _ = tr.query(X[gb], k=1)
            r["面元覆蓋"].append(f"{100 * cov[gb].mean():.0f}%")
            r["中心1m內"].append(f"{100 * (d1 < 0.01).mean():.0f}%")
            near = cov[gb]
            al = np.abs((md["n"][cj[gb][near]] * N[gb][near]).sum(1)) > 0.9
            r["朝向對齊"].append(f"{100 * al.mean():.0f}%" if near.sum() >= 10 else "—")
            r["離表面>2m"].append(f"{100 * (dg[mb] > 0.02).mean():.0f}%" if mb.sum() >= 10 else "—")
            # 立面 1 m 內的顆粒：拉長
            dfac, jf = cKDTree(X[fac]).query(mm[mb], k=1) if mb.sum() else (np.zeros(0), None)
            sel = np.where(mb)[0][dfac < 0.01] if mb.sum() else np.zeros(0, int)
            if len(sel) >= 10:
                r["長短軸比"].append(f"{np.median(mr[sel]):.1f}")
                r["直的拉長"].append(f"{100 * ((mr[sel] > 3) & (np.abs(ma[sel, 2]) > 0.9)).mean():.0f}%")
            else:
                r["長短軸比"].append("—"); r["直的拉長"].append("—")
            semi = mo & (md["o"] > 0.05) & (md["o"] <= 0.5) & (md["m"][:, 2] >= BANDS[i]) & (md["m"][:, 2] < BANDS[i + 1])
            r["半透明數"].append(f"{int(semi.sum()):,}")
            SUM.setdefault((md["lab"], i), [0, 0, 0, 0]); sm = SUM[(md["lab"], i)]
            sm[0] += int(cov[gb].sum()); sm[1] += ng; sm[2] += int(al.sum()); sm[3] += int(near.sum())
        per.append((md["lab"], r))
    print(hdr)
    print(f"  {'GT 立面點':14}" + "".join(f"{x:>14}" for x in per[0][1]["GT 立面點"]))
    for key in list(rows)[1:]:
        for lab, r in per:
            print(f"  {(key + ' ' + lab)[:14]:14}" + "".join(f"{x:>14}" for x in r[key]))

print("\n=== 總結：所有高樓立面（GT 點加權）依高度帶 ===")
print(f"{'':14}" + "".join(f"{bl(i):>14}" for i in range(len(BANDS) - 1)))
def pct(lab, i, a_, b_):
    v = SUM.get((lab, i))
    return f"{100 * v[a_] / max(v[b_], 1):.0f}%" if v else "—"
for name, a_, b_ in [("面元覆蓋", 0, 1), ("朝向對齊", 2, 3)]:
    for md in models:
        print(f"{(name + ' ' + md['lab'])[:14]:14}" + "".join(f"{pct(md['lab'], i, a_, b_):>14}" for i in range(len(BANDS) - 1)))
print("DONE tower_gt")
