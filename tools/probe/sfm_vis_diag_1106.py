"""11-06 SfM 視線一致性 v3 讓 b3 高樓破洞變多（100～200 m：棟 6 10 -> 49%）：在已訓練好的模型上重算投票，看被判掉的是哪種票、落在哪裡，
以及候選修法（空區票改比例制／只信深度一致的格）能不能把「真的浮塊／地下層」和「高樓立面」分開。（GPU；cwd＝repo 根；只讀）
類別（只用 SfM 定義，同 sfm_dist_premise_1105）：地下、海面上方、高樓 50~200 m（10 m 格 SfM 頂高 > 100 m、膨脹 1 格）、高樓其他高度、其他。
規則：v3＝空區票 >= 2 或（有證據 >= 3 台且被擋 >= 90%）；比例＝空區票 >= 2 且空區比例 >= --frac；一致格＝只數格內 SfM 深度差 <= 容差的格；
代表性 r＝比例，且地下票另要求「有證據台數 >= r × 視野內台數」（證據只來自被擋的視角時不下判斷）。
用法：python sfm_vis_diag_1106.py --block 3 --cams outputs/.../cameras.json --models 標籤=ckpt ...
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from scipy.ndimage import maximum_filter
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--cams", required=True, help="跑次輸出的 cameras.json（本塊訓練相機）")
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--cell", type=int, default=8)
ap.add_argument("--px", type=float, default=4.0)
ap.add_argument("--frac", type=float, default=0.5)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = "cuda"
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
bb1 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(x).double() @ Rt.T)[:, :2])[B].numpy()

# ---- 深度格（同 _sfm_vis_build，但另存格內深度差）----
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
print(f"b{B}｜{K} 台本塊訓練相機｜{g} px 格、有 SfM 的格 {100 * (~emp).mean():.1f}%｜容差 {a.px:g} px 足跡")


@torch.no_grad()
def votes(X):
    """回傳 (free, behind, cover) 兩組：全部格／只數一致格（格內 SfM 深度差 <= 容差）；外加每顆的「最大空區深度差」中位用不到就不算。"""
    N = X.shape[0]; z0 = lambda: torch.zeros(N, dtype=torch.int32, device=dev)
    fa, ba, ca, fh, bh, ch, inf = z0(), z0(), z0(), z0(), z0(), z0(), z0()
    for k in range(K):
        fx, fy, cx, cy, W, H = I_t[k]
        Xc = X @ R_t[k].T + T_t[k]; z = Xc[:, 2]; zs = z.clamp_min(1e-6)
        u = fx * Xc[:, 0] / zs + cx; v = fy * Xc[:, 1] / zs + cy
        ok = (z > 1e-6) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
        cu = (u / g).long().clamp(0, Wc - 1); cv = (v / g).long().clamp(0, Hc - 1)
        dm = dmin_t[k][cv, cu]; dM = dmax_t[k][cv, cu]
        has = ok & torch.isfinite(dm); tol = a.px * z / fx
        fr = has & (z < dm - tol); be = has & (z > dM + tol)
        hom = has & ((dM - dm) <= tol)
        fa += fr.int(); ba += be.int(); ca += has.int()
        fh += (fr & hom).int(); bh += (be & hom).int(); ch += hom.int(); inf += ok.int()
    return [x.cpu().numpy() for x in (fa, ba, ca, fh, bh, ch, inf)]


# ---- 類別（同 sfm_dist_premise_1105；外加高樓格）----
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
gd = (TRK >= 3) & (ERR < 1.5)
zg = P[gd, 2]; e = np.arange(-0.10, 0.0501, 0.001); h, _ = np.histogram(zg, bins=e); hs = np.convolve(h, np.ones(5) / 5, mode="same")
pk = [i for i in range(2, len(hs) - 2) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.02 * hs.max()]
ZS = e[pk[0]] + 0.0005
S = P[gd & inb(P, bb1)]


def cellmed(cell):
    o0 = np.floor(S[:, :2].min(0) / cell).astype(int) - 2
    ci = np.floor(S[:, :2] / cell).astype(int) - o0; shp = tuple(ci.max(0) + 3)
    cid = ci[:, 0] * shp[1] + ci[:, 1]; o = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[o], return_index=True, return_counts=True)
    med = np.full(shp[0] * shp[1], np.nan); med[u] = S[o, 2][f + c // 2]
    return o0, shp, med.reshape(shp)


def look(x, cell, o0, shp, grid):
    ci = np.floor(x[:, :2] / cell).astype(int) - o0
    ok = (ci[:, 0] >= 0) & (ci[:, 0] < shp[0]) & (ci[:, 1] >= 0) & (ci[:, 1] < shp[1])
    r = np.full(len(x), np.nan); r[ok] = grid[ci[ok, 0], ci[ok, 1]]; return r


G2 = cellmed(0.02); G4 = cellmed(0.04)
o10 = np.floor(S[:, :2].min(0) / 0.1).astype(int) - 2; ci10 = np.floor(S[:, :2] / 0.1).astype(int) - o10
shp10 = tuple(ci10.max(0) + 3); top = np.full(shp10, -1e9); np.maximum.at(top, (ci10[:, 0], ci10[:, 1]), S[:, 2])
tall = maximum_filter((top > 1.0).astype(np.uint8), size=3).astype(float)

RULES = ["v3", "比例", "一致格", "代表性0.1", "代表性0.2", "代表性0.3"]
print(f"欄：顆數｜各規則判掉的比例（浮塊票／地下票），比例門檻 {a.frac:g}")
print(f"{'模型':12}{'類別':14}{'顆數':>9}" + "".join(f"{r:>16}" for r in RULES) + f"{'視野內':>8}{'有證據':>8}{'證據/視野':>10}{'空區比例(v3浮塊)':>18}")
for spec in a.models:
    lab, pat = spec.split("=", 1); f = sorted(glob.glob(pat))[-1]
    sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
    m = sd["gaussian_model.gaussians.means"].double().numpy(); o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy(); del sd
    k = inb(m, bb0) & (o > 0.5); m = m[k]
    fa, ba, ca, fh, bh, ch, inf = votes(torch.from_numpy(m).float().to(dev))
    gm = look(m, 0.02, *G2); sm = look(m, 0.04, *G4); tl = look(m, 0.1, o10, shp10, tall) > 0
    under = np.isfinite(gm) & (gm < 0.03) & (m[:, 2] < gm - 0.01)
    sea = np.isfinite(sm) & (np.abs(sm - ZS) < 0.01) & (m[:, 2] > ZS + 0.05) & ~under
    tmid = tl & (m[:, 2] > 0.5) & (m[:, 2] < 2.0) & ~under & ~sea
    toth = tl & ~tmid & ~under & ~sea
    cats = {"地下": under, "海面上方": sea, "高樓 50~200m": tmid, "高樓其他高度": toth, "其他": ~under & ~sea & ~tl}
    und = lambda b, c: (c >= 3) & (b >= 0.9 * c)
    rules = {"v3": (fa >= 2, und(ba, ca)), "比例": ((fa >= 2) & (fa >= a.frac * ca), und(ba, ca)),
             "一致格": (fh >= 2, und(bh, ch))}
    for r in (0.1, 0.2, 0.3):
        rules[f"代表性{r:g}"] = ((fa >= 2) & (fa >= a.frac * ca), und(ba, ca) & (ca >= r * inf))
    for c, s in cats.items():
        if s.sum() == 0:
            continue
        row = "".join(f"{f'{100 * np.mean(fl[s]):.0f}/{100 * np.mean(un[s] & ~fl[s]):.0f}%':>16}" for fl, un in rules.values())
        fl = rules["v3"][0] & s
        rat = np.median(fa[fl] / np.maximum(ca[fl], 1)) if fl.any() else float("nan")
        print(f"{lab:12}{c:14}{int(s.sum()):>9,}{row}{int(np.median(inf[s])):>8}{int(np.median(ca[s])):>8}"
              f"{np.median(ca[s] / np.maximum(inf[s], 1)):>10.2f}{rat:>18.2f}", flush=True)
print("DONE sfm_vis_diag")
