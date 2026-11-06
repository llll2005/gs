"""11-06 使用者「v5 B 是不是可以先作一個測試：直接依照判準切掉符合條件的，然後 test 看結構有沒有崩」（GPU 投票＋CPU；cwd＝repo 根）。
不訓練：在已訓練的模型上算每顆的 SfM 視線證據（同 sfm_vis_votes：全部本塊訓練相機、8 px 格、4 px 足跡容差）與離可靠 SfM 的距離，
距離換成「像素足跡」（距離 ÷（看得到它的相機的平均深度 / fx））=> 沒有公尺常數（使用者 11-05：換場景不能靠具體數字）。
切法（刪掉＝True）：
  沒證據        視野內 >= 3 台、格內有 SfM 的 0 台
  沒證據且遠    沒證據 且 離 SfM > --dpx 像素足跡
  遠            離 SfM > --dpx 像素足跡（「只限增生 5 m」的無因次版；b3 500 m 外 1 px ≈ 0.4 m => 12 px ≈ 5 m）
  空區過半      v4 的浮塊規則：空區票 >= 2 且 >= 0.5 x 有證據
每種切法存一個 ckpt（只刪顆粒、丟掉 optimizer 狀態）到 <out>/<切法>/blocks/block_<B>/checkpoints/，給 held-out／viewer／各量測工具用。
另印每種切法在各類（海面上方、高樓 50～200 m、路燈帶、車輛帶、地面、其他）各刪掉多少（o > 0.3 的比例）。
用法：python tools/probe/cut_test_1106.py --block 3 --ckpt <ckpt> --cams <run 的 cameras.json> --out outputs/local/cuttest_b3_1106
"""
import argparse, json, os, sys
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
sys.path.insert(0, ".")
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cams", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cell", type=int, default=8)
ap.add_argument("--px", type=float, default=4.0)
ap.add_argument("--dpx", type=float, default=12.0)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = "cuda"
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
bb1 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.from_numpy(x).double() @ Rt.T)[:, :2])[B].numpy()

# ---- 深度格（同 _sfm_vis_build）----
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
good = {pid: q.xyz for pid, q in pts.items() if len(q.image_ids) >= 3 and q.error < 1.5}
P = np.stack([q.xyz for q in pts.values()]); gd = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
ims = read_images_binary(f"{D}/sparse/0/images.bin"); cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names = {os.path.basename(c["img_name"]) for c in json.load(open(a.cams))}
sel = [im for im in ims.values() if os.path.basename(im.name) in names]
g = a.cell; K = len(sel)
Hc = (max(cams[im.camera_id].height for im in sel) + g - 1) // g; Wc = (max(cams[im.camera_id].width for im in sel) + g - 1) // g
dmin = np.full((K, Hc, Wc), np.inf, np.float32); dmax = np.full((K, Hc, Wc), -np.inf, np.float32)
R = np.zeros((K, 3, 3), np.float32); T = np.zeros((K, 3), np.float32); intr = np.zeros((K, 6), np.float32)
for k, im in enumerate(sel):
    cam = cams[im.camera_id]; pr = cam.params
    fx, fy, cx, cy = (pr[0], pr[0], pr[1], pr[2]) if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL") else (pr[0], pr[1], pr[2], pr[3])
    Rk = qvec2rotmat(im.qvec); R[k] = Rk; T[k] = im.tvec; intr[k] = (fx, fy, cx, cy, cam.width, cam.height)
    ids = [int(i) for i in im.point3D_ids if int(i) in good]
    if not ids:
        continue
    Pc = np.array([good[i] for i in ids]) @ Rk.T + im.tvec
    z = Pc[:, 2]; m = z > 1e-6
    u = fx * Pc[m, 0] / z[m] + cx; v = fy * Pc[m, 1] / z[m] + cy; zz = z[m]
    ok = (u >= 0) & (v >= 0) & (u < cam.width) & (v < cam.height)
    np.minimum.at(dmin[k], ((v[ok] // g).astype(int), (u[ok] // g).astype(int)), zz[ok])
    np.maximum.at(dmax[k], ((v[ok] // g).astype(int), (u[ok] // g).astype(int)), zz[ok])
emp = ~np.isfinite(dmin); dmin[emp] = np.nan; dmax[emp] = np.nan
t = lambda x: torch.from_numpy(x).to(dev)
dmin_t, dmax_t, R_t, T_t, I_t = t(dmin), t(dmax), t(R), t(T), t(intr)
FX = float(np.median(intr[:, 0]))


@torch.no_grad()
def votes(X):
    N = X.shape[0]; z0 = lambda: torch.zeros(N, dtype=torch.int32, device=dev)
    fr_, be_, cv_, iv_ = z0(), z0(), z0(), z0(); zs = torch.zeros(N, dtype=torch.float64, device=dev)
    for k in range(K):
        fx, fy, cx, cy, W, H = I_t[k]
        Xc = X @ R_t[k].T + T_t[k]; z = Xc[:, 2]; zc = z.clamp_min(1e-6)
        u = fx * Xc[:, 0] / zc + cx; v = fy * Xc[:, 1] / zc + cy
        ok = (z > 1e-6) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
        cu = (u / g).long().clamp(0, Wc - 1); cv = (v / g).long().clamp(0, Hc - 1)
        dm = dmin_t[k][cv, cu]; dM = dmax_t[k][cv, cu]
        has = ok & torch.isfinite(dm); tol = a.px * z / fx
        fr_ += (has & (z < dm - tol)).int(); be_ += (has & (z > dM + tol)).int(); cv_ += has.int(); iv_ += ok.int()
        zs += torch.where(ok, z.double(), torch.zeros_like(zs))
    return [x.cpu().numpy() for x in (fr_, be_, cv_, iv_, zs)]


# ---- 類別（只用 SfM；同 smallobj_probe_1106／sea_opacity_1106）----
S = P[gd & inb(P, bb1)]
zg = P[gd, 2]; e = np.arange(-0.10, 0.0501, 0.001); h, _ = np.histogram(zg, bins=e); hs = np.convolve(h, np.ones(5) / 5, mode="same")
pk = [i for i in range(2, len(hs) - 2) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.02 * hs.max()]
ZS = e[pk[0]] + 0.0005


def grid_med(cell):
    o0 = np.floor(S[:, :2].min(0) / cell).astype(int) - 4
    ci = np.floor(S[:, :2] / cell).astype(int) - o0; shp = tuple(ci.max(0) + 5)
    cid = ci[:, 0] * shp[1] + ci[:, 1]; o = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[o], return_index=True, return_counts=True)
    med = np.full(shp[0] * shp[1], np.nan); med[u] = S[o, 2][f + c // 2]; top = np.full(shp[0] * shp[1], -1e9); np.maximum.at(top, cid, S[:, 2])
    return o0, shp, med.reshape(shp), top.reshape(shp)


def look(x, cell, o0, shp, grid, fill=np.nan):
    ci = np.floor(x[:, :2] / cell).astype(int) - o0
    ok = (ci[:, 0] >= 0) & (ci[:, 0] < shp[0]) & (ci[:, 1] >= 0) & (ci[:, 1] < shp[1])
    r = np.full(len(x), fill, dtype=float); r[ok] = grid[ci[ok, 0], ci[ok, 1]]; return r


o2, s2, m2, _ = grid_med(0.02); o4, s4, m4, _ = grid_med(0.04); o10, s10, _, top10 = grid_med(0.1)
ground2 = np.isfinite(m2) & (m2 < 0.03); open2 = ground2 & ~(maximum_filter((np.isfinite(m2) & ~ground2).astype(np.uint8), size=5) > 0)
nosfm4 = ~(maximum_filter(np.isfinite(m4).astype(np.uint8), size=3) > 0)
tall10 = maximum_filter((top10 > 1.0).astype(np.uint8), size=3) > 0


def categories(x):
    tl = look(x, 0.1, o10, s10, tall10.astype(float), 0) > 0
    sm = look(x, 0.04, o4, s4, m4); ns = look(x, 0.04, o4, s4, nosfm4.astype(float), 0) > 0
    sea = ~tl & ((np.isfinite(sm) & (np.abs(sm - ZS) < 0.01)) | ns) & (x[:, 2] > ZS + 0.05)
    gm = look(x, 0.02, o2, s2, m2); og = look(x, 0.02, o2, s2, open2.astype(float), 0) > 0
    hg = x[:, 2] - gm
    lamp = og & (hg > 0.03) & (hg < 0.15) & ~sea; car = og & (hg > 0.005) & (hg < 0.025) & ~sea
    grd = np.isfinite(gm) & (gm < 0.03) & (np.abs(hg) <= 0.005) & ~sea
    tmid = tl & (x[:, 2] > 0.5) & (x[:, 2] < 2.0)
    other = ~(sea | lamp | car | grd | tmid)
    return {"海面上方": sea, "高樓50~200m": tmid, "路燈帶3~15m": lamp, "車輛帶0.5~2.5m": car, "地面±0.5m": grd, "其他": other}


# ---- 模型 ----
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd = ck["state_dict"]; X = sd["gaussian_model.gaussians.means"].double().numpy(); N = X.shape[0]
O = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy()
fr, be, cv, iv, zs = votes(torch.from_numpy(X).float().to(dev))
zmean = zs / np.maximum(iv, 1)
dist = cKDTree(P[gd]).query(X, k=1)[0]
dpx = np.where(iv > 0, dist / np.maximum(zmean / FX, 1e-9), np.inf)
cuts = {
    "沒證據": (iv >= 3) & (cv == 0),
    "沒證據且遠": (iv >= 3) & (cv == 0) & (dpx > a.dpx),
    "遠": dpx > a.dpx,
    "空區過半": (fr >= 2) & (fr >= 0.5 * cv),
}
IN = inb(X, bb0); cat = categories(X); vis = O > 0.3
print(f"b{B}｜{K} 台本塊訓練相機｜原模型 {N:,} 顆（塊內 {int(IN.sum()):,}）｜距離門檻 {a.dpx:g} 像素足跡（平均 fx {FX:.0f}；"
      f"塊內顆粒的像素足跡中位 {np.median((zmean / FX)[IN & (iv > 0)]) * 100:.2f} m => {a.dpx:g} px ≈ {a.dpx * np.median((zmean / FX)[IN & (iv > 0)]) * 100:.1f} m）", flush=True)
print("各類 o > 0.3 的塊內顆數：" + "、".join(f"{c} {int((m & IN & vis).sum()):,}" for c, m in cat.items()), flush=True)
print(f"{'切法':12}{'刪掉（塊內）':>14}" + "".join(f"{c:>14}" for c in cat), flush=True)
for nm, cut in cuts.items():
    row = "".join(f"{100 * (cut & m & IN & vis).sum() / max((m & IN & vis).sum(), 1):>13.1f}%" for m in cat.values())
    print(f"{nm:12}{int((cut & IN).sum()):>14,}{row}", flush=True)
    keep = torch.from_numpy(~cut)
    sd2 = {k: (v[keep] if hasattr(v, "shape") and v.dim() and v.shape[0] == N else v) for k, v in sd.items()}
    ck2 = {k: v for k, v in ck.items() if k not in ("state_dict", "optimizer_states")}
    ck2["state_dict"] = sd2; ck2["optimizer_states"] = []
    od = f"{a.out}/{nm}/blocks/block_{B}/checkpoints"; os.makedirs(od, exist_ok=True)
    torch.save(ck2, f"{od}/{os.path.basename(a.ckpt)}")
for c in ("cameras.json",):
    src = os.path.join(os.path.dirname(a.cams), c)
    for nm in cuts:
        dst = f"{a.out}/{nm}/blocks/block_{B}/{c}"
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
print("（各類欄＝該類 o > 0.3 的塊內顆粒被這種切法刪掉的比例）")
print("DONE cut_test")
