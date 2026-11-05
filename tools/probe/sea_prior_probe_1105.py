"""11-05 使用者「海平面先驗的測試可以安排」：不靠真深度，能不能估出海平面高度與海面範圍？（cwd＝gs；CPU；只讀）
**真深度只拿來評分，不進任何估計。** 海平面真值 Z_SEA＝−0.0341（sealevel_probe_1102.py）。1 場景單位＝100 m。

問三件事：
  R1 現況：真海面格裡有沒有 SfM 點、在什麼高度；單目（現行每張 SfM 對齊）把海面放在哪個高度（離真值幾公尺）。
  R2 不靠真值的海面遮罩：格裡沒有可靠 SfM（3x3 膨脹）＋單目算出來是水平＋單目高度低 => 對真海面的 precision／recall。
  R3 不靠真值的海平面估計，對真值的誤差（公尺）：
       E1 遮罩格裡單目高度的中位數
       E2 海岸環（遮罩外 1～2 格的陸地格）SfM 點高度的低百分位（碼頭牆往下延伸到水線）
       E3 遮罩格裡的 SfM 點（若有）高度中位數
     遮罩用「不靠真值的」與「真值的」兩種各算一次 => 分開「遮罩錯」與「估計方法錯」。
格子 0.04（約 4 m）。STRIDE（張）／ST（像素）環境變數可調。輸出：logs/sea_prior_probe_1105.log（stdout）。
"""
import os, sys, json, glob, re, time, gc
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import cv2
from multiprocessing import Pool
from scipy.ndimage import maximum_filter, binary_dilation
sys.path.insert(0, ".")
import torch
torch.set_num_threads(1); cv2.setNumThreads(1)
from internal.utils.colmap import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

D = "data/matrix_city/aerial/train/block_all"
GT = "../mc_depth_f32/aerial/train"
Z_SEA, SEA_TOL = -0.0341, 0.004
K_UNIT = 1.000009e-4                      # monodepth_sea_1102 [檢查 b] 量到的場景單位／cm
ST = int(os.environ.get("ST", "4"))
CELL = 0.04
M = 100.0                                 # 場景單位 -> 公尺

t0 = time.time()
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
imgs = read_images_binary(f"{D}/sparse/0/images.bin")
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
PX = np.stack([p.xyz for p in pts.values()]).astype(np.float64)
PERR = np.fromiter((p.error for p in pts.values()), dtype=np.float64, count=len(pts))
PTRK = np.fromiter((len(p.image_ids) for p in pts.values()), dtype=np.int64, count=len(pts))
del pts; gc.collect()
GOODP = (PTRK >= 3) & (PERR < 1.5)
FMAP = json.load(open("data/matrix_city/aerial/global_frame_map_correct.json"))
SCALES = json.load(open(f"{D}/estimated_depth_scales.json"))
MED = float(np.median([v["scale"] for v in SCALES.values()]))
LB, UB = 0.2, 5.0
GTIDX = {}
for p in glob.glob(f"{GT}/block_*/**/*.exr", recursive=True):
    mb = re.search(r"block_(\d+)", os.path.relpath(p, GT)); ms = re.search(r"(\d+)$", os.path.splitext(os.path.basename(p))[0])
    if mb and ms:
        GTIDX[(int(mb.group(1)), int(ms.group(1)))] = p
part = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
PC = PartitionCoordinates(id=part["partition_coordinates"]["id"], xy=part["partition_coordinates"]["xy"])
BB = PC.get_bounding_boxes(part["scene_config"]["partition_size"], enlarge=0.)
RT = part["extra_data"]["rotation_transform"][:3, :3].double().numpy()

# 0.04 格網（範圍取全部 SfM 點再放寬 2 格）
O0 = np.floor(PX[:, :2].min(0) / CELL).astype(np.int64) - 2
SHP = tuple((np.floor(PX[:, :2].max(0) / CELL).astype(np.int64) - O0 + 3).tolist())
NC = SHP[0] * SHP[1]
pci = np.floor(PX[:, :2] / CELL).astype(np.int64) - O0
PCID = pci[:, 0] * SHP[1] + pci[:, 1]
SFM_GOOD = np.bincount(PCID[GOODP], minlength=NC).reshape(SHP)
SFM_ALL = np.bincount(PCID, minlength=NC).reshape(SHP)
BYNAME = {im.name: im for im in imgs.values()}
NAMES = sorted(BYNAME)[::int(os.environ.get("STRIDE", "4"))]
if os.environ.get("LIMIT"):
    NAMES = NAMES[:int(os.environ["LIMIT"])]
print(f"SfM {len(PX):,}（可靠 {int(GOODP.sum()):,}）｜影像取 {len(NAMES):,} 張（STRIDE {os.environ.get('STRIDE', '4')}、像素步長 {ST}）｜格網 {SHP}｜{time.time() - t0:.0f} s", flush=True)


def to_key(name):
    return f"{int(os.path.splitext(name)[0]) + 1:06d}.png"


def agg(cid, *vals):
    u, inv = np.unique(cid, return_inverse=True)
    return (u,) + tuple(np.bincount(inv, weights=v, minlength=len(u)) for v in vals)


def one(name):
    im = BYNAME[name]; sc = SCALES.get(name)
    if sc is None or not (LB * MED <= sc["scale"] <= UB * MED):
        return None
    m = FMAP[to_key(name)]; p = GTIDX.get((m["block"], m["frame_index"]))
    if p is None:
        return None
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = (d[..., 0] if d.ndim == 3 else d).astype(np.float64)
    raw = np.load(f"{D}/estimated_depths/{name}.npy").astype(np.float64)
    raw = raw[..., 0] if raw.ndim == 3 else raw
    c = cams[im.camera_id]; W, H = c.width, c.height; fx, fy, cx, cy = c.params[:4]
    R = qvec2rotmat(im.qvec); t = np.asarray(im.tvec, dtype=np.float64)
    if d.shape != (H, W):
        return None
    if raw.shape != (H, W):
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
    uu, vv = np.meshgrid(np.arange(0, W, ST), np.arange(0, H, ST))
    ray = np.stack([(uu + 0.5 - cx) / fx, (vv + 0.5 - cy) / fy, np.ones(uu.shape)], -1)
    dg = d[::ST, ::ST]; vg = np.isfinite(dg) & (dg > 0) & (dg < 1e8)
    zg = np.where(vg, dg, 1.0) * K_UNIT
    Xw = (ray * zg[..., None] - t) @ R                       # 真
    inv = raw[::ST, ::ST] * sc["scale"] + sc["offset"]       # ① 現行對齊（訓練用的那份）
    vm = inv > 1e-6
    zm = np.where(vm, 1.0 / np.where(vm, inv, 1.0), 1.0)
    Xm = (ray * zm[..., None] - t) @ R                       # 單目
    def nz_of(X):
        du = np.zeros_like(X); dv = np.zeros_like(X)
        du[:, 1:-1] = X[:, 2:] - X[:, :-2]; dv[1:-1] = X[2:] - X[:-2]
        n = np.cross(du, dv); nn = np.linalg.norm(n, axis=-1)
        return np.abs(n[..., 2]) / np.maximum(nn, 1e-30)
    sea = vg & (np.abs(Xw[..., 2] - Z_SEA) < SEA_TOL) & (nz_of(Xw) > 0.8)
    hor = nz_of(Xm) > 0.9
    def cid_of(X, ok):
        ci = np.floor(X[..., :2] / CELL).astype(np.int64) - O0
        ok = ok & (ci[..., 0] >= 0) & (ci[..., 0] < SHP[0]) & (ci[..., 1] >= 0) & (ci[..., 1] < SHP[1])
        return (ci[..., 0] * SHP[1] + ci[..., 1])[ok], ok
    cg, okg = cid_of(Xw, vg)
    mz = np.where(vm, Xm[..., 2], 0.0)[okg]
    A = agg(cg, np.ones(len(cg)), sea[okg].astype(np.float64), mz * vm[okg], mz * mz * vm[okg], vm[okg].astype(np.float64))
    cm, okm = cid_of(Xm, vm)
    B = agg(cm, np.ones(len(cm)), Xm[..., 2][okm], hor[okm].astype(np.float64))
    return A, B


if __name__ == "__main__":
    nproc = int(os.environ.get("NPROC", "16"))
    t1 = time.time(); res = []
    with Pool(nproc) as pool:
        for k, r in enumerate(pool.imap_unordered(one, NAMES, chunksize=4)):
            if r is not None:
                res.append(r)
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(NAMES)}｜{time.time() - t1:.0f} s", flush=True)
    print(f"逐張完成 {len(res)}／{len(NAMES)} 張｜{time.time() - t1:.0f} s", flush=True)
    A_all = np.zeros(NC); A_sea = np.zeros(NC); A_mz = np.zeros(NC); A_mn = np.zeros(NC)
    B_n = np.zeros(NC); B_z = np.zeros(NC); B_h = np.zeros(NC)
    for (A, B) in res:
        u = A[0]; A_all[u] += A[1]; A_sea[u] += A[2]; A_mz[u] += A[3]; A_mn[u] += A[5]
        u = B[0]; B_n[u] += B[1]; B_z[u] += B[2]; B_h[u] += B[3]
    del res; gc.collect()
    A_all, A_sea, A_mz, A_mn = (x.reshape(SHP) for x in (A_all, A_sea, A_mz, A_mn))
    B_n, B_z, B_h = (x.reshape(SHP) for x in (B_n, B_z, B_h))

    # 每格屬於哪一塊（格中心、海平面高度）
    BLK = np.full(SHP, -1, dtype=np.int16)
    ix = np.flatnonzero(((A_all > 0) | (B_n > 0)).ravel())
    gx, gy = ix // SHP[1], ix % SHP[1]
    cen = np.stack([(gx + O0[0] + 0.5) * CELL, (gy + O0[1] + 0.5) * CELL, np.full(len(ix), Z_SEA)], -1)
    inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=BB, coordinates=torch.from_numpy((cen @ RT.T)[:, :2])).numpy()
    BLK.ravel()[ix] = np.where(inb.any(0), inb.argmax(0), -1); del cen, inb, gx, gy, ix

    TRUE = (A_all >= 20) & (A_sea >= 0.95 * A_all)
    LAND = (A_all >= 20) & (A_sea < 0.05 * A_all)
    OBS = A_all >= 20
    mono_true = np.where(A_mn > 0, A_mz / np.maximum(A_mn, 1), np.nan)      # 真位置格上的單目高度
    mono_cell = np.where(B_n > 0, B_z / np.maximum(B_n, 1), np.nan)         # 單目位置格的單目高度
    hfrac = np.where(B_n > 0, B_h / np.maximum(B_n, 1), 0)
    nosfm = ~(maximum_filter(SFM_GOOD > 0, size=3))

    def pct(x, q=(5, 50, 95)):
        x = x[np.isfinite(x)]
        return "／".join(f"{v:+.1f}" for v in np.percentile(x, q)) if len(x) else "—"

    print("\n=== R1 現況：真海面格（>=95% 像素是海、>=20 像素；0.04 格≈4 m） ===")
    print(f"真海面格 {int(TRUE.sum()):,}（觀測到的格 {int(OBS.sum()):,}）")
    sel = TRUE.ravel()[PCID]
    print(f"格內 SfM 點：可靠 {int((sel & GOODP).sum()):,}、全部 {int(sel.sum()):,}；"
          f"高度相對海平面 p5／p50／p95（m）可靠 {pct((PX[sel & GOODP, 2] - Z_SEA) * M)}｜全部 {pct((PX[sel, 2] - Z_SEA) * M)}")
    print(f"單目（現行對齊）在真海面格的高度相對海平面 p5／p50／p95（m）：{pct((mono_true[TRUE] - Z_SEA) * M)}")
    for b in range(16):
        tb = TRUE & (BLK == b)
        if tb.sum() >= 200:
            sb = tb.ravel()[PCID]
            print(f"  b{b:<2} 海面格 {int(tb.sum()):>7,}｜SfM 可靠 {int((sb & GOODP).sum()):>6,} 全部 {int(sb.sum()):>6,}｜"
                  f"單目高度 p5／p50／p95 {pct((mono_true[tb] - Z_SEA) * M)} m")

    print("\n=== R2 不靠真值的海面遮罩（在單目位置的格；只評有觀測的格） ===")
    masks = {
        "C0 無可靠 SfM（3x3）": (B_n >= 20) & nosfm,
        "C1 ＋單目水平 >=80%": (B_n >= 20) & nosfm & (hfrac >= 0.8),
        "C2 ＋單目高度 < 0.03": (B_n >= 20) & nosfm & (hfrac >= 0.8) & (mono_cell < 0.03),
        "C3 ＋單目高度 < 0.0": (B_n >= 20) & nosfm & (hfrac >= 0.8) & (mono_cell < 0.0),
    }
    for k, cmask in masks.items():
        ev = cmask & OBS
        prec = (ev & TRUE).sum() / max(ev.sum(), 1); rec = (cmask & TRUE).sum() / max(TRUE.sum(), 1)
        land = (ev & LAND).sum() / max(ev.sum(), 1)
        print(f"  {k:24} 格 {int(cmask.sum()):>9,}｜precision {prec:6.1%}（誤含陸地 {land:5.1%}）｜recall {rec:6.1%}")

    print("\n=== R3 海平面估計（m，相對真值；+ 表示估得比真海面高） ===")
    def ring(mask, r0, r1):   # 不靠真值的「陸地」＝有可靠 SfM 的格（⚠ scipy iterations=0 是「膨脹到收斂」，所以 r0=0 要直接用 mask）
        inner = mask if r0 == 0 else binary_dilation(mask, iterations=r0)
        return binary_dilation(mask, iterations=r1) & ~inner & (SFM_GOOD > 0)
    def est(mask, tag, blk=None, mono=None):
        mk = mask if blk is None else mask & (BLK == blk)
        if mk.sum() < 100:
            return None
        mono = mono_cell if mono is None else mono
        e1 = np.nanmedian(mono[mk]) if np.isfinite(mono[mk]).any() else np.nan
        rg = ring(mk, 0, 2); sr = rg.ravel()[PCID] & GOODP; sa = rg.ravel()[PCID]
        e2 = np.percentile(PX[sr, 2], [1, 5, 10]) if sr.sum() >= 50 else [np.nan] * 3
        e2a = np.percentile(PX[sa, 2], [1, 5]) if sa.sum() >= 50 else [np.nan] * 2
        si = mk.ravel()[PCID] & GOODP
        e3 = np.median(PX[si, 2]) if si.sum() >= 20 else np.nan
        f = lambda v: f"{(v - Z_SEA) * M:+6.1f}" if np.isfinite(v) else "   —"
        return (f"{tag:22} 格 {int(mk.sum()):>8,}｜E1 單目中位 {f(e1)}｜E2 海岸環 SfM 可靠 p1／p5／p10 {f(e2[0])}／{f(e2[1])}／{f(e2[2])}"
                f"（全部 p1／p5 {f(e2a[0])}／{f(e2a[1])}；{int(sr.sum()):,} 點）｜E3 格內可靠 SfM 中位 {f(e3)}（{int(si.sum()):,} 點）")
    # E4：可靠 SfM 點高度直方圖（0.1 m 一格）的最低強峰＝海平面候選（不靠真值）
    zg = PX[GOODP, 2]; edges = np.arange(-0.10, 0.0501, 0.001); h, _ = np.histogram(zg, bins=edges)
    hs = np.convolve(h, np.ones(5) / 5, mode="same")
    peaks = [i for i in range(2, len(hs) - 2) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.02 * hs.max()]
    print("可靠 SfM 高度直方圖的峰（m 相對真海面、點數／0.1 m）：" + "，".join(f"{(edges[i] + 0.0005 - Z_SEA) * M:+.1f}:{h[i]:,}" for i in peaks))
    Z4 = edges[peaks[0]] + 0.0005 if peaks else np.nan
    print(f"E4 最低強峰＝{Z4:.4f}（相對真值 {(Z4 - Z_SEA) * M:+.1f} m）")
    # C4：格內可靠 SfM 高度中位在 E4 ±1 m、單目水平 => 不靠真值的海面遮罩
    gcid = PCID[GOODP]; o = np.lexsort((zg, gcid)); sc_, sz = gcid[o], zg[o]
    u, first, cnt = np.unique(sc_, return_index=True, return_counts=True)
    CMED = np.full(NC, np.nan); CMED[u] = sz[first + cnt // 2]; CMED = CMED.reshape(SHP)
    for tol in (0.01, 0.005):
        c4 = np.isfinite(CMED) & (np.abs(CMED - Z4) < tol) & (hfrac >= 0.5)
        ev = c4 & OBS
        print(f"  C4 可靠 SfM 中位在 E4±{tol * M:.1f} m＋單目水平 格 {int(c4.sum()):>9,}｜precision {(ev & TRUE).sum() / max(ev.sum(), 1):6.1%}"
              f"（誤含陸地 {(ev & LAND).sum() / max(ev.sum(), 1):5.1%}）｜recall {(c4 & TRUE).sum() / max(TRUE.sum(), 1):6.1%}")
        masks[f"C4 E4±{tol * M:.1f} m"] = c4
    cands = {"真值遮罩": TRUE, "不靠真值 C4±1m": masks["C4 E4±1.0 m"], "不靠真值 C2": masks["C2 ＋單目高度 < 0.03"]}
    for k, cmask in cands.items():
        line = est(cmask, f"全場景 {k}", mono=mono_true if k == "真值遮罩" else None)
        print(line if line else f"全場景 {k}：格太少")
    for b in range(16):
        if (TRUE & (BLK == b)).sum() < 200:
            continue
        for k, cmask in cands.items():
            line = est(cmask, f"b{b} {k}", b, mono=mono_true if k == "真值遮罩" else None)
            if line:
                print("  " + line)
    # R4：真海面格裡，初始化與訓練後模型的顆粒在什麼高度（只看該塊的海面格）
    print("\n=== R4 真海面格裡的顆粒高度（相對海平面；o>0.5＝不透明；init PLY 沒有 opacity 只算點數） ===")
    print(f"{'模型':34}{'塊':>3}{'海面格':>8}{'海面±1m':>10}{'其中不透明':>10}{'+1~10m':>9}{'>+10m':>9}{'<-1m':>8}{'有不透明的格(±1m)':>18}")
    for spec in os.environ.get("SEA_MODELS", "").split():
        lab, rest = spec.split("=", 1); b, pat = rest.split(":", 1); b = int(b)
        fs = sorted(glob.glob(pat))
        if not fs:
            print(f"{lab:34} 找不到 {pat}"); continue
        f = fs[-1]
        if f.endswith(".ply"):
            from plyfile import PlyData
            v = PlyData.read(f)["vertex"]; m = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64); o = None
        else:
            sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
            m = sd["gaussian_model.gaussians.means"].double().numpy()
            o = torch.sigmoid(sd["gaussian_model.gaussians.opacities"].float()).reshape(-1).numpy(); del sd
        ci = np.floor(m[:, :2] / CELL).astype(np.int64) - O0
        ok = (ci[:, 0] >= 0) & (ci[:, 0] < SHP[0]) & (ci[:, 1] >= 0) & (ci[:, 1] < SHP[1])
        cid = np.where(ok, ci[:, 0] * SHP[1] + ci[:, 1], 0)
        sea_b = (TRUE & (BLK == b)).ravel()
        on = ok & sea_b[cid]; dz = (m[:, 2] - Z_SEA) * M
        lay = on & (np.abs(dz) <= 1); op = lay if o is None else lay & (o > 0.5)
        cov = len(np.unique(cid[op])) / max(sea_b.sum(), 1)
        print(f"{lab:34}{b:>3}{int(sea_b.sum()):>8,}{int(lay.sum()):>10,}{('—' if o is None else f'{int(op.sum()):,}'):>10}"
              f"{int((on & (dz > 1) & (dz <= 10)).sum()):>9,}{int((on & (dz > 10)).sum()):>9,}{int((on & (dz < -1)).sum()):>8,}{cov:>17.1%}", flush=True)
        del m, o
    print(f"\nDONE sea_prior_probe｜{time.time() - t0:.0f} s")
