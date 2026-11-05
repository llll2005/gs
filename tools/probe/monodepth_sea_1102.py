"""11-02 使用者「照你說的排」①：海面假說驗證（monodepth_diag_1101.py 的複本，只改海面定義＋加顆粒統計）。
11-01 診斷的「水面」用 |z|<0.02 => 漏掉真正的海面（真深度量到海平面 z＝−0.0341，比地面低約 3.4 m；4482.png 海面 e 中位 −0.39、100% 判前方）。
這版：海面像素＝真深度世界 z 在海平面 ±0.004、水平；並累計 0.02 格的海面／全部像素數 => 海面格（>=95% 是海），
再看各模型在海面格裡的顆粒：海面層（±0.01）不透明／半透明數、opacity 質量、面積覆蓋指標、躺平比例、上方浮塊、下方顆粒。
假說：深度支撐閘門把海面本身判成「在前方」推透明（＝使用者看到的霧狀海面）。

（以下是 11-01 原說明）11-01 使用者「作單目深度誤差診斷（在 lab 作，MatrixCity 的資料也直接在那下載）」。
現行單目深度（Depth Anything V2＋每張一組 SfM 縮放；訓練時 batch[2] 的那份）對 MatrixCity 真實深度（float32 EXR、cm、Z 深度）錯在哪。
**只用來診斷：真實深度不進任何訓練。** cwd＝gs；CPU（runner 的 [cpu] 任務）。

e = z_真 / z_單目 − 1 ＝「剛好在真表面上的一顆」在深度支撐閘門裡的殘差：
  e < −5%  ⇒ 會被判「一直在前方」（被當浮塊壓掉；塔頂煙柱就是這個）
  |e| <= 5% ⇒ 落在閘門的貼面帶內
  e > +10% ⇒ 單側不能看穿（margin 0.1）會懲罰真表面、把東西往相機推
四種對齊（同一張單目深度圖，只換縮放）：
  ① 現行：每張一組，用 SfM 點擬合（訓練用的就是這個）
  ② 每張一組，用真深度擬合（全域對齊的上限）           ①−② ＝ SfM 擬合本身的誤差
  ③ 每 60 px 一組，用真深度擬合（局部對齊的上限）       ②−③ ＝「每張只有一組」造成的誤差
  ④ 每 120 px 一組，用該 patch 內的 SfM 點擬合（實際可做的局部校正；點不夠就退回 ①）
  ③ 之後還剩的 ＝ 網路本身的錯（邊緣糊、材質、細結構），對齊救不了
輸出：logs/monodepth_diag_1101.log（表）、outputs/lab/monodepth_diag_1101/（逐張 CSV、示意圖、直方圖 npz）
"""
import os, sys, json, glob, re, hashlib, time, gc
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"                  # 平行靠多行程；fork 前不讓 OpenMP 開執行緒池（避免子行程死鎖）
import numpy as np
import cv2
from multiprocessing import Pool
from scipy.ndimage import maximum_filter
sys.path.insert(0, ".")
import torch
torch.set_num_threads(1); cv2.setNumThreads(1)
from internal.utils.colmap import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.metrics.geom_gate import SfMVoxels, quat_normals

D = "data/matrix_city/aerial/train/block_all"
GT = "../mc_depth_f32/aerial/train"
OUT = "outputs/lab/monodepth_sea_1102"
Z_SEA, SEA_TOL = -0.0341, 0.004           # 海平面（sealevel_probe_1102.py：4482.png 海面 z 5～95 百分位 −0.0341～−0.0340）
ST, PL, PS, NMIN = 2, 30, 120, 8          # 取樣步長；③ patch（取樣後 px＝原圖 60 px）；④ patch（原圖 px）；④ 每 patch 最少 SfM 點
EB = np.linspace(-1.0, 1.0, 401)          # e 的直方圖格（0.005）；夾到 ±0.9999
NB = len(EB) - 1
ALIGN = ["①現行（每張一組，SfM 擬合）", "②每張一組（真深度擬合）", "③每 60 px 一組（真深度擬合）", "④每 120 px 一組（SfM 擬合）"]
BANDS = [-1e9, 0.05, 1.0, 2.0, 3.0, 1e9]
BAND_N = ["z<0.05 地面／水面", "z 0.05~1", "z 1~2", "z 2~3", "z>=3 超高塔頂"]
ORI_N = ["水平（屋頂／地面）", "斜面", "立面"]
NCODE = 17 * 5 * 3 * 8                    # (塊+1, 高度帶, 朝向, 邊緣, 近 SfM, 水面)
os.makedirs(OUT, exist_ok=True)


def code_of(blk, band, ori, edge, near, water):
    return ((((blk + 1) * 5 + band) * 3 + ori) * 8) + edge * 4 + near * 2 + water


def robust_fit(x, y):
    """get_depth_scales.py 同一套（中位數／平均絕對偏差）：y ≈ s·x + o。"""
    tx, ty = np.median(x), np.median(y)
    sx, sy = np.mean(np.abs(x - tx)), np.mean(np.abs(y - ty))
    if sx < 1e-12:
        return None
    s = sy / sx
    return s, ty - s * tx


# ───────────────────────── 主行程：載入共用資料（fork 給 worker）
t0 = time.time()
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
imgs = read_images_binary(f"{D}/sparse/0/images.bin")
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
pid = np.fromiter(pts.keys(), dtype=np.int64, count=len(pts))
PX = np.stack([p.xyz for p in pts.values()]).astype(np.float64)
PERR = np.fromiter((p.error for p in pts.values()), dtype=np.float64, count=len(pts))
PTRK = np.fromiter((len(p.image_ids) for p in pts.values()), dtype=np.int64, count=len(pts))
del pts; gc.collect()
id2i = np.full(pid.max() + 1, -1, dtype=np.int64); id2i[pid] = np.arange(len(pid))
GOODP = (PTRK >= 3) & (PERR < 1.5)
print(f"SfM 點 {len(PX):,}（track>=3 且誤差<1.5：{int(GOODP.sum()):,}）｜影像 {len(imgs):,}｜{time.time() - t0:.0f} s", flush=True)

FMAP = json.load(open("data/matrix_city/aerial/global_frame_map_correct.json"))
SCALES = json.load(open(f"{D}/estimated_depth_scales.json"))
MED = float(np.median([v["scale"] for v in SCALES.values()]))
LB, UB = 0.2, 5.0                         # dataparser 的 depth_scale_lower/upper_bound（超界那張訓練時不載深度）

GTIDX = {}
for p in glob.glob(f"{GT}/block_*/**/*.exr", recursive=True):
    mb = re.search(r"block_(\d+)", os.path.relpath(p, GT)); ms = re.search(r"(\d+)$", os.path.splitext(os.path.basename(p))[0])
    if mb and ms:
        GTIDX[(int(mb.group(1)), int(ms.group(1)))] = p
print(f"真深度 EXR {len(GTIDX):,} 張（例：{next(iter(GTIDX.items())) if GTIDX else '無'}）", flush=True)
if not GTIDX:
    sys.exit("⛔ 找不到真深度 EXR")

part = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
PC = PartitionCoordinates(id=part["partition_coordinates"]["id"], xy=part["partition_coordinates"]["xy"])
BB = PC.get_bounding_boxes(part["scene_config"]["partition_size"], enlarge=0.)
RT = part["extra_data"]["rotation_transform"][:3, :3].double().numpy()
CONTRACT, AABB = bool(part["scene_config"]["contract"]), part["scene_config"]["aabb"]

# 水面（推定）：float2 同一套「2 cm 格裡沒有可靠 SfM 點」，再加 3x3 鄰格膨脹
CELL = 0.02
G = PX[GOODP]
O0 = np.floor(G[:, :2].min(0) / CELL).astype(np.int64)
gi = np.floor(G[:, :2] / CELL).astype(np.int64) - O0
HAS = np.zeros(tuple(gi.max(0) + 1), dtype=bool); HAS[gi[:, 0], gi[:, 1]] = True
HAS = maximum_filter(HAS, size=3)

NAMES = sorted(im.name for im in imgs.values())[::int(os.environ.get("STRIDE", "2"))]
BYNAME = {im.name: im for im in imgs.values()}


def to_key(name):
    return f"{int(os.path.splitext(name)[0]) + 1:06d}.png"


if os.environ.get("ONLY_BLOCK"):          # 冒煙測試：只用某個 MatrixCity block 的前 LIMIT 張，輸出到 _smoke
    _ob = int(os.environ["ONLY_BLOCK"])
    NAMES = [n for n in NAMES if FMAP[to_key(n)]["block"] == _ob][: int(os.environ.get("LIMIT", "1000000"))]
    OUT = OUT + "_smoke"; os.makedirs(OUT, exist_ok=True)
    print(f"⚠ 冒煙測試：MatrixCity block_{_ob} 的 {len(NAMES)} 張", flush=True)


# ───────────────────────── 前置檢查（都不過就停）
# (a) 影像 ↔ MatrixCity 原始影格：md5 逐位元（第三份 metadata，不依賴姿態）
rng = np.random.default_rng(0)
chk = [NAMES[i] for i in rng.choice(len(NAMES), min(30, len(NAMES)), replace=False)]
bad = []
for n in chk:
    m = FMAP[to_key(n)]
    a = hashlib.md5(open(f"{D}/input/{n}", "rb").read()).hexdigest()
    src = f"data/matrix_city/aerial/train/block_{m['block']}/input/{m['frame_index']:04d}.png"
    b = hashlib.md5(open(src, "rb").read()).hexdigest() if os.path.exists(src) else "缺"
    if a != b:
        bad.append((n, src))
print(f"[檢查 a] 影像 md5 對 MatrixCity 原始影格：{len(chk) - len(bad)}/{len(chk)} 相同" + (f" ⛔ 不同：{bad[:3]}" if bad else ""), flush=True)
if bad:
    sys.exit("⛔ 影像與 MatrixCity 影格對不上 => 不做（真深度會配錯張）")


def load_gt(name):
    m = FMAP[to_key(name)]
    p = GTIDX.get((m["block"], m["frame_index"]))
    if p is None:
        return None
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d[..., 0] if d.ndim == 3 else d
    return d.astype(np.float64)


def cam_of(im):
    c = cams[im.camera_id]
    fx, fy, cx, cy = c.params[:4]
    return c.width, c.height, fx, fy, cx, cy, qvec2rotmat(im.qvec), np.asarray(im.tvec, dtype=np.float64)


def sfm_obs(im, R, t):
    ok = im.point3D_ids >= 0
    ii = id2i[im.point3D_ids[ok]]
    keep = ii >= 0
    ii = ii[keep]; uv = im.xys[ok][keep]
    zc = (PX[ii] @ R.T + t)[:, 2]
    return uv, zc, ii


# (b) 單位：SfM 點的相機 z ÷ 真深度（cm）——姿態是 GT => 應該是一個常數；分散大 ＝ 單位或配對錯
rat = []
for n in NAMES[::140]:
    im = BYNAME[n]; d = load_gt(n)
    if d is None:
        continue
    W, H, fx, fy, cx, cy, R, t = cam_of(im)
    uv, zc, ii = sfm_obs(im, R, t)
    u = np.clip(np.floor(uv[:, 0]).astype(int), 0, d.shape[1] - 1); v = np.clip(np.floor(uv[:, 1]).astype(int), 0, d.shape[0] - 1)
    dv = d[v, u]; g = np.isfinite(dv) & (dv > 0) & (zc > 0) & GOODP[ii]
    rat.append(zc[g] / dv[g])
rat = np.concatenate(rat)
K_UNIT = float(np.median(rat)); mad = float(np.median(np.abs(rat / K_UNIT - 1)))
print(f"[檢查 b] 場景單位 / cm = {K_UNIT:.6e}（MatrixCity 的 /10000 是 {1e-4:.0e}）；相對 MAD {100 * mad:.2f}%（{len(rat):,} 個 SfM 觀測）", flush=True)
if mad > 0.02:
    sys.exit("⛔ SfM 深度與真深度不成比例 => 單位或配對不對，不做")


# ───────────────────────── 逐張
def one(name, want_maps=False):
    im = BYNAME[name]
    sc = SCALES.get(name)
    if sc is None or not (LB * MED <= sc["scale"] <= UB * MED):
        return None, "訓練時沒載深度（縮放超界／缺）"
    d = load_gt(name)
    if d is None:
        return None, "缺真深度"
    raw = np.load(f"{D}/estimated_depths/{name}.npy").astype(np.float64)
    raw = raw[..., 0] if raw.ndim == 3 else raw
    W, H, fx, fy, cx, cy, R, t = cam_of(im)
    if d.shape != (H, W):
        return None, f"真深度大小 {d.shape} != 相機 {(H, W)}"
    if raw.shape != (H, W):
        raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
    uu, vv = np.meshgrid(np.arange(0, W, ST), np.arange(0, H, ST))
    dg = d[::ST, ::ST]; x = raw[::ST, ::ST]
    vg = np.isfinite(dg) & (dg > 0) & (dg < 1e8)
    zg = np.where(vg, dg, 1.0) * K_UNIT
    ig = 1.0 / zg
    # ① 現行
    inv = [x * sc["scale"] + sc["offset"]]
    # ② 全域、真深度
    m = vg & (inv[0] > 1e-6)
    if m.sum() < 1000:
        return None, "有效像素太少"
    A = np.stack([x[m], np.ones(int(m.sum()))], 1)
    ab = np.linalg.lstsq(A, ig[m], rcond=None)[0]
    r = np.abs(A @ ab - ig[m]); k90 = r <= np.quantile(r, 0.9)
    ab = np.linalg.lstsq(A[k90], ig[m][k90], rcond=None)[0]
    inv.append(x * ab[0] + ab[1])
    # ③ 每 60 px、真深度（加權最小平方，patch 內有效 < 200 就用 ②）
    h2, w2 = x.shape; ph, pw = h2 // PL, w2 // PL
    def pat(a):
        return a[:ph * PL, :pw * PL].reshape(ph, PL, pw, PL)
    wv = pat(m.astype(np.float64)); xx = pat(np.where(m, x, 0)); yy = pat(np.where(m, ig, 0))
    S1 = wv.sum((1, 3)); Sx = xx.sum((1, 3)); Sy = yy.sum((1, 3)); Sxx = (xx * xx).sum((1, 3)); Sxy = (xx * yy).sum((1, 3))
    den = S1 * Sxx - Sx * Sx
    okp = (S1 >= 200) & (den > 1e-12 * np.maximum(S1 * Sxx, 1e-30))
    a3 = np.where(okp, (S1 * Sxy - Sx * Sy) / np.where(okp, den, 1), ab[0]); b3 = np.where(okp, (Sy - a3 * Sx) / np.maximum(S1, 1), ab[1])
    A3 = np.full(x.shape, ab[0]); B3 = np.full(x.shape, ab[1])
    A3[:ph * PL, :pw * PL] = np.repeat(np.repeat(a3, PL, 0), PL, 1); B3[:ph * PL, :pw * PL] = np.repeat(np.repeat(b3, PL, 0), PL, 1)
    inv.append(x * A3 + B3)
    # ④ 每 120 px、SfM 點（中位數／MAD；不夠 NMIN 點就用 ①）
    uv, zc, ii = sfm_obs(im, R, t)
    g = (zc > 0) & GOODP[ii]
    uv, zc = uv[g], zc[g]
    ui = np.clip(np.floor(uv[:, 0]).astype(int), 0, W - 1); vi = np.clip(np.floor(uv[:, 1]).astype(int), 0, H - 1)
    xs, ys = raw[vi, ui], 1.0 / zc
    gh, gw = (H + PS - 1) // PS, (W + PS - 1) // PS
    pidx = (vi // PS) * gw + (ui // PS)
    s4 = np.full(gh * gw, sc["scale"]); o4 = np.full(gh * gw, sc["offset"]); n4 = 0
    for k in np.unique(pidx):
        sel = pidx == k
        if sel.sum() >= NMIN:
            f = robust_fit(xs[sel], ys[sel])
            if f is not None and f[0] > 0:
                s4[k], o4[k] = f; n4 += 1
    pk = (vv // PS) * gw + (uu // PS)
    inv.append(x * s4[pk] + o4[pk])
    # 世界座標、分類
    Xc = np.stack([(uu + 0.5 - cx) / fx * zg, (vv + 0.5 - cy) / fy * zg, zg], -1)
    Xw = (Xc - t) @ R
    xy = Xw.reshape(-1, 3) @ RT.T
    xy = torch.from_numpy(xy[:, :2])
    if CONTRACT:
        xy = CityGSPartitioning.contract_to_unisphere(xy, AABB, ord=torch.inf)
    inb = CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=BB, coordinates=xy).numpy()
    blk = np.where(inb.any(0), inb.argmax(0), -1).reshape(x.shape)
    band = np.clip(np.searchsorted(BANDS, Xw[..., 2], side="right") - 1, 0, 4)
    du = np.zeros_like(Xw); dv_ = np.zeros_like(Xw)
    du[:, 1:-1] = Xw[:, 2:] - Xw[:, :-2]; dv_[1:-1] = Xw[2:] - Xw[:-2]
    nrm = np.cross(du, dv_); nn = np.linalg.norm(nrm, axis=-1)
    nz = np.abs(nrm[..., 2]) / np.maximum(nn, 1e-30)
    ori = np.where(nn <= 0, 1, np.where(nz > 0.8, 0, np.where(nz < 0.3, 2, 1)))
    jump = np.zeros(x.shape, dtype=bool)
    for sl0, sl1 in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))), ((slice(None), slice(1, None)), (slice(None), slice(None, -1)))):
        j = np.abs(zg[sl0] - zg[sl1]) / np.minimum(zg[sl0], zg[sl1]) > 0.03
        jump[sl0] |= j; jump[sl1] |= j
    edge = (maximum_filter((jump | ~vg).astype(np.uint8), size=5) > 0) & vg
    near = SFMV.near(torch.from_numpy(Xw.reshape(-1, 3).astype(np.float32))).numpy().reshape(x.shape)
    ci = np.floor(Xw[..., :2] / CELL).astype(np.int64) - O0
    inside = (ci[..., 0] >= 0) & (ci[..., 0] < HAS.shape[0]) & (ci[..., 1] >= 0) & (ci[..., 1] < HAS.shape[1])
    hs = np.zeros(x.shape, dtype=bool)
    hs[inside] = HAS[ci[..., 0][inside], ci[..., 1][inside]]
    water = (np.abs(Xw[..., 2] - Z_SEA) < SEA_TOL) & (ori == 0)
    cid = ci[..., 0] * HAS.shape[1] + ci[..., 1]
    cells = (np.unique(cid[vg & water & inside], return_counts=True), np.unique(cid[vg & inside], return_counts=True))   # 稀疏：格號、像素數
    code = code_of(blk, band, ori, edge.astype(np.int64), near.astype(np.int64), water.astype(np.int64))
    hist = np.zeros((4, NCODE * NB), dtype=np.int64)
    es = []
    for j in range(4):
        ok = vg & (inv[j] > 1e-6)
        e = np.clip(zg * inv[j] - 1.0, -0.9999, 0.9999)
        b = np.clip(np.searchsorted(EB, e[ok], side="right") - 1, 0, NB - 1)
        hist[j] += np.bincount(code[ok] * NB + b, minlength=NCODE * NB)
        es.append(np.where(ok, e, np.nan))
    cc = np.corrcoef(x[m], ig[m])[0, 1] if m.sum() > 100 else np.nan
    zr = (1.0 / np.maximum(xs * sc["scale"] + sc["offset"], 1e-9)) / zc
    vb = blk[vg]
    row = {"name": name, "block44": int(np.bincount(vb[vb >= 0], minlength=16).argmax()) if (vb >= 0).any() else -1,
           "n": int(vg.sum()), "corr_mono_gt": float(cc), "sfm_patches": n4, "n_sfm": int(len(zc)),
           "sfm_gt_ratio": sfm_ratio(zc, d[vi, ui]),
           "mono_at_sfm_med_abs": float(np.median(np.abs(zr - 1))) if len(zc) else np.nan,
           "tower_px": int((vg & (band == 4)).sum()), "water_px": int((vg & water).sum()),
           "water_inv0": int((vg & water & ~(inv[0] > 1e-6)).sum())}
    for j in range(4):
        e = es[j][vg]; e = e[np.isfinite(e)]
        row[f"a{j + 1}_bad5"] = float(np.mean(np.abs(e) > 0.05)) if len(e) else np.nan
        row[f"a{j + 1}_front5"] = float(np.mean(e < -0.05)) if len(e) else np.nan
    if want_maps:
        return (hist, row, {"x": x, "ig": ig, "vg": vg, "e1": es[0], "e4": es[3], "e3": es[2]}), ""
    return (hist, row, cells), ""


def sfm_ratio(zc, dd):
    ok = np.isfinite(dd) & (dd > 0) & (dd < 1e8)
    return float(np.median(zc[ok] / (dd[ok] * K_UNIT))) if ok.sum() > 10 else np.nan


def chunk(names):
    torch.set_num_threads(1)
    H = np.zeros((4, NCODE * NB), dtype=np.int64); rows = []; skip = {}
    SC, AC = [], []
    for n in names:
        try:
            r, why = one(n)
        except Exception as ex:          # 單張壞掉不拖垮整批，但要記下來
            r, why = None, f"例外 {type(ex).__name__}: {ex}"
        if r is None:
            skip[why] = skip.get(why, 0) + 1; continue
        H += r[0]; rows.append(r[1]); SC.append(r[2][0]); AC.append(r[2][1])
    return H, rows, skip, merge_cells(SC), merge_cells(AC)


def merge_cells(L):
    if not L:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    u = np.concatenate([x[0] for x in L]); c = np.concatenate([x[1] for x in L])
    uu, ix = np.unique(u, return_inverse=True)
    return uu, np.bincount(ix, weights=c).astype(np.int64)


def dense(uc):
    a = np.zeros(HAS.size, dtype=np.int64); a[uc[0]] = uc[1]
    return a


SFMV = SfMVoxels(PX, 0.03, "cpu")         # 與訓練的 SfM 豁免同一套（全部 SfM 點、0.03 單位＝約 3 m；11-01 版註解寫 3 cm 是錯的）


def inbox(xy3, b):
    xy = torch.from_numpy(np.asarray(xy3, dtype=np.float64) @ RT.T)[:, :2]
    if CONTRACT:
        xy = CityGSPartitioning.contract_to_unisphere(xy, AABB, ord=torch.inf)
    return CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=BB, coordinates=xy)[b].numpy()


def sea_particles(SEAC, ALLC):
    """海面格（真深度：該 0.02 格的像素 >=95% 是海面、至少 20 個）裡，各模型的顆粒。環境變數 SEA_RUNS＝'<塊>:<標籤>=<跑次> ...'。"""
    sea = (SEAC >= 20) & (SEAC >= 0.95 * ALLC)
    print(f"\n=== 海面格的顆粒（海面格 {int(sea.sum()):,} 個 0.02 格＝約 2 m；海面層＝海平面 ±0.01（約 1 m）；o>0.5 不透明、0.05~0.5 半透明）===")
    print("覆蓋指標＝Σ o·π·(2s1)(2s2) ÷ 該塊海面格面積（>1 表示疊好幾層；只看相對變化）；躺平＝|法向 z|>0.8 的不透明海面層顆粒比例")
    print(f"{'塊 標籤':24}{'塊內 N':>11} {'海面格':>7} {'海面層不透明':>10} {'半透明':>9} {'opacity質量':>11} {'覆蓋指標':>8} {'有不透明的格':>10} {'躺平':>6} {'上方>10m浮塊':>11} {'海面下':>8}")
    for spec in os.environ.get("SEA_RUNS", "").split():
        b, rest = spec.split(":", 1); lab, run = rest.split("=", 1); b = int(b)
        cks = [c for c in glob.glob(f"outputs/lab/{run}/blocks/block_{b}/checkpoints/*.ckpt") if "_inbox" not in c]
        cks = sorted(cks, key=lambda c: int(re.search(r"step=(\d+)", c).group(1)))
        if not cks:
            print(f"b{b:<2} {lab:20} 沒有 ckpt（{run}）"); continue
        sd = torch.load(cks[-1], map_location="cpu", weights_only=False)["state_dict"]
        g = lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
        m = g("means").numpy().astype(np.float64); inb = inbox(m, b)
        m = m[inb]; op = torch.sigmoid(g("opacities")).reshape(-1)[inb].numpy()
        s = torch.exp(g("scales"))[inb][:, :2].numpy(); nz = quat_normals(g("rotations")[inb])[:, 2].abs().numpy()
        del sd
        ci = np.floor(m[:, :2] / CELL).astype(np.int64) - O0
        ins = (ci[:, 0] >= 0) & (ci[:, 1] >= 0) & (ci[:, 0] < HAS.shape[0]) & (ci[:, 1] < HAS.shape[1])
        cid = np.full(len(m), -1, dtype=np.int64); cid[ins] = ci[ins, 0] * HAS.shape[1] + ci[ins, 1]
        ons = np.zeros(len(m), bool); ons[ins] = sea[cid[ins]]
        # 該塊的海面格（格中心在塊盒內）
        si = np.flatnonzero(sea); cx = (si // HAS.shape[1] + O0[0] + 0.5) * CELL; cy = (si % HAS.shape[1] + O0[1] + 0.5) * CELL
        sb = si[inbox(np.stack([cx, cy, np.full(len(si), Z_SEA)], 1), b)] if len(si) else si
        dz = m[:, 2] - Z_SEA
        surf = ons & (np.abs(dz) < 0.01); opq = surf & (op > 0.5); semi = surf & (op > 0.05) & (op <= 0.5)
        area = float((op[surf] * np.pi * 4 * s[surf, 0] * s[surf, 1]).sum()) / max(len(sb) * CELL * CELL, 1e-12)
        hit = np.zeros(HAS.size, bool); hit[cid[opq]] = True
        cov = float(hit[sb].mean()) if len(sb) else float("nan")
        flat = float((nz[opq] > 0.8).mean()) if opq.any() else float("nan")
        above = ons & (dz > 0.1) & (op > 0.2); below = ons & (dz < -0.01) & (op > 0.2)
        print(f"b{b:<2} {lab:20}{int(inb.sum()):>11,} {len(sb):>7,} {int(opq.sum()):>10,} {int(semi.sum()):>9,} {float(op[surf].sum()):>11,.0f} {area:>8.2f} {100 * cov:>9.1f}% {100 * flat:>5.1f}% {int(above.sum()):>11,} {int(below.sum()):>8,}", flush=True)
if __name__ == "__main__":
    t1 = time.time()
    NP = int(os.environ.get("NPROC", "12"))
    parts = [NAMES[i::2 * NP] for i in range(2 * NP)]
    with Pool(NP) as pool:
        res = pool.map(chunk, parts)
    Hs = sum(r[0] for r in res).reshape(4, NCODE, NB)
    rows = [x for r in res for x in r[1]]
    SEAC = dense(merge_cells([r[3] for r in res])); ALLC = dense(merge_cells([r[4] for r in res]))
    print(f"0.02 格網 {HAS.shape}（{HAS.size:,} 格）", flush=True)
    skip = {}
    for r in res:
        for k, v in r[2].items():
            skip[k] = skip.get(k, 0) + v
    np.savez_compressed(f"{OUT}/hist.npz", H=Hs, EB=EB)
    keys = list(rows[0].keys())
    with open(f"{OUT}/per_image.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\n逐張完成：{len(rows):,} 張、略過 {skip}｜{time.time() - t1:.0f} s", flush=True)

    ratio = np.array([r["sfm_gt_ratio"] for r in rows]); corr = np.array([r["corr_mono_gt"] for r in rows])
    print(f"[檢查 c] 每張的 SfM/真深度比 偏離 > 2% 的張數：{int(np.sum(np.abs(ratio - 1) > 0.02))}／{len(rows)}（配對錯開會在這裡出現）")
    print(f"[檢查 d] 單目逆深度與真逆深度的相關 中位 {np.nanmedian(corr):.3f}、< 0.5 的張數 {int(np.sum(corr < 0.5))}（單目圖配錯張會在這裡出現）")
    print(f"[參考] 現行對齊在 SfM 點上的 |z單目/zSfM − 1| 中位（逐張中位再取中位）{100 * np.nanmedian([r['mono_at_sfm_med_abs'] for r in rows]):.1f}%")

    # 解碼 code => 各維度
    cidx = np.arange(NCODE)
    c_wat = cidx % 2; c_near = (cidx // 2) % 2; c_edge = (cidx // 4) % 2
    c_ori = (cidx // 8) % 3; c_band = (cidx // 24) % 5; c_blk = cidx // 120 - 1
    mids = 0.5 * (EB[:-1] + EB[1:])

    def stats(h):
        n = h.sum()
        if n == 0:
            return None
        cdf = np.cumsum(h[np.argsort(np.abs(mids))]) / n
        med = np.sort(np.abs(mids))[np.searchsorted(cdf, 0.5)]
        return n, med, h[np.abs(mids) <= 0.05].sum() / n, h[mids < -0.05].sum() / n, h[mids > 0.10].sum() / n

    def table(title, groups, aligns=(0, 1, 2, 3)):
        print(f"\n=== {title} ===")
        print(f"{'':28}{'像素佔比':>8}  " + "  ".join(f"{ALIGN[j][0]} 中位|e|／貼面帶內／判前方／被推前" for j in aligns))
        tot = Hs[0].sum()
        for name, sel in groups:
            out = []
            for j in aligns:
                s = stats(Hs[j][sel].sum(0))
                out.append("—" if s is None else f"{100 * s[1]:5.1f}%／{100 * s[2]:5.1f}%／{100 * s[3]:5.1f}%／{100 * s[4]:5.1f}%")
            share = Hs[0][sel].sum() / max(tot, 1)
            print(f"{name:28}{100 * share:7.1f}%  " + "  ".join(f"{o:>34}" for o in out))

    print("\n欄位：中位|e|＝單目深度相對誤差中位；貼面帶內＝|e|<=5%（真表面上的顆粒會被閘門判有支撐）；"
          "判前方＝e<−5%（真表面會被當成浮塊壓掉）；被推前＝e>+10%（單側不能看穿會懲罰真表面）")
    print("對齊：" + "；".join(ALIGN))
    table("全部（塊內＋塊外，所有訓練影像）", [("全部", np.ones(NCODE, bool))])
    table("依高度（世界 z，場景單位）", [(BAND_N[b], c_band == b) for b in range(5)])
    table("依表面朝向（真深度算的法向）", [(ORI_N[o], c_ori == o) for o in range(3)])
    table("邊緣（真深度高低落差 3% 以上、±4 px）", [("邊緣帶", c_edge == 1), ("非邊緣", c_edge == 0)])
    table("離 SfM 點 3 cm 內（＝SfM 豁免會蓋到的地方）", [("近 SfM", c_near == 1), ("遠離 SfM", c_near == 0)])
    wp = sum(r["water_px"] for r in rows); w0 = sum(r["water_inv0"] for r in rows)
    print(f"\n海面像素（真深度 z 在 {Z_SEA}±{SEA_TOL}、水平）：{wp:,}；其中單目逆深度 <=0（訓練時當無效、閘門／不能看穿都不看）{100 * w0 / max(wp, 1):.1f}%（下表不含這些）")
    table("海面（真深度：海平面±0.004、水平）", [("海面", c_wat == 1), ("非海面", c_wat == 0)])
    table("海面 × 塊（只列有海面的塊）", [(f"b{b} 海面", (c_blk == b) & (c_wat == 1)) for b in range(16) if Hs[0][(c_blk == b) & (c_wat == 1)].sum() > 0], aligns=(0, 1, 2, 3))
    table("依 4x4 塊（像素的真 3D 點落在哪一塊）", [(f"b{b}", c_blk == b) for b in range(16)] + [("塊外（場景外）", c_blk == -1)], aligns=(0, 2, 3))
    sea_particles(SEAC, ALLC)

    # 示意圖：塔頂最多 2 張、b12 被判前方最多 2 張、水面最多 1 張
    pick = [r["name"] for r in sorted(rows, key=lambda r: -r["water_px"])[:3]]

    def cmap_e(e):
        tt = np.clip(np.nan_to_num(e, nan=0.0) / 0.2, -1, 1)
        w = 1 - np.abs(tt)
        img = np.stack([w + (tt < 0) * np.abs(tt), w, w + (tt > 0) * np.abs(tt)], -1)   # BGR：負＝藍、正＝紅
        img[np.isnan(e)] = 0.3
        return (255 * np.clip(img, 0, 1)).astype(np.uint8)

    for n in dict.fromkeys(pick):
        r, _ = one(n, want_maps=True)
        if r is None:
            continue
        mp = r[2]
        rgb = cv2.imread(f"{D}/input/{n}")[::ST, ::ST]
        lo, hi = np.percentile(mp["ig"][mp["vg"]], [2, 98])
        def dmap(a):
            return cv2.applyColorMap((255 * np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1)).astype(np.uint8), cv2.COLORMAP_TURBO)
        sc = SCALES[n]
        tiles = [rgb, dmap(np.where(mp["vg"], mp["ig"], lo)), dmap(mp["x"] * sc["scale"] + sc["offset"]), cmap_e(mp["e1"]), cmap_e(mp["e3"]), cmap_e(mp["e4"])]
        labs = ["RGB", "GT inverse depth", "mono inverse depth (current)", "err current (blue: mono too far => gated)", "err local GT fit (network itself)", "err local SfM fit"]
        tiles = [cv2.putText(tl.copy(), lb, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5) for tl, lb in zip(tiles, labs)]
        tiles = [cv2.putText(tl, lb, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2) for tl, lb in zip(tiles, labs)]
        big = np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:], 1)], 0)
        cv2.imwrite(f"{OUT}/{os.path.splitext(n)[0]}_b{r[1]['block44']}.jpg", cv2.resize(big, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\n示意圖：{OUT}/*.jpg（{len(dict.fromkeys(pick))} 張：海面像素最多的 3 張）")
    print("DONE monodepth_sea", flush=True)
