"""11-28 使用者的海面假說（「同一片海面要擬合多視角；斜看的視角 SfM 沒點 => 細節被壓到近處」）的檢查。只渲染、不訓練。

對一塊（預設 b8）：
  (1) 訓練視角的海面像素：視線打到海平面的點落在「海面格」（同 tools/probe/sea_blocks_1116.py：4 cm 格，格內 SfM 中位在海平面 ±1 m
      或 3x3 鄰域沒有可靠 SfM；扣掉高樓格）且在本塊分區盒內。逐像素記：視線俯角、GT 亮度、渲染亮度、
      渲染深度／海平面深度（< 0.8＝前面有東西擋，多半是霧或浮塊）、校正深度可靠度（凸包內／外）。
      => 按俯角、按凸包內外分組：GT 亮度隨俯角變多少（視角外觀假說）、霧出現在哪類像素（沒有深度約束假說）。
  (2) 測試幀（預設 0386～0398，塊 8 海面掉分最多的那幾幀）：海面像素裡渲染深度 < 0.8 x 海平面深度的「霧點」（渲染表面的 3D 點）；
      把霧點投到本塊訓練相機：看得到它的相機，俯角多少、那個像素在不在 SfM 凸包外、那條視線打到的海面在不在本塊盒內。
      對照組＝同一批測試像素上「海平面上的點」。
用法（lab，cwd＝gs）：python tools/probe/sea_haze_1128.py --run outputs/lab/fs1122_base10_f32 --block 8 [--frames 0386 0387 ...]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter

sys.path.insert(0, "."); sys.path.insert(0, "tools")
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="outputs/lab/fs1122_base10_f32")
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--frames", nargs="*", default=[f"{i:04d}" for i in range(386, 399)])
ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--max_train", type=int, default=400)
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
a = ap.parse_args()

from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.gaussian_model_loader import GaussianModelLoader
import eval_official_test as eot

D = "data/matrix_city/aerial/train/block_all"
dev = torch.device("cuda")
B = a.block
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pcd = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pcd.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb0, coordinates=(torch.from_numpy(np.ascontiguousarray(x)).double() @ Rt.T)[:, :2])[B].numpy()

# ── 海平面與海面格（同 sea_blocks_1116.py）──
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ERR = np.array([q.error for q in pts.values()]); TRK = np.array([len(q.image_ids) for q in pts.values()]); del pts
gd = (TRK >= 3) & (ERR < 1.5)
zg = P[gd, 2]; e = np.arange(-0.10, 0.0501, 0.001); h, _ = np.histogram(zg, bins=e); hs = np.convolve(h, np.ones(5) / 5, mode="same")
pk = [i for i in range(2, len(hs) - 2) if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.02 * hs.max()]
ZS = e[pk[0]] + 0.0005
S = P[gd]


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


o4, shp4, med4 = cellmed(0.04)
NOSFM = (~(maximum_filter(np.isfinite(med4).astype(np.uint8), size=3) > 0)).astype(float)
o10 = np.floor(S[:, :2].min(0) / 0.1).astype(int) - 2; ci10 = np.floor(S[:, :2] / 0.1).astype(int) - o10
shp10 = tuple(ci10.max(0) + 3); top = np.full(shp10, -1e9); np.maximum.at(top, (ci10[:, 0], ci10[:, 1]), S[:, 2])
TALL = maximum_filter((top > 1.0).astype(np.uint8), size=3).astype(float)
SEA = (np.isfinite(med4) & (np.abs(med4 - ZS) < 0.01)) | (NOSFM > 0)


def is_sea(x):
    s = look(x, 0.04, o4, shp4, SEA.astype(float))
    t = look(x, 0.1, o10, shp10, TALL)
    return (s > 0) & ~(t > 0)


print(f"海平面估 z={ZS:.4f}（{ZS * 100:.1f} m）；海面格 4 m", flush=True)

# ── 模型與相機 ──
mck = os.path.join(a.run, "checkpoints", "merged.ckpt")
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(mck, device=dev, eval_mode=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
bck = sorted(glob.glob(os.path.join(a.run, "blocks", f"block_{B}", "checkpoints", "*.ckpt")))[-1]
ck = torch.load(bck, map_location="cpu", weights_only=False); dmh = ck["datamodule_hyper_parameters"]; del ck
if getattr(dmh["parser"], "points_from", None) == "ply":
    dmh["parser"].points_from = "random"
ts = dmh["parser"].instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
tnames, tcams, tpaths = list(ts.image_names), ts.cameras, list(ts.image_paths)
sel = np.linspace(0, len(tnames) - 1, min(a.max_train, len(tnames))).round().astype(int).tolist()
print(f"[模型] {mck} N={model.n_gaussians:,}｜第 {B} 塊訓練相機 {len(tnames)} 台，取 {len(sel)} 台", flush=True)
bg = torch.zeros(3, device=dev)
REL = {}


def rel_of(name):
    st = os.path.splitext(os.path.basename(str(name)))[0]
    if st not in REL:
        f = f"{D}/estimated_depths_tri_rel/{st}.npy"
        REL[st] = np.load(f) if os.path.exists(f) else None
    return REL[st]


def rays(cam, s):
    H, W = int(cam.height), int(cam.width)
    v, u = np.mgrid[0:H:s, 0:W:s]
    dc = np.stack([(u + 0.5 - float(cam.cx)) / float(cam.fx), (v + 0.5 - float(cam.cy)) / float(cam.fy), np.ones_like(u, dtype=float)], -1).reshape(-1, 3)
    R = cam.R.cpu().numpy().astype(np.float64); T = cam.T.cpu().numpy().astype(np.float64)
    o = -R.T @ T
    dw = dc @ R            # 相機座標方向 -> 世界（x = R^T (p - T)）
    return u.reshape(-1), v.reshape(-1), dc, dw, o, H, W


def render(cam):
    with torch.no_grad():
        out = renderer(cam.to_device(dev), model, bg_color=bg)
    return out["render"].clamp(0, 1).cpu().numpy(), out["surf_depth"][0].cpu().numpy()


EB = [0, 20, 30, 40, 50, 60, 90]
rec = []        # 每列：俯角、凸包外、深度比、GT 亮度、渲染亮度、se
for k, i in enumerate(sel):
    cam = tcams[i]
    u, v, dc, dw, o, H, W = rays(cam, a.stride)
    t = (ZS - o[2]) / np.where(dw[:, 2] < -1e-6, dw[:, 2], np.nan)
    Pw = o[None] + t[:, None] * dw
    ok = np.isfinite(t) & (t > 0)
    if not ok.any():
        continue
    ok[ok] = is_sea(Pw[ok]) & inb(Pw[ok])
    if not ok.any():
        continue
    img, dep = render(cam)
    gt = np.asarray(Image.open(tpaths[i]).convert("RGB").resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.
    zr = dep[v[ok], u[ok]]; zp = t[ok]           # 海平面點的相機 z＝t（dc 的 z 分量是 1）
    el = np.degrees(np.arcsin(np.clip(-dw[ok, 2] / np.linalg.norm(dw[ok], axis=1), -1, 1)))
    rl = rel_of(tnames[i])
    if rl is None:
        out_h = np.full(ok.sum(), np.nan)
    else:
        hh, ww = rl.shape
        out_h = ((rl[np.clip((v[ok] * hh) // H, 0, hh - 1), np.clip((u[ok] * ww) // W, 0, ww - 1)] & 7) >= 4).astype(float)
    g = gt[v[ok], u[ok]]; r = img[:, v[ok], u[ok]].T
    rec.append(np.stack([el, out_h, zr / zp, g.mean(1), r.mean(1), ((r - g) ** 2).mean(1)], 1))
    if (k + 1) % 100 == 0:
        print(f"  訓練視角 {k + 1}/{len(sel)}", flush=True)
R_ = np.concatenate(rec) if rec else np.zeros((0, 6))
print(f"\n=== (1) 第 {B} 塊訓練視角的海面像素（{len(R_):,} 個取樣像素；視線打到海平面的點在海面格且在本塊盒內）===")


def table(rows, key, labels):
    print(f"  {'':<12} {'佔比':>6} {'GT 亮度':>8} {'渲染亮度':>8} {'PSNR':>7} {'前面有東西(<0.8)':>16} {'凸包外':>7}")
    for lab, m in zip(labels, key):
        if m.sum() == 0:
            continue
        x = rows[m]
        print(f"  {lab:<12} {100 * m.mean():>5.1f}% {np.median(x[:, 3]):>8.3f} {np.median(x[:, 4]):>8.3f} {-10 * np.log10(max(x[:, 5].mean(), 1e-12)):>7.2f}"
              f" {100 * np.mean(x[:, 2] < 0.8):>15.1f}% {100 * np.nanmean(x[:, 1]):>6.1f}%")


if len(R_):
    table(R_, [(R_[:, 0] >= EB[j]) & (R_[:, 0] < EB[j + 1]) for j in range(len(EB) - 1)], [f"俯角 {EB[j]}~{EB[j + 1]}°" for j in range(len(EB) - 1)])
    print("  ——按 SfM 凸包內外——")
    table(R_, [R_[:, 1] == 0, R_[:, 1] == 1], ["凸包內", "凸包外"])

# ── (2) 測試幀的霧點 ──
names_t, cams_t = eot.load_test_cameras(a.test_dir, 1.2)
idx = {os.path.splitext(os.path.basename(str(n)))[0]: j for j, n in enumerate(names_t)}
fog, ref = [], []
for fn in a.frames:
    if fn not in idx:
        continue
    cam = cams_t[idx[fn]]
    u, v, dc, dw, o, H, W = rays(cam, a.stride)
    t = (ZS - o[2]) / np.where(dw[:, 2] < -1e-6, dw[:, 2], np.nan)
    Pw = o[None] + t[:, None] * dw
    ok = np.isfinite(t) & (t > 0)
    ok[ok] = is_sea(Pw[ok]) & inb(Pw[ok])
    img, dep = render(cam)
    zr = dep[v[ok], u[ok]]
    ratio = zr / t[ok]
    fz = (ratio < 0.8) & (zr > 0)
    Pf = o[None] + (zr[fz])[:, None] * dw[ok][fz]          # 霧點（渲染表面）
    fog.append(Pf); ref.append(Pw[ok][~fz][:: max(1, int((~fz).sum() // max(1, fz.sum())))])
    print(f"  測試幀 {fn}：海面取樣像素 {ok.sum():,}，前面有東西（渲染深度 < 0.8 x 海平面深度）{100 * fz.mean():.1f}%", flush=True)
FOG = np.concatenate(fog) if fog else np.zeros((0, 3)); REF = np.concatenate(ref) if ref else np.zeros((0, 3))
print(f"\n=== (2) 測試幀 {a.frames[0]}～{a.frames[-1]} 的霧點 {len(FOG):,} 個（對照：海平面點 {len(REF):,} 個）===")
if len(FOG):
    hz = (FOG[:, 2] - ZS) * 100
    print(f"  霧點離海平面高度（m）：p10 {np.percentile(hz, 10):.1f}／中位 {np.median(hz):.1f}／p90 {np.percentile(hz, 90):.1f}")


def seen_by(X):
    """X 投到本塊訓練相機：每個（點, 相機）配對的俯角、該像素在不在凸包外、那條視線打到的海面在不在本塊盒內。"""
    if len(X) == 0:
        return None
    X = X[np.random.default_rng(0).permutation(len(X))[:3000]]
    out = []
    for i in sel:
        cam = tcams[i]
        R = cam.R.cpu().numpy().astype(np.float64); T = cam.T.cpu().numpy().astype(np.float64)
        pc = X @ R.T + T
        z = pc[:, 2]
        uu = pc[:, 0] / np.maximum(z, 1e-9) * float(cam.fx) + float(cam.cx); vv = pc[:, 1] / np.maximum(z, 1e-9) * float(cam.fy) + float(cam.cy)
        H, W = int(cam.height), int(cam.width)
        m = (z > 0.2) & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        if not m.any():
            continue
        o = -R.T @ T
        d = X[m] - o[None]; d /= np.linalg.norm(d, axis=1, keepdims=True)
        el = np.degrees(np.arcsin(np.clip(-d[:, 2], -1, 1)))
        t = (ZS - o[2]) / np.where(d[:, 2] < -1e-6, d[:, 2], np.nan)
        hit = o[None] + t[:, None] * d
        hb = np.zeros(m.sum()); okh = np.isfinite(t)
        hb[okh] = inb(hit[okh]).astype(float)
        rl = rel_of(tnames[i])
        if rl is None:
            oh = np.full(m.sum(), np.nan)
        else:
            hh, ww = rl.shape
            oh = ((rl[np.clip((vv[m] * hh / H).astype(int), 0, hh - 1), np.clip((uu[m] * ww / W).astype(int), 0, ww - 1)] & 7) >= 4).astype(float)
        out.append(np.stack([el, oh, hb], 1))
    return np.concatenate(out) if out else None


for lab, X in (("霧點", FOG), ("對照：海平面點", REF)):
    s = seen_by(X)
    if s is None:
        print(f"  {lab}：沒有相機看得到"); continue
    q = np.percentile(s[:, 0], [10, 50, 90])
    print(f"  {lab}：（點, 相機）配對 {len(s):,}｜相機俯角 p10/中位/p90 {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}°｜"
          f"該像素在 SfM 凸包外 {100 * np.nanmean(s[:, 1]):.1f}%｜視線打到的海面在本塊盒內 {100 * s[:, 2].mean():.1f}%")
    for j in range(len(EB) - 1):
        m = (s[:, 0] >= EB[j]) & (s[:, 0] < EB[j + 1])
        if m.sum():
            print(f"      俯角 {EB[j]:>2}~{EB[j + 1]:<2}°：{100 * m.mean():5.1f}% 的配對｜凸包外 {100 * np.nanmean(s[m, 1]):5.1f}%")
