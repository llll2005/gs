"""11-06 使用者 viewer：「當前的機制會把路面的路燈幾乎消除、b3 的車輛也重建失敗；CityGS 有路燈、車也清楚 => 是我們的問題」。
找出是哪一代開始壞的（GPU 渲染＋CPU 量測；cwd＝repo 根；只讀）。第一個模型必須是 CityGS（＝參考）。
量測（只用 SfM 定義地面，不靠真值）：
  地面格＝2 m 格 SfM 中位 z < 3 m；建築格＝其餘有 SfM 的格，往外擴 2 格（4 m）後排除 => 「開闊地面」。
  路燈帶＝開闊地面上方 3～15 m；車輛帶＝0.5～2.5 m（含矮樹叢等，近似）。孤立柱＝路燈帶裡周圍 8 格的參考顆粒 < 中心格一半的格（路燈像、樹不像）。
  參考＝CityGS 在各帶的顆粒（o > 0.3）；缺口＝參考顆粒 R 內沒有本模型 o > 0.3 的顆粒的比例（R＝0.5／1 m）。
對照圖：路燈帶孤立柱最多的幾個格、車輛帶最密的幾個格，各放一台低角度相機，每個模型渲染一張，拼成一張圖（列＝地點、欄＝模型）。
用法：python tools/probe/smallobj_probe_1106.py --block 3 --models CityGS=... 標籤=ckpt ... --out <資料夾>
"""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter, uniform_filter
sys.path.insert(0, ".")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary
from internal.cameras.cameras import Cameras
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nlamp", type=int, default=4)
ap.add_argument("--ncar", type=int, default=3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda"); M = 100.0
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
bb1 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double() @ Rt.T)[:, :2])[B].numpy()

# ---- 地面格／建築格（2 m 格，可靠 SfM）----
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); ok = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = P[ok & inb(P, bb1)]
G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u] = S[order, 2][f + c // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03)
bldg = maximum_filter((np.isfinite(med) & ~ground).astype(np.uint8), size=5) > 0
open_g = ground & ~bldg
# 地面高度：開闊地面格的中位，空格用 5x5 鄰域平均補
gz = np.where(ground, med, np.nan)
num = uniform_filter(np.nan_to_num(gz), size=5); den = uniform_filter(np.isfinite(gz).astype(float), size=5)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))


def cell_of(x):
    cc = np.floor(x[:, :2] / G).astype(int) - o0
    okc = (cc[:, 0] >= 0) & (cc[:, 0] < shp[0]) & (cc[:, 1] >= 0) & (cc[:, 1] < shp[1])
    cc = np.clip(cc, 0, np.array(shp) - 1)
    return cc, okc


def bands(x):
    cc, okc = cell_of(x)
    h = x[:, 2] - gfill[cc[:, 0], cc[:, 1]]
    og = okc & open_g[cc[:, 0], cc[:, 1]] & np.isfinite(h)
    return cc, og & (h > 0.03) & (h < 0.15), og & (h > 0.005) & (h < 0.025), h


def load(spec):
    lab, pat = spec.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    model, rnd, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
    pr = {k: v.detach().float() for k, v in model.properties.items()}
    k = torch.from_numpy(inb(pr["means"].numpy(), bb0))
    model.properties = {kk: v[k] for kk, v in pr.items()}
    return lab, model, rnd


# ---- 參考（CityGS）----
labs, models = [], []
lab0, ref, rnd0 = load(a.models[0])
for fl in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
    if hasattr(rnd0, fl):
        setattr(rnd0, fl, False)
X0 = ref.properties["means"].detach().numpy().astype(np.float64); O0 = torch.sigmoid(ref.properties["opacities"].detach()).reshape(-1).numpy()
cc0, lamp0, car0, _ = bands(X0)
lamp0 &= O0 > 0.3; car0 &= O0 > 0.3
cnt = np.zeros(shp); np.add.at(cnt, (cc0[lamp0, 0], cc0[lamp0, 1]), 1)
ring = uniform_filter(cnt, size=3) * 9 - cnt
iso_cell = (cnt >= 15) & (ring < 0.5 * cnt)
iso0 = lamp0 & iso_cell[cc0[:, 0], cc0[:, 1]]
print(f"b{B}｜開闊地面 2 m 格 {int(open_g.sum()):,}｜參考 {lab0}（o > 0.3）：路燈帶 3～15 m {int(lamp0.sum()):,} 顆、其中孤立柱 {int(iso0.sum()):,} 顆"
      f"（{int(iso_cell.sum())} 格）、車輛帶 0.5～2.5 m {int(car0.sum()):,} 顆", flush=True)

# ---- 地點與相機 ----
ims = read_images_binary(f"{D}/sparse/0/images.bin"); cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
c0 = cams[next(iter(ims.values())).camera_id]; W0, H0 = round(c0.width / 1.2), round(c0.height / 1.2); f0 = c0.params[0] * W0 / c0.width


def mk(R, T):
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(R), T=t(T), fx=t([f0])[0], fy=t([f0])[0], cx=t([W0 / 2])[0], cy=t([H0 / 2])[0],
                   width=torch.tensor([W0], dtype=torch.int16), height=torch.tensor([H0], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1),
                   distortion_params=None, camera_type=torch.zeros(1, dtype=torch.int8))[0]


def pick(score, n, sep):
    sc = score.copy(); out = []
    while len(out) < n and sc.max() > 0:
        i, j = np.unravel_index(np.argmax(sc), sc.shape); out.append((i, j))
        r = int(sep / G); sc[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = 0
    return out


ccar = np.zeros(shp); np.add.at(ccar, (cc0[car0, 0], cc0[car0, 1]), 1); ccar = uniform_filter(ccar, size=3)
spots = [("燈", ij) for ij in pick(np.where(iso_cell, cnt, 0), a.nlamp, 0.3)] + [("車", ij) for ij in pick(ccar, a.ncar, 0.4)]
views = []
for kind, (i, j) in spots:
    x, y = (np.array([i, j]) + o0 + 0.5) * G; g = gfill[i, j]
    dist, hc, ht = (0.22, 0.05, 0.04) if kind == "燈" else (0.25, 0.08, 0.008)
    tgt = np.array([x, y, g + ht]); best = None
    for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        C = np.array([x + dist * np.cos(th), y + dist * np.sin(th), g + hc])
        cc, okc = cell_of(C[None]);
        if okc[0] and open_g[cc[0, 0], cc[0, 1]] and inb(C[None], bb0)[0]:
            best = C; break
    if best is None:
        best = np.array([x + dist, y, g + hc])
    z = tgt - best; z /= np.linalg.norm(z); xx = np.cross(z, [0, 0, 1.0]); xx /= np.linalg.norm(xx); yy = np.cross(z, xx)
    R = np.stack([xx, yy, z]); views.append((f"{kind}{len(views) + 1}（{x * M:.0f},{y * M:.0f} m）", mk(R, -R @ best)))
print("地點：" + "、".join(v[0] for v in views), flush=True)

# ---- 每個模型：量測＋渲染 ----
T0 = cKDTree(X0)
rows = {}
print(f"{'模型':16}{'路燈帶缺口 0.5/1 m':>18}{'孤立柱缺口 0.5/1 m':>18}{'車輛帶缺口 0.5/1 m':>18}{'路燈帶顆數':>10}{'車輛帶顆數':>10}{'路燈帶 o 中位':>12}", flush=True)
for mi, spec in enumerate(a.models):
    if mi == 0:
        lab, model = lab0, ref
    else:
        lab, model, _ = load(spec)
    X = model.properties["means"].detach().cpu().numpy().astype(np.float64); O = torch.sigmoid(model.properties["opacities"].detach()).reshape(-1).cpu().numpy()
    _, lamp, car, _ = bands(X)
    tr = cKDTree(X[O > 0.3])
    gap = lambda ref_m, r: 100 * np.mean(np.isinf(tr.query(X0[ref_m], k=1, distance_upper_bound=r)[0])) if ref_m.any() else float("nan")
    print(f"{lab:16}{gap(lamp0, 0.005):>9.0f}/{gap(lamp0, 0.01):<3.0f}%{gap(iso0, 0.005):>12.0f}/{gap(iso0, 0.01):<3.0f}%"
          f"{gap(car0, 0.005):>12.0f}/{gap(car0, 0.01):<3.0f}%{int((lamp & (O > 0.3)).sum()):>10,}{int((car & (O > 0.3)).sum()):>10,}"
          f"{np.median(O[lamp]) if lamp.any() else float('nan'):>12.2f}", flush=True)
    model.to(dev)
    for nm, cam in views:
        with torch.no_grad():
            o = rnd0(cam.to_device(dev), model, bg_color=torch.zeros(3, device=dev))
        rows.setdefault(nm, []).append((lab, (o["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)))
    model.to(torch.device("cpu")); del model; torch.cuda.empty_cache()

# ---- 拼圖 ----
tw = 480; th = round(H0 * tw / W0); pad = 22
font = None
for fp in ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, 16); break
names = list(rows)
cols = [l for l, _ in rows[names[0]]]
canvas = Image.new("RGB", (len(cols) * tw, pad + len(names) * (th + pad)), (20, 20, 20))
dr = ImageDraw.Draw(canvas)
for ci_, l in enumerate(cols):
    dr.text((ci_ * tw + 6, 2), l, fill=(230, 230, 230), font=font)
for ri, nm in enumerate(names):
    y0 = pad + ri * (th + pad)                     # 每列：地點標籤一條＋圖
    dr.text((6, y0 + 2), nm, fill=(150, 200, 255), font=font)
    for ci_, (l, img) in enumerate(rows[nm]):
        canvas.paste(Image.fromarray(img).resize((tw, th), Image.LANCZOS), (ci_ * tw, y0 + pad))
canvas.save(f"{a.out}/grid.jpg", quality=90)
for ri, nm in enumerate(names):
    for l, img in rows[nm]:
        Image.fromarray(img).save(f"{a.out}/{ri}_{l}.jpg", quality=90)
print(f"對照圖：{a.out}/grid.jpg（列＝地點、欄＝模型）；單張：{a.out}/<列>_<模型>.jpg", flush=True)
print("DONE smallobj_probe")
