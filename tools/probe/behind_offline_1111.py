"""11-11 規格 §3（`紀錄/new_archived/規格與交接/規格_組合臂_表面起點增生限制深度上限_1111.md`）：訓練前的離線檢查——「只擋後面的增生限制」的 behind 票會不會誤判。
同一條判準（`internal/metrics/geom_gate.py` DepthSupport 的 behind 票），但用**全部**本塊訓練相機（訓練時每個視窗只輪到約 50 台）：
  r＝(顆粒深度 − 表面深度)／表面深度；深度＝dataparser 給的那份（校正深度＋可靠度過濾，不可靠像素＝0＝不算觀測）
  有效觀測＝z > near、投影在畫面內、深度 > 0；behind 票＝margin < r < r_max；有效觀測 >= 3 且 behind 票過半 => 判 behind。
量的東西（分類只用 SfM 定義地面，不靠真值；**量測用公尺可以，機制不行**——類別的公尺門檻只在這支腳本）：
  地面格＝2 m 格可靠 SfM 中位 z < 3 m；建築格＝其餘有 SfM 的格往外擴 2 格；開闊地面＝地面格扣掉建築格（同 smallobj_cpu_1109）。
  h＝顆粒 z − 地面高度（開闊地面格中位、5x5 補洞；補不到用本塊地面格中位）。
  類別：貼地 |h|<0.5 m／淺地下 −1～−0.5 m／真地下 < −1 m（地面格；o>0.5）、車輛帶 0.5～2.5 m／路燈帶 3～15 m（開闊地面；o>0.3，同 smallobj 的參考門檻）、
        樓頂 h>3 m 且 |n_z|>0.85／立面 h>3 m 且 |n_z|<0.3／斜的 h>3 m 其餘（不在開闊地面；o>0.5）。
過關（規格）：CityGS 每一類 behind <= 約 3%（路燈帶、車輛帶特別列）；我方基準「真地下」behind >= 約 50%。
另列幾個變體（同一次投影順便算，沒過時拿來討論）：窄帶換 (0.5,5)／(1,3)／(2,5)／(1,10)%、「貼面票 < 20%」（|r| <= margin 的觀測少＝這顆幾乎沒有在哪個視角是表面），
以及「不計被擋」：分母扣掉 r >= r_max 的觀測（被別的東西擋住的視角不算票，只在看得到附近表面的視角裡過半）——15 台試跑時貼地／真地下的觀測一半以上是 r >= 5%。
用法（lab，cwd＝gs，CPU）：
  conda run -n gspl --no-capture-output python tools/probe/behind_offline_1111.py --block 1 \
     --cams outputs/lab/<基準>/blocks/block_1/cameras.json --models CityGS=<ckpt 樣式> 基準=<ckpt 樣式>
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import maximum_filter, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--cams", required=True, help="本塊訓練相機（跑次的 blocks/block_<B>/cameras.json）")
ap.add_argument("--models", nargs="+", required=True, help="標籤=ckpt 樣式（glob，取排序最後一個）；第一個當參考（CityGS）")
ap.add_argument("--margin", type=float, default=0.01)
ap.add_argument("--rmax", type=float, default=0.05)
ap.add_argument("--min_count", type=int, default=3)
ap.add_argument("--near", type=float, default=0.2)
ap.add_argument("--depth_dir", default="estimated_depths_tri")
ap.add_argument("--rel_dir", default="estimated_depths_tri_rel")
ap.add_argument("--scale_name", default="estimated_depth_scales_tri")
ap.add_argument("--slope_max", type=int, default=1)
ap.add_argument("--size_max", type=int, default=3)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--max_cams", type=int, default=0, help=">0：只取前幾台（快速試跑）")
ap.add_argument("--save", default="", help="給了就把每顆的判定存成 npz（標籤_b<B>.npz）到這個資料夾")
a = ap.parse_args()
torch.set_num_threads(a.threads)
D = "data/matrix_city/aerial/train/block_all"; B = a.block; MU = 100.0          # 1 場景單位＝100 m（只用在類別定義）
t0 = time.time()

# ---- 分塊、地面格（同 smallobj_cpu_1109）----
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
bb1 = pc.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double() @ Rt.T)[:, :2])[B].numpy()

pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); okp = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = P[okp & inb(P, bb1)]
G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u_, f_, c_ = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u_] = S[order, 2][f_ + c_ // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03)
bldg = maximum_filter((np.isfinite(med) & ~ground).astype(np.uint8), size=5) > 0
open_g = ground & ~bldg
gz = np.where(ground, med, np.nan)
num = uniform_filter(np.nan_to_num(gz), size=5); den = uniform_filter(np.isfinite(gz).astype(float), size=5)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
GL = float(np.nanmedian(gz))


def categories(m, o, nz):
    cc = np.floor(m[:, :2] / G).astype(int) - o0
    okc = (cc[:, 0] >= 0) & (cc[:, 0] < shp[0]) & (cc[:, 1] >= 0) & (cc[:, 1] < shp[1])
    cc = np.clip(cc, 0, np.array(shp) - 1)
    g = gfill[cc[:, 0], cc[:, 1]]
    h = (m[:, 2] - np.where(okc & np.isfinite(g), g, GL)) * MU                     # 公尺
    isg = okc & ground[cc[:, 0], cc[:, 1]]
    iso = okc & open_g[cc[:, 0], cc[:, 1]]
    op5, op3 = o > 0.5, o > 0.3
    hi = op5 & ~iso & (h > 3)
    return [("貼地 |h|<0.5m", op5 & isg & (np.abs(h) < 0.5)),
            ("淺地下 −1～−0.5m", op5 & isg & (h <= -0.5) & (h > -1)),
            ("真地下 < −1m", op5 & isg & (h <= -1)),
            ("車輛帶 0.5～2.5m", op3 & iso & (h > 0.5) & (h < 2.5)),
            ("路燈帶 3～15m", op3 & iso & (h > 3) & (h < 15)),
            ("樓頂 h>3m", hi & (nz > 0.85)),
            ("立面 h>3m", hi & (nz < 0.3)),
            ("斜的 h>3m", hi & (nz >= 0.3) & (nz <= 0.85))], h


# ---- 模型 ----
def nrm_z(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.abs(1 - 2 * (x * x + y * y))                                          # 面元法向（局部 z 軸）的世界 z 分量


BANDS = [(a.margin, a.rmax), (0.005, 0.05), (0.01, 0.03), (0.02, 0.05), (0.01, 0.10)]
models = []
for spec in a.models:
    lab, pat = spec.split("=", 1)
    fs = sorted(glob.glob(pat))
    if not fs:
        raise SystemExit(f"⛔ {lab}：找不到 {pat}")
    sd = torch.load(fs[-1], map_location="cpu", weights_only=False)["state_dict"]
    gg = lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
    m = gg("means").double().numpy(); o = torch.sigmoid(gg("opacities")).reshape(-1).numpy(); q = gg("rotations").double().numpy()
    del sd
    k = inb(m, bb0) & (o > 0.3)
    m, o, nz = m[k], o[k], nrm_z(q[k])
    cats, h = categories(m, o, nz)
    st = {"cnt": torch.zeros(len(m), dtype=torch.int32), "front": torch.zeros(len(m), dtype=torch.int32),
          "on": torch.zeros(len(m), dtype=torch.int32), "far": torch.zeros(len(m), dtype=torch.int32)}
    for bi in range(len(BANDS)):
        st[f"b{bi}"] = torch.zeros(len(m), dtype=torch.int32)
    models.append({"lab": lab, "ckpt": fs[-1], "X": torch.from_numpy(m), "o": o, "cats": cats, "h": h, "st": st})
    print(f"[{lab}] {fs[-1]}｜塊內 o>0.3 {len(m):,} 顆｜" + "、".join(f"{nm} {int(c.sum()):,}" for nm, c in cats), flush=True)

# ---- 相機與深度（與 dataparser load_depth 相同：raw x scale + offset，可靠度不合格的像素設 0）----
ims = read_images_binary(f"{D}/sparse/0/images.bin"); cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names = [os.path.basename(c["img_name"]) for c in json.load(open(a.cams))]
byname = {os.path.basename(im.name): im for im in ims.values()}
sel = [byname[n] for n in names if n in byname]
if a.max_cams > 0:
    sel = sel[:a.max_cams]
scales = json.load(open(f"{D}/{a.scale_name}.json"))
med_s = float(np.median([v["scale"] for v in scales.values()]))
print(f"b{B}｜本塊訓練相機 {len(names)} 台（images.bin 對到 {len(sel)}）｜深度 {a.depth_dir}＋可靠度 {a.rel_dir}（斜率級 <= {a.slope_max}、大小級 <= {a.size_max}）"
      f"｜behind 窄帶 ({a.margin:g}, {a.rmax:g})、觀測 >= {a.min_count}、near {a.near:g}", flush=True)
n_nodepth = 0; valid_px = []
for kk, im in enumerate(sel):
    nm = os.path.basename(im.name)
    sc = scales.get(im.name, scales.get(nm))
    if sc is None or not (0.2 * med_s <= sc["scale"] <= 5.0 * med_s):
        n_nodepth += 1; continue
    f = f"{D}/{a.depth_dir}/{nm}.npy"
    if not os.path.isfile(f):
        n_nodepth += 1; continue
    dep = np.load(f).astype(np.float32)
    dep = dep[..., 0] if dep.ndim == 3 else dep
    dep = dep * sc["scale"] + sc["offset"]
    if a.rel_dir:
        rel = np.load(f"{D}/{a.rel_dir}/{nm}.npy")
        dep = np.where(((rel & 7) <= a.slope_max) & ((rel >> 3) <= a.size_max), dep, 0.0).astype(np.float32)
    valid_px.append(float((dep > 1e-6).mean()))
    inv = torch.from_numpy(dep)
    hd, wd = inv.shape
    cam = cams[im.camera_id]; pr = cam.params
    fx, fy, cx, cy = (pr[0], pr[0], pr[1], pr[2]) if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL") else (pr[0], pr[1], pr[2], pr[3])
    Rk = torch.from_numpy(qvec2rotmat(im.qvec)); Tk = torch.from_numpy(np.asarray(im.tvec, dtype=np.float64))
    for md in models:
        Xc = md["X"] @ Rk.T + Tk
        z = Xc[:, 2]
        ok = z > a.near
        zs = z.clamp_min(1e-6)
        u = fx * Xc[:, 0] / zs + cx; v = fy * Xc[:, 1] / zs + cy
        ud = (u * (wd / float(cam.width))).floor().long(); vd = (v * (hd / float(cam.height))).floor().long()
        ok &= (ud >= 0) & (ud < wd) & (vd >= 0) & (vd < hd)
        dinv = inv[vd.clamp(0, hd - 1), ud.clamp(0, wd - 1)].double()
        ok &= dinv > 1e-6
        zm = 1.0 / dinv.clamp_min(1e-6)
        r = torch.where(ok, (z - zm) / zm, torch.zeros_like(z))
        s = md["st"]
        s["cnt"] += ok.int()
        s["front"] += (ok & (r < -a.margin)).int()
        s["on"] += (ok & (r.abs() <= a.margin)).int()
        s["far"] += (ok & (r >= a.rmax)).int()
        for bi, (lo, hi) in enumerate(BANDS):
            s[f"b{bi}"] += (ok & (r > lo) & (r < hi)).int()
    if (kk + 1) % 50 == 0:
        print(f"  相機 {kk + 1}/{len(sel)}（{time.time() - t0:.0f} s）", flush=True)
print(f"有深度的相機 {len(sel) - n_nodepth}/{len(sel)}｜可靠像素比例中位 {100 * np.median(valid_px):.1f}%｜{time.time() - t0:.0f} s", flush=True)


def pct(x, k):
    return f"{100 * x[k].mean():6.2f}" if k.sum() else "     —"


hdr_b = [f"({100 * lo:g},{100 * hi:g})%" for lo, hi in BANDS[1:]]
for md in models:
    s = {k: v.numpy() for k, v in md["st"].items()}
    cnt = s["cnt"]; seen = cnt >= a.min_count
    beh = {bi: seen & (s[f"b{bi}"] > 0.5 * cnt) for bi in range(len(BANDS))}
    strict = beh[0] & (s["on"] < 0.2 * cnt)
    vis = cnt - s["far"]
    nofar = (vis >= a.min_count) & (s["b0"] > 0.5 * vis)
    fr = lambda key: np.where(cnt > 0, s[key] / np.maximum(cnt, 1), np.nan)
    print(f"\n== {md['lab']}｜behind（主判準 ({100 * a.margin:g}, {100 * a.rmax:g})%）{100 * beh[0].mean():.2f}%＝{int(beh[0].sum()):,} 顆")
    print(f"  {'類別':16}{'顆數':>10}{'看過>=3%':>9}{'behind%':>9}{'貼面票<20%':>11}{'不計被擋':>9}  | 變體 behind%：" + "".join(f"{x:>12}" for x in hdr_b)
          + "  | 觀測票平均：前方／貼面／窄帶／更後面 %")
    rows = [("全部", np.ones(len(cnt), bool))] + md["cats"]
    for nm, k in rows:
        if k.sum() == 0:
            print(f"  {nm:16}{0:>10}"); continue
        kv = k & (cnt > 0)
        mf = lambda key: f"{100 * np.nanmean(fr(key)[kv]):5.1f}" if kv.sum() else "  —"
        print(f"  {nm:16}{int(k.sum()):>10,}{pct(seen, k):>9}{pct(beh[0], k):>9}{pct(strict, k):>11}{pct(nofar, k):>9}  | "
              + "".join(f"{pct(beh[bi], k):>12}" for bi in range(1, len(BANDS)))
              + f"  | {mf('front')}／{mf('on')}／{mf('b0')}／{mf('far')}", flush=True)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        np.savez_compressed(f"{a.save}/{md['lab']}_b{B}.npz", X=md["X"].numpy().astype(np.float32), o=md["o"].astype(np.float32), h=md["h"].astype(np.float32),
                            behind=beh[0], strict=strict, nofar=nofar, **{k: v.astype(np.int16) for k, v in s.items()})
print(f"\n過關條件（規格 §3）：第一個模型（CityGS）每一類 behind <= 約 3%（路燈帶、車輛帶特別看）；我方基準「真地下 < −1m」behind >= 約 50%｜{time.time() - t0:.0f} s")
print("DONE")
