"""11-07 穿透率判斷刪除（record_T）只刪到 938 顆：不透明顆粒 Tmax 分位 q10／50／90＝0.60／0.82／0.98 => 九成在某一台訓練相機站在前面。
那建築內部／地下的實心顆粒（中心判斷「全部被擋」的 37k）是被誰、怎麼看到的？兩個假說：
  ①牆有縫（不透明小圓片鋪不滿平面）或斜看時漏出來 => 只在少數台看得到、看得到的那台是斜看（入射角大）
  ②中心判斷誤判（面元大、邊緣露出來；或它其實就是表面那層）=> 很多台看得到、正面看
逐台 899 台訓練相機：一次一般渲染（中心判斷：該像素 α > 0.95 且比表面深 > --margin_px 個像素足跡＝被擋）＋一次 record_T（逐顆覆蓋像素平均 T）。
報（盒內 o > 0.5，依分區與「中心被擋比例」分組）：T > 0.3 的台數與佔畫面內台數的比例、Tmax、Tmax 那台的入射角（視線與面元法向夾角，90°＝貼著面看）與俯角。
GPU；只讀。用法：python tools/probe/hidden_view_probe_1107.py --block 3 --models 標籤=ckpt ...
"""
import argparse, glob, sys, time
import numpy as np, torch
from scipy.ndimage import uniform_filter
sys.path.insert(0, ".")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.general_utils import build_rotation
from internal.utils.colmap import read_images_binary, read_cameras_binary, read_points3D_binary
from internal.cameras.cameras import Cameras
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--margin_px", type=float, default=5.0)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda")
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
bb0 = pc.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.); bb1 = pc.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.05)
Rt = P["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double() @ Rt.T)[:, :2])[B].numpy()

# 地面格／建築格（同 hidden_probe_1107）
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
Pp = np.stack([q.xyz for q in pts.values()]); ok = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = Pp[ok & inb(Pp, bb1)]; G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u, f, c = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u] = S[order, 2][f + c // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03); bldg = np.isfinite(med) & ~ground
gz = np.where(ground, med, np.nan); num = uniform_filter(np.nan_to_num(gz), size=9); den = uniform_filter(np.isfinite(gz).astype(float), size=9)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nanmedian(gz)))


def regions(x):
    cc = np.clip(np.floor(x[:, :2] / G).astype(int) - o0, 0, np.array(shp) - 1)
    h = x[:, 2] - gfill[cc[:, 0], cc[:, 1]]
    g = ground[cc[:, 0], cc[:, 1]]; b = bldg[cc[:, 0], cc[:, 1]]
    R = {"地下": g & (h < -0.005), "地面": g & (np.abs(h) <= 0.005), "建築範圍內": b & (h > 0.03)}
    R["其他"] = ~(R["地下"] | R["地面"] | R["建築範圍內"])
    return R


names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}; cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names = [n for n in names if n in ims]


def mk(nm):
    im = ims[nm]; cc = cams[im.camera_id]; W, H = round(cc.width / 1.2), round(cc.height / 1.2); s = W / cc.width
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(im.qvec2rotmat()), T=t(im.tvec), fx=t([cc.params[0] * s])[0], fy=t([cc.params[1] * s])[0], cx=t([cc.params[2] * s])[0],
                   cy=t([cc.params[3] * s])[0], width=torch.tensor([W], dtype=torch.int16), height=torch.tensor([H], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1), distortion_params=None,
                   camera_type=torch.zeros(1, dtype=torch.int8))[0]


CAMS = [mk(n) for n in names]
print(f"b{B}｜{len(CAMS)} 台訓練相機（全部）｜被擋＝中心像素 α > 0.95 且比表面深 > {a.margin_px:g} px 足跡｜看得到＝該台覆蓋像素平均 T > 0.3｜只看盒內 o > 0.5", flush=True)
rnd = None
for spec in a.models:                                   # ⚠ 第一個要是我方模型：全部用它的 renderer（CityGS 的 ckpt 帶的是上游 renderer、沒有 record_T）
    lab, pat = spec.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    model, rnd_m, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
    if rnd is None:
        rnd = rnd_m
        for fl in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
            if hasattr(rnd, fl):
                setattr(rnd, fl, False)
    rnd.alpha_max_999 = bool(getattr(rnd_m, "alpha_max_999", False))   # 用該模型訓練時的 alpha 上限
    model.to(dev)
    X = model.get_xyz.detach().float(); N = X.shape[0]; Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    F = torch.zeros(N, device=dev); Hc = torch.zeros(N, device=dev); n30 = torch.zeros(N, device=dev); n05 = torch.zeros(N, device=dev)
    Tm = torch.zeros(N, device=dev); Ka = torch.full((N,), -1, dtype=torch.long, device=dev); Cc = torch.zeros(len(CAMS), 3, device=dev)
    bg = torch.zeros(3, device=dev); t0 = time.perf_counter()
    for k, cam in enumerate(CAMS):
        cd = cam.to_device(dev)
        with torch.no_grad():
            o = rnd(cd, model, bg_color=bg)
            al = o["rend_alpha"].reshape(o["rend_alpha"].shape[-2:]); zs = o["surf_depth"].reshape(al.shape); Hh, Ww = al.shape
            W2C = cd.world_to_camera.float(); Cc[k] = torch.linalg.inv(W2C)[3, :3]
            p = Xh @ W2C; zc = p[:, 2]
            uu = (float(cd.fx) * p[:, 0] / zc.clamp_min(1e-6) + float(cd.cx)).floor().long(); vv = (float(cd.fy) * p[:, 1] / zc.clamp_min(1e-6) + float(cd.cy)).floor().long()
            inn = (zc > 1e-3) & (uu >= 0) & (uu < Ww) & (vv >= 0) & (vv < Hh)
            uu, vv = uu.clamp(0, Ww - 1), vv.clamp(0, Hh - 1)
            F += inn.float()
            Hc += (inn & (al[vv, uu] > 0.95) & (zs[vv, uu] > 0) & (zc > zs[vv, uu] * (1 + a.margin_px / float(cd.fx)))).float()
            tT = rnd(cd, model, bg_color=bg, record_transmittance=True, record_T=True).float().clamp_min(0)
            n30 += (tT > 0.3).float(); n05 += (tT > 0.05).float()
            up = tT > Tm; Ka[up] = k; Tm = torch.where(up, tT, Tm)
    torch.cuda.synchronize(); tj = time.perf_counter() - t0
    if float(Tm.max()) > 1.0001:
        raise RuntimeError("Tmax > 1 => record_T 沒生效（光柵器沒重編？）")
    nrm = build_rotation(model.get_rotations().detach().float())[:, :, 2]
    has = Ka >= 0
    d = X - Cc[Ka.clamp_min(0)]; d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
    inc = torch.rad2deg(torch.acos((d * nrm).sum(1).abs().clamp(0, 1)))                  # 入射角：0°＝正對面元、90°＝貼著面看
    dep = torch.rad2deg(torch.asin((-d[:, 2]).clamp(-1, 1)))                              # 俯角（世界 +z 朝上）
    op = (model.get_opacities().detach().reshape(-1) > 0.5)
    F, Hc, n30, n05, Tm, inc, dep, has, op = (t.cpu().numpy() for t in (F, Hc, n30, n05, Tm, inc, dep, has, op))
    Xn = X.cpu().numpy().astype(np.float64); IN = inb(Xn, bb0); R = regions(Xn)
    base = IN & op & (F >= 3)
    grp = {"中心全部被擋": base & (Hc == F), "中心 50～99% 被擋": base & (Hc >= 0.5 * F) & (Hc < F), "中心多半看得到（< 50% 被擋）": base & (Hc < 0.5 * F)}
    print(f"\n[{lab}] alpha 上限 {0.999 if rnd.alpha_max_999 else 0.99}｜{N:,} 顆｜盒內不透明且畫面內 >= 3 台：{int(base.sum()):,}｜兩次渲染 x {len(CAMS)} 台 {tj:.1f} s", flush=True)
    print(f"   {'分區':6} {'分組':22} {'顆數':>8} {'看得到台數 q50/q90':>18} {'看得到/畫面內 q50':>17} {'<= 3 台看得到':>13} {'Tmax q50':>9} {'Tmax 那台入射角 q50/q90':>22} {'> 75°':>7} {'俯角 q50':>8}")
    for rn, rm in R.items():
        for gn, gm in grp.items():
            m = rm & gm & has
            if m.sum() < 20:
                continue
            fr = n30[m] / np.maximum(F[m], 1)
            print(f"   {rn:6} {gn:22} {int(m.sum()):>8,} {np.percentile(n30[m], 50):>8.0f}/{np.percentile(n30[m], 90):<9.0f} {np.percentile(fr, 50):>17.3f} {100 * (n30[m] <= 3).mean():>12.1f}% "
                  f"{np.percentile(Tm[m], 50):>9.2f} {np.percentile(inc[m], 50):>10.0f}°/{np.percentile(inc[m], 90):<10.0f}° {100 * (inc[m] > 75).mean():>6.1f}% {np.percentile(dep[m], 50):>7.0f}°", flush=True)
    del model; torch.cuda.empty_cache()
print("DONE")
