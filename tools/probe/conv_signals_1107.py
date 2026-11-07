"""11-07 步數方向 A 第 1 步（使用者「照你說的作」＋另一個 session 的修正）：找「訓練中拿得到、不靠 GT」又跟得上幾何的收斂訊號。
驗法：同一步數、不同配方（F1 0.32～0.38）＋同配方重跑（雜訊）——訊號要能排出跟 F1／低角度破洞一樣的順序才算有鑑別力；
沿一條訓練曲線找相關沒有用（幾乎所有量都隨步數單調變化）。GPU 渲染＋CPU；cwd＝repo 根；只讀。
訊號（全部只用訓練相機、SfM、模型自己；誤差單位＝像素足跡 depth/fx，沒有公尺常數）：
  mv     多視角深度一致：錨點訓練相機 i 的實心像素（alpha > 0.5、世界點在本塊盒內）用 i 的深度投到最近的同向訓練相機 j，
         |d_j − z_proj| 以像素足跡計；報中位與 < 5 px 的比例。和 SfM 機制無關（避開循環）。
  vlow   虛擬低角度透空：錨點相機的光軸打到地面的點 P，同方位、仰角 20°、距離減半放一台虛擬相機；本塊可靠 SfM 投進去（8 px 格取最近），
         格中心 alpha < 0.5（透空）或渲染深度 > 1.3 倍 SfM 深度（破洞）的比例＝低角度破洞的無 GT 版。
  sfm    SfM 一致：訓練視角裡本塊可靠 SfM（4 px 格取最近）對渲染深度，< 5 px 足跡的比例。⚠ 循環：視線一致性／不准往上本身把顆粒往 SfM 推。
  tsee   訓練視角透空：SfM 格中心 alpha < 0.5 的比例（另一個 session 問：空拍訓練視角透空本來就少，跟得上低角度破洞嗎）。
  nc     深度法向一致：實心像素 1 − rend_normal·surf_normal 的平均（法向 loss 本身會壓它）。
  tpsnr  訓練視角 PSNR（實心且盒內像素）：反例對照（照片擬合，不是幾何）。
用法：python tools/probe/conv_signals_1107.py --block 3 --models 標籤=ckpt ... --out <json>
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, ".")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_images_binary, read_cameras_binary, read_points3D_binary
from internal.cameras.cameras import Cameras
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--nanchor", type=int, default=32)
ap.add_argument("--elev", type=float, default=20.0, help="虛擬低角度相機的仰角（度）")
ap.add_argument("--out", required=True)
a = ap.parse_args()
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda")
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
coords = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
bb0 = coords.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.0)
RTm = P["extra_data"]["rotation_transform"][:3, :3].double()
inbox = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb0, coordinates=(torch.as_tensor(x).double() @ RTm.T)[:, :2])[B]

names_all = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
names_all = [n for n in names_all if n in ims]


def mk(R, T, fx, fy, cx, cy, W, H):
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(R), T=t(T), fx=t([fx])[0], fy=t([fy])[0], cx=t([cx])[0], cy=t([cy])[0],
                   width=torch.tensor([W], dtype=torch.int16), height=torch.tensor([H], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1),
                   distortion_params=None, camera_type=torch.zeros(1, dtype=torch.int8))[0]


def cam_of(nm):
    im = ims[nm]; c = cams[im.camera_id]; W, H = round(c.width / 1.2), round(c.height / 1.2); s = W / c.width
    R = im.qvec2rotmat(); T = im.tvec
    return dict(R=R, T=T, C=-R.T @ T, fx=c.params[0] * s, fy=c.params[1] * s, cx=c.params[2] * s, cy=c.params[3] * s, W=W, H=H)


CI = {nm: cam_of(nm) for nm in names_all}
Cs = np.stack([CI[n]["C"] for n in names_all]); Zs = np.stack([CI[n]["R"][2] for n in names_all])   # 光軸（世界）
anchors = [names_all[i] for i in np.linspace(0, len(names_all) - 1, a.nanchor).round().astype(int)]
pairs = []
for nm in anchors:
    i = names_all.index(nm)
    d = np.linalg.norm(Cs - Cs[i], axis=1); cosang = Zs @ Zs[i]
    d[i] = np.inf; d[cosang < np.cos(np.radians(20))] = np.inf
    j = int(np.argmin(d)); pairs.append((nm, names_all[j] if np.isfinite(d[j]) else None))

# 本塊可靠 SfM
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
S = np.stack([q.xyz for q in pts.values() if len(q.image_ids) >= 3 and q.error < 1.5]); del pts
S = S[inbox(S).numpy()]
zg = float(np.percentile(S[:, 2], 5))                                  # 地面高度（本塊可靠 SfM 的 5 分位）
St = torch.from_numpy(S).float()

# 虛擬低角度相機
vlow = []
for nm in anchors:
    c = CI[nm]; z = c["R"][2]
    if z[2] >= -1e-3:
        continue
    t = (zg - c["C"][2]) / z[2]; Pg = c["C"] + t * z
    hz = c["C"][:2] - Pg[:2]; hz = hz / max(np.linalg.norm(hz), 1e-9); dist = 0.5 * np.linalg.norm(c["C"] - Pg)
    el = np.radians(a.elev)
    Cv = np.array([Pg[0] + dist * np.cos(el) * hz[0], Pg[1] + dist * np.cos(el) * hz[1], Pg[2] + dist * np.sin(el)])
    zz = Pg - Cv; zz /= np.linalg.norm(zz); xx = np.cross(zz, [0, 0, 1.0]); xx /= np.linalg.norm(xx); yy = np.cross(zz, xx)
    R = np.stack([xx, yy, zz])
    vlow.append(dict(R=R, T=-R @ Cv, C=Cv, fx=c["fx"], fy=c["fy"], cx=c["W"] / 2, cy=c["H"] / 2, W=c["W"], H=c["H"]))
print(f"b{B}｜本塊訓練相機 {len(names_all)}｜錨點 {len(anchors)}（有同向鄰居 {sum(p[1] is not None for p in pairs)}）｜虛擬低角度 {len(vlow)}（仰角 {a.elev:g}°）"
      f"｜本塊可靠 SfM {len(S):,}｜地面 z {zg * 100:.1f} m", flush=True)


def sfm_cells(c, cell):
    """SfM 投到相機 c，每 cell px 格取最近的一個：回傳 (u, v, z) 張量。"""
    R = torch.from_numpy(c["R"]).float(); T = torch.from_numpy(c["T"]).float()
    Xc = St @ R.T + T; z = Xc[:, 2]
    u = c["fx"] * Xc[:, 0] / z.clamp_min(1e-6) + c["cx"]; v = c["fy"] * Xc[:, 1] / z.clamp_min(1e-6) + c["cy"]
    ok = (z > 1e-4) & (u >= 0) & (u < c["W"]) & (v >= 0) & (v < c["H"])
    u, v, z = u[ok], v[ok], z[ok]
    cid = (v // cell).long() * ((c["W"] + cell - 1) // cell) + (u // cell).long()
    order = torch.argsort(z); cid_o = cid[order]
    srt, idx = torch.sort(cid_o, stable=True); keep = torch.ones_like(srt, dtype=torch.bool); keep[1:] = srt[1:] != srt[:-1]
    sel = order[idx[keep]]
    return u[sel], v[sel], z[sel]


rnd0 = None
OUT = {}
TO = lambda c: mk(c["R"], c["T"], c["fx"], c["fy"], c["cx"], c["cy"], c["W"], c["H"])
for spec in a.models:
    lab, pat = spec.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    model, rnd, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
    pr = {k: v.detach().float() for k, v in model.properties.items()}
    if rnd0 is None:
        rnd0 = rnd
        for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
            if hasattr(rnd0, f):
                setattr(rnd0, f, False)
    k = inbox(pr["means"]); model.properties = {kk: v[k] for kk, v in pr.items()}; model.to(dev); del pr

    cache = {}
    def render(c, key=None):
        if key is not None and key in cache:
            return cache[key]
        with torch.no_grad():
            o = rnd0(TO(c).to_device(dev), model, bg_color=torch.zeros(3, device=dev))
        r = dict(img=o["render"].clamp(0, 1).cpu(), dep=o["surf_depth"][0].float().cpu(), al=o["rend_alpha"][0].float().cpu(),
                 nr=o["rend_normal"].float().cpu() if "rend_normal" in o else None, ns=o["surf_normal"].float().cpu() if "surf_normal" in o else None)
        if key is not None:
            cache[key] = r
        return r

    mv_e, sfm_e, tsee, nc_s, nc_n, se, sn = [], [], [0, 0], 0.0, 0, 0.0, 0
    for nm, nb in pairs:
        c = CI[nm]; r = render(c, nm)
        H, W = r["dep"].shape
        uu, vv = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5, indexing="xy")
        ray = torch.stack([(uu - c["cx"]) / c["fx"], (vv - c["cy"]) / c["fy"], torch.ones_like(uu)], -1)
        R = torch.from_numpy(c["R"]).float(); T = torch.from_numpy(c["T"]).float()
        Xw = (ray * r["dep"][..., None] - T) @ R
        sol = (r["al"] > 0.5) & (r["dep"] > 0)
        ib = torch.zeros_like(sol); ib[sol] = inbox(Xw[sol].reshape(-1, 3))
        # 訓練 PSNR（反例對照）
        gt = Image.open(f"{D}/input/{nm}").convert("RGB")
        if gt.size != (W, H):
            gt = gt.resize((W, H), Image.LANCZOS)
        gt = torch.from_numpy(np.array(gt, np.float32) / 255.).permute(2, 0, 1)
        e2 = ((r["img"] - gt) ** 2).mean(0)
        se += float(e2[ib].sum()); sn += int(ib.sum())
        # 深度法向一致
        if r["nr"] is not None and r["ns"] is not None:
            d = 1 - (r["nr"] * r["ns"]).sum(0)
            nc_s += float(d[ib].sum()); nc_n += int(ib.sum())
        # SfM 一致＋訓練視角透空
        u, v, z = sfm_cells(c, 4)
        ui, vi = u.long().clamp(0, W - 1), v.long().clamp(0, H - 1)
        al_s = r["al"][vi, ui]; dp_s = r["dep"][vi, ui]
        tsee[0] += int((al_s < 0.5).sum()); tsee[1] += int(len(z))
        okd = al_s >= 0.5
        sfm_e.append(((dp_s[okd] - z[okd]).abs() / (z[okd] / c["fx"])).numpy())
        # 多視角深度一致
        if nb is not None:
            cj = CI[nb]; rj = render(cj, nb)
            Rj = torch.from_numpy(cj["R"]).float(); Tj = torch.from_numpy(cj["T"]).float()
            Xs = Xw[ib]
            if len(Xs) > 200000:
                Xs = Xs[torch.randperm(len(Xs))[:200000]]
            Xc = Xs @ Rj.T + Tj; zp = Xc[:, 2]
            uj = cj["fx"] * Xc[:, 0] / zp.clamp_min(1e-6) + cj["cx"]; vj = cj["fy"] * Xc[:, 1] / zp.clamp_min(1e-6) + cj["cy"]
            ok = (zp > 1e-4) & (uj >= 0) & (uj < cj["W"]) & (vj >= 0) & (vj < cj["H"])
            uj, vj, zp = uj[ok].long(), vj[ok].long(), zp[ok]
            okj = rj["al"][vj, uj] > 0.5
            mv_e.append(((rj["dep"][vj, uj][okj] - zp[okj]).abs() / (zp[okj] / cj["fx"])).numpy())
    # 虛擬低角度
    vs = [0, 0, 0]
    for c in vlow:
        r = render(c)
        H, W = r["dep"].shape
        u, v, z = sfm_cells(c, 8)
        ui, vi = u.long().clamp(0, W - 1), v.long().clamp(0, H - 1)
        al_s = r["al"][vi, ui]; dp_s = r["dep"][vi, ui]
        vs[0] += int((al_s < 0.5).sum()); vs[1] += int(((al_s >= 0.5) & (dp_s > 1.3 * z)).sum()); vs[2] += int(len(z))
    mv = np.concatenate(mv_e) if mv_e else np.zeros(0); sf = np.concatenate(sfm_e) if sfm_e else np.zeros(0)
    res = dict(mv_med=float(np.median(mv)), mv_lt5=float((mv < 5).mean()), sfm_med=float(np.median(sf)), sfm_lt5=float((sf < 5).mean()),
               tsee=tsee[0] / max(tsee[1], 1), vlow_see=vs[0] / max(vs[2], 1), vlow_hole=vs[1] / max(vs[2], 1),
               nc=nc_s / max(nc_n, 1), tpsnr=float(-10 * np.log10(max(se / max(sn, 1), 1e-12))))
    OUT[lab] = res
    print(f"{lab:18} 多視角 中位 {res['mv_med']:6.2f} px <5px {100 * res['mv_lt5']:5.1f}%｜SfM 中位 {res['sfm_med']:6.2f} px <5px {100 * res['sfm_lt5']:5.1f}%"
          f"｜訓練透空 {100 * res['tsee']:5.2f}%｜虛擬低角度 透空 {100 * res['vlow_see']:5.2f}% 破洞 {100 * res['vlow_hole']:5.2f}%"
          f"｜深度法向 {res['nc']:.4f}｜訓練 PSNR {res['tpsnr']:.2f}", flush=True)
    del model, cache; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(OUT, open(a.out, "w"), ensure_ascii=False, indent=1)
print(f"DONE -> {a.out}")
