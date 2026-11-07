"""11-07「看得到才往 1 推」的二元熵（vis gate）定門檻用：訓練中累加的 V（Σ T·α）與 C（實際算過的像素數）在一個窗口後的
VS＝V／(C·o)＝覆蓋像素上平均 T·G（G＝面元高斯權重）。全看得到的不透明面元 VS 約 0.1～0.2；躲在一層表面後面的約 0.001～0.01。
這裡用 899 台訓練相機離線算同一個量（record_transmittance＋record_coverage，與訓練中 train_record 同一種計數），
依「中心被擋比例」與分區分組，看各組 VS 的分位、以及各候選門檻會把多少顆推到閘門外。GPU；只讀。
用法：python tools/probe/vis_score_probe_1107.py --block 3 --models 標籤=ckpt ...（第一個要是我方模型）
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
TAUS = [0.002, 0.005, 0.01, 0.02, 0.05]
print(f"b{B}｜{len(CAMS)} 台訓練相機｜VS＝ΣT·α／(ΣC·o)｜分組同 hidden_view_probe（中心像素 α > 0.95 且比表面深 > {a.margin_px:g} px 足跡＝被擋）｜只看盒內 o > 0.5", flush=True)
rnd = None
for spec in a.models:
    lab, pat = spec.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    model, rnd_m, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
    if rnd is None:
        rnd = rnd_m
        for fl in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
            if hasattr(rnd, fl):
                setattr(rnd, fl, False)
    rnd.alpha_max_999 = bool(getattr(rnd_m, "alpha_max_999", False))
    model.to(dev)
    X = model.get_xyz.detach().float(); N = X.shape[0]; Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    F = torch.zeros(N, device=dev); Hc = torch.zeros(N, device=dev); SV = torch.zeros(N, device=dev, dtype=torch.float64); SC = torch.zeros_like(SV)
    bg = torch.zeros(3, device=dev); t0 = time.perf_counter()
    for cam in CAMS:
        cd = cam.to_device(dev)
        with torch.no_grad():
            o = rnd(cd, model, bg_color=bg)
            al = o["rend_alpha"].reshape(o["rend_alpha"].shape[-2:]); zs = o["surf_depth"].reshape(al.shape); Hh, Ww = al.shape
            p = Xh @ cd.world_to_camera.float(); zc = p[:, 2]
            uu = (float(cd.fx) * p[:, 0] / zc.clamp_min(1e-6) + float(cd.cx)).floor().long(); vv = (float(cd.fy) * p[:, 1] / zc.clamp_min(1e-6) + float(cd.cy)).floor().long()
            inn = (zc > 1e-3) & (uu >= 0) & (uu < Ww) & (vv >= 0) & (vv < Hh)
            uu, vv = uu.clamp(0, Ww - 1), vv.clamp(0, Hh - 1)
            F += inn.float()
            Hc += (inn & (al[vv, uu] > 0.95) & (zs[vv, uu] > 0) & (zc > zs[vv, uu] * (1 + a.margin_px / float(cd.fx)))).float()
            tr, cn = rnd(cd, model, bg_color=bg, record_transmittance=True, record_coverage=True)
            SV += tr.double().clamp_min(0) * cn.double(); SC += cn.double()
    torch.cuda.synchronize(); tj = time.perf_counter() - t0
    oo = model.get_opacities().detach().reshape(-1).double()
    VS = torch.where(SC > 0, SV / (SC * oo).clamp_min(1e-12), torch.full_like(SV, float("nan")))
    op = oo > 0.5
    F, Hc, VS, op = (t.cpu().numpy() for t in (F, Hc, VS, op))
    Xn = X.cpu().numpy().astype(np.float64); IN = inb(Xn, bb0); R = regions(Xn)
    base = IN & op & (F >= 3)
    grp = {"中心全部被擋": base & (Hc == F), "中心 50～99% 被擋": base & (Hc >= 0.5 * F) & (Hc < F), "中心多半看得到": base & (Hc < 0.5 * F)}
    ok = base & np.isfinite(VS)
    print(f"\n[{lab}] {N:,} 顆｜盒內不透明且畫面內 >= 3 台 {int(base.sum()):,}（C＝0 的 {int((base & ~np.isfinite(VS)).sum()):,}）｜{tj:.1f} s"
          f"｜全體 VS 分位 q10/50/90 {np.percentile(VS[ok], [10, 50, 90]).round(4).tolist()}", flush=True)
    print(f"   {'分區':6} {'分組':16} {'顆數':>8}  {'VS q10/q50/q90':>24}  " + "  ".join(f"VS<{t:g}" for t in TAUS))
    for rn, rm in list(R.items()) + [("全部", np.ones_like(base))]:
        for gn, gm in grp.items():
            m = rm & gm & ok
            if m.sum() < 20:
                continue
            q = np.percentile(VS[m], [10, 50, 90])
            print(f"   {rn:6} {gn:16} {int(m.sum()):>8,}  {q[0]:>7.4f}/{q[1]:>7.4f}/{q[2]:>7.4f}  " + "  ".join(f"{100 * (VS[m] < t).mean():>7.1f}%" for t in TAUS), flush=True)
    print(f"   => 各門檻推到閘門外的盒內不透明顆粒（佔全部 {int(ok.sum()):,}）：" + "、".join(f"{t:g}：{100 * (VS[ok] < t).mean():.1f}%" for t in TAUS), flush=True)
    del model; torch.cuda.empty_cache()
print("DONE")
