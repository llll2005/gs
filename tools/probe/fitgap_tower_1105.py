"""11-05 使用者「我們是不是高度擬合視角、甚至過擬合，不在視角中就崩」＋「藍色高樓的診斷」（GPU 渲染；cwd＝repo 根；只讀）。
① 擬合 vs 外插：同一個塊內模型，在「訓練視角」與「MatrixCity test 視角（held-out，同高度的空拍）」上，依高度帶算 PSNR。
   像素＝參考模型（CityGS）的渲染表面落在本塊盒內；高度＝該表面的世界 z（各模型共用同一把尺）。
   另算「棟 3（藍色玻璃塔）」的像素。訓練 PSNR − test PSNR＝擬合缺口；缺口大＝記住訓練視角。
② 低角度外插：環繞棟 3 的 8 台相機（高 Hc、距離 Rc、看塔身中段），算本模型在 CityGS 實心像素上的
   「透空」（alpha<0.5）與「破洞」（深度比 CityGS 深 30% 以上）比例，並存對照圖。
用法：python fitgap_tower_1105.py --models 標籤=ckpt ...（第一個必須是 CityGS＝參考）--out <資料夾>
"""
import argparse, glob, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_images_binary, read_cameras_binary
from internal.cameras.cameras import Cameras
from internal.utils.citygs_partitioning_utils import CityGSPartitioning, PartitionCoordinates
from eval_official_test import load_test_cameras, block_bounds

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, default=3)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--ntrain", type=int, default=80)
ap.add_argument("--ntest", type=int, default=0, help="0＝全部")
ap.add_argument("--tower", type=float, nargs=2, default=[3.30, -2.82], help="棟 3 的中心 x y（tower_probe）")
ap.add_argument("--tower_r", type=float, default=0.25, help="棟 3 足跡半徑（場景單位）")
ap.add_argument("--Hc", type=float, default=1.6)
ap.add_argument("--Rc", type=float, default=2.2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda"); M = 100.0
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
coords = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
boxes = coords.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.0)
RTm = P["extra_data"]["rotation_transform"][:3, :3].double()
inbox = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=(x.double() @ RTm.T)[:, :2])[B]

# 訓練視角（本塊分區清單，均勻抽）
names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
names = [names[i] for i in np.linspace(0, len(names) - 1, a.ntrain).round().astype(int)]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
def mk(R, T, fx, fy, cx, cy, W, H):
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(R), T=t(T), fx=t([fx])[0], fy=t([fy])[0], cx=t([cx])[0], cy=t([cy])[0],
                   width=torch.tensor([W], dtype=torch.int16), height=torch.tensor([H], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1),
                   distortion_params=None, camera_type=torch.zeros(1, dtype=torch.int8))[0]
views = []          # (種類, 名稱, Cameras, GT 路徑或 None)
for nm in names:
    im = ims[nm]; c = cams[im.camera_id]; W, H = round(c.width / 1.2), round(c.height / 1.2); s = W / c.width
    views.append(("train", nm, mk(im.qvec2rotmat(), im.tvec, c.params[0] * s, c.params[1] * s, c.params[2] * s, c.params[3] * s, W, H), f"{D}/input/{nm}"))
TD = "data/matrix_city/aerial/test/block_all_test_official2"
tn, tc = load_test_cameras(TD, 1.2)
lo, hi, _ = block_bounds(D, B, [4, 4])
cen = np.array([(-tc.R[i].numpy().T @ tc.T[i].numpy()) for i in range(len(tn))])
sel = [i for i in range(len(tn)) if np.all(cen[i] >= lo) and np.all(cen[i] <= hi)]
if a.ntest:
    sel = [sel[j] for j in np.linspace(0, len(sel) - 1, a.ntest).round().astype(int)]
for i in sel:
    views.append(("test", tn[i], tc[i], f"{TD}/images_1.2/{tn[i]}"))
# 低角度環繞棟 3
c0 = cams[ims[names[0]].camera_id]; W0, H0 = round(c0.width / 1.2), round(c0.height / 1.2); f0 = c0.params[0] * W0 / c0.width
tgt = np.array([a.tower[0], a.tower[1], 2.0])
for k in range(8):
    th = 2 * np.pi * k / 8; C = np.array([a.tower[0] + a.Rc * np.cos(th), a.tower[1] + a.Rc * np.sin(th), a.Hc])
    z = tgt - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z]); views.append(("low", f"low{k}", mk(R, -R @ C, f0, f0, W0 / 2, H0 / 2, W0, H0), None))
print(f"b{B}｜訓練視角 {sum(v[0] == 'train' for v in views)}｜test 視角 {sum(v[0] == 'test' for v in views)}｜低角度 {sum(v[0] == 'low' for v in views)}"
      f"（高 {a.Hc * M:.0f} m、距棟 3 {a.Rc * M:.0f} m）", flush=True)

def gt_of(p, W, H):
    im = Image.open(p).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    return torch.from_numpy(np.array(im, np.float32) / 255.).permute(2, 0, 1)

BANDS = [-9, 0.5, 1.0, 1.5, 2.0, 3.0, 9]
BN = ["<50m", "50~100", "100~150", "150~200", "200~300", ">300m"]
REF = {}            # 參考（CityGS）每個視角：世界 z、塊內、棟 3、alpha、深度
res = {}
rnd0 = None
for mi, spec in enumerate(a.models):
    lab, pat = spec.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    model, rnd, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
    pr = {k: v.detach().float() for k, v in model.properties.items()}
    if rnd0 is None:
        rnd0 = rnd
        for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
            if hasattr(rnd0, f):
                setattr(rnd0, f, False)
    k = inbox(pr["means"]); model.properties = {kk: v[k] for kk, v in pr.items()}; model.to(dev); del pr
    se = {}; low = []
    for kind, nm, cam, gp in views:
        with torch.no_grad():
            o = rnd0(cam.to_device(dev), model, bg_color=torch.zeros(3, device=dev))
        img = o["render"].clamp(0, 1).cpu(); dep = o["surf_depth"][0].float().cpu(); al = o["rend_alpha"][0].float().cpu()
        Hh, Ww = dep.shape
        if mi == 0:
            uu, vv = torch.meshgrid(torch.arange(Ww) + 0.5, torch.arange(Hh) + 0.5, indexing="xy")
            ray = torch.stack([(uu - float(cam.cx)) / float(cam.fx), (vv - float(cam.cy)) / float(cam.fy), torch.ones_like(uu)], -1)
            Rw = cam.R.cpu().float(); Tw = cam.T.cpu().float()
            Xw = (ray * dep[..., None] - Tw) @ Rw
            ok = (al > 0.95) & (dep > 0)
            ib = torch.zeros_like(ok); ib[ok] = inbox(Xw[ok].reshape(-1, 3))
            tw = ok & (((Xw[..., 0] - a.tower[0]) ** 2 + (Xw[..., 1] - a.tower[1]) ** 2) < a.tower_r ** 2) & (Xw[..., 2] > 0.5)
            band = torch.bucketize(Xw[..., 2].contiguous(), torch.tensor(BANDS[1:-1]))
            REF[nm] = (ib, tw, band, al, dep)
        ib, tw, band, ral, rdep = REF[nm]
        if kind == "low":
            sol = tw                                    # CityGS 在棟 3 的實心像素
            n = int(sol.sum())
            if n:
                low.append((n, int((sol & (al < 0.5)).sum()), int((sol & (al >= 0.5) & (dep > 1.3 * rdep)).sum())))
            if nm in ("low0", "low2", "low4", "low6"):
                Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f"{a.out}/{nm}_{lab}.jpg", quality=88)
            continue
        gt = gt_of(gp, Ww, Hh)
        err = ((img - gt) ** 2).mean(0)
        for key, msk in (("all", ib), ("tower", tw)):
            for bi in range(len(BN)):
                mm = msk & (band == bi)
                if mm.any():
                    s = se.setdefault((kind, key, bi), [0.0, 0])
                    s[0] += float(err[mm].sum()); s[1] += int(mm.sum())
    res[lab] = (se, low)
    del model; torch.cuda.empty_cache()
    print(f"[{lab}] 完成", flush=True)

ps = lambda s: -10 * np.log10(max(s[0] / max(s[1], 1), 1e-12)) if s and s[1] > 500 else float("nan")
for key, title in (("all", "塊內全部像素"), ("tower", "棟 3（藍色玻璃塔）像素")):
    print(f"\n=== ① {title}：訓練 PSNR／test PSNR（缺口）依高度帶（高度＝CityGS 表面） ===")
    print(f"{'模型':16}" + "".join(f"{b:>22}" for b in BN))
    for lab, (se, _) in res.items():
        row = ""
        for bi in range(len(BN)):
            tr, te = ps(se.get(("train", key, bi))), ps(se.get(("test", key, bi)))
            row += f"{(f'{tr:.2f}／{te:.2f}（{tr - te:+.2f}）' if np.isfinite(tr) and np.isfinite(te) else '—'):>22}"
        print(f"{lab:16}{row}")
    se0 = res[next(iter(res))][0]
    cnt = [f"{se0.get(('train', key, bi), [0, 0])[1] // 1000}k／{se0.get(('test', key, bi), [0, 0])[1] // 1000}k" for bi in range(len(BN))]
    print(f"{'像素數 訓練／test':16}" + "".join(f"{c:>22}" for c in cnt))
print(f"\n=== ② 低角度環繞棟 3（CityGS 實心像素上；8 台）：透空 alpha<0.5／破洞 深度比 CityGS 深 30% 以上 ===")
for lab, (_, low) in res.items():
    n = sum(x[0] for x in low); t = sum(x[1] for x in low); h = sum(x[2] for x in low)
    print(f"{lab:16} 參考像素 {n:>9,}｜透空 {100 * t / max(n, 1):5.1f}%｜破洞 {100 * h / max(n, 1):5.1f}%")
print(f"對照圖：{a.out}/low{{0,2,4,6}}_<模型>.jpg\nDONE fitgap_tower")
