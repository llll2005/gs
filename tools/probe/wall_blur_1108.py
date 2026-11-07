"""11-08 使用者「看 test 圖就會發現大樓牆面有些地方是糊的」＋「高處側面擬合失敗、樓頂正常」：牆面清晰度量測（GPU 渲染；cwd＝repo 根；只讀）。
清晰度＝16 px tile 上的高頻（灰階減掉 σ 1.5 px 高斯模糊）：
  能量比 = Σ hp_render² / Σ hp_GT²（< 0.5＝糊：細節不到一半）；相關 = hp_render 與 hp_GT 的相關（細節對不對得上；噪點會讓能量高但相關低）。
像素分類用參考模型（第一個，CityGS）的渲染：世界法向 |n_z| < 0.3＝立面、> 0.85＝平面（地面／樓頂）；高度＝表面世界 z；只取塊內、alpha > 0.95。
tile 有效：該類像素 ≥ 75%、塊內 ≥ 90%、GT 高頻 RMS ≥ --min_hp（無紋理的牆不算）。
視角：本塊訓練視角（均勻抽 --ntrain）＝擬合得好不好；MatrixCity test 塊內視角＝新視角。兩者都糊＝擬合問題，只有 test 糊＝外插問題。
輸出：各模型 × 視角種類 × 類別 × 高度帶的糊 tile 比例／能量比中位／相關中位／PSNR；玻璃塔（棟 3）立面另列；
     「只有我們糊、CityGS 不糊」的 test 例子（GT｜各模型）拼圖；逐 tile 的世界座標／法向／入射角／各模型能量比存 npz（之後 CPU 分析用）。
用法：python wall_blur_1108.py --models CityGS=ckpt 標籤=ckpt ... --out <資料夾>（第一個必須是 CityGS；第二個＝要找例子的主模型）
"""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
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
ap.add_argument("--ntrain", type=int, default=60)
ap.add_argument("--ntest", type=int, default=0, help="0＝全部")
ap.add_argument("--tile", type=int, default=16)
ap.add_argument("--min_hp", type=float, default=0.02, help="GT 高頻 RMS 門檻（灰階 0～1）")
ap.add_argument("--tower", type=float, nargs=2, default=[3.30, -2.82])
ap.add_argument("--tower_r", type=float, default=0.25)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
D = "data/matrix_city/aerial/train/block_all"; B = a.block; dev = torch.device("cuda"); M = 100.0; T = a.tile
P = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
coords = PartitionCoordinates(id=P["partition_coordinates"]["id"], xy=P["partition_coordinates"]["xy"])
boxes = coords.get_bounding_boxes(P["scene_config"]["partition_size"], enlarge=0.0)
RTm = P["extra_data"]["rotation_transform"][:3, :3].double()
inbox = lambda x: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=boxes, coordinates=(x.double() @ RTm.T)[:, :2])[B]

names = [l.strip() for l in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{B % 4:03d}_{B // 4:03d}.txt") if l.strip()]
names = [names[i] for i in np.linspace(0, len(names) - 1, a.ntrain).round().astype(int)]
ims = {im.name: im for im in read_images_binary(f"{D}/sparse/0/images.bin").values()}
cams = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
def mk(R, T_, fx, fy, cx, cy, W, H):
    t = lambda x: torch.tensor(np.array([x]), dtype=torch.float32)
    return Cameras(R=t(R), T=t(T_), fx=t([fx])[0], fy=t([fy])[0], cx=t([cx])[0], cy=t([cy])[0],
                   width=torch.tensor([W], dtype=torch.int16), height=torch.tensor([H], dtype=torch.int16),
                   appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1),
                   distortion_params=None, camera_type=torch.zeros(1, dtype=torch.int8))[0]
views = []
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
print(f"b{B}｜訓練視角 {sum(v[0] == 'train' for v in views)}｜test 視角 {sum(v[0] == 'test' for v in views)}｜tile {T} px｜GT 高頻 RMS ≥ {a.min_hp}", flush=True)

def gt_of(p, W, H):
    im = Image.open(p).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    return torch.from_numpy(np.array(im, np.float32) / 255.).permute(2, 0, 1)
_gk = torch.exp(-0.5 * (torch.arange(-4, 5, dtype=torch.float32) / 1.5) ** 2); _gk /= _gk.sum()
def hp(img):                                            # (3,H,W) => 灰階高頻 (H,W)
    g = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])[None, None]
    b = F.conv2d(F.pad(g, (4, 4, 0, 0), mode="replicate"), _gk.view(1, 1, 1, 9).to(g))
    b = F.conv2d(F.pad(b, (0, 0, 4, 4), mode="replicate"), _gk.view(1, 1, 9, 1).to(g))
    return (g - b)[0, 0]
tsum = lambda x: F.avg_pool2d(x[None, None].float(), T, T)[0, 0] * T * T          # tile 加總 (H/T, W/T)

BANDS = [0.5, 1.5, 2.5]; BN = ["<50m", "50~150", "150~250", ">250m"]
REF = {}; rows = {}; ex = []; recs = []; cgimg = {}
rnd0 = None
for mi, spec in enumerate(a.models):
    lab, pat = spec.split("=", 1); ck = sorted(glob.glob(pat))[-1]
    model, rnd, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=torch.device("cpu"), eval_mode=True)
    pr = {k: v.detach().float() for k, v in model.properties.items()}
    if rnd0 is None:
        rnd0 = rnd
        for f_ in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
            if hasattr(rnd0, f_):
                setattr(rnd0, f_, False)
    k = inbox(pr["means"]); model.properties = {kk: v[k] for kk, v in pr.items()}; model.to(dev); del pr
    for vi, (kind, nm, cam, gp) in enumerate(views):
        with torch.no_grad():
            o = rnd0(cam.to_device(dev), model, bg_color=torch.zeros(3, device=dev))
        img = o["render"].clamp(0, 1).float().cpu()
        Hh, Ww = img.shape[1:]; Ht, Wt = Hh // T, Ww // T
        if mi == 0:
            dep = o["surf_depth"][0].float().cpu(); al = o["rend_alpha"][0].float().cpu(); nw = o["rend_normal"].float().cpu()
            nw = nw / nw.norm(dim=0, keepdim=True).clamp_min(1e-6)
            uu, vv = torch.meshgrid(torch.arange(Ww) + 0.5, torch.arange(Hh) + 0.5, indexing="xy")
            ray = torch.stack([(uu - float(cam.cx)) / float(cam.fx), (vv - float(cam.cy)) / float(cam.fy), torch.ones_like(uu)], -1)
            Rw = cam.R.cpu().float(); Tw = cam.T.cpu().float()
            Xw = (ray * dep[..., None] - Tw) @ Rw
            ok = (al > 0.95) & (dep > 0)
            ib = torch.zeros_like(ok); ib[ok] = inbox(Xw[ok].reshape(-1, 3))
            nz = nw[2].abs()
            fac = ib & (nz < 0.3); flat = ib & (nz > 0.85)
            tw = ib & (((Xw[..., 0] - a.tower[0]) ** 2 + (Xw[..., 1] - a.tower[1]) ** 2) < a.tower_r ** 2) & (Xw[..., 2] > 0.5)
            gt = gt_of(gp, Ww, Hh); hg = hp(gt)
            Eg = tsum(hg ** 2)[:Ht, :Wt]
            frac = lambda m: (tsum(m) / (T * T))[:Ht, :Wt]
            fib, ffac, fflat, ftw = frac(ib), frac(fac), frac(flat), frac(tw)
            zt = (tsum(torch.where(ib, Xw[..., 2], torch.zeros_like(dep))) / tsum(ib).clamp_min(1))[:Ht, :Wt]
            valid = (fib >= 0.9) & (Eg / (T * T) >= a.min_hp ** 2)
            cls = torch.full((Ht, Wt), -1, dtype=torch.long)
            cls[valid & (fflat >= 0.75)] = 1; cls[valid & (ffac >= 0.75)] = 0
            band = torch.bucketize(zt.contiguous(), torch.tensor(BANDS))
            Ccen = (-Rw.T @ Tw).numpy()
            yy, xx = torch.nonzero(cls == 0, as_tuple=True)
            xc = []
            for y_, x_ in zip(yy.tolist(), xx.tolist()):          # 立面 tile：世界中心、法向、入射角、距離
                m_ = ib[y_ * T:(y_ + 1) * T, x_ * T:(x_ + 1) * T]
                p_ = Xw[y_ * T:(y_ + 1) * T, x_ * T:(x_ + 1) * T][m_].median(0).values.numpy()
                n_ = nw[:, y_ * T:(y_ + 1) * T, x_ * T:(x_ + 1) * T][:, m_].mean(1).numpy(); n_ /= max(np.linalg.norm(n_), 1e-6)
                d_ = p_ - Ccen; dist = np.linalg.norm(d_)
                xc.append((p_, n_, float(np.degrees(np.arccos(min(1.0, abs(d_ @ n_) / dist)))), dist, bool(ftw[y_, x_] >= 0.5)))
            REF[vi] = (cls, band, ftw >= 0.5, (gt * 255).round().to(torch.uint8), None, Eg, (yy, xx), xc, (ib, Ww, Hh))
        cls, band, istw, gt, _, Eg, (yy, xx), xc, (ib, Ww, Hh) = REF[vi][:9]
        gt = gt.float() / 255.; hg = hp(gt)
        hr = hp(img)
        Er = tsum(hr ** 2)[:Ht, :Wt]; Ex = tsum(hr * hg)[:Ht, :Wt]
        ratio = Er / Eg.clamp_min(1e-9); corr = Ex / (Er * Eg).sqrt().clamp_min(1e-9)
        se = tsum(((img - gt) ** 2).mean(0))[:Ht, :Wt]
        for c_, cn in ((0, "立面"), (1, "平面")):
            for bi in range(len(BN)):
                m_ = (cls == c_) & (band == bi)
                if m_.any():
                    r = rows.setdefault((lab, kind, cn, BN[bi]), [[], [], 0.0, 0]); r[0] += ratio[m_].tolist(); r[1] += corr[m_].tolist()
                    r[2] += float(se[m_].sum()); r[3] += int(m_.sum()) * T * T
        m_ = (cls == 0) & istw
        if m_.any():
            r = rows.setdefault((lab, kind, "玻璃塔立面", "全部"), [[], [], 0.0, 0]); r[0] += ratio[m_].tolist(); r[1] += corr[m_].tolist()
            r[2] += float(se[m_].sum()); r[3] += int(m_.sum()) * T * T
        rf = ratio[yy, xx].numpy(); cf = corr[yy, xx].numpy()
        for j in range(len(xc)):
            if mi == 0:
                recs.append([kind == "test", vi, *xc[j][0], *xc[j][1], xc[j][2], xc[j][3], xc[j][4], float(Eg[yy[j], xx[j]]) / (T * T)])
        if mi == 0:
            REF[vi] = REF[vi] + (rf, cf)
            if kind == "test":
                cgimg[vi] = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            REF[vi] = REF[vi] + (rf, cf)
            if mi == 1 and kind == "test":            # 只有主模型糊、CityGS 不糊的例子
                rcg = REF[vi][9]
                for j in np.nonzero((rf < 0.4) & (rcg > 0.8))[0]:
                    ex.append((float(Eg[yy[j], xx[j]]), vi, int(yy[j]), int(xx[j]), float(rf[j]), float(rcg[j])))
    del model; torch.cuda.empty_cache()
    print(f"[{lab}] 完成", flush=True)

labs = [s.split("=", 1)[0] for s in a.models]
med = lambda x: float(np.median(x)) if len(x) else float("nan")
for kind in ("train", "test"):
    print(f"\n=== {'訓練視角（擬合）' if kind == 'train' else 'MatrixCity test 視角（新視角）'}：糊 tile 比例（能量比 < 0.5）｜能量比中位｜相關中位｜PSNR｜tile 數 ===")
    for cn, bns in (("立面", BN), ("平面", BN), ("玻璃塔立面", ["全部"])):
        for b_ in bns:
            if (labs[0], kind, cn, b_) not in rows:
                continue
            print(f"[{cn} {b_}] tile {len(rows[(labs[0], kind, cn, b_)][0]):,}")
            for lab in labs:
                r = rows.get((lab, kind, cn, b_))
                if not r:
                    continue
                ra = np.array(r[0]); ps = -10 * np.log10(max(r[2] / max(r[3], 1), 1e-12))
                print(f"   {lab:14} 糊 {100 * np.mean(ra < 0.5):5.1f}%｜能量比 {med(ra):.2f}｜相關 {med(r[1]):.2f}｜PSNR {ps:5.2f}")
# 逐 tile 紀錄
R = np.array(recs, dtype=np.float64)
per = np.stack([np.concatenate([REF[vi][9 + 2 * mi] for vi in range(len(views))]) for mi in range(len(labs))], 1)
pcr = np.stack([np.concatenate([REF[vi][10 + 2 * mi] for vi in range(len(views))]) for mi in range(len(labs))], 1)
np.savez_compressed(f"{a.out}/facade_tiles.npz", rec=R, ratio=per, corr=pcr, labs=np.array(labs),
                    cols=np.array(["is_test", "view", "x", "y", "z", "nx", "ny", "nz", "incid_deg", "dist", "tower3", "gt_hp_ms"]))
print(f"\n逐 tile 紀錄：{a.out}/facade_tiles.npz（立面 tile {len(R):,}）")
# 例子拼圖：只有主模型糊、CityGS 不糊（依 GT 細節量排，同一視角最多 2 個）
ex.sort(key=lambda t: -t[0]); pick = []; cnt = {}
for e in ex:
    if cnt.get(e[1], 0) < 2:
        pick.append(e); cnt[e[1]] = cnt.get(e[1], 0) + 1
    if len(pick) >= 12:
        break
print(f"只有 {labs[1]} 糊、CityGS 不糊的 test 立面 tile：{len(ex):,} 個（佔 test 立面 tile {100 * len(ex) / max(int(R[:, 0].sum()), 1):.1f}%）")
if pick:
    C = 128; pad = 22
    try:
        font = ImageFont.truetype("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
    # 第二輪：只渲染被選到的視角（主模型；CityGS 用存下的）
    lab1, pat1 = a.models[1].split("=", 1)
    m1, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(sorted(glob.glob(pat1))[-1], device=torch.device("cpu"), eval_mode=True)
    pr = {k: v.detach().float() for k, v in m1.properties.items()}; k = inbox(pr["means"]); m1.properties = {kk: v[k] for kk, v in pr.items()}; m1.to(dev); del pr
    canvas = Image.new("RGB", (3 * (C + 4) + 4, len(pick) * (C + pad) + pad), (255, 255, 255)); d = ImageDraw.Draw(canvas)
    for ci_, t_ in enumerate(["GT", lab1, labs[0]]):
        d.text((4 + ci_ * (C + 4), 2), t_, fill=(0, 0, 0), font=font)
    for ri, (_, vi, ty, tx, r1, rc) in enumerate(pick):
        kind, nm, cam, gp = views[vi]
        with torch.no_grad():
            im1 = (rnd0(cam.to_device(dev), m1, bg_color=torch.zeros(3, device=dev))["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt = REF[vi][3].permute(1, 2, 0).numpy()
        cy_, cx_ = ty * T + T // 2, tx * T + T // 2
        y0 = int(np.clip(cy_ - C // 2, 0, gt.shape[0] - C)); x0 = int(np.clip(cx_ - C // 2, 0, gt.shape[1] - C))
        y1 = pad + ri * (C + pad)
        d.text((4, y1 - 2), f"{nm}  能量比 {r1:.2f} vs CityGS {rc:.2f}", fill=(0, 0, 0), font=font)
        for ci_, src in enumerate([gt, im1, cgimg[vi]]):
            canvas.paste(Image.fromarray(src[y0:y0 + C, x0:x0 + C]), (4 + ci_ * (C + 4), y1 + pad - 4))
    canvas.save(f"{a.out}/only_ours_blur.jpg", quality=92)
    print(f"例子：{a.out}/only_ours_blur.jpg")
print("DONE wall_blur")
