"""11-11 規格 §5（`紀錄/規格_組合臂_表面起點增生限制深度上限_1111.md`）：「只開逐貢獻深度上限」掉畫質，是哪一個原因？
  候選 ①：顆粒被推過頭、浮在路面上方（b1 地上覆蓋 0.72 → 1.81 層）；候選 ②：路面層畫不出原本的紋理（顏色）。
做法（只渲染、只讀；GPU；cwd＝repo 根）：
  (a) CPU：兩個候選模型的本塊顆粒，在地面格（2 m 格可靠 SfM 中位 z < 3 m）上依離地高度分帶：貼地 |h|<0.5 m／路面上方 0.5～1、1～2、2～5 m／地下，
      不透明（o>0.5）顆數與覆蓋（Σo·2π·su·sv ÷ 地面格面積；同 ground_layer_1103）。
  (b) GPU：換塊（同 tools/swap_block_eval.py：full44_best 合併模型只把第 B 塊換成候選的盒內顆粒），第 B 塊 held-out 視角，
      幾何區域（基準換塊後的表面深度反推回世界、落在第 B 塊盒內、alpha>0.5）的每個像素依「第一個候選（基準）渲染的表面」分類：
        路面＝地面格、離地 < 2 m、|n_z| > 0.8；屋頂＝|n_z| > 0.8、離地 > 3 m；立面＝|n_z| < 0.3、離地 > 3 m；其他。
      可靠度：把該像素的世界點投到本塊訓練相機中「相機中心最近的 16 台」，用校正深度（未過濾）判看得到（|r| < 2%），
        看得到的那些相機在可靠度圖上多數可靠（斜率級 <= 1）=> 可靠；都看不到 => 未知。
      每類各模型的 PSNR（該類像素的 MSE）、差值，以及「渲染位置和顏色」拆解：
        顏色誤差 vs 形狀誤差的代理＝各模型自己的表面深度和基準差多少（|z_cap/z_base − 1| > 1% 的像素比例）。
用法（lab，佇列 [solo]）：
  conda run -n gspl --no-capture-output python tools/probe/cap_error_split_1111.py --block 1 \
     --cands 基準=outputs/lab/g44s30ksfm10nbnp16_best1maskdistrfed10b4/blocks/block_1/checkpoints/'*.ckpt' 上限權重2=outputs/lab/g44tn_s30g50np16tricap2_best1maskdistrfed10b4/blocks/block_1/checkpoints/'*.ckpt'
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter, uniform_filter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from internal.utils.colmap import read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--cands", nargs=2, required=True, help="標籤=ckpt 樣式 x 2（第一個＝基準，像素分類用它的表面）")
ap.add_argument("--merged_run", default="outputs/lab/full44_best")
ap.add_argument("--train_dir", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
ap.add_argument("--depth_dir", default="estimated_depths_tri")
ap.add_argument("--rel_dir", default="estimated_depths_tri_rel")
ap.add_argument("--slope_max", type=int, default=1)
ap.add_argument("--ncam", type=int, default=16)
ap.add_argument("--stride", type=int, default=2, help="可靠度只在每 stride 個像素算一次（其餘像素沿用最近的）")
ap.add_argument("--max_views", type=int, default=100000)
ap.add_argument("--cpu_only", action="store_true")
a = ap.parse_args()
B = a.block; D = a.train_dir; MU = 100.0
t0 = time.time()

# ---- 分塊與地面格（同 smallobj_cpu_1109／behind_offline_1111）----
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc_ = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
bb0 = pc_.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.)
bb1 = pc_.get_bounding_boxes(p["scene_config"]["partition_size"], enlarge=0.05)
Rt = p["extra_data"]["rotation_transform"][:3, :3].double()
inb = lambda x, bb: CityGSPartitioning.is_in_bounding_boxes(bounding_boxes=bb, coordinates=(torch.as_tensor(x).double().cpu() @ Rt.T)[:, :2])[B].numpy()
pts = read_points3D_binary(f"{D}/sparse/0/points3D.bin")
P = np.stack([q.xyz for q in pts.values()]); okp = np.array([len(q.image_ids) >= 3 and q.error < 1.5 for q in pts.values()]); del pts
S = P[okp & inb(P, bb1)]
G = 0.02
o0 = np.floor(S[:, :2].min(0) / G).astype(int) - 4
ci = np.floor(S[:, :2] / G).astype(int) - o0; shp = tuple(ci.max(0) + 5)
cid = ci[:, 0] * shp[1] + ci[:, 1]; order = np.lexsort((S[:, 2], cid)); u_, f_, c_ = np.unique(cid[order], return_index=True, return_counts=True)
med = np.full(shp[0] * shp[1], np.nan); med[u_] = S[order, 2][f_ + c_ // 2]; med = med.reshape(shp)
ground = np.isfinite(med) & (med < 0.03)
gz = np.where(ground, med, np.nan)
num = uniform_filter(np.nan_to_num(gz), size=5); den = uniform_filter(np.isfinite(gz).astype(float), size=5)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
GL = float(np.nanmedian(gz))


def height_and_ground(xyz):
    cc = np.floor(xyz[:, :2] / G).astype(int) - o0
    okc = (cc[:, 0] >= 0) & (cc[:, 0] < shp[0]) & (cc[:, 1] >= 0) & (cc[:, 1] < shp[1])
    cc = np.clip(cc, 0, np.array(shp) - 1)
    g = gfill[cc[:, 0], cc[:, 1]]
    h = (xyz[:, 2] - np.where(okc & np.isfinite(g), g, GL)) * MU
    return h, okc & ground[cc[:, 0], cc[:, 1]], cc


labs, cks = [], []
for spec in a.cands:
    lab, pat = spec.split("=", 1); fs = sorted(glob.glob(pat))
    if not fs:
        raise SystemExit(f"⛔ {lab}：找不到 {pat}")
    labs.append(lab); cks.append(fs[-1])

# ---- (a) CPU：路面上方的顆粒 ----
print(f"== (a) b{B} 地面格上的不透明顆粒（o>0.5）依離地高度；覆蓋＝Σo·2π·su·sv ÷ 本塊地面格面積（層）", flush=True)
bands = [("地下 < −0.5m", -1e9, -0.5), ("貼地 ±0.5m", -0.5, 0.5), ("上方 0.5～1m", 0.5, 1), ("上方 1～2m", 1, 2), ("上方 2～5m", 2, 5)]
print(f"  {'模型':12}" + "".join(f"{nm:>22}" for nm, _, _ in bands) + "   （每格：顆數／覆蓋）")
for lab, ck in zip(labs, cks):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    gg = lambda k: sd[f"gaussian_model.gaussians.{k}"].float()
    m = gg("means").double().numpy(); o = torch.sigmoid(gg("opacities")).reshape(-1).numpy(); s = torch.exp(gg("scales")).numpy() * MU
    del sd
    k = inb(m, bb0) & (o > 0.5)
    m, o, s = m[k], o[k], s[k]
    h, isg, cc = height_and_ground(m)
    area = len(np.unique((cc[isg, 0].astype(np.int64) << 32) + cc[isg, 1])) * 4.0
    cov = o * 2 * np.pi * s[:, 0] * s[:, 1]
    print(f"  {lab:12}" + "".join(f"{int((isg & (h > lo) & (h <= hi)).sum()):>12,}／{cov[isg & (h > lo) & (h <= hi)].sum() / area:6.2f}" for _, lo, hi in bands), flush=True)
if a.cpu_only:
    print("DONE"); sys.exit(0)

# ---- (b) GPU：換塊渲染 ----
from internal.utils.gaussian_model_loader import GaussianModelLoader
import eval_official_test as eot
from swap_block_eval import block_mask
dev = torch.device("cuda")
mck = sorted(glob.glob(os.path.join(a.merged_run, "checkpoints", "*.ckpt")))
mck = [c for c in mck if os.path.basename(c) == "merged.ckpt"] or mck
model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(mck[-1], device=dev, eval_mode=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
    setattr(renderer, f, False)
base = {k: v.detach().cpu() for k, v in model.properties.items()}
model.properties = {k: v[:1].to(dev) for k, v in base.items()}; torch.cuda.empty_cache()
variants = []
mmask = pc0 = None
for lab, ck in zip(labs, cks):
    cko = torch.load(ck, map_location="cpu"); pc = cko["datamodule_hyper_parameters"]["parser"]; del cko
    pc0 = pc
    if mmask is None:
        mmask = block_mask(base["means"], pc)[B]
    cm, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck, device=dev, eval_mode=True)
    cp = dict(cm.properties)
    for _k in ("shs_rest", "shs"):
        if _k in cp and _k in base and cp[_k].shape[1] < base[_k].shape[1]:
            r = cp[_k]; cp[_k] = torch.cat([r, torch.zeros((r.shape[0], base[_k].shape[1] - r.shape[1], r.shape[2]), dtype=r.dtype, device=r.device)], 1)
    cmask = block_mask(cp["means"], pc)[B].to(cp["means"].device)
    variants.append({k: torch.cat([base[k][~mmask], cp[k][cmask].detach().cpu()], 0) for k in base})
    print(f"[{lab}] {ck}｜塊內 {int(cmask.sum()):,} 顆", flush=True)
    del cm, cp
del base
variants = [{k: v.to(dev) for k, v in vv.items()} for vv in variants]     # 兩個換塊後模型各約 4 GB，都放 GPU（佇列 [solo]）
torch.cuda.empty_cache()

names_t, cams = eot.load_test_cameras(a.test_dir, 1.2)
centres = np.array([(-cams.R[i].numpy().T @ cams.T[i].numpy()) for i in range(len(names_t))])
lo, hi, _ = eot.block_bounds(a.train_dir, B, [4, 4])
selB = [i for i in range(len(names_t)) if np.all(centres[i] >= lo) and np.all(centres[i] <= hi)][:a.max_views]

# 訓練相機（可靠度用）：本塊訓練相機（partition 清單）
by, bx = B // 4, B % 4
wanted = {ln.strip() for ln in open(f"{D}/partition/partitions-dim_4_4_visibility_0.08/{bx:03d}_{by:03d}.txt") if ln.strip()}
ims = read_images_binary(f"{D}/sparse/0/images.bin"); tcam = read_cameras_binary(f"{D}/sparse/0/cameras.bin")
tr = [im for im in ims.values() if im.name in wanted or os.path.basename(im.name) in wanted]
TC = np.array([-qvec2rotmat(im.qvec).T @ im.tvec for im in tr])
cache = {}


def tri_maps(im):
    nm = os.path.basename(im.name)
    if nm not in cache:
        dep = np.load(f"{D}/{a.depth_dir}/{nm}.npy").astype(np.float32)
        dep = dep[..., 0] if dep.ndim == 3 else dep
        rel = np.load(f"{D}/{a.rel_dir}/{nm}.npy")
        cache[nm] = (torch.from_numpy(dep), torch.from_numpy(((rel & 7) <= a.slope_max)))
    return cache[nm]


@torch.no_grad()
def reliability(Xw):
    """Xw [M,3]（世界）=> 0 未知、1 可靠、2 不可靠。"""
    c = Xw.mean(0).cpu().numpy()
    near = np.argsort(np.linalg.norm(TC - c, axis=1))[:a.ncam]
    good = torch.zeros(Xw.shape[0], device=dev); bad = torch.zeros(Xw.shape[0], device=dev)
    for j in near:
        im = tr[j]; cam = tcam[im.camera_id]; pr = cam.params
        fx, fy, cx, cy = (pr[0], pr[0], pr[1], pr[2]) if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL") else (pr[0], pr[1], pr[2], pr[3])
        dep, ok_r = tri_maps(im)
        hd, wd = dep.shape
        R = torch.from_numpy(qvec2rotmat(im.qvec)).float().to(dev); T = torch.from_numpy(np.asarray(im.tvec)).float().to(dev)
        Xc = Xw @ R.T + T; z = Xc[:, 2]
        u = fx * Xc[:, 0] / z.clamp_min(1e-6) + cx; v = fy * Xc[:, 1] / z.clamp_min(1e-6) + cy
        ud = (u * (wd / float(cam.width))).floor().long(); vd = (v * (hd / float(cam.height))).floor().long()
        ok = (z > 0.2) & (ud >= 0) & (ud < wd) & (vd >= 0) & (vd < hd)
        uc, vc = ud.clamp(0, wd - 1).cpu(), vd.clamp(0, hd - 1).cpu()
        dinv = dep[vc, uc].to(dev); rr = ok_r[vc, uc].to(dev)
        ok &= dinv > 1e-6
        vis = ok & ((z * dinv - 1).abs() < 0.02)
        good += (vis & rr).float(); bad += (vis & ~rr).float()
    out = torch.zeros(Xw.shape[0], dtype=torch.long, device=dev)
    out[(good + bad > 0) & (good >= bad)] = 1
    out[(good + bad > 0) & (good < bad)] = 2
    return out


CLS = ["路面", "屋頂", "立面", "其他"]
REL = ["未知", "可靠", "不可靠"]
acc = np.zeros((len(CLS), len(REL), len(labs)))   # Σ SE
cnt = np.zeros((len(CLS), len(REL)))
zdiff = np.zeros((len(CLS), len(REL)))           # 第二個模型表面深度和基準差 > 1% 的像素數
bg = torch.zeros(3, device=dev)
with torch.no_grad():
    for ii, i in enumerate(selB):
        cam = cams[i].to_device(dev)
        pil = Image.open(os.path.join(a.test_dir, "images_1.2", names_t[i])).convert("RGB")
        outs, zs = [], []
        for vi, props in enumerate(variants):
            model.properties = props
            oo = renderer(cam, model, bg_color=bg)
            out = oo["render"].clamp(0, 1); H_, W_ = out.shape[1:]
            outs.append(out); zs.append(oo["surf_depth"][0].reshape(H_, W_))
            if vi == 0:
                al = oo["rend_alpha"].reshape(H_, W_)
        if pil.size != (W_, H_):
            pil = pil.resize((W_, H_), Image.LANCZOS)
        gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
        zz = zs[0]
        v2, u2 = torch.meshgrid(torch.arange(H_, device=dev), torch.arange(W_, device=dev), indexing="ij")
        pcm = torch.stack([(u2 + 0.5 - cam.cx) / cam.fx * zz, (v2 + 0.5 - cam.cy) / cam.fy * zz, zz], -1)
        xw = (pcm.reshape(-1, 3) - cams.T[i].to(dev).float()) @ cams.R[i].to(dev).float()
        xw = xw.reshape(H_, W_, 3)
        region = block_mask(xw.reshape(-1, 3), pc0)[B].reshape(H_, W_).to(dev) & (zz > 0) & (al > 0.5)
        du = torch.zeros_like(xw); dv = torch.zeros_like(xw)
        du[:, 1:-1] = xw[:, 2:] - xw[:, :-2]; dv[1:-1] = xw[2:] - xw[:-2]
        nrm = torch.cross(du, dv, dim=-1); nl = nrm.norm(dim=-1)
        nz = (nrm[..., 2].abs() / nl.clamp_min(1e-12))
        h, isg, _ = height_and_ground(xw.reshape(-1, 3).double().cpu().numpy())
        h = torch.from_numpy(h).float().to(dev).reshape(H_, W_); isg = torch.from_numpy(isg).to(dev).reshape(H_, W_)
        cls = torch.full((H_, W_), 3, dtype=torch.long, device=dev)
        cls[(nz < 0.3) & (h > 3)] = 2
        cls[(nz > 0.8) & (h > 3)] = 1
        cls[isg & (h < 2) & (nz > 0.8)] = 0
        cls[nl <= 0] = 3
        # 可靠度：每 stride 個像素算一次，其餘取最近的
        s_ = a.stride
        sub = xw[::s_, ::s_].reshape(-1, 3)
        rl = reliability(sub).reshape(xw[::s_, ::s_].shape[:2])
        rel = rl.repeat_interleave(s_, 0).repeat_interleave(s_, 1)[:H_, :W_]
        se = [((o_ - gt) ** 2).mean(0) for o_ in outs]
        zd = ((zs[1] / zs[0].clamp_min(1e-6) - 1).abs() > 0.01)
        for ci_ in range(len(CLS)):
            for ri in range(len(REL)):
                mm = region & (cls == ci_) & (rel == ri)
                n_ = int(mm.sum())
                if n_ == 0:
                    continue
                cnt[ci_, ri] += n_
                zdiff[ci_, ri] += int((mm & zd).sum())
                for vi in range(len(labs)):
                    acc[ci_, ri, vi] += float(se[vi][mm].sum())
        if (ii + 1) % 20 == 0:
            print(f"  視角 {ii + 1}/{len(selB)}（{time.time() - t0:.0f} s）", flush=True)

psnr = lambda s, n: -10 * np.log10(max(s / max(n, 1), 1e-12))
tot = cnt.sum()
print(f"\n== (b) b{B} held-out 第 {B} 塊視角 {len(selB)} 幀｜幾何區域像素 {int(tot):,}｜PSNR 只算該類像素；Δ＝{labs[1]} − {labs[0]}"
      f"｜深度差＝{labs[1]} 表面深度和 {labs[0]} 差 > 1% 的像素比例")
print(f"  {'類別':6}{'可靠度':8}{'像素佔比':>9}{labs[0]:>12}{labs[1]:>12}{'ΔPSNR':>9}{'Δ×佔比（對整區 SE 的貢獻 %）':>28}{'深度差%':>9}")
se_tot = [acc[..., vi].sum() for vi in range(len(labs))]
print(f"  {'全部':6}{'':8}{100.0:>8.1f}%{psnr(se_tot[0], tot):>12.3f}{psnr(se_tot[1], tot):>12.3f}{psnr(se_tot[1], tot) - psnr(se_tot[0], tot):>+9.3f}{100.0:>27.1f}%{100 * zdiff.sum() / max(tot, 1):>8.1f}%")
for ci_ in range(len(CLS)):
    rows = [(REL[ri], acc[ci_, ri], cnt[ci_, ri], zdiff[ci_, ri]) for ri in range(len(REL))]
    rows.insert(0, ("合計", acc[ci_].sum(0), cnt[ci_].sum(), zdiff[ci_].sum()))
    for rn, sv, n_, zd_ in rows:
        if n_ == 0:
            continue
        p0, p1 = psnr(sv[0], n_), psnr(sv[1], n_)
        share = 100 * (sv[1] - sv[0]) / max(se_tot[1] - se_tot[0], 1e-12)
        print(f"  {CLS[ci_] if rn == '合計' else '':6}{rn:8}{100 * n_ / tot:>8.1f}%{p0:>12.3f}{p1:>12.3f}{p1 - p0:>+9.3f}{share:>27.1f}%{100 * zd_ / n_:>8.1f}%", flush=True)
print(f"（整區 SE 增量裡各列的佔比：正＝這一類讓 {labs[1]} 變差的份量；{time.time() - t0:.0f} s）")
print("DONE")
