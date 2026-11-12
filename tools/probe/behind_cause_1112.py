"""11-12 規格 3 §1（規格 4 §6 照做）：被深度上限罰到的「落後貢獻」到底是什麼（只渲染、不訓練；GPU、no_grad；cwd＝repo 根）。
MatrixCity 真深度只用來量（分桶），不進任何機制。上限圖與訓練同一張（校正深度＋可靠度過濾＋塊外遮罩 A）。

D1（每個模型；本塊訓練相機，每 --stride 台取一台）：每個設了上限的像素量
  W（rend_capw）、B（rend_cap）、T_final＝1 − rend_alpha、T_cap ≈ W + T_final（抵達先驗表面時剩下的光；混合依中心深度排、上限用交點深度 => 近似）、
  e＝z_真 x 逆深度_先驗 − 1、先驗表面朝向（校正深度反投影的法向，躺平＝|n_z| > 0.8）、像素類別（先驗表面：路面／屋頂／立面／其他）、可靠度級。
  分桶（margin m＝3%、1% 各一次）：① e > m 先驗太淺｜② e < −m 先驗太深｜③ |e|<=m 且 T_cap > 0.9 前面是空的｜④ 0.1 < T_cap <= 0.9 前面漏光｜
  ⑤ T_cap <= 0.1 前面擋住｜⑥ 沒有真深度。每桶：像素佔比、ΣW 佔比、平均超出量 ΣB／ΣW、超出量分帶（margin 1／3／5／10% 各渲染一次，W 相減）。
  逐顆：ρ(m)＝Vb(m)／Vc（m＝1、3、5%；證據門檻同 renderer.behind_rho）；S3（ρ(3%) > 0.5）的 Vb 有幾成來自 ①／③／④
  （把上限圖限制在該桶的像素再渲染一次）；同一朝向類別內 ρ > 0.5 與 ρ < 0.1 的 SH 高階／DC 能量比。
D2（只對 --d2 那個模型）：把一群顆粒的 opacity 設成 0（就地改 logit，用完還原；不存 ckpt），量
  訓練視角 ΔPSNR（塊外遮罩 A 像素）與誤差增量依 D1 的桶（margin 3%）、躺平、類別怎麼分；換塊 held-out ΔPSNR（同 tools/swap_block_eval.py 的換法）；
  地下覆蓋與貼地覆蓋少多少（同 cap_error_split (a) 的定義：o>0.5、地面格、離地 < −0.5 m／±0.5 m）；顆數。
  群：S3＝ρ(3%)>0.5｜S3F／S3N＝S3 裡躺平／不躺平（面元 |n_z| > 0.8）｜S_near＝ρ(1%)>0.5 且 ρ(5%)<0.25｜S_far＝ρ(5%)>0.5。
  D2 量的是不重新訓練時那群撐著多少分；B 臂在訓練中拿掉、模型可以補 => 只比方向和群與群的相對大小（規格 3 §1.2）。
判讀見規格 3 §1.3（一個量要在 b1、b4 都過半才算成立）。
用法（lab，佇列 [solo]）：
  conda run -n gspl --no-capture-output python tools/probe/behind_cause_1112.py --block 1 \
     --base 'outputs/lab/g44s30ksfm10nbnp16_best1maskdistrfed10b4/blocks/block_1/checkpoints/*.ckpt' \
     --models CityGS='<CityGS 塊 ckpt>' 基準='<同 base>' 上限0.5='outputs/lab/g44tn_s30g50np16tricap0p5_best1maskdistrfed10b4/blocks/block_1/checkpoints/*.ckpt'
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "probe"))
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.renderers.sep_depth_trim_2dgs_renderer import behind_rho
from make_cap_masks import world_normals

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--base", required=True, help="我方基準 ckpt 樣式（渲染器、訓練相機、塊外遮罩路徑）")
ap.add_argument("--models", nargs="+", required=True, help="標籤=ckpt 樣式；標籤含 CityGS 的只做 D1 像素表（不算 Vb 分桶）")
ap.add_argument("--d2", default="", help="D2 的模型標籤（預設第一個不是 CityGS 的）")
ap.add_argument("--stride", type=int, default=2, help="訓練相機每幾台取一台")
ap.add_argument("--evidence", type=float, default=0.1)
ap.add_argument("--depth_dir", default="estimated_depths_tri")
ap.add_argument("--rel_dir", default="estimated_depths_tri_rel")
ap.add_argument("--scale_name", default="estimated_depth_scales_tri")
ap.add_argument("--merged_run", default="outputs/lab/full44_best")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
ap.add_argument("--max_cams", type=int, default=0)
ap.add_argument("--no_d2", action="store_true")
a = ap.parse_args()
B = a.block; D = "data/matrix_city/aerial/train/block_all"; MU = 100.0; dev = torch.device("cuda")
t0 = time.time()
import monodepth_sea_1102 as M            # 真深度配對與單位檢查（import 時會跑；真深度只量測）

# ---- 分塊、地面格（同 cap_error_split_1111）----
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


def nrm_z(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.abs(1 - 2 * (x * x + y * y))


# ---- 渲染器、訓練相機、塊外遮罩：取自我方基準 ckpt（同 behind_acc_offline_1111）----
base_ck = sorted(glob.glob(a.base))[-1]
_, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(base_ck, device=dev, eval_mode=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "skip_unscored_pixels", "trim_running_vc"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
ck = torch.load(base_ck, map_location="cpu", weights_only=False)
dmh = ck["datamodule_hyper_parameters"]; mcfg = ck["hyper_parameters"].get("metric")
mask_dir = getattr(getattr(mcfg, "init_args", mcfg), "block_mask_dir", "") or ""
del ck
parser = dmh["parser"]
if getattr(parser, "points_from", None) == "ply":
    parser.points_from = "random"
parser.depth_dir = a.depth_dir; parser.depth_scale_name = a.scale_name; parser.depth_rel_dir = a.rel_dir
ts = parser.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
cams = ts.cameras; names = list(ts.image_names)
sel = list(range(0, len(names), max(1, a.stride)))
if a.max_cams > 0:
    sel = sel[: a.max_cams]
print(f"b{B}｜本塊訓練相機 {len(names)} 台（用 {len(sel)}）｜深度 {a.depth_dir}＋{a.rel_dir}｜遮罩 {mask_dir or '⚠⚠ 沒有（全畫面當 A）'}", flush=True)


def prior_and_mask(i, H, W):
    """訓練時的上限先驗（過濾後的逆深度，不在 A 的像素＝0）與 A 遮罩（bool）。"""
    d = ts.extra_data_processor(ts.extra_data[i]) if ts.extra_data[i] is not None else None
    if isinstance(d, tuple):
        d = d[0]
    if d is None:
        return None, None
    inv = d.reshape(d.shape[-2:]).float()
    inv = torch.nn.functional.interpolate(inv[None, None], size=(H, W), mode="nearest")[0, 0].to(dev)
    A = torch.ones(H, W, dtype=torch.bool, device=dev)
    if mask_dir:
        mp = os.path.join(mask_dir, os.path.basename(str(names[i])))
        if os.path.exists(mp):
            mt = torch.from_numpy(np.array(Image.open(mp)))[None, None].float()
            A = torch.nn.functional.interpolate(mt, size=(H, W), mode="nearest")[0, 0].to(dev) == 1
    return inv * A, A


CLS = ["路面", "屋頂", "立面", "其他"]
BK = ["①先驗太淺", "②先驗太深", "③前面是空的", "④前面漏光", "⑤前面擋住", "⑥沒有真深度"]
MB = [0.03, 0.01]                 # 分桶用的 margin
MR = [0.01, 0.03, 0.05, 0.10]     # 渲染的 margin（超出量分帶）


def attrs(i, H, W):
    """先驗表面的像素屬性（render 解析度、GPU）：類別 [H,W] long、躺平 bool、斜率級、大小級、e（無真深度＝nan）。"""
    nm = os.path.basename(str(names[i])); im = M.BYNAME.get(nm)
    if im is None:
        return None
    W0, H0, fx, fy, cx, cy, R, t = M.cam_of(im)
    raw = np.load(f"{D}/{a.depth_dir}/{nm}.npy").astype(np.float64); raw = raw[..., 0] if raw.ndim == 3 else raw
    rel = np.load(f"{D}/{a.rel_dir}/{nm}.npy")
    h, w = raw.shape
    n, okn = world_normals(raw, fx, fy, cx, cy, w / W0, h / H0, R)
    nz = np.abs(n[..., 2])
    z = np.where(raw > 0, 1.0 / np.maximum(raw, 1e-12), 0.0)
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * W0 / w, (np.arange(h) + 0.5) * H0 / h)
    Xc = np.stack([z * (uu - cx) / fx, z * (vv - cy) / fy, z], -1).reshape(-1, 3)
    Xw = (Xc - t) @ R                                    # 世界 = R^T (X_c − t)
    hh, isg, _ = height_and_ground(Xw)
    hh = hh.reshape(h, w); isg = isg.reshape(h, w)
    cls = np.full((h, w), 3, np.int64)
    cls[okn & (nz < 0.3) & (hh > 3)] = 2
    cls[okn & (nz > 0.8) & (hh > 3)] = 1
    cls[okn & isg & (hh < 2) & (nz > 0.8)] = 0
    flat = okn & (nz > 0.8)
    up = lambda x, mode="nearest": torch.nn.functional.interpolate(torch.from_numpy(np.ascontiguousarray(x)).float()[None, None], size=(H, W), mode=mode)[0, 0].to(dev)
    out = {"cls": up(cls).long(), "flat": up(flat.astype(np.float32)) > 0.5, "sl": up((rel & 7).astype(np.float32)).long(),
           "sz": up((rel >> 3).astype(np.float32)).long()}
    dg = M.load_gt(nm)
    if dg is None or dg.shape != (H0, W0):
        out["zt"] = None
    else:
        iy = np.clip(((np.arange(H) + 0.5) * H0 / H).astype(int), 0, H0 - 1); ix = np.clip(((np.arange(W) + 0.5) * W0 / W).astype(int), 0, W0 - 1)
        d = dg[iy][:, ix].astype(np.float64)
        v = np.isfinite(d) & (d > 0) & (d < 1e8)
        out["zt"] = torch.from_numpy(np.where(v, d * M.K_UNIT, np.nan)).float().to(dev)
    return out


def buckets(e, Tcap, mb):
    b = torch.full(Tcap.shape, 4, dtype=torch.long, device=dev)
    b[Tcap > 0.1] = 3
    b[Tcap > 0.9] = 2
    b[e < -mb] = 1
    b[e > mb] = 0
    b[torch.isnan(e)] = 5
    return b


# ---- 模型 ----
specs = []
for sp in a.models:
    lab, pat = sp.split("=", 1); fs = sorted(glob.glob(pat))
    if not fs:
        raise SystemExit(f"⛔ {lab}：找不到 {pat}")
    specs.append((lab, fs[-1]))
ours = [l for l, _ in specs if "CityGS" not in l]
d2lab = a.d2 or (ours[0] if ours else "")
models = {}
for lab, ckf in specs:
    m, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ckf, device=dev, eval_mode=True)
    n = m.get_xyz.shape[0]
    models[lab] = {"m": m, "ck": ckf, "n": n, "vc": torch.zeros(n, device=dev),
                   "vb": {mg: torch.zeros(n, device=dev) for mg in MR[:3]},
                   "acc": torch.zeros(7, 2 * 6 * 2 * 4, dtype=torch.float64, device=dev),
                   "acc2": torch.zeros(2, 2 * 6 * 2 * 4, dtype=torch.float64, device=dev), "dummy": torch.zeros(n, device=dev)}
    print(f"[{lab}] {ckf}｜N {n:,}", flush=True)
bg = torch.zeros(3, device=dev)

# ---- 第一輪：D1 像素表＋逐顆帳 ----
n_used = 0
with torch.no_grad():
    for ii, i in enumerate(sel):
        cam = cams[i].to_device(dev); H, W = int(cam.height), int(cam.width)
        inv, Am = prior_and_mask(i, H, W)
        if inv is None:
            continue
        at = attrs(i, H, W)
        if at is None:
            continue
        on = inv > 1e-6
        if int(on.sum()) == 0:
            continue
        n_used += 1
        e = (at["zt"] * inv - 1) if at["zt"] is not None else torch.full_like(inv, float("nan"))
        fl = at["flat"].long(); cl = at["cls"]; sl = at["sl"].clamp(0, 1); sz = at["sz"].clamp(0, 3)
        for lab, md in models.items():
            Wm, Bm = {}, {}
            for mg in MR:
                cap = torch.where(on, inv / (1.0 + mg), torch.zeros_like(inv)).contiguous()
                kw = {"_depth_cap_acc": (md["vb"][mg], md["vc"] if mg == MR[0] else md["dummy"])} if mg in md["vb"] else {}
                out = renderer(cam, md["m"], bg_color=bg, _depth_cap=cap, **kw)
                Wm[mg] = out["rend_capw"].reshape(H, W); Bm[mg] = out["rend_cap"].reshape(H, W)
                Tf = 1.0 - out["rend_alpha"].reshape(H, W)
            for mi, mb in enumerate(MB):
                bk = buckets(e, Wm[mb] + Tf, mb)
                key = (((mi * 6 + bk) * 2 + fl) * 4 + cl)[on]
                for j, val in enumerate([torch.ones_like(Tf), Wm[mb], Bm[mb], Wm[0.01], Wm[0.03], Wm[0.05], Wm[0.10]]):
                    md["acc"][j] += torch.bincount(key, weights=val[on].double(), minlength=96)
                key2 = (((mi * 6 + bk) * 2 + sl) * 4 + sz)[on]
                md["acc2"][0] += torch.bincount(key2, minlength=96).double()
                md["acc2"][1] += torch.bincount(key2, weights=Wm[mb][on].double(), minlength=96)
        if n_used % 50 == 0:
            print(f"  第一輪 相機 {n_used}／{len(sel)}（{time.time() - t0:.0f} s）", flush=True)
print(f"第一輪完成：有上限的相機 {n_used}｜{time.time() - t0:.0f} s", flush=True)

# ---- D1 印表 ----
for lab, md in models.items():
    acc = md["acc"].cpu().numpy().reshape(7, 2, 6, 2, 4)        # 量, margin, 桶, 躺平, 類別
    print(f"\n== D1 {lab}｜{md['ck']}")
    for mi, mb in enumerate(MB):
        A_ = acc[:, mi]
        tot_n = A_[0].sum(); tot_w = A_[1].sum()
        print(f"  margin {100 * mb:g}%｜上限像素 {int(tot_n):,}｜平均 W {tot_w / max(tot_n, 1):.4f}")
        print(f"    {'桶':10}{'像素%':>8}{'ΣW%':>8}{'ΣB/ΣW':>9}{'W 分帶（佔該桶 W1%）1~3%':>24}{'3~5%':>7}{'5~10%':>7}{'>10%':>7}")
        for b in range(6):
            n_, w_, b_, w1, w3, w5, w10 = [A_[j, b].sum() for j in range(7)]
            if n_ == 0:
                continue
            print(f"    {BK[b]:10}{100 * n_ / tot_n:>8.1f}{100 * w_ / max(tot_w, 1e-12):>8.1f}{b_ / max(w_, 1e-12):>9.4f}"
                  f"{100 * (w1 - w3) / max(w1, 1e-12):>24.1f}{100 * (w3 - w5) / max(w1, 1e-12):>7.1f}{100 * (w5 - w10) / max(w1, 1e-12):>7.1f}{100 * w10 / max(w1, 1e-12):>7.1f}")
        print(f"    ΣW 佔比（%）依 桶 x 躺平／不躺平 x 類別：" + "".join(f"{('躺' if f_ else '不躺') + CLS[c]:>8}" for f_ in (1, 0) for c in range(4)))
        for b in range(6):
            if A_[0, b].sum() == 0:
                continue
            print(f"    {BK[b]:10}" + "".join(f"{100 * A_[1, b, f_, c] / max(tot_w, 1e-12):>8.2f}" for f_ in (1, 0) for c in range(4)))
        print(f"    像素佔比（%）依 躺平／不躺平 x 類別：" + "".join(f"{100 * A_[0, :, f_, c].sum() / max(tot_n, 1):>8.2f}" for f_ in (1, 0) for c in range(4)))
    a2 = md["acc2"].cpu().numpy().reshape(2, 2, 6, 2, 4)        # 量, margin, 桶, 斜率級, 大小級
    print(f"  可靠度組合（margin 3%）：像素佔上限像素%｜該組裡 ① 的像素%｜該組 ΣW 裡 ① 的%")
    for s_ in range(2):
        for z_ in range(4):
            n_all = a2[0, 0, :, s_, z_].sum()
            if n_all == 0:
                continue
            print(f"    斜率級 {s_}、大小級 {z_}：{100 * n_all / max(a2[0, 0].sum(), 1):6.1f}｜{100 * a2[0, 0, 0, s_, z_] / n_all:6.1f}｜"
                  f"{100 * a2[1, 0, 0, s_, z_] / max(a2[1, 0, :, s_, z_].sum(), 1e-12):6.1f}", flush=True)

# ---- 逐顆：ρ、群、SH 能量比 ----
for lab, md in models.items():
    m = md["m"]
    rho = {}; ok = None
    for mg in MR[:3]:
        r_, o_ = behind_rho(md["vc"], md["vb"][mg], a.evidence); rho[mg] = r_; ok = o_
    nzg = torch.from_numpy(nrm_z(m.get_rotations().detach().double().cpu().numpy())).to(dev)
    op = m.get_opacities().reshape(-1)
    xyz = m.get_xyz.detach().double().cpu().numpy(); inbox = torch.from_numpy(inb(xyz, bb0)).to(dev)
    md.update(rho=rho, ok=ok, nzg=nzg, op=op, inbox=inbox)
    flat_g = nzg > 0.8
    grp = {"S3": rho[0.03] > 0.5}
    grp["S3F"] = grp["S3"] & flat_g; grp["S3N"] = grp["S3"] & ~flat_g
    grp["S_near"] = (rho[0.01] > 0.5) & (rho[0.05] < 0.25); grp["S_far"] = rho[0.05] > 0.5
    md["grp"] = grp
    print(f"\n== 逐顆 {lab}｜有證據 {100 * float(ok.float().mean()):.1f}%｜ρ>0.5 比例 1%／3%／5%："
          + "／".join(f"{100 * float((rho[mg] > 0.5).float().mean()):.2f}%" for mg in MR[:3]))
    for g, k in grp.items():
        print(f"    {g:7}{int(k.sum()):>10,} 顆（塊內 {int((k & inbox).sum()):,}、塊內 o>0.5 {int((k & inbox & (op > 0.5)).sum()):,}）", flush=True)
    pr = m.properties
    dc = pr["shs_dc"].reshape(md["n"], -1).pow(2).sum(1); rest = pr["shs_rest"].reshape(md["n"], -1).pow(2).sum(1) if "shs_rest" in pr else torch.zeros_like(dc)
    ratio = rest / dc.clamp_min(1e-12)
    print("    SH 高階／DC 能量比（中位）：朝向類別｜ρ(3%)>0.5｜ρ(3%)<0.1（有證據）")
    for nm_, kk in (("躺平", nzg > 0.8), ("立面", nzg < 0.3), ("斜的", (nzg >= 0.3) & (nzg <= 0.8))):
        hi_ = kk & ok & (rho[0.03] > 0.5); lo_ = kk & ok & (rho[0.03] < 0.1)
        f = lambda s: float(ratio[s].median()) if int(s.sum()) > 0 else float("nan")
        print(f"      {nm_}｜{f(hi_):.4f}（{int(hi_.sum()):,}）｜{f(lo_):.4f}（{int(lo_.sum()):,}）", flush=True)

np.savez_compressed(f"outputs/lab/behind_cause_1112_b{B}.npz", **{f"{lab}_rho{int(100 * mg)}": models[lab]["rho"][mg].cpu().numpy()
                                                                   for lab in models for mg in MR[:3]})

# ---- 第二輪：S3 的 Vb 分桶（我方模型）＋ D2 訓練視角 ----
do_d2 = (not a.no_d2) and d2lab in models
G2 = ["S3", "S3F", "S3N", "S_near", "S_far"]
vbk = {lab: {k: torch.zeros(models[lab]["n"], device=dev) for k in (0, 2, 3)} for lab in ours}
if do_d2:
    md2 = models[d2lab]
    op_logit0 = md2["m"].properties["opacities"].detach().clone()
    se_tot = torch.zeros(1 + len(G2), dtype=torch.float64, device=dev); n_px = 0
    # 誤差增量分解：key＝桶(6，margin 3%；7＝沒設上限的 A 像素)x 躺平 x 類別
    dse = torch.zeros(len(G2), 7 * 2 * 4, dtype=torch.float64, device=dev)
n2 = 0
with torch.no_grad():
    for ii, i in enumerate(sel):
        cam = cams[i].to_device(dev); H, W = int(cam.height), int(cam.width)
        inv, Am = prior_and_mask(i, H, W)
        if inv is None:
            continue
        at = attrs(i, H, W)
        if at is None:
            continue
        on = inv > 1e-6
        n2 += 1
        e = (at["zt"] * inv - 1) if at["zt"] is not None else torch.full_like(inv, float("nan"))
        cap3 = torch.where(on, inv / 1.03, torch.zeros_like(inv)).contiguous()
        for lab in ours:
            md = models[lab]
            out = renderer(cam, md["m"], bg_color=bg, _depth_cap=cap3)
            bk = buckets(e, out["rend_capw"].reshape(H, W) + 1.0 - out["rend_alpha"].reshape(H, W), 0.03)
            for k in (0, 2, 3):
                capk = (cap3 * (bk == k)).contiguous()
                if int((capk > 0).sum()) == 0:
                    continue
                renderer(cam, md["m"], bg_color=bg, _depth_cap=capk, _depth_cap_acc=(vbk[lab][k], md["dummy"]))
            if do_d2 and lab == d2lab:
                pil = Image.open(ts.image_paths[i]).convert("RGB")
                if pil.size != (W, H):
                    pil = pil.resize((W, H), Image.LANCZOS)
                gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
                se0 = ((out["render"].clamp(0, 1) - gt) ** 2).mean(0)
                se_tot[0] += float(se0[Am].double().sum()); n_px += int(Am.sum())
                kb = torch.where(on, bk, torch.full_like(bk, 6))
                key = ((kb * 2 + at["flat"].long()) * 4 + at["cls"])[Am]
                props = dict(md["m"].properties)
                for gi, g in enumerate(G2):
                    opl = op_logit0.clone(); opl[md["grp"][g]] = -30.0
                    md["m"].properties = {**props, "opacities": opl}
                    sg = ((renderer(cam, md["m"], bg_color=bg)["render"].clamp(0, 1) - gt) ** 2).mean(0)
                    se_tot[1 + gi] += float(sg[Am].double().sum())
                    dse[gi] += torch.bincount(key, weights=(sg - se0)[Am].double(), minlength=56)
                md["m"].properties = {**props, "opacities": op_logit0}
        if n2 % 50 == 0:
            print(f"  第二輪 相機 {n2}／{len(sel)}（{time.time() - t0:.0f} s）", flush=True)

for lab in ours:
    md = models[lab]; vb3 = md["vb"][0.03]
    print(f"\n== S3 的 Vb 來自哪一桶（margin 3%）｜{lab}：①先驗太淺／③前面是空的／④前面漏光／其他（②⑤⑥）")
    for g in ("S3", "S3F", "S3N"):
        k = md["grp"][g]; den_ = float(vb3[k].sum())
        if den_ <= 0:
            continue
        sh = [float(vbk[lab][b][k].sum()) / den_ for b in (0, 2, 3)]
        print(f"    {g:5}{int(k.sum()):>10,} 顆｜" + "／".join(f"{100 * x:.1f}%" for x in sh) + f"／{100 * (1 - sum(sh)):.1f}%", flush=True)

psnr = lambda s, n: -10 * np.log10(max(s / max(n, 1), 1e-12))
if do_d2:
    md = models[d2lab]
    print(f"\n== D2 {d2lab}（opacity 設 0，不重新訓練）｜訓練視角 {n2} 台、塊外遮罩 A 像素 {n_px:,}")
    s = se_tot.cpu().numpy(); p0 = psnr(s[0], n_px)
    xyz = md["m"].get_xyz.detach().double().cpu().numpy(); o = md["op"].cpu().numpy()
    sc = md["m"].get_scales().detach().double().cpu().numpy() * MU
    h_, isg_, _ = height_and_ground(xyz)
    ib = md["inbox"].cpu().numpy()
    cov = o * 2 * np.pi * sc[:, 0] * sc[:, 1]
    ug = ib & (o > 0.5) & isg_ & (h_ < -0.5); ng = ib & (o > 0.5) & isg_ & (np.abs(h_) < 0.5)
    print(f"    {'群':7}{'顆數':>10}{'塊內 o>0.5':>12}{'訓練視角 ΔPSNR ↑':>18}{'地下覆蓋少 %':>14}{'貼地覆蓋少 %':>14}")
    for gi, g in enumerate(G2):
        k = md["grp"][g].cpu().numpy()
        print(f"    {g:7}{int(k.sum()):>10,}{int((k & ib & (o > 0.5)).sum()):>12,}{psnr(s[1 + gi], n_px) - p0:>+18.3f}"
              f"{100 * cov[ug & k].sum() / max(cov[ug].sum(), 1e-12):>14.1f}{100 * cov[ng & k].sum() / max(cov[ng].sum(), 1e-12):>14.1f}", flush=True)
    dd = dse.cpu().numpy().reshape(len(G2), 7, 2, 4)
    BK7 = BK + ["沒設上限"]
    print("    誤差增量的去處（%，佔該群 ΣΔSE）：桶（margin 3%）｜躺平／不躺平｜類別")
    for gi, g in enumerate(G2):
        tot_ = dd[gi].sum()
        if tot_ <= 0:
            print(f"    {g}：誤差沒有增加"); continue
        print(f"    {g:7}桶 " + " ".join(f"{BK7[b][:3]}{100 * dd[gi, b].sum() / tot_:.0f}" for b in range(7))
              + f"｜躺 {100 * dd[gi, :, 1].sum() / tot_:.0f}／不躺 {100 * dd[gi, :, 0].sum() / tot_:.0f}｜"
              + " ".join(f"{CLS[c]}{100 * dd[gi, :, :, c].sum() / tot_:.0f}" for c in range(4)), flush=True)

    # ---- D2 換塊 held-out ----
    import eval_official_test as eot
    from swap_block_eval import block_mask
    for lab_ in list(models):                  # 只留 D2 的模型，其他放掉（換塊的合併模型約 4 GB）
        if lab_ != d2lab:
            del models[lab_]
    m = None
    torch.cuda.empty_cache()
    mck = sorted(glob.glob(os.path.join(a.merged_run, "checkpoints", "*.ckpt")))
    mck = [c for c in mck if os.path.basename(c) == "merged.ckpt"] or mck
    mm, rr, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(mck[-1], device=dev, eval_mode=True)
    for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        setattr(rr, f, False)
    basep = {k: v.detach() for k, v in mm.properties.items()}
    pcfg = torch.load(md["ck"], map_location="cpu", weights_only=False)["datamodule_hyper_parameters"]["parser"]
    keep_other = ~block_mask(basep["means"], pcfg)[B].to(dev)
    cp = dict(md["m"].properties)
    for _k in ("shs_rest",):
        if _k in cp and _k in basep and cp[_k].shape[1] < basep[_k].shape[1]:
            r_ = cp[_k]; cp[_k] = torch.cat([r_, torch.zeros((r_.shape[0], basep[_k].shape[1] - r_.shape[1], r_.shape[2]), dtype=r_.dtype, device=r_.device)], 1)
    cmask = block_mask(cp["means"], pcfg)[B].to(dev)
    merged = {k: torch.cat([basep[k][keep_other], cp[k][cmask].detach()], 0) for k in basep}
    n_other = int(keep_other.sum())
    del basep, cp; torch.cuda.empty_cache()
    mm.properties = merged
    opm0 = merged["opacities"].clone()
    names_t, tcams = eot.load_test_cameras(a.test_dir, 1.2)
    centres = np.array([(-tcams.R[j].numpy().T @ tcams.T[j].numpy()) for j in range(len(names_t))])
    lo, hi, _ = eot.block_bounds(D, B, [4, 4])
    selB = [j for j in range(len(names_t)) if np.all(centres[j] >= lo) and np.all(centres[j] <= hi)]
    se_f = np.zeros(1 + len(G2)); se_r = np.zeros(1 + len(G2)); nf = 0; nr = 0
    with torch.no_grad():
        for jj, j in enumerate(selB):
            cam = tcams[j].to_device(dev)
            pil = Image.open(os.path.join(a.test_dir, "images_1.2", names_t[j])).convert("RGB")
            mm.properties = {**merged, "opacities": opm0}
            o0_ = rr(cam, mm, bg_color=bg)
            img0 = o0_["render"].clamp(0, 1); H_, W_ = img0.shape[1:]
            if pil.size != (W_, H_):
                pil = pil.resize((W_, H_), Image.LANCZOS)
            gt = torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1).to(dev) / 255.
            zz = o0_["surf_depth"][0].reshape(H_, W_); al = o0_["rend_alpha"].reshape(H_, W_)
            v2, u2 = torch.meshgrid(torch.arange(H_, device=dev), torch.arange(W_, device=dev), indexing="ij")
            pcm = torch.stack([(u2 + 0.5 - cam.cx) / cam.fx * zz, (v2 + 0.5 - cam.cy) / cam.fy * zz, zz], -1)
            xw = (pcm.reshape(-1, 3) - tcams.T[j].to(dev).float()) @ tcams.R[j].to(dev).float()
            region = block_mask(xw, pcfg)[B].reshape(H_, W_).to(dev) & (zz > 0) & (al > 0.5)
            s0 = ((img0 - gt) ** 2).mean(0)
            se_f[0] += float(s0.double().sum()); se_r[0] += float(s0[region].double().sum()); nf += s0.numel(); nr += int(region.sum())
            for gi, g in enumerate(G2):
                opl = opm0.clone(); idx = torch.nonzero(md["grp"][g][cmask]).reshape(-1) + n_other
                opl[idx] = -30.0
                mm.properties = {**merged, "opacities": opl}
                sg = ((rr(cam, mm, bg_color=bg)["render"].clamp(0, 1) - gt) ** 2).mean(0)
                se_f[1 + gi] += float(sg.double().sum()); se_r[1 + gi] += float(sg[region].double().sum())
            if (jj + 1) % 50 == 0:
                print(f"  換塊 held-out 視角 {jj + 1}／{len(selB)}（{time.time() - t0:.0f} s）", flush=True)
    print(f"\n== D2 換塊 held-out（{a.merged_run} 的第 {B} 塊換成 {d2lab}；第 {B} 塊視角 {len(selB)} 幀；PSNR＝合併 MSE）")
    print(f"    {'群':7}{'全畫面 ΔPSNR ↑':>16}{'幾何區域 ΔPSNR ↑':>18}")
    for gi, g in enumerate(G2):
        print(f"    {g:7}{psnr(se_f[1 + gi], nf) - psnr(se_f[0], nf):>+16.3f}{psnr(se_r[1 + gi], nr) - psnr(se_r[0], nr):>+18.3f}", flush=True)
    print(f"    （基準換塊後：全畫面 {psnr(se_f[0], nf):.3f}、幾何區域 {psnr(se_r[0], nr):.3f}）")
print(f"\n判讀見規格 3 §1.3｜{time.time() - t0:.0f} s")
print("DONE")
