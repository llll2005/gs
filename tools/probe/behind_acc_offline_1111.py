"""11-11 規格 2 §4：落後貢獻計價的離線檢查（用重編後的光柵器，不訓練；GPU、no_grad）。
對現成模型把本塊全部訓練相機各渲染一次（完整路徑、校正深度＋可靠度過濾、塊外遮罩 A 像素——與訓練時 metric.depth_cap_map 同一張上限圖），
光柵器逐顆累加 Vc（有上限的像素上的 Σ T·α）與 Vb（其中交點深度 > (1+margin)·先驗深度的部分）=> ρ＝Vb／Vc（證據不足＝0，同 renderer.behind_rho）。
被擋住的顆粒 T≈0 沒有份量；斜看牆面用的是交點深度。margin 3% 與 1% 各一次。
分類同 tools/probe/behind_offline_1111.py（量測用公尺可以，機制不行）。
過關（規格 2 §4）：CityGS 地面以上各類（貼地、立面、斜面、樓頂、路燈帶、車輛帶）ρ > 0.5 的比例 <= 約 3%；我方基準真地下 >= 約 50%（30% 以上也可排，要註明）。
用法（lab，佇列 [solo]）：python tools/probe/behind_acc_offline_1111.py --block 1 --base <我方基準 ckpt> --models CityGS=<ckpt> 基準=<ckpt>
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import maximum_filter, uniform_filter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.renderers.sep_depth_trim_2dgs_renderer import behind_rho

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--base", required=True, help="我方基準 ckpt（取渲染器、訓練相機清單、塊外遮罩路徑）")
ap.add_argument("--models", nargs="+", required=True, help="標籤=ckpt 樣式；第一個＝CityGS（判誤判），其餘＝我方（判真地下命中）")
ap.add_argument("--margins", default="0.03,0.01")
ap.add_argument("--evidence", type=float, default=0.1)
ap.add_argument("--depth_dir", default="estimated_depths_tri")
ap.add_argument("--rel_dir", default="estimated_depths_tri_rel")
ap.add_argument("--scale_name", default="estimated_depth_scales_tri")
ap.add_argument("--max_cams", type=int, default=0)
a = ap.parse_args()
B = a.block; D = "data/matrix_city/aerial/train/block_all"; MU = 100.0; dev = torch.device("cuda")
t0 = time.time()

# ---- 分類（同 behind_offline_1111）----
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
bldg = maximum_filter((np.isfinite(med) & ~ground).astype(np.uint8), size=5) > 0
open_g = ground & ~bldg
gz = np.where(ground, med, np.nan)
num = uniform_filter(np.nan_to_num(gz), size=5); den = uniform_filter(np.isfinite(gz).astype(float), size=5)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
GL = float(np.nanmedian(gz))


def nrm_z(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); w, x, y, z = q.T
    return np.abs(1 - 2 * (x * x + y * y))


def categories(m, o, nz):
    cc = np.floor(m[:, :2] / G).astype(int) - o0
    okc = (cc[:, 0] >= 0) & (cc[:, 0] < shp[0]) & (cc[:, 1] >= 0) & (cc[:, 1] < shp[1])
    cc = np.clip(cc, 0, np.array(shp) - 1)
    g = gfill[cc[:, 0], cc[:, 1]]
    h = (m[:, 2] - np.where(okc & np.isfinite(g), g, GL)) * MU
    isg = okc & ground[cc[:, 0], cc[:, 1]]; iso = okc & open_g[cc[:, 0], cc[:, 1]]
    op5, op3 = o > 0.5, o > 0.3
    hi = op5 & ~iso & (h > 3)
    return [("貼地 |h|<0.5m", op5 & isg & (np.abs(h) < 0.5)), ("淺地下 −1～−0.5m", op5 & isg & (h <= -0.5) & (h > -1)),
            ("真地下 < −1m", op5 & isg & (h <= -1)), ("車輛帶 0.5～2.5m", op3 & iso & (h > 0.5) & (h < 2.5)),
            ("路燈帶 3～15m", op3 & iso & (h > 3) & (h < 15)), ("樓頂 h>3m", hi & (nz > 0.85)),
            ("立面 h>3m", hi & (nz < 0.3)), ("斜的 h>3m", hi & (nz >= 0.3) & (nz <= 0.85))]


# ---- 渲染器、訓練相機（含校正深度＋可靠度）、塊外遮罩：都取自我方基準 ckpt ----
_, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.base, device=dev, eval_mode=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "skip_unscored_pixels", "trim_running_vc"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
ck = torch.load(a.base, map_location="cpu", weights_only=False)
dmh = ck["datamodule_hyper_parameters"]; mcfg = ck["hyper_parameters"].get("metric")
mask_dir = getattr(getattr(mcfg, "init_args", mcfg), "block_mask_dir", "") or ""
del ck
parser = dmh["parser"]
if getattr(parser, "points_from", None) == "ply":
    parser.points_from = "random"
parser.depth_dir = a.depth_dir; parser.depth_scale_name = a.scale_name; parser.depth_rel_dir = a.rel_dir
ts = parser.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
cams = ts.cameras; names = list(ts.image_names)
sel = list(range(len(names)))[: a.max_cams] if a.max_cams > 0 else list(range(len(names)))
print(f"b{B}｜本塊訓練相機 {len(names)} 台（用 {len(sel)}）｜深度 {a.depth_dir}＋{a.rel_dir}｜遮罩 {mask_dir or '⚠⚠ 沒有（全畫面當 A）'}｜margin {a.margins}", flush=True)

from PIL import Image


def prior_and_mask(i, H, W):
    d = ts.extra_data_processor(ts.extra_data[i]) if ts.extra_data[i] is not None else None
    if isinstance(d, tuple):
        d = d[0]
    if d is None:
        return None
    inv = d.reshape(d.shape[-2:]).float()
    inv = torch.nn.functional.interpolate(inv[None, None], size=(H, W), mode="nearest")[0, 0].to(dev)
    if mask_dir:
        mp = os.path.join(mask_dir, os.path.basename(str(names[i])))
        if os.path.exists(mp):
            mt = torch.from_numpy(np.array(Image.open(mp)))[None, None].float()
            m = torch.nn.functional.interpolate(mt, size=(H, W), mode="nearest")[0, 0].to(dev)
            inv = inv * (m == 1)
    return inv


margins = [float(x) for x in a.margins.split(",")]
bg = torch.zeros(3, device=dev)
for spec in a.models:
    lab, pat = spec.split("=", 1)
    ckf = sorted(glob.glob(pat))[-1]
    model, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ckf, device=dev, eval_mode=True)
    n = model.get_xyz.shape[0]
    acc = {mg: (torch.zeros(n, device=dev), torch.zeros(n, device=dev)) for mg in margins}
    wsum = {mg: [0.0, 0.0] for mg in margins}
    n_used = 0
    with torch.no_grad():
        for i in sel:
            cam = cams[i].to_device(dev)
            H, W = int(cam.height), int(cam.width)
            inv = prior_and_mask(i, H, W)
            if inv is None:
                continue
            n_used += 1
            for mg in margins:
                cap = torch.where(inv > 1e-6, inv / (1.0 + mg), torch.zeros_like(inv)).contiguous()
                vb, vc = acc[mg]
                out = renderer(cam, model, bg_color=bg, _depth_cap=cap, _depth_cap_acc=(vb, vc))
                if "rend_capw" in out:
                    wsum[mg][0] += float(out["rend_capw"].sum()); wsum[mg][1] += float((cap > 0).sum())
            if n_used % 200 == 0:
                print(f"  [{lab}] 相機 {n_used}（{time.time() - t0:.0f} s）", flush=True)
    xyz = model.get_xyz.detach().double().cpu().numpy(); o = model.get_opacities().reshape(-1).detach().cpu().numpy()
    nz = nrm_z(model.get_rotations().detach().double().cpu().numpy())
    keep = inb(xyz, bb0) & (o > 0.3)
    cats = categories(xyz[keep], o[keep], nz[keep])
    print(f"\n== {lab}｜{ckf}｜N {n:,}、塊內 o>0.3 {int(keep.sum()):,}｜有深度的相機 {n_used}／{len(sel)}")
    for mg in margins:
        vb, vc = acc[mg]
        rho, ok = behind_rho(vc, vb, a.evidence)
        rho, ok = rho.cpu().numpy()[keep], ok.cpu().numpy()[keep]
        print(f"  margin {100 * mg:g}%｜有上限像素上的平均 W {wsum[mg][0] / max(wsum[mg][1], 1):.4f}｜有證據 {100 * ok.mean():.1f}%")
        print(f"    {'類別':16}{'顆數':>10}{'有證據%':>9}{'ρ>0.5 %':>9}{'ρ>0.3 %':>9}{'ρ>0.7 %':>9}{'ρ 中位(有證據)':>14}")
        for nm, k in cats:
            if k.sum() == 0:
                continue
            r_ = rho[k]
            print(f"    {nm:16}{int(k.sum()):>10,}{100 * ok[k].mean():>9.1f}{100 * (r_ > 0.5).mean():>9.2f}{100 * (r_ > 0.3).mean():>9.2f}"
                  f"{100 * (r_ > 0.7).mean():>9.2f}{np.median(r_[ok[k]]) if ok[k].any() else float('nan'):>14.3f}", flush=True)
        ug = dict(cats)["真地下 < −1m"]; ab = np.zeros_like(ug)
        for nm, k in cats:
            if nm.startswith(("貼地", "車輛", "路燈", "樓頂", "立面", "斜的")):
                ab |= k
        edges = np.linspace(0, 1, 11)
        h1 = np.histogram(rho[ug & ok], edges)[0]; h2 = np.histogram(rho[ab & ok], edges)[0]
        print("    ρ 直方圖（有證據；0～1 十格，%）真地下：" + " ".join(f"{100 * x / max(h1.sum(), 1):4.1f}" for x in h1))
        print("                                    地面以上：" + " ".join(f"{100 * x / max(h2.sum(), 1):4.1f}" for x in h2), flush=True)
    del model, acc
    torch.cuda.empty_cache()
print(f"\n過關（規格 2 §4）：第一個模型（CityGS）各類 ρ>0.5 <= 約 3%；我方真地下 >= 約 50%（>= 30% 也可排、要註明）｜{time.time() - t0:.0f} s")
print("DONE")
