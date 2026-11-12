"""11-12 規格 4 §2 前提檢查：「表面前面很遠也罰」（渲染的期望深度 z_r < 先驗 z_p／(1+f) 就罰）在訓練前先量三件事（只渲染、不訓練；GPU；cwd＝repo 根）。
MatrixCity 真深度只用來分類（海面像素、誤差），不進機制。先驗＝訓練時的上限圖那一份（校正深度＋可靠度過濾＋塊外遮罩 A）。
  (1) 海面像素（真深度世界 z 在海平面 ±0.004、真深度法向躺平；定義同 monodepth_sea_1102）裡有可靠先驗的比例、先驗誤差，
      以及「真的海面本身會被罰」的比例（z_真 < z_p／(1+f)，先驗太深）；全部 A 像素的同一個比例＝這條規則對真表面的誤傷率。
  (2) 我方模型（--ours，例：上限 0.5）：海面上方浮塊所在的像素（真表面是海、渲染的期望深度反投影後高於海平面 5 m、α > 0.5）有多少有可靠先驗、多少會被命中。
  (3) 細物像素（**真表面**落在路燈帶＝開闊地面格離地 3～15 m、車輛帶 0.5～2.5 m、樓頂突出物＝建築格高於該格 SfM 最高點 3 m 以上；真深度只分類）被命中的比例：
      「完美模型」（渲染深度＝真深度）與 CityGS 同塊各一欄，要 <= 約 3%（規格 4 §2）。f＝5%（規格預設）、3%、10% 各算一次。
      （11-12 第一版用 CityGS 自己的渲染表面分類 => 半透明筆觸浮在路面上方也被算成路燈帶，循環論證，改成用真表面。）
(1) 很低或 (3) 沒過 => 數字貼給使用者，不要硬排（規格 4 §2）。
用法（lab，佇列 [solo]）：
  conda run -n gspl --no-capture-output python -u tools/probe/front_cap_premise_1112.py --block 1 \
     --ours 'outputs/lab/g44tn_s30g50np16tricap0p5_best1maskdistrfed10b4/blocks/block_1/checkpoints/*.ckpt' \
     --citygs '../cityGS_origin/outputs/citygsv2_mc_aerial_sh2_trim/blocks/block_1/checkpoints/*.ckpt'
"""
import argparse
import glob
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
sys.path.insert(0, os.path.join(ROOT, "tools", "probe"))
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.gaussian_model_loader import GaussianModelLoader
from make_cap_masks import world_normals

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--ours", required=True)
ap.add_argument("--citygs", required=True)
ap.add_argument("--fronts", default="0.05,0.03,0.10")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--max_cams", type=int, default=0)
ap.add_argument("--depth_dir", default="estimated_depths_tri")
ap.add_argument("--rel_dir", default="estimated_depths_tri_rel")
ap.add_argument("--scale_name", default="estimated_depth_scales_tri")
a = ap.parse_args()
B = a.block; D = "data/matrix_city/aerial/train/block_all"; MU = 100.0; dev = torch.device("cuda")
FR = [float(x) for x in a.fronts.split(",")]
t0 = time.time()
import monodepth_sea_1102 as M            # 真深度配對、單位檢查、海平面 Z_SEA／SEA_TOL（真深度只量測）

# ---- 地面格（同 behind_offline_1111）＋每格 SfM 最高點 ----
p = torch.load(f"{D}/partition/partitions-dim_4_4_visibility_0.08/partitions.pt", map_location="cpu", weights_only=False)
pc_ = PartitionCoordinates(id=p["partition_coordinates"]["id"], xy=p["partition_coordinates"]["xy"])
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
zmax = np.full(shp[0] * shp[1], np.nan); zmax[u_] = S[order, 2][f_ + c_ - 1]; zmax = zmax.reshape(shp)
ground = np.isfinite(med) & (med < 0.03)
bldg = maximum_filter((np.isfinite(med) & ~ground).astype(np.uint8), size=5) > 0
open_g = ground & ~bldg
gz = np.where(ground, med, np.nan)
num = uniform_filter(np.nan_to_num(gz), size=5); den = uniform_filter(np.isfinite(gz).astype(float), size=5)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
GL = float(np.nanmedian(gz))
T = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
t_gfill = T(np.where(np.isfinite(gfill), gfill, GL)).float(); t_open = T(open_g); t_ground = T(ground)
t_zmax = T(np.where(np.isfinite(zmax), zmax, np.inf)).float(); t_bcell = T(np.isfinite(med) & ~ground)
t_o0 = torch.tensor(o0, device=dev); t_shp = torch.tensor(shp, device=dev)


def cell_info(xw):
    """xw [M,3]（世界）=> (離地高度 m, 開闊地面格, 建築格, 高於該格 SfM 最高點 m)。格外的點都當 False。"""
    cc = torch.floor(xw[:, :2] / G).long() - t_o0
    ok = (cc[:, 0] >= 0) & (cc[:, 0] < t_shp[0]) & (cc[:, 1] >= 0) & (cc[:, 1] < t_shp[1])
    cc = torch.minimum(torch.maximum(cc, torch.zeros_like(cc)), t_shp - 1)
    h = (xw[:, 2] - t_gfill[cc[:, 0], cc[:, 1]]) * MU
    above = (xw[:, 2] - t_zmax[cc[:, 0], cc[:, 1]]) * MU
    return h, ok & t_open[cc[:, 0], cc[:, 1]], ok & t_bcell[cc[:, 0], cc[:, 1]], above


# ---- 渲染器（期望深度）、訓練相機、塊外遮罩：取自 --ours ----
ours_ck = sorted(glob.glob(a.ours))[-1]; cg_ck = sorted(glob.glob(a.citygs))[-1]
m_ours, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ours_ck, device=dev, eval_mode=True)
m_cg, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(cg_ck, device=dev, eval_mode=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "skip_unscored_pixels", "trim_running_vc"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
renderer.depth_ratio = 0.0               # surf_depth＝期望深度（規格 4 §2：半透明浮塊的中位深度抓不到）
ck = torch.load(ours_ck, map_location="cpu", weights_only=False)
dmh = ck["datamodule_hyper_parameters"]; mcfg = ck["hyper_parameters"].get("metric")
mask_dir = getattr(getattr(mcfg, "init_args", mcfg), "block_mask_dir", "") or ""
del ck
parser = dmh["parser"]
if getattr(parser, "points_from", None) == "ply":
    parser.points_from = "random"
parser.depth_dir = a.depth_dir; parser.depth_scale_name = a.scale_name; parser.depth_rel_dir = a.rel_dir
ts = parser.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
cams = ts.cameras; names = list(ts.image_names)
sel = list(range(0, len(names), max(1, a.stride)))[: a.max_cams or None]
print(f"b{B}｜訓練相機 {len(names)} 台（用 {len(sel)}）｜我方 {ours_ck}｜CityGS {cg_ck}｜遮罩 {mask_dir or '⚠⚠ 沒有'}｜海平面 {M.Z_SEA}±{M.SEA_TOL}", flush=True)

NF = len(FR)
acc = {k: 0 for k in ("A", "Ap", "sea", "sea_p", "flt", "flt_p")}
acc["true_hit"] = np.zeros(NF); acc["sea_hit"] = np.zeros(NF); acc["flt_hit"] = np.zeros(NF)
e_sea = []
CG = ["路燈帶 3～15m", "車輛帶 0.5～2.5m", "樓頂突出物 >3m", "全部"]
cg_n = np.zeros(len(CG)); cg_p = np.zeros(len(CG)); cg_hit = np.zeros((len(CG), NF)); tr_hit = np.zeros((len(CG), NF))
bg = torch.zeros(3, device=dev)


def backproject(cam, z, H, W):
    v2, u2 = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    pcm = torch.stack([(u2 + 0.5 - cam.cx) / cam.fx * z, (v2 + 0.5 - cam.cy) / cam.fy * z, z], -1).reshape(-1, 3)
    w2c = cam.world_to_camera                                # 列向量版：[p,1] @ w2c
    R = w2c[:3, :3]; t = w2c[3, :3]
    return (pcm - t) @ R.T


with torch.no_grad():
    for ii, i in enumerate(sel):
        nm = os.path.basename(str(names[i])); im = M.BYNAME.get(nm)
        dg = M.load_gt(nm) if im is not None else None
        if dg is None:
            continue
        cam = cams[i].to_device(dev); H, W = int(cam.height), int(cam.width)
        d = ts.extra_data_processor(ts.extra_data[i]) if ts.extra_data[i] is not None else None
        d = d[0] if isinstance(d, tuple) else d
        if d is None:
            continue
        inv = torch.nn.functional.interpolate(d.reshape(d.shape[-2:]).float()[None, None], size=(H, W), mode="nearest")[0, 0].to(dev)
        A = torch.ones(H, W, dtype=torch.bool, device=dev)
        if mask_dir:
            mp = os.path.join(mask_dir, nm)
            if os.path.exists(mp):
                A = torch.nn.functional.interpolate(torch.from_numpy(np.array(Image.open(mp)))[None, None].float(), size=(H, W), mode="nearest")[0, 0].to(dev) == 1
        pv = A & (inv > 1e-6)
        zp = torch.where(pv, 1.0 / inv.clamp_min(1e-12), torch.zeros_like(inv))
        # 真深度 => 世界點、海面像素
        W0, H0, fx, fy, cx, cy, R0, t0_ = M.cam_of(im)
        iy = np.clip(((np.arange(H) + 0.5) * H0 / H).astype(int), 0, H0 - 1); ix = np.clip(((np.arange(W) + 0.5) * W0 / W).astype(int), 0, W0 - 1)
        dd = dg[iy][:, ix].astype(np.float64)
        vt = np.isfinite(dd) & (dd > 0) & (dd < 1e8)
        zt_np = np.where(vt, dd * M.K_UNIT, 0.0)
        nrm, okn = world_normals(np.where(vt, 1.0 / np.maximum(zt_np, 1e-12), 0.0), fx, fy, cx, cy, W / W0, H / H0, R0)
        zt = T(zt_np).float()
        xt = backproject(cam, zt, H, W).reshape(H, W, 3)
        sea = T(vt).bool() & ((xt[..., 2] - M.Z_SEA).abs() < M.SEA_TOL) & T(okn & (np.abs(nrm[..., 2]) > 0.8)) & A
        acc["A"] += int(A.sum()); acc["Ap"] += int(pv.sum())
        tv = pv & (zt > 0)
        for k, f in enumerate(FR):
            acc["true_hit"][k] += int((tv & (zt < zp / (1 + f))).sum())
            acc["sea_hit"][k] += int((sea & pv & (zt < zp / (1 + f))).sum())
        acc["sea"] += int(sea.sum()); acc["sea_p"] += int((sea & pv).sum())
        if bool((sea & pv).any()):
            e_sea.append((zt[sea & pv] / zp[sea & pv] - 1).float().cpu().numpy())
        # (2) 我方：海面上方的浮塊像素
        out = renderer(cam, m_ours, bg_color=bg)
        zr = out["surf_depth"].reshape(H, W); al = out["rend_alpha"].reshape(H, W)
        xr = backproject(cam, zr, H, W).reshape(H, W, 3)
        flt = sea & (al > 0.5) & (zr > 0) & (xr[..., 2] > M.Z_SEA + 0.05)
        acc["flt"] += int(flt.sum()); acc["flt_p"] += int((flt & pv).sum())
        for k, f in enumerate(FR):
            acc["flt_hit"][k] += int((flt & pv & (zr < zp / (1 + f))).sum())
        # (3) 細物像素（真表面分類）：完美模型與 CityGS
        oc = renderer(cam, m_cg, bg_color=bg)
        zc = oc["surf_depth"].reshape(H, W); ac = oc["rend_alpha"].reshape(H, W)
        h, iso, isb, above = cell_info(xt.reshape(-1, 3))
        h = h.reshape(H, W); iso = iso.reshape(H, W); isb = isb.reshape(H, W); above = above.reshape(H, W)
        base_ok = A & (zt > 0)
        cls = [base_ok & iso & (h > 3) & (h < 15), base_ok & iso & (h > 0.5) & (h < 2.5), base_ok & isb & (above > 3), base_ok]
        cg_ok = (ac > 0.5) & (zc > 0)
        for j, mk in enumerate(cls):
            cg_n[j] += int(mk.sum()); cg_p[j] += int((mk & pv).sum())
            for k, f in enumerate(FR):
                tr_hit[j, k] += int((mk & pv & (zt < zp / (1 + f))).sum())
                cg_hit[j, k] += int((mk & pv & cg_ok & (zc < zp / (1 + f))).sum())
        if (ii + 1) % 50 == 0:
            print(f"  相機 {ii + 1}／{len(sel)}（{time.time() - t0:.0f} s）", flush=True)

pc = lambda x, y: 100.0 * x / max(y, 1)
print(f"\n== (1) 先驗在海面上可不可靠（b{B}；A 像素 {acc['A']:,}、其中有可靠先驗 {pc(acc['Ap'], acc['A']):.1f}%）")
print(f"  海面像素 {acc['sea']:,}（佔 A {pc(acc['sea'], acc['A']):.2f}%）｜其中有可靠先驗 {pc(acc['sea_p'], acc['sea']):.1f}%")
if e_sea:
    e = np.concatenate(e_sea)
    print(f"  海面上先驗誤差 e＝z_真／z_先驗 − 1：中位 {100 * np.median(e):+.2f}%、|e| 中位 {100 * np.median(np.abs(e)):.2f}%、p10／p90 {100 * np.percentile(e, 10):+.2f}／{100 * np.percentile(e, 90):+.2f}%")
for k, f in enumerate(FR):
    print(f"  f {100 * f:g}%：真的海面本身會被罰 {pc(acc['sea_hit'][k], acc['sea_p']):.2f}%（佔有先驗的海面像素）｜全部 A 像素的真表面會被罰 {pc(acc['true_hit'][k], acc['Ap']):.2f}%")
print(f"\n== (2) 我方模型海面上方的浮塊像素（真表面是海、渲染表面高於海平面 5 m、α>0.5）：{acc['flt']:,}（佔海面像素 {pc(acc['flt'], acc['sea']):.2f}%）｜其中有可靠先驗 {pc(acc['flt_p'], acc['flt']):.1f}%")
for k, f in enumerate(FR):
    print(f"  f {100 * f:g}%：命中 {pc(acc['flt_hit'][k], acc['flt_p']):.1f}%（佔有先驗的浮塊像素）＝佔全部浮塊像素 {pc(acc['flt_hit'][k], acc['flt']):.1f}%")
print(f"\n== (3) 細物像素（真表面分類）被命中的比例（佔有先驗的像素；要 <= 約 3%）｜每格：完美模型／CityGS")
print(f"  {'類別':18}{'像素':>12}{'有先驗%':>9}" + "".join(f"{'f ' + format(100 * f, 'g') + '%':>18}" for f in FR))
for j, nm in enumerate(CG):
    print(f"  {nm:18}{int(cg_n[j]):>12,}{pc(cg_p[j], cg_n[j]):>9.1f}"
          + "".join(f"{pc(tr_hit[j, k], cg_p[j]):>9.2f}／{pc(cg_hit[j, k], cg_p[j]):<8.2f}" for k in range(NF)), flush=True)
print(f"\n判讀（規格 4 §2）：(1) 海面有可靠先驗的比例很低，或 (3) 任一類 > 約 3% => 數字貼給使用者，不排臂｜{time.time() - t0:.0f} s")
print("DONE")
