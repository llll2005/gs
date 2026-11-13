"""★ 2026-11-13 合併借鑑三點＋線狀合併的離線前提檢查（使用者「三點採用、記得標來源，排實驗跟測試」；cwd＝repo 根；只渲染、不訓練）。
對一個我方單塊 ckpt（原始參數，不 pre-activate），每個變體從原樣重新載入、跑一次 merge_step（max_frac 1＝所有互為最佳的配對一次併完、顏色門檻同底 0.1），
刪掉 j，然後用本塊訓練相機（--stride）量塊外遮罩 A 像素的 PSNR 變化。=> 「同樣減掉 1% 顆數，畫質掉多少」就是合併規則的好壞。
變體：現行（union 不透明度）｜L2 最佳不透明度｜只在平滑區（tex_tol 0.02／0.03／0.05）｜線狀合併｜線狀＋L2。
另外印：平面候選的 k 近鄰顏色平均絕對偏差分位數（校準 tex_tol 用）、配對的高度帶（離 SfM 地面；只是量測分類）、合併誤差中位（L2 版＝sin²α）。
用法（lab，佇列 [solo]）：conda run -n gspl --no-capture-output python tools/probe/merge_variants_1113.py --block 1 --ckpt '<ckpt 樣式>'
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
from internal.utils.colmap import read_points3D_binary
from internal.utils.citygs_partitioning_utils import PartitionCoordinates, CityGSPartitioning
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.density_controllers.merge_ops import merge_step
from internal.metrics.geom_gate import morton_neighbors

ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--color_tol", type=float, default=0.1)
a = ap.parse_args()
B = a.block; D = "data/matrix_city/aerial/train/block_all"; MU = 100.0; dev = torch.device("cuda")
t0 = time.time()
ckf = sorted(glob.glob(a.ckpt))[-1]

# ---- SfM 地面（同 dead_weight_1113；只用來分類）----
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
ground = np.isfinite(med) & (med < 0.03)
gz = np.where(ground, med, np.nan)
num = uniform_filter(np.nan_to_num(gz), size=5); den = uniform_filter(np.isfinite(gz).astype(float), size=5)
gfill = np.where(np.isfinite(gz), gz, np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
GL = float(np.nanmedian(gz))


def height(xyz):
    cc = np.floor(xyz[:, :2] / G).astype(int) - o0
    okc = (cc[:, 0] >= 0) & (cc[:, 0] < shp[0]) & (cc[:, 1] >= 0) & (cc[:, 1] < shp[1])
    cc = np.clip(cc, 0, np.array(shp) - 1)
    g = gfill[cc[:, 0], cc[:, 1]]
    return (xyz[:, 2] - np.where(okc & np.isfinite(g), g, GL)) * MU


BANDS = [("地下 <−0.5", -1e9, -0.5), ("貼地 ±0.5", -0.5, 0.5), ("0.5～3 m", 0.5, 3), ("3～15 m", 3, 15), ("> 15 m", 15, 1e9)]

# ---- 渲染器、訓練相機、塊外遮罩 ----
_, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ckf, device=dev, eval_mode=True)
for f in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "skip_unscored_pixels", "trim_running_vc"):
    if hasattr(renderer, f):
        setattr(renderer, f, False)
ck = torch.load(ckf, map_location="cpu", weights_only=False)
dmh = ck["datamodule_hyper_parameters"]; mcfg = ck["hyper_parameters"].get("metric")
mask_dir = getattr(getattr(mcfg, "init_args", mcfg), "block_mask_dir", "") or ""
del ck
parser = dmh["parser"]
if getattr(parser, "points_from", None) == "ply":
    parser.points_from = "random"
ts = parser.instantiate(path=dmh["path"], output_path="/tmp", global_rank=0).get_outputs().train_set
cams = ts.cameras; names = list(ts.image_names)
sel = list(range(0, len(names), max(1, a.stride)))
print(f"b{B}｜{ckf}｜訓練相機 {len(names)} 台（用 {len(sel)}）｜遮罩 {mask_dir or '沒有'}", flush=True)
bg = torch.zeros(3, device=dev)
views = []
with torch.no_grad():
    for i in sel:
        cam = cams[i].to_device(dev); H, W = int(cam.height), int(cam.width)
        A = torch.ones(H, W, dtype=torch.bool, device=dev)
        if mask_dir:
            mp = os.path.join(mask_dir, os.path.basename(str(names[i])))
            if os.path.exists(mp):
                mt = torch.from_numpy(np.array(Image.open(mp)))[None, None].float()
                A = torch.nn.functional.interpolate(mt, size=(H, W), mode="nearest")[0, 0].to(dev) == 1
        pil = Image.open(ts.image_paths[i]).convert("RGB")
        if pil.size != (W, H):
            pil = pil.resize((W, H), Image.LANCZOS)
        gt = (torch.from_numpy(np.array(pil, np.uint8)).float().permute(2, 0, 1) / 255.).half()
        views.append((cam, A.cpu(), gt))


def load():
    m, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ckf, device=dev, eval_mode=True, pre_activate=False)
    return m


def score(m):
    se = 0.0; n = 0
    with torch.no_grad():
        for cam, A, gt in views:
            img = renderer(cam, m, bg_color=bg)["render"].clamp(0, 1)
            A_ = A.to(dev); g_ = gt.to(dev).float()
            se += float(((img - g_) ** 2).mean(0)[A_].double().sum()); n += int(A_.sum())
    return -10 * np.log10(max(se / max(n, 1), 1e-12))


m0 = load(); N0 = m0.get_xyz.shape[0]
p0 = score(m0)
print(f"原樣：N {N0:,}｜訓練視角 A 像素 PSNR {p0:.4f}｜{time.time() - t0:.0f} s", flush=True)
# 校準 tex_tol：不透明顆粒的 k 近鄰 DC 顏色平均絕對偏差
with torch.no_grad():
    O = torch.sigmoid(m0.gaussians["opacities"]).reshape(-1); cand = torch.nonzero(O > 0.5, as_tuple=True)[0]
    rgb = m0.gaussians["shs_dc"][cand, 0, :].float() * 0.28209479177387814 + 0.5
    nb = morton_neighbors(m0.gaussians["means"][cand].float(), 12).long()
    mad = torch.cat([(rgb[nb[c:c + 65536]] - rgb[nb[c:c + 65536]].mean(1, keepdim=True)).abs().mean((1, 2)) for c in range(0, cand.numel(), 65536)])
    qs = torch.quantile(mad[:2_000_000].float(), torch.tensor([.1, .25, .5, .75, .9], device=dev)).tolist()
print("不透明顆粒的 k 近鄰顏色平均絕對偏差（RGB 0～1）分位 10/25/50/75/90%：" + "／".join(f"{q:.4f}" for q in qs), flush=True)
del m0; torch.cuda.empty_cache()

VARS = [("現行（union）", {}), ("L2 最佳不透明度", {"opacity_mode": "l2"}),
        ("平滑區 0.02", {"tex_tol": 0.02}), ("平滑區 0.03", {"tex_tol": 0.03}), ("平滑區 0.05", {"tex_tol": 0.05}),
        ("線狀", {"line": True}), ("線狀＋L2", {"line": True, "opacity_mode": "l2"})]
print(f"\n{'變體':16}{'配對':>9}{'其中線狀':>9}{'N 減少':>9}{'訓練視角 ΔPSNR ↑':>18}{'每減 1% 掉 dB':>14}{'合併誤差中位':>13}  配對的高度帶（留下那顆）：" + "／".join(b[0] for b in BANDS), flush=True)
for nm, kw in VARS:
    m = load()
    with torch.no_grad():
        I, J, info = merge_step(m, k=12, max_frac=1.0, color_tol=a.color_tol, chunk=65536, render_eps=1e9, **kw)
        keep = torch.ones(N0, dtype=torch.bool, device=dev); keep[J] = False
        hI = height(m.gaussians["means"][I].double().cpu().numpy()) if I.numel() else np.zeros(0)
        m.properties = {k: v[keep] for k, v in m.properties.items()}
    pm = score(m)
    red = 100 * J.numel() / N0
    bands = "／".join(f"{int(((hI >= lo) & (hI < hi)).sum()):,}" for _, lo, hi in BANDS)
    print(f"{nm:16}{I.numel():>9,}{info.get('line_pairs', 0):>9,}{red:>8.2f}%{pm - p0:>+18.4f}{(pm - p0) / max(red, 1e-9):>+14.4f}"
          f"{info.get('err_med', float('nan')):>13.4f}  {bands}", flush=True)
    del m; torch.cuda.empty_cache()
print(f"\n{time.time() - t0:.0f} s")
print("DONE")
