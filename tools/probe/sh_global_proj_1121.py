#!/usr/bin/env python
"""★ 11-21 SH 投影到「相機幾何算出的全域可觀測基底」後的 test PSNR（40 §1.3）。不訓練。

s' = B_k^T B_k s（係數空間的正交投影；B_k＝M̄ 前 k 個特徵向量，M̄ 只用相機幾何算，tools/probe 的 sh_geo）。
這保住的是「整個觀測仰角帶」上的顏色（任何相機可能看的方向），不是只保住某顆實際被看到的方向（逐顆最小平方會把沒被看到的成分歸零 => test 變差）。
對全場景所有顆粒都投影，渲染看向某塊的 test 幀：比 GT PSNR 變化、與完整 SH 渲染之間的 PSNR。
用法：python tools/probe/sh_global_proj_1121.py --ckpt <合併> --basis <evec.npy> --block 2 --k 9 10 12 [--ntest 16]
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--basis", required=True); ap.add_argument("--block", type=int, required=True)
ap.add_argument("--k", type=int, nargs="+", default=[9, 10, 12]); ap.add_argument("--ntest", type=int, default=16)
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
a = ap.parse_args()
from internal.utils.gaussian_model_loader import GaussianModelLoader
import eval_official_test as eot

model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=torch.device("cpu"), eval_mode=True)
for f_ in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "exact_strip_cull", "inv_homog"):
    setattr(renderer, f_, False)
G = {k: v.data for k, v in model.gaussians.items()}
X = G["means"].double().numpy(); sc = model.get_scaling.detach().double().numpy(); rad = 3.0 * sc.max(axis=1)
V = torch.from_numpy(np.load(a.basis)).float()                    # [16,16]，欄＝特徵向量（大到小）
P = torch.load(os.path.join(a.D, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
Rp = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
ids = P["partition_coordinates"]["id"].numpy(); xys = P["partition_coordinates"]["xy"].double().numpy(); size = P["scene_config"]["partition_size"].double().numpy()
k_ = int(np.where((ids[:, 0] == max(a.block, 0) % 4) & (ids[:, 1] == max(a.block, 0) // 4))[0][0])
tn, tc = eot.load_test_cameras(a.test_dir, 1.2)
pick = []
for i in range(len(tn)):
    R = tc.R[i].double().numpy(); T = tc.T[i].double().numpy(); c = -R.T @ T; f = R[2]
    if f[2] >= -1e-3: continue
    t = (0.0 - c[2]) / f[2]; p = ((c + t * f) @ Rp.T)[:2]
    if xys[k_][0] <= p[0] < xys[k_][0] + size[k_][0] and xys[k_][1] <= p[1] < xys[k_][1] + size[k_][1]:
        pick.append(i)
pick = pick[:: max(1, len(pick) // a.ntest)][: a.ntest]
if a.block < 0:
    pick = list(range(len(tn)))          # ★ 全部 741 幀
dev = torch.device("cuda"); bg = torch.zeros(3, device=dev); keys = list(G.keys())
psnr = lambda x, y: float(-10 * torch.log10(((x - y) ** 2).mean().clamp_min(1e-12)))


def rend(i):
    R = tc.R[i].double().numpy(); T = tc.T[i].double().numpy()
    W, H = int(tc.width[i]), int(tc.height[i]); fx, fy, cx, cy = (float(tc.fx[i]), float(tc.fy[i]), float(tc.cx[i]), float(tc.cy[i]))
    Xc = X @ R.T + T; z = Xc[:, 2]
    u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy; r = fx * rad / np.maximum(z, 1e-6)
    idx = torch.from_numpy(np.where((z > 0.2) & (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H))[0])
    sub = {k2: torch.nn.Parameter(G[k2][idx].to(dev), requires_grad=False) for k2 in keys}
    old = model.gaussians
    model.__dict__["_modules"] = dict(model._modules); model.gaussians = torch.nn.ParameterDict(sub)
    try:
        with torch.no_grad():
            o = renderer(tc[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
    finally:
        model.gaussians = old
    del sub; torch.cuda.empty_cache(); return o


print(f"看向 b{a.block} 的 test 幀 {len(pick)} 張：{[tn[i] for i in pick]}", flush=True)
shs0 = G["shs"].clone()
PROJS = {k: torch.einsum("jl,nlc->njc", V[:, :k] @ V[:, :k].T, shs0) for k in a.k}   # 係數空間正交投影
base, d_gt, d_full = [], {k: [] for k in a.k}, {k: [] for k in a.k}
for n_, i in enumerate(pick):
    gt = torch.from_numpy(np.array(Image.open(os.path.join(a.test_dir, "images_1.2", tn[i])).convert("RGB"))).float().permute(2, 0, 1).to(dev) / 255
    G["shs"] = shs0; f0 = rend(i); b0 = psnr(f0, gt); base.append(b0)
    for k in a.k:
        G["shs"] = PROJS[k]; o = rend(i); d_gt[k].append(psnr(o, gt) - b0); d_full[k].append(psnr(o, f0))
    if (n_ + 1) % 100 == 0:
        print(f"  {n_ + 1}／{len(pick)}：" + "｜".join(f"k={k} {np.mean(d_gt[k]):+.4f}" for k in a.k), flush=True)
G["shs"] = shs0
print(f"完整 SH vs GT：PSNR 平均 {np.mean(base):.3f}（{len(pick)} 張）")
for k in a.k:
    j = int(np.argmin(d_gt[k]))
    print(f"  k={k:>2}（{3 * k} 個 float）：對 GT 的 PSNR 變化 平均 {np.mean(d_gt[k]):+.4f}、最差 {d_gt[k][j]:+.4f}（{tn[pick[j]]}）"
          f"｜與完整 SH 之間 PSNR 平均 {np.mean(d_full[k]):.1f}、最低 {min(d_full[k]):.1f} dB", flush=True)
