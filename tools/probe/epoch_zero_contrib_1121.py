#!/usr/bin/env python
"""★ 11-21 「一整輪（該塊每台訓練相機各一次）的累計貢獻＝0」的顆粒有多少（使用者：增生期中間把累計貢獻 0 的清掉、下一輪再清一次）。

用合併模型＋某塊的全部訓練相機，每台只取畫面內的顆粒，走 trim 的 record pass（V＝Σ T·α、C＝處理到的像素數，在 alpha 判斷之前就計）累加。
報：本塊相機看得到範圍內的顆粒裡 V＝0／V 很小／C＝0（完全被擋）各多少、Σ o·面積、以及遠處集合（clip_far）的 V。
用法：python tools/probe/epoch_zero_contrib_1121.py --ckpt <合併 ckpt> --block 3 [--every 1]
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--block", type=int, required=True)
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--every", type=int, default=1, help="每幾台相機取一台（1＝整輪）")
ap.add_argument("--f", type=float, default=0.25)
ap.add_argument("--out", default="")
a = ap.parse_args()

from internal.utils.colmap import read_points3D_binary
from internal.utils.gaussian_model_loader import GaussianModelLoader
import eval_official_test as eot

model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=torch.device("cpu"), eval_mode=True)
for f_ in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "exact_strip_cull", "inv_homog"):
    setattr(renderer, f_, False)
G = {k: v.data for k, v in model.gaussians.items()}
X = G["means"].double().numpy(); N = len(X)
sc = model.get_scaling.detach().double().numpy(); rad = 3.0 * sc.max(axis=1)
op = model.get_opacity.detach().double().numpy().reshape(-1); area = np.pi * sc[:, 0] * sc[:, 1]
P = torch.load(os.path.join(a.D, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
Rp = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
pts = read_points3D_binary(os.path.join(a.D, "sparse/0/points3D.bin"))
q = np.stack([x.xyz for x in pts.values()])[np.array([len(x.image_ids) >= 3 and x.error < 1.5 for x in pts.values()])] @ Rp.T
lo, hi = np.percentile(q[:, :2], 0.5, axis=0), np.percentile(q[:, :2], 99.5, axis=0); mf = a.f * float((hi - lo).max())
xy = (X @ Rp.T)[:, :2]
far = ~((xy[:, 0] >= lo[0] - mf) & (xy[:, 0] <= hi[0] + mf) & (xy[:, 1] >= lo[1] - mf) & (xy[:, 1] <= hi[1] + mf))

names, cams = eot.load_test_cameras(a.D, 1.2)
want = [l.strip() for l in open(os.path.join(a.D, a.part, "%03d_%03d.txt" % (a.block % 4, a.block // 4))) if l.strip()][:: a.every]
pos = {n: i for i, n in enumerate(names)}
dev = torch.device("cuda"); bg = torch.zeros(3, device=dev)
V = np.zeros(N); C = np.zeros(N); Fv = np.zeros(N, np.int32)
keys = list(G.keys()); t0 = time.time()
for k, nm in enumerate(want):
    i = pos[nm]
    R = cams.R[i].double().numpy(); T = cams.T[i].double().numpy()
    W, H = int(cams.width[i]), int(cams.height[i]); fx, fy, cx, cy = (float(cams.fx[i]), float(cams.fy[i]), float(cams.cx[i]), float(cams.cy[i]))
    Xc = X @ R.T + T; z = Xc[:, 2]
    u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy; r = fx * rad / np.maximum(z, 1e-6)
    idx = np.where((z > 0.2) & (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H))[0]
    Fv[idx] += 1
    sub = {kk: torch.nn.Parameter(G[kk][torch.from_numpy(idx)].to(dev), requires_grad=False) for kk in keys}
    old = model.gaussians
    model.__dict__["_modules"] = dict(model._modules); model.gaussians = torch.nn.ParameterDict(sub)
    try:
        with torch.no_grad():
            mean, cov = renderer(cams[i].to_device(dev), model, bg_color=bg, record_transmittance=True, record_coverage=True)
    finally:
        model.gaussians = old
    cov = cov.double().cpu().numpy(); V[idx] += mean.double().cpu().numpy() * cov; C[idx] += cov
    del sub; torch.cuda.empty_cache()
    if (k + 1) % 100 == 0:
        print(f"  {k + 1}／{len(want)} 台（{time.time() - t0:.0f} s）", flush=True)

inf = Fv > 0
print(f"\n══ b{a.block}：{len(want)} 台訓練相機（{'整輪' if a.every == 1 else f'每 {a.every} 台取 1'}）｜N={N:,}｜至少進過一次畫面 {int(inf.sum()):,}")
for lab, sel in (("V＝0（一次都沒混合到）", inf & (V == 0)), ("C＝0（完全被擋，從沒被算到）", inf & (C == 0)),
                 ("V < 1e-3", inf & (V < 1e-3)), ("V < 1e-2", inf & (V < 1e-2)), ("V < 0.1", inf & (V < 0.1))):
    print(f"  {lab:<26} {int(sel.sum()):>9,} 顆（畫面內的 {100 * sel.sum() / max(inf.sum(), 1):5.2f}%）｜o>0.5 {int((sel & (op > 0.5)).sum()):>8,}｜Σ o·面積 {float((op * area)[sel].sum()):.3g}")
fi = far & inf
print(f"  遠處集合（> {a.f} x 最長邊）進過本塊畫面的 {int(fi.sum())} 顆：V 總和 {float(V[fi].sum()):.3g}、V＝0 的 {int((fi & (V == 0)).sum())}、C＝0 的 {int((fi & (C == 0)).sum())}、最大 V {float(V[fi].max()) if fi.any() else 0:.3g}")
if a.out:
    np.savez_compressed(a.out, V=V.astype(np.float32), C=C.astype(np.float32), F=Fv, far=far)
    print("=>", a.out)
