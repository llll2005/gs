#!/usr/bin/env python
"""★ 11-21 場景範圍外的遠處顆粒到底有沒有貢獻（使用者：「既然沒被貢獻度 trim 剪掉，不就代表有貢獻？」）。

1. 遠處集合 F＝離可靠 SfM 平面範圍超過 f x 最長邊（同 slim_view_ckpt --clip_far）。
2. 所有訓練相機（down_sample 1.2）裡，F 有任何一顆的中心投進畫面（含投影半徑）的有幾台、是哪幾塊的相機。
3. 挑投影面積最大的 K 台，對每台只拿畫面內的顆粒（本機 6 GB 放不下全場景）渲染兩次：有 F／沒有 F，
   報 GT PSNR 的變化、畫面被改到（|差| > 1/255）的像素比例。
用法：python tools/probe/far_contrib_1121.py --ckpt <合併 ckpt> [--f 0.25] [--K 16]
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--f", type=float, default=0.25)
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--random", type=int, default=0, help=">0：改成在「F 投進畫面」的相機裡隨機抽這麼多台（seed 0）")
a = ap.parse_args()

from internal.utils.colmap import read_points3D_binary
from internal.utils.gaussian_model_loader import GaussianModelLoader
import eval_official_test as eot

P = torch.load(os.path.join(a.D, a.part, "partitions.pt"), map_location="cpu", weights_only=False)
Rp = P["extra_data"]["rotation_transform"][:3, :3].double().numpy()
pts = read_points3D_binary(os.path.join(a.D, "sparse/0/points3D.bin"))
q = np.stack([x.xyz for x in pts.values()])[np.array([len(x.image_ids) >= 3 and x.error < 1.5 for x in pts.values()])] @ Rp.T
lo, hi = np.percentile(q[:, :2], 0.5, axis=0), np.percentile(q[:, :2], 99.5, axis=0); mf = a.f * float((hi - lo).max())

model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(a.ckpt, device=torch.device("cpu"), eval_mode=True)
for f_ in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
    setattr(renderer, f_, False)
G = {k: v.data for k, v in model.gaussians.items()}
X = G["means"].double().numpy()
xy = (X @ Rp.T)[:, :2]
far = ~((xy[:, 0] >= lo[0] - mf) & (xy[:, 0] <= hi[0] + mf) & (xy[:, 1] >= lo[1] - mf) & (xy[:, 1] <= hi[1] + mf))
sc = model.get_scaling.detach().double().numpy() if hasattr(model, "get_scaling") else np.exp(G["scales"].double().numpy())
rad = 3.0 * sc.max(axis=1)
print(f"N={len(X):,}｜遠處集合 F（> {a.f} x 最長邊）{int(far.sum())} 顆")

names, cams = eot.load_test_cameras(a.D, 1.2)
part_of = {}
for b in range(16):
    fp = os.path.join(a.D, a.part, "%03d_%03d.txt" % (b % 4, b // 4))
    if os.path.exists(fp):
        for l in open(fp):
            if l.strip():
                part_of.setdefault(l.strip(), []).append(b)
Fi = np.where(far)[0]; XF = X[Fi]; rF = rad[Fi]
seen = []
for i in range(len(names)):
    R = cams.R[i].double().numpy(); T = cams.T[i].double().numpy()
    W, H = int(cams.width[i]), int(cams.height[i]); fx, fy, cx, cy = (float(cams.fx[i]), float(cams.fy[i]), float(cams.cx[i]), float(cams.cy[i]))
    # cams.R 是 world->camera 的轉置慣例？用 eval_official_test 的定義：Cameras(R=qvec2rotmat, T=tvec) => x_c = R X + T
    Xc = XF @ R.T + T
    z = Xc[:, 2]; ok = z > 0.2
    u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy; r = fx * rF / np.maximum(z, 1e-6)
    vis = ok & (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H)
    if vis.any():
        seen.append((float(np.minimum(np.pi * r[vis] ** 2, W * H).sum()), i, int(vis.sum())))
print(f"訓練相機 {len(names)} 台裡，F 至少一顆投進畫面的：{len(seen)} 台")
blk = {}
for _, i, _ in seen:
    for b in part_of.get(names[i], [-1]):
        blk[b] = blk.get(b, 0) + 1
print("  這些相機屬於哪幾塊的訓練清單：", ", ".join(f"b{b}:{c}" for b, c in sorted(blk.items())))
seen.sort(reverse=True)
if a.random > 0:
    rng = np.random.default_rng(0); seen = [seen[j] for j in rng.choice(len(seen), min(a.random, len(seen)), replace=False)]; a.K = len(seen)
dev = torch.device("cuda"); bg = torch.zeros(3, device=dev)
mod_keys = list(G.keys())


def render_subset(idx, cam):
    sub = {k: torch.nn.Parameter(G[k][idx].to(dev), requires_grad=False) for k in mod_keys}
    old = model.gaussians
    model.__dict__["_modules"] = dict(model._modules); model.gaussians = torch.nn.ParameterDict(sub)
    try:
        with torch.no_grad():
            out = renderer(cam, model, bg_color=bg)["render"].clamp(0, 1)
    finally:
        model.gaussians = old
    del sub; torch.cuda.empty_cache()
    return out


print(f"\n{'隨機抽' if a.random else '投影面積最大的'} {min(a.K, len(seen))} 台：")
dps = []
for area, i, nf in seen[: a.K]:
    R = cams.R[i].double().numpy(); T = cams.T[i].double().numpy()
    W, H = int(cams.width[i]), int(cams.height[i]); fx, fy, cx, cy = (float(cams.fx[i]), float(cams.fy[i]), float(cams.cx[i]), float(cams.cy[i]))
    Xc = X @ R.T + T; z = Xc[:, 2]
    u = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx; v = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy; r = fx * rad / np.maximum(z, 1e-6)
    inview = (z > 0.2) & (u + r > 0) & (u - r < W) & (v + r > 0) & (v - r < H)
    cam = cams[i].to_device(dev)
    A_ = render_subset(torch.from_numpy(np.where(inview)[0]), cam)
    B_ = render_subset(torch.from_numpy(np.where(inview & ~far)[0]), cam)
    gt = torch.from_numpy(np.array(Image.open(os.path.join(a.D, "input", names[i])).convert("RGB").resize((W, H), Image.LANCZOS))).float().permute(2, 0, 1).to(dev) / 255
    psnr = lambda x: float(-10 * torch.log10(((x - gt) ** 2).mean()))
    ch = float(((A_ - B_).abs().amax(0) > 1 / 255).float().mean())
    dps.append(psnr(B_) - psnr(A_))
    if a.random and ch == 0: continue
    print(f"  {names[i]}（b{part_of.get(names[i], ['-'])}）F 投進 {nf} 顆｜畫面內顆數 {int(inview.sum()):,}｜被改到的像素 {100 * ch:.2f}%｜"
          f"PSNR 有 F {psnr(A_):.3f} -> 拿掉 F {psnr(B_):.3f}（{psnr(B_) - psnr(A_):+.3f}）")
if dps:
    print(f"\n拿掉 F 的 PSNR 變化：中位 {np.median(dps):+.4f}、最差 {min(dps):+.4f}、最好 {max(dps):+.4f} dB")
