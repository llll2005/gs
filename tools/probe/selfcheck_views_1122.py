#!/usr/bin/env python
"""★ 11-22 交 viewer 前的自檢（本機 6 GB 也跑得動：每個視角只把畫面內的顆粒搬上 GPU）。不訓練。
同一組視角渲染幾個 ckpt，存成「各模型橫排、各視角直排」的 PNG，並印出模型兩兩之間的 PSNR。
視角：
  test 幀（--test，MatrixCity 741 幀的檔名，不含副檔名）：訓練仰角帶內，GT 放第一欄
  低角度（--low N）：場景中心四周 N 個方位、離地 --low_h x 訓練鏡頭高度中位，看向場景中心（訓練仰角帶外：SH 低維化最可能露餡的角度）
  斜看（--oblique）：場景中心上方 0.8 x 高度、往外看
距離與高度都用「訓練鏡頭高度中位」的倍數（不用場景公尺）。地面高度＝SfM 點 z 的第 10 百分位（中位會被屋頂拉高）。
用法：python tools/probe/selfcheck_views_1122.py --models 預設=<ckpt> 新底10=<ckpt> --test 0244 0350 --low 4 --out <png>
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--test", nargs="*", default=[])
ap.add_argument("--low", type=int, default=4); ap.add_argument("--low_h", type=float, default=0.15)
ap.add_argument("--oblique", action="store_true")
ap.add_argument("--W", type=int, default=800); ap.add_argument("--H", type=int, default=450); ap.add_argument("--fov", type=float, default=60.0)
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
ap.add_argument("--test_img_sub", default="images_1.2"); ap.add_argument("--test_down", type=float, default=1.2)
a = ap.parse_args()
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_images_binary
from internal.cameras.cameras import Cameras
import eval_official_test as eot
from internal.utils.colmap_fast import points3D_arrays

# ── 場景尺度（訓練鏡頭、可靠 SfM）──
ims = read_images_binary(os.path.join(a.D, "sparse/0/images.bin"))
C = np.array([-im.qvec2rotmat().T @ np.array(im.tvec) for im in ims.values()])
P = points3D_arrays(os.path.join(a.D, "sparse/0/points3D.bin"))
xyz = P["xyz"] if isinstance(P, dict) else P[0]
zg = float(np.percentile(xyz[:, 2], 10)); hmed = float(np.median(C[:, 2] - zg)); ctr = np.r_[np.median(C[:, :2], 0), zg]
R_xy = float(np.percentile(np.linalg.norm(C[:, :2] - ctr[:2], axis=1), 60))
print(f"地面 z {zg:.4f}｜鏡頭高度中位 {hmed:.4f}｜場景中心 {ctr.round(3)}｜鏡頭水平半徑 p60 {R_xy:.3f}", flush=True)


def look_at(eye, tgt):
    f = tgt - eye; f = f / np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.99 else np.array([0, 1.0, 0])
    x = np.cross(f, up); x /= np.linalg.norm(x); y = np.cross(f, x)
    Rw = np.stack([x, y, f]); return Rw, -Rw @ eye


views = []   # (名稱, R, T, fx, fy, cx, cy, W, H, gt 路徑或 None)
fx = 0.5 * a.W / np.tan(np.radians(a.fov) / 2)
if a.test:
    tn, tc = eot.load_test_cameras(a.test_dir, a.test_down)
    for nm in a.test:
        i = next((k for k, x in enumerate(tn) if os.path.splitext(x)[0] == nm), None)
        if i is None:
            print(f"⚠ test 幀 {nm} 找不到"); continue
        views.append((f"test {nm}", tc.R[i].double().numpy(), tc.T[i].double().numpy(), float(tc.fx[i]), float(tc.fy[i]), float(tc.cx[i]), float(tc.cy[i]),
                      int(tc.width[i]), int(tc.height[i]), os.path.join(a.test_dir, a.test_img_sub, tn[i])))
for k in range(a.low):
    th = 2 * np.pi * k / max(a.low, 1)
    eye = ctr + np.r_[R_xy * np.cos(th), R_xy * np.sin(th), a.low_h * hmed]
    tgt = ctr + np.r_[0, 0, 0.1 * hmed]
    Rw, T = look_at(eye, tgt); views.append((f"low {a.low_h:g}h az{int(np.degrees(th))}", Rw, T, fx, fx, a.W / 2, a.H / 2, a.W, a.H, None))
if a.oblique:
    eye = ctr + np.r_[0, 0, 0.8 * hmed]; tgt = ctr + np.r_[R_xy, 0, 0]
    Rw, T = look_at(eye, tgt); views.append(("oblique 0.8h", Rw, T, fx, fx, a.W / 2, a.H / 2, a.W, a.H, None))

dev = torch.device("cuda"); bg = torch.zeros(3, device=dev)
t_ = lambda x, d=torch.float32: torch.tensor(np.array(x), dtype=d)


def cam_of(v):
    _, Rw, T, fx_, fy_, cx_, cy_, W, H, _ = v
    return Cameras(R=t_([Rw]), T=t_([T]), fx=t_([fx_]), fy=t_([fy_]), cx=t_([cx_]), cy=t_([cy_]), width=t_([W]).to(torch.int16),
                   height=t_([H]).to(torch.int16), appearance_id=torch.zeros(1, dtype=torch.int), normalized_appearance_id=torch.zeros(1),
                   distortion_params=None, camera_type=torch.zeros(1, dtype=torch.int8))[0]


cols, labels = [], []
if any(v[-1] for v in views):
    cols.append([np.array(Image.open(v[-1]).convert("RGB").resize((a.W, a.H))) if v[-1] else np.zeros((a.H, a.W, 3), np.uint8) for v in views])
    labels.append("GT")
for spec in a.models:
    lab, path = spec.split("=", 1)
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(path, device=torch.device("cpu"), eval_mode=True)
    for f_ in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train", "exact_strip_cull", "inv_homog"):
        if hasattr(renderer, f_):
            setattr(renderer, f_, False)
    G = {k: v.data for k, v in model.gaussians.items()}
    X = G["means"].double().numpy(); rad = 3.0 * model.get_scaling.detach().double().numpy().max(axis=1)
    print(f"[{lab}] N={len(X):,}｜shs_rest {tuple(G['shs_rest'].shape) if 'shs_rest' in G else '-'}", flush=True)
    col = []
    for v in views:
        _, Rw, T, fx_, fy_, cx_, cy_, W, H, _ = v
        Xc = X @ Rw.T + T; z = Xc[:, 2]
        u = fx_ * Xc[:, 0] / np.maximum(z, 1e-6) + cx_; vv = fy_ * Xc[:, 1] / np.maximum(z, 1e-6) + cy_; r = fx_ * rad / np.maximum(z, 1e-6)
        idx = torch.from_numpy(np.where((z > 0.2) & (u + r > 0) & (u - r < W) & (vv + r > 0) & (vv - r < H))[0])
        old = model.gaussians
        model.gaussians = torch.nn.ParameterDict({k: torch.nn.Parameter(G[k][idx].to(dev), requires_grad=False) for k in G})
        try:
            with torch.no_grad():
                o = renderer(cam_of(v).to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
        finally:
            model.gaussians = old
        col.append(np.array(Image.fromarray((o.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).resize((a.W, a.H))))
        print(f"  {v[0]}：畫面內 {len(idx):,} 顆", flush=True)
        torch.cuda.empty_cache()
    cols.append(col); labels.append(lab)
    del model, renderer, G; torch.cuda.empty_cache()

mc = [i for i, l in enumerate(labels) if l != "GT"]
psnr = lambda x, y: float(-10 * np.log10(max(((x.astype(np.float64) - y) ** 2).mean() / 255 ** 2, 1e-12)))
for ri, v in enumerate(views):
    s = f"{v[0]:>18}"
    if labels[0] == "GT" and v[-1]:
        s += "｜對 GT " + " ".join(f"{labels[c]} {psnr(cols[c][ri], cols[0][ri]):.2f}" for c in mc)
    if len(mc) >= 2:
        s += f"｜{labels[mc[0]]} vs {labels[mc[1]]} {psnr(cols[mc[0]][ri], cols[mc[1]][ri]):.1f} dB"
    print(s)
canvas = Image.new("RGB", (a.W * len(cols), (a.H + 20) * len(views) + 20), (20, 20, 20)); dr = ImageDraw.Draw(canvas)
for ci, (lab, col) in enumerate(zip(labels, cols)):
    dr.text((ci * a.W + 6, 4), lab, fill=(255, 255, 0))
    for ri, img in enumerate(col):
        canvas.paste(Image.fromarray(img), (ci * a.W, 20 + ri * (a.H + 20) + 20)); dr.text((ci * a.W + 6, 20 + ri * (a.H + 20) + 4), views[ri][0], fill=(200, 200, 200))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); canvas.save(a.out); print("=>", a.out)
