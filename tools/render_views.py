#!/usr/bin/env python
"""★ 11-18 交給使用者 viewer 之前的自我檢查（使用者：v5～v7 每次都被一眼看出問題 => 先自己渲染固定視角比對）。GPU；lab 上跑。
對同一塊的幾個 ckpt，用同一組固定視角渲染，存成「各模型橫排、各視角直排」的一張 PNG。
視角由本塊的海面物件與訓練鏡頭自動決定（不用場景公尺；距離用訓練鏡頭高度中位的倍數）：
  V1 俯視海面：海面磚重心正上方、高度＝1.3 x 鏡頭高度中位，正下看
  V2 海面低角度看城市：從海面磚重心往外推、高度＝0.15 x 鏡頭高度中位，看向訓練鏡頭重心（離地 0.3 x 高度）——使用者看港口煙團的角度
  V3 斜看交界：訓練鏡頭重心上方 0.8 x 高度，往海面磚重心看（看近海、交界空洞帶、外圍）
  另可 --test 加幾張 MatrixCity test 幀（GT 放第一欄）
用法：python tools/render_views.py --block 3 --obj <sea npz> --models 名=<ckpt glob> ... --out logs/views_b3.png [--test 0226 0269]
"""
import argparse, glob, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--block", type=int, required=True); ap.add_argument("--obj", default="", help="海面物件 npz（給 V1～V3 海面視角）；空＝不畫 V1～V3，只畫 --train／--test")
ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--train", nargs="*", default=[], help="★ 訓練影像名（例 2033.png）：用該訓練鏡頭的視角渲染，GT＝input/ 原圖")
ap.add_argument("--test", nargs="*", default=[]); ap.add_argument("--W", type=int, default=960); ap.add_argument("--H", type=int, default=540)
ap.add_argument("--fov", type=float, default=60.0)
ap.add_argument("--D", default="data/matrix_city/aerial/train/block_all")
ap.add_argument("--part", default="partition/partitions-dim_4_4_visibility_0.08")
ap.add_argument("--test_dir", default="data/matrix_city/aerial/test/block_all_test_official2")
a = ap.parse_args()
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.colmap import read_images_binary
from internal.cameras.cameras import Cameras
import eval_official_test as eot

if a.obj:
    z = np.load(a.obj); n = z["n"]; c = float(z["c"])
else:
    z = None; n = np.array([0, 0, 1.0]); c = -0.02565
tiles = (z["tiles"][: int(z["n_tiles0"])] if "n_tiles0" in z.files else z["tiles"]) if z is not None else None
imgs = read_images_binary(os.path.join(a.D, "sparse/0/images.bin"))
names = {l.strip() for l in open(os.path.join(a.D, a.part, "%03d_%03d.txt" % (a.block % 4, a.block // 4))) if l.strip()}
C = np.array([-v.qvec2rotmat().T @ np.array(v.tvec) for v in imgs.values() if v.name in names])
hmed = float(np.median(C @ n - c)); cam_c = C.mean(0)
sea_c = tiles.mean(0) if tiles is not None else cam_c - n * ((cam_c @ n) - c)
land = cam_c - n * ((cam_c @ n - c) - 0.3 * hmed)                  # 鏡頭重心正下方、離海面 0.3 x 高度
out_dir = sea_c - cam_c; out_dir = out_dir - (out_dir @ n) * n; out_dir /= max(np.linalg.norm(out_dir), 1e-9)

def look_at(eye, tgt):
    f = tgt - eye; f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.99 else np.array([0, 1.0, 0])
    x = np.cross(f, up); x /= np.linalg.norm(x); y = np.cross(f, x)
    Rw2c = np.stack([x, y, f]); T = -Rw2c @ eye
    return Rw2c, T
views = [("V1 top-down sea", sea_c + n * 1.3 * hmed, sea_c),
         ("V2 low from sea -> city", sea_c + out_dir * 0.5 * hmed + n * 0.15 * hmed, land),
         ("V3 oblique -> boundary", land + n * 0.8 * hmed, sea_c)]
if z is None:
    views = []
fx = 0.5 * a.W / np.tan(np.radians(a.fov) / 2)
t_ = lambda x, d=torch.float32: torch.tensor(np.array(x), dtype=d)
nv = len(views)
Rs, Ts = zip(*[look_at(e, t) for _, e, t in views]) if nv else ([], [])
cams = None if nv == 0 else Cameras(R=t_(Rs), T=t_(Ts), fx=t_([fx] * nv), fy=t_([fx] * nv), cx=t_([a.W / 2] * nv), cy=t_([a.H / 2] * nv),
               width=t_([a.W] * nv).to(torch.int16), height=t_([a.H] * nv).to(torch.int16), appearance_id=torch.zeros(nv, dtype=torch.int),
               normalized_appearance_id=torch.zeros(nv), distortion_params=None, camera_type=torch.zeros(nv, dtype=torch.int8))
tcams = None; tnames = []; gt_paths = []
if a.test:
    nt, tc = eot.load_test_cameras(a.test_dir, 1.2)
    sel = [i for i, x in enumerate(nt) if os.path.splitext(x)[0] in set(a.test)]
    tcams = tc; tnames = [(i, nt[i]) for i in sel]
    gt_paths = [os.path.join(a.test_dir, "images_1.2", nm) for _, nm in tnames]
elif a.train:
    # 訓練鏡頭：同 eval_official_test.load_test_cameras 的慣例（down_sample 1.2、內參依實際四捨五入比例縮）
    nt, tc = eot.load_test_cameras(a.D, 1.2)
    want = set(a.train)
    sel = [i for i, x in enumerate(nt) if x in want]
    tcams = tc; tnames = [(i, nt[i]) for i in sel]
    gt_paths = [os.path.join(a.D, "input", nm) for _, nm in tnames]
    print(f"[render_views] 訓練視角 {len(tnames)}／{len(want)}：{[nm for _, nm in tnames]}", flush=True)
dev = torch.device("cuda"); bg = torch.zeros(3, device=dev)
cols = []
labels = []
if tnames:
    gtcol = [np.zeros((a.H, a.W, 3), np.uint8)] * nv
    for gp in gt_paths:
        im = Image.open(gp).convert("RGB").resize((a.W, a.H))
        gtcol.append(np.array(im))
    cols.append(gtcol); labels.append("GT")
for spec in a.models:
    lab, pat = spec.split("=", 1); fs = sorted(x for x in glob.glob(pat) if "xyz" not in x)
    assert fs, f"找不到 {pat}"
    model, renderer, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(fs[-1], device=dev, eval_mode=True)
    for f_ in ("absgrad_gate", "bwd_sat_skip", "bwd_reduce", "record_reduce", "tile_cull", "lean_train"):
        setattr(renderer, f_, False)
    col = []
    with torch.no_grad():
        for i in range(nv):
            o = renderer(cams[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
            col.append((o.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
        for i, nm in tnames:
            o = renderer(tcams[i].to_device(dev), model, bg_color=bg)["render"].clamp(0, 1)
            col.append(np.array(Image.fromarray((o.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).resize((a.W, a.H))))
    cols.append(col); labels.append(f"M{len(labels) + (0 if tnames else 1)}"); print(f"[render_views] 欄 {labels[-1]} = {lab}（{fs[-1]}）", flush=True)
    del model, renderer; torch.cuda.empty_cache()
rows = [v[0] for v in views] + [f"{'test' if a.test else 'train'} {os.path.splitext(nm)[0]}" for _, nm in tnames]
canvas = Image.new("RGB", (a.W * len(cols), (a.H + 24) * len(rows) + 24), (20, 20, 20))
dr = ImageDraw.Draw(canvas)
for ci, (lab, col) in enumerate(zip(labels, cols)):
    dr.text((ci * a.W + 6, 4), lab, fill=(255, 255, 0))
    for ri, img in enumerate(col):
        canvas.paste(Image.fromarray(img), (ci * a.W, 24 + ri * (a.H + 24) + 24))
        dr.text((ci * a.W + 6, 24 + ri * (a.H + 24) + 4), rows[ri], fill=(200, 200, 200))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
canvas.save(a.out)
print(f"[render_views] {a.out}｜{len(cols)} 欄 x {len(rows)} 列｜鏡頭高度中位 {hmed:.3f}｜海面磚重心 {sea_c.round(3)}｜鏡頭重心 {cam_c.round(3)}", flush=True)
