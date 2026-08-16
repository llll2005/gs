"""Why do a handful of views stay at 17 dB while the rest reach 25-29?

`1729` has scored 17.3 / 17.4 / 17.5 across three recipes that differ by the GT-mapping fix, the
normal prior, the depth loss and the SSIM weight. It is immune to everything tried, so it is not a
tuning problem and another parameter sweep will not touch it. Same for `2999` (18.9) and `1607`
(19.2). Meanwhile `2643` went 26.8 -> 29.0 on the same changes.

Candidate explanations, each with a measurable signature. Nothing here needs the GPU: it reads the
saved side-by-side renders plus the camera geometry.

  content     the view is simply denser -- more GT gradient, more small structure
  altitude    higher camera, so each pixel spans more ground and needs finer primitives
  obliquity   looking sideways at vertical facades, which a nadir-flown capture sees at grazing
              angles and 2D surfels are badly placed to represent
  support     few training cameras look at what this view looks at, so the geometry is
              under-constrained regardless of capacity
  ownership   the frame is mostly other blocks' territory (the partition threshold is 8% content),
              so this block's model was never asked to represent most of what is on screen

The last one was checked once before on a different model and came back near zero (corr -0.04), but
that was under the GT bug, so it is re-measured here rather than assumed.
"""
import argparse, glob, os, re, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "."); sys.path.insert(0, "tools")
from eval_official_test import load_test_cameras

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="noprior_b12")
ap.add_argument("--block", type=int, default=12)
ap.add_argument("--block_dim", type=int, nargs=2, default=[5, 5])
ap.add_argument("--data", default="data/matrix_city/aerial/train/block_all")
a = ap.parse_args()

names, cams = load_test_cameras(a.data, 1.2)
by, bx = a.block // a.block_dim[0], a.block % a.block_dim[0]
want = {l.strip() for l in open(os.path.join(a.data, "partition",
        f"partitions-dim_{a.block_dim[0]}_{a.block_dim[1]}_visibility_0.08", f"{bx:03d}_{by:03d}.txt")) if l.strip()}
blk = [i for i, n in enumerate(names) if n in want]
C = np.stack([(cams[i].world_to_camera.T.inverse()[:3, 3]).cpu().numpy() for i in blk])
D = np.stack([(cams[i].world_to_camera[:3, :3].T @ np.array([0, 0, 1.0])) for i in blk])
ctr = C.mean(0)

ck = glob.glob(f"outputs/{a.run}/blocks/block_{a.block}/checkpoints/*.ckpt")
ck.sort(key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
sd = torch.load(ck[-1], map_location="cpu")["state_dict"]
xyz = sd[[k for k in sd if k.endswith("means")][0]]

d = sorted(glob.glob(f"outputs/{a.run}/blocks/block_{a.block}/test/*/"))[-1]
rows = []
for p in sorted(glob.glob(d + "*.png")):
    nm = os.path.basename(p).replace(".png.png", ".png")
    if nm not in names: continue
    i = names.index(nm); k = blk.index(i) if i in blk else None
    if k is None: continue
    im = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.
    w = im.shape[1] // 2
    gt, rd = im[:, :w], im[:, w:]
    psnr = -10 * np.log10(max(float(((gt - rd) ** 2).mean()), 1e-12))
    grad = float(np.abs(np.diff(gt, axis=0)).mean() + np.abs(np.diff(gt, axis=1)).mean())
    tilt = float(np.degrees(np.arccos(np.clip(-D[k][2], -1, 1))))     # 0 = 正下方
    # 有多少訓練相機「看向類似的地方」：位置近且朝向近
    sim = int((((np.linalg.norm(C - C[k], axis=1) < 1.0) & (D @ D[k] > 0.9)).sum()))
    proj = cams.full_projection[i]
    pr = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ proj
    ok = pr[:, 3] > 1e-6
    uv = pr[ok, :2] / pr[ok, 3:4]
    own = float(((uv.abs() < 1).all(1)).float().mean())
    rows.append((psnr, grad, C[k][2], tilt, sim, own, nm))

rows.sort()
P = np.array([r[0] for r in rows])
cols = [("GT梯度", 1), ("相機高度", 2), ("傾角°", 3), ("鄰近同向相機數", 4), ("本塊幾何在畫面內", 5)]
print(f"[{a.run}] {len(rows)} 視角   PSNR {P.min():.1f}~{P.max():.1f}")
print("\n相關係數 vs PSNR：")
for nm, j in cols:
    v = np.array([r[j] for r in rows], dtype=float)
    print(f"  {nm:<18} {np.corrcoef(v, P)[0,1]:+.3f}")
print(f"\n{'PSNR':>6}{'GT梯度':>9}{'高度':>7}{'傾角':>7}{'同向相機':>9}{'畫面內':>8}  檔名")
for r in rows[:4] + rows[-3:]:
    print(f"{r[0]:>6.1f}{r[1]:>9.4f}{r[2]:>7.2f}{r[3]:>7.1f}{r[4]:>9}{100*r[5]:>7.1f}%  {r[6]}")
print("\n  相關最強的那一欄就是主因；若都很弱，則失敗是這些軸都沒捕捉到的東西。")
