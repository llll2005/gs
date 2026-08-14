"""Two questions about the outputs the renderer computes and the loss no longer consumes.

With lambda_normal 0 (the winning ablation) and lambda_dist unset (default 0), the renderer still
produces every step: `depth_to_normal` (unproject + cross products over a 1600x900 depth map),
the CUDA distortion accumulation inside the blend loop, and `view_normal`, which no file outside
the renderer reads at all.

A. WHAT DOES IT COST?  Time one forward with the auxiliary work and one without. If it is a few
   percent, skip it; if it is 15%, removing it is worth doing on its own.

B. IS IT A SIGNAL WE ALREADY HAVE?  `rend_dist` is a per-pixel measure of how spread the samples
   are along a ray -- that is overdraw, the c_i this project hunted for days. But we already have a
   per-primitive footprint count (`num_covered_pixels`), so the question is whether rend_dist adds
   anything. Correlate it against per-pixel error and against the alpha map.
   ⚠ If rend_dist tracks the footprint count closely it is not new information, whatever its name.
"""
import argparse, glob, os, re, sys, time
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
from eval_official_test import load_test_cameras
from internal.utils.gaussian_model_loader import GaussianModelLoader

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="dssim05_b12")
ap.add_argument("--block", type=int, default=12)
ap.add_argument("--views", type=int, default=6)
ap.add_argument("--reps", type=int, default=8)
a = ap.parse_args()

D = "data/matrix_city/aerial/train/block_all"
names, cams = load_test_cameras(D, 1.2)
want = {l.strip() for l in open(f"{D}/partition/partitions-dim_5_5_visibility_0.08/002_002.txt") if l.strip()}
idx = [i for i, n in enumerate(names) if n in want][:: max(1, len(want) // a.views)][:a.views]
files = sorted(f for f in os.listdir(D + "/images_1.2") if f.endswith(".png"))

ck = glob.glob(f"outputs/{a.run}/blocks/block_{a.block}/checkpoints/*.ckpt")
ck.sort(key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
m, rend, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(ck[-1], "cuda", eval_mode=True)
bg = torch.zeros(3, device="cuda")
print(f"[{a.run}] N={m.get_xyz.shape[0]:,}  {len(idx)} 視角")

# --- A. cost of the auxiliary outputs ---------------------------------------------------------
def timeit(fn, reps):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps

cam = cams[idx[len(idx)//2]].to_device("cuda")
with torch.no_grad():
    full = timeit(lambda: rend(cam, m, bg_color=bg), a.reps)
    # record_transmittance skips the colour path AND all the auxiliary maps -> lower bound on
    # what is spent outside the core rasterisation
    lean = timeit(lambda: rend(cam, m, bg_color=bg, record_transmittance=True), a.reps)
print(f"\n[A 成本] 完整 forward {1000*full:.1f} ms   精簡（跳過顏色與輔助圖）{1000*lean:.1f} ms"
      f"   差 {100*(full-lean)/full:.1f}%")
print("  ⚠ 精簡路徑同時跳過顏色計算，所以這是輔助圖成本的【上界】，不是它單獨的成本。")

# --- B. is rend_dist new information? ----------------------------------------------------------
from PIL import Image
R, E, A_, C = [], [], [], []
with torch.no_grad():
    for i in idx:
        c = cams[i].to_device("cuda")
        o = rend(c, m, bg_color=bg)
        gt = torch.from_numpy(np.array(Image.open(f"{D}/images_1.2/{files[int(names[i][:-4])]}").convert("RGB")
              .resize((o["render"].shape[2], o["render"].shape[1])), np.uint8)).float().permute(2,0,1).cuda()/255.
        err = ((o["render"].clamp(0,1) - gt) ** 2).mean(0)
        k = 16
        pool = lambda x: torch.nn.functional.avg_pool2d(x[None,None], k)[0,0].flatten()
        R.append(pool(o["rend_dist"][0])); E.append(pool(err)); A_.append(pool(o["rend_alpha"][0]))
        _, cov = rend(c, m, bg_color=bg, record_transmittance=True, record_coverage=True)
        C.append(float(cov.sum()))
        del o; torch.cuda.empty_cache()
r = torch.cat(R).cpu().numpy(); e = torch.cat(E).cpu().numpy(); al = torch.cat(A_).cpu().numpy()
cc = lambda x, y: float(np.corrcoef(x, y)[0, 1])
print(f"\n[B 訊號] {len(r):,} 個 {k}x{k} 格")
print(f"  corr(rend_dist, 逐像素誤差) = {cc(r, e):+.3f}   ← 若高，它是免費的誤差偵測器")
print(f"  corr(rend_dist, alpha)      = {cc(r, al):+.3f}   ← 若高，它只是在量覆蓋，不是新資訊")
print(f"  rend_dist 值域 {r.min():.4f} ~ {r.max():.4f}  中位 {np.median(r):.4f}")
